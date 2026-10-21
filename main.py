import cv2
import numpy as np
import os
import time
from ultralytics import YOLO


# ============================================================
# 1. SETTINGS
# ============================================================

# Laptop webcam device index
WEBCAM_INDEX = 0

# YOLO segmentation model
MODEL_PATH = "yolo26n-seg.pt"

# ByteTrack configuration
TRACKER = "bytetrack.yaml"

# Confidence threshold
CONFIDENCE = 0.40

# Image size
IMG_SIZE = 640

# Output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output video
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "yolo_segmentation_bytetrack_output.mp4"
)


# ============================================================
# 2. LOAD YOLO SEGMENTATION MODEL
# ============================================================

print("Loading YOLO segmentation model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# 3. CHECK DEVICE
# ============================================================

try:
    import torch

    if torch.cuda.is_available():
        DEVICE = 0
        print("GPU available:")
        print(torch.cuda.get_device_name(0))
    else:
        DEVICE = "cpu"
        print("GPU not available. Using CPU.")

except Exception:
    DEVICE = "cpu"
    print("Could not check GPU. Using CPU.")


# ============================================================
# 4. CONNECT TO MOBILE IP CAMERA
# ============================================================

print("\nConnecting to laptop webcam...")
print("Webcam index:", WEBCAM_INDEX)

cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("\nERROR: Cannot open laptop webcam.")
    print("Check:")
    print("1. The webcam is connected and not used by another app.")
    print("2. The camera permission is enabled for Python/OpenCV.")
    print("3. Try changing WEBCAM_INDEX to 0, 1, or 2.")

    exit()


print("Laptop webcam connected successfully.")


# ============================================================
# 5. GET VIDEO INFORMATION
# ============================================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or np.isnan(fps):
    fps = 20.0


print("\nVideo information:")
print("Width :", width)
print("Height:", height)
print("FPS   :", fps)


# ============================================================
# 6. CREATE VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 7. VARIABLES
# ============================================================

frame_count = 0

previous_time = time.time()


# ============================================================
# 8. MAIN HYBRID ALGORITHM
# ============================================================

print("\nStarting YOLO Segmentation + ByteTrack...")
print("Press Q to stop.\n")


while True:

    # --------------------------------------------------------
    # Read frame from phone
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("Failed to receive frame from webcam.")

        # Try to reconnect
        cap.release()

        time.sleep(1)

        cap = cv2.VideoCapture(WEBCAM_INDEX)

        if not cap.isOpened():
            cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

        continue


    frame_count += 1


    # --------------------------------------------------------
    # YOLO SEGMENTATION + BYTETRACK
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )


    result = results[0]


    # --------------------------------------------------------
    # Create output frame
    # --------------------------------------------------------

    output_frame = frame.copy()


    # --------------------------------------------------------
    # Check segmentation masks
    # --------------------------------------------------------

    if result.masks is not None:

        class_counts = {}

        masks = result.masks.data.cpu().numpy()


        # ----------------------------------------------------
        # Get tracking IDs
        # ----------------------------------------------------

        if result.boxes.id is not None:

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

        else:

            track_ids = [
                None
                for _ in range(len(masks))
            ]


        # ----------------------------------------------------
        # Get classes
        # ----------------------------------------------------

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Get confidence
        # ----------------------------------------------------

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )


        # ====================================================
        # PROCESS EVERY DETECTED OBJECT
        # ====================================================

        for i, mask in enumerate(masks):


            # ------------------------------------------------
            # Convert mask to binary
            # ------------------------------------------------

            mask = (
                mask > 0.5
            ).astype(np.uint8) * 255


            # ------------------------------------------------
            # Resize mask to original frame
            # ------------------------------------------------

            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


            # ------------------------------------------------
            # Find actual object contour
            # ------------------------------------------------

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )


            # ------------------------------------------------
            # Draw actual object-shaped border
            # ------------------------------------------------

            cv2.drawContours(
                output_frame,
                contours,
                -1,
                (0, 255, 0),
                3
            )


            # ------------------------------------------------
            # Object information
            # ------------------------------------------------

            class_id = class_ids[i]

            class_name = model.names[
                class_id
            ]

            confidence = confidences[i]

            track_id = track_ids[i]

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            display_name = f"{class_name.title()} {class_counts[class_name]}"


            # ------------------------------------------------
            # Create label
            # ------------------------------------------------

            label = (
                f"{display_name} "
                f"{confidence:.2f}"
            )




            # ------------------------------------------------
            # Find label position
            # ------------------------------------------------

            if len(contours) > 0:

                largest_contour = max(
                    contours,
                    key=cv2.contourArea
                )

                x, y, w, h = cv2.boundingRect(
                    largest_contour
                )


                # ------------------------------------------------
                # Label background
                # ------------------------------------------------

                text_size = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )[0]

                text_width = text_size[0]
                text_height = text_size[1]


                cv2.rectangle(
                    output_frame,
                    (
                        x,
                        max(
                            0,
                            y - text_height - 12
                        )
                    ),
                    (
                        x + text_width + 10,
                        y
                    ),
                    (0, 255, 0),
                    -1
                )


                # ------------------------------------------------
                # Label text
                # ------------------------------------------------

                cv2.putText(
                    output_frame,
                    label,
                    (
                        x + 5,
                        y - 7
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    elapsed_time = (
        current_time - previous_time
    )

    if elapsed_time > 0:

        current_fps = (
            1 / elapsed_time
        )

    else:

        current_fps = 0


    previous_time = current_time


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.rectangle(
        output_frame,
        (10, 10),
        (470, 60),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        output_frame,
        "YOLO Segmentation + ByteTrack",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    cv2.putText(
        output_frame,
        f"FPS: {current_fps:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SAVE OUTPUT VIDEO
    # ========================================================

    video_writer.write(
        output_frame
    )


    # ========================================================
    # SHOW LIVE OUTPUT
    # ========================================================

    cv2.imshow(
        "YOLO Segmentation + ByteTrack",
        output_frame
    )


    # ========================================================
    # STOP WITH Q
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\nStopping...")

        break


# ============================================================
# 9. RELEASE EVERYTHING
# ============================================================

cap.release()

video_writer.release()

cv2.destroyAllWindows()


print("\n========================================")
print("Processing finished")
print("Frames processed:", frame_count)
print("Output video:")
print(OUTPUT_VIDEO)
print("========================================")