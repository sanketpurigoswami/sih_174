import cv2
import math
import mediapipe as mp
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. PATHS
# ============================================================

HAND_MODEL_PATH = r"F:\sih\models\hand_landmarker.task"

# Your trained YOLO model
YOLO_MODEL_PATH = r"F:\sih\runs\detect\train\weights\best.pt"

# Change this if you want to use a video file instead of webcam.
VIDEO_SOURCE = 1
# Example:
# VIDEO_SOURCE = r"F:\sih\screw.mp4"


# ============================================================
# 2. SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.25

# Start with 50 pixels as a simple proximity threshold.
# We will tune this later using your actual camera setup.
PROXIMITY_THRESHOLD = 50


# ============================================================
# 3. PROXIMITY FUNCTIONS
# ============================================================

def point_to_box_distance(px, py, box):
    """
    Calculate the distance from a point to the nearest edge
    of a bounding box.

    If the point is inside the box, distance = 0.

    box = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box

    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)

    return math.sqrt(dx * dx + dy * dy)


def point_inside_box(px, py, box):
    """Return True if the point lies inside the bounding box."""
    x1, y1, x2, y2 = box

    return x1 <= px <= x2 and y1 <= py <= y2


def draw_distance_info(frame, point, box, distance, label):
    """
    Draw the hand-to-object line and distance on the frame.
    """
    px, py = point
    x1, y1, x2, y2 = box

    # Object bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    # Line from fingertip to nearest point region
    cv2.line(
        frame,
        (px, py),
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (255, 255, 0),
        2
    )

    # Distance text
    cv2.putText(
        frame,
        f"{label}: {distance:.1f}px",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# 4. LOAD YOLO
# ============================================================

print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)


# ============================================================
# 5. LOAD MEDIAPIPE HAND LANDMARKER
# ============================================================

print("Loading MediaPipe Hand Landmarker...")

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 6. OPEN CAMERA / VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open camera/video source")


fps = cap.get(cv2.CAP_PROP_FPS)

# Webcam FPS can sometimes return 0.
if fps <= 0:
    fps = 30.0

frame_number = 0


# ============================================================
# 7. MAIN LOOP
# ============================================================

cv2.namedWindow(
    "YOLO + MediaPipe Hand Proximity",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "YOLO + MediaPipe Hand Proximity",
    1280,
    720
)

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video / could not read frame.")
        break

    h, w, _ = frame.shape

    # --------------------------------------------------------
    # A. YOLO OBJECT DETECTION
    # --------------------------------------------------------

    yolo_results = yolo_model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    # Store detections so we can compare them with hands.
    detections = []

    for result in yolo_results:

        if result.boxes is None:
            continue

        for box_data in result.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = box_data.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            confidence = float(box_data.conf[0])

            class_id = int(box_data.cls[0])

            label = yolo_model.names[class_id]

            detections.append({
                "label": label,
                "box": (x1, y1, x2, y2),
                "confidence": confidence
            })


    # --------------------------------------------------------
    # B. MEDIAPIPE HAND LANDMARK DETECTION
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (frame_number / fps) * 1000
    )

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # C. PROCESS EACH HAND
    # --------------------------------------------------------

    if hand_result.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):

            # ------------------------------------------------
            # Get important landmarks
            # ------------------------------------------------

            wrist = hand_landmarks[0]
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]

            # Convert normalized MediaPipe coordinates
            # to image pixel coordinates.

            wrist_xy = (
                int(wrist.x * w),
                int(wrist.y * h)
            )

            thumb_xy = (
                int(thumb_tip.x * w),
                int(thumb_tip.y * h)
            )

            index_xy = (
                int(index_tip.x * w),
                int(index_tip.y * h)
            )

            middle_xy = (
                int(middle_tip.x * w),
                int(middle_tip.y * h)
            )


            # ------------------------------------------------
            # Draw all 21 hand landmarks
            # ------------------------------------------------

            for landmark in hand_landmarks:

                lx = int(landmark.x * w)
                ly = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (lx, ly),
                    3,
                    (0, 255, 0),
                    -1
                )


            # Highlight index fingertip
            cv2.circle(
                frame,
                index_xy,
                8,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # D. CHECK PROXIMITY TO EACH YOLO OBJECT
            # ------------------------------------------------

            for detection in detections:

                label = detection["label"]
                box = detection["box"]
                confidence = detection["confidence"]

                distance = point_to_box_distance(
                    index_xy[0],
                    index_xy[1],
                    box
                )

                inside = point_inside_box(
                    index_xy[0],
                    index_xy[1],
                    box
                )

                # Draw object + distance
                draw_distance_info(
                    frame,
                    index_xy,
                    box,
                    distance,
                    label
                )

                # Determine simple proximity state
                if inside:
                    status = "CONTACT / INSIDE BOX"

                elif distance < PROXIMITY_THRESHOLD:
                    status = "NEAR"

                else:
                    status = "FAR"


                # Print useful information in terminal
                print(
                    f"Hand {hand_index} -> "
                    f"{label} | "
                    f"distance={distance:.1f}px | "
                    f"status={status}"
                )


                # Display state near the object
                x1, y1, x2, y2 = box

                cv2.putText(
                    frame,
                    status,
                    (x1, min(h - 10, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )


            # ------------------------------------------------
            # Optional: display fingertip coordinates
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Index: {index_xy}",
                (10, 30 + hand_index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


    # --------------------------------------------------------
    # E. DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "YOLO + MediaPipe Hand Proximity",
        frame
    )


    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1


# ============================================================
# 8. CLEANUP
# ============================================================

cap.release()
hand_detector.close()
cv2.destroyAllWindows()

print("Finished.")
