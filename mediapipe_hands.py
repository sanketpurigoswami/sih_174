import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = r"F:\sih\models\hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


# Use your video file here
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Calculate timestamp in milliseconds
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    timestamp_ms = int((frame_number / fps) * 1000)

    # VIDEO mode requires timestamp
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    h, w, _ = frame.shape

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # Index fingertip = landmark 8
            index_tip = hand_landmarks[8]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            cv2.circle(
                frame,
                (index_x, index_y),
                8,
                (0, 0, 255),
                -1
            )

    cv2.imshow("MediaPipe Hands - VIDEO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1


cap.release()
detector.close()
cv2.destroyAllWindows()