from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

MODEL_PATH = r"F:\sih\runs\detect\train\weights\best.pt"
HAND_MODEL_PATH = r"F:\sih\models\hand_landmarker.task"
# VIDEO_INPUT_PATH = r"F:\sih\tester.mp4"
# VIDEO_OUTPUT_PATH = r"F:\sih\annotated_output.mp4"
STATE_LOG_PATH = r"F:\sih\object_states.txt"
FSM_EVENT_LOG_PATH = r"F:\sih\cv_observed.jsonl"
CAMERA_INDEX = 1

CONFIDENCE = 0.35
IOU = 0.45

# Movement is normalized by the current object's bounding-box diagonal.
MOVEMENT_HISTORY_SIZE = 5
MOVE_ON_THRESHOLD = 0.04
MOVE_OFF_THRESHOLD = 0.02

# Fingertip/object distancing logic is unchanged.
CONTACT_DISTANCE = 30.0
GRASP_FINGERS_REQUIRED = 4

PICKUP_SIZE_CHANGE = 0.4
PICKUP_OBJECT_MOVEMENT = 8.0
PICKUP_CONFIRM_FRAMES = 3
BASELINE_FRAMES = 8

PLACE_OVERLAP_THRESHOLD = 0.20
PLACE_STABLE_FRAMES = 5

RELEASE_DISTANCE_MULTIPLIER = 1.35
RELEASE_CONFIRM_FRAMES = 3

MAX_FAILED_FRAMES = 30

CLASS_NAMES = {
    0: "orange_circular_cap",
    1: "red_computer_mouse",
    2: "wireless_earbuds_case",
    3: "dark_blue_square",
    4: "yellow_square",
    5: "pink_square",
    6: "green_square",
    7: "white_box",
    8: "person",
    9: "hand",
}

OBJECT_IDS = {0, 1, 2, 3}
PLACE_IDS = {4, 5, 6, 7}
HAND_ID = 9


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[float, float]


@dataclass
class ObjectState:
    class_id: int
    current_center: Optional[Tuple[float, float]] = None
    previous_center: Optional[Tuple[float, float]] = None
    current_area: float = 0.0
    previous_area: float = 0.0
    baseline_area: Optional[float] = None
    baseline_count: int = 0
    action: str = "IDLE"
    grasped: bool = False
    picked_up: bool = False
    placed: bool = False
    pickup_counter: int = 0
    stable_counter: int = 0
    release_counter: int = 0
    target_place_id: Optional[int] = None
    last_seen_frame: int = -1

    # Five-frame normalized movement history for stable MOVE detection.
    movement_history: Deque[float] = None
    moving: bool = False

    def __post_init__(self):
        self.movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalized_object_movement(movement, box):
    """Normalize center movement by the object's bounding-box diagonal."""
    x1, y1, x2, y2 = box
    diagonal = math.hypot(x2 - x1, y2 - y1)

    if diagonal <= 0:
        return 0.0

    return movement / diagonal


def update_movement_state(state, normalized_movement):
    """
    Maintain a five-frame median movement value with hysteresis.

    MOVE turns on only when the median reaches MOVE_ON_THRESHOLD.
    MOVE turns off only when the median falls to MOVE_OFF_THRESHOLD.
    Between those thresholds, the previous movement state is retained.
    """
    state.movement_history.append(normalized_movement)

    median_movement = sorted(state.movement_history)[
        len(state.movement_history) // 2
    ]

    if not state.moving and median_movement >= MOVE_ON_THRESHOLD:
        state.moving = True
    elif state.moving and median_movement <= MOVE_OFF_THRESHOLD:
        state.moving = False

    return median_movement


def point_to_box_distance(px, py, box):
    """Distance from a point to the nearest edge of a bounding box."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


def fingertip_box_distances(fingertips, box):
    return [point_to_box_distance(x, y, box) for x, y in fingertips]

def record_red_mouse_state(
    frame,
    hand_result,
    frame_number,
    mouse_state,
    output_file="red_mouse_states.txt"
):
    """
    Append the red computer mouse fingertip coordinates and state
    to a text file.

    Fingertips:
        thumb, index, middle, ring, pinky

    If no hand is detected, all coordinates are (-1, -1).
    """

    fingertip_indices = (4, 8, 12, 16, 20)
    fingertip_names = ("thumb", "index", "middle", "ring", "pinky")

    # Default: no hand detected
    fingertips = {
        name: (-1, -1)
        for name in fingertip_names
    }

    # Extract fingertips from the first detected hand
    if hand_result.hand_landmarks:
        hand_landmarks = hand_result.hand_landmarks[0]

        h, w = frame.shape[:2]

        for name, index in zip(fingertip_names, fingertip_indices):
            landmark = hand_landmarks[index]

            fingertips[name] = (
                int(landmark.x * w),
                int(landmark.y * h)
            )

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(
            f"Frame {frame_number} | "
            f"Thumb={fingertips['thumb']} | "
            f"Index={fingertips['index']} | "
            f"Middle={fingertips['middle']} | "
            f"Ring={fingertips['ring']} | "
            f"Pinky={fingertips['pinky']} | "
            f"State={mouse_state}\n"
        )


def box_area(box):
    x1, y1, x2, y2 = box
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def overlap_ratio(object_box, place_box):
    area = box_area(object_box)
    if area <= 0:
        return 0.0
    return intersection_area(object_box, place_box) / area


def extract_detections(result):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        confidence = float(confidence)
        if confidence < CONFIDENCE:
            continue

        class_id = int(class_id)
        x1, y1, x2, y2 = map(int, box)
        bbox = (x1, y1, x2, y2)

        detections.append(
            Detection(
                class_id=class_id,
                name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                confidence=confidence,
                box=bbox,
                center=center_of_box(bbox),
            )
        )

    return detections


def find_hand(detections):
    hands = [d for d in detections if d.class_id == HAND_ID]
    return max(hands, key=lambda d: d.confidence) if hands else None


def find_objects(detections):
    return [d for d in detections if d.class_id in OBJECT_IDS]


def find_places(detections):
    return [d for d in detections if d.class_id in PLACE_IDS]


def get_best_place(obj, places):
    best_place = None
    best_overlap = 0.0

    for place in places:
        overlap = overlap_ratio(obj.box, place.box)
        if overlap > best_overlap:
            best_overlap = overlap
            best_place = place

    return best_place, best_overlap


class ActionDetector:
    def __init__(self):
        self.states: Dict[int, ObjectState] = {}

    def get_state(self, class_id):
        if class_id not in self.states:
            self.states[class_id] = ObjectState(class_id=class_id)
        return self.states[class_id]

    def update(self, obj, fingertips, places, frame_number):
        state = self.get_state(obj.class_id)

        state.previous_center = state.current_center
        state.current_center = obj.center
        state.last_seen_frame = frame_number

        state.previous_area = state.current_area
        state.current_area = box_area(obj.box)

        object_movement = point_distance(
            state.previous_center, state.current_center
        )

        normalized_movement = normalized_object_movement(
            object_movement,
            obj.box,
        )

        smoothed_normalized_movement = update_movement_state(
            state,
            normalized_movement,
        )

        object_moving = state.moving

        if not state.picked_up and state.baseline_count < BASELINE_FRAMES:
            if state.baseline_area is None:
                state.baseline_area = state.current_area
            else:
                state.baseline_area = (
                    0.85 * state.baseline_area
                    + 0.15 * state.current_area
                )
            state.baseline_count += 1

        if state.baseline_area and state.baseline_area > 0:
            size_ratio = state.current_area / state.baseline_area
            relative_size_change = abs(size_ratio - 1.0)
        else:
            relative_size_change = 0.0

        finger_distances = (
            fingertip_box_distances(fingertips, obj.box)
            if fingertips
            else []
        )
        close_fingers = sum(
            distance <= CONTACT_DISTANCE for distance in finger_distances
        )
        fingers_in_contact = close_fingers >= GRASP_FINGERS_REQUIRED
        fingers_separated = (
            not fingertips
            or (
                bool(finger_distances)
                and all(
                    distance >= CONTACT_DISTANCE * RELEASE_DISTANCE_MULTIPLIER
                    for distance in finger_distances
                )
            )
        )


        place, place_overlap = get_best_place(obj, places)

        if not state.grasped:
            if fingers_in_contact:
                state.grasped = True
                state.picked_up = True
                state.action = "GRASP"
                state.pickup_counter = 0
                if place is not None:
                    state.target_place_id = place.class_id
                return state

            if object_moving:
                state.action = "MOVE"
            else:
                state.action = "IDLE"

            return state

        if fingers_separated and not object_moving:
            state.release_counter += 1
        else:
            state.release_counter = max(0, state.release_counter - 1)

        if state.release_counter >= RELEASE_CONFIRM_FRAMES:
            state.action = "RELEASE"
            state.grasped = False
            state.picked_up = False
            state.placed = False
            state.pickup_counter = 0
            state.stable_counter = 0
            state.release_counter = 0
            return state

        if object_moving:
            state.action = "MOVE"
        else:
            state.action = "GRASP"

        return state


def draw_detection(frame, detection, action=None):
    x1, y1, x2, y2 = detection.box
    label = f"{detection.name} {detection.confidence:.2f}"
    if action:
        label += f" | {action}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_fsm_event(
    event_log,
    timestamp,
    frame_number,
    obj,
    state,
    action,
    place=None,
):
    """Write one semantic CV event using the FSM vocabulary."""
    target_name = None
    target_id = None

    if place is not None:
        target_name = place.name
        target_id = place.class_id
    elif state.target_place_id is not None:
        target_id = state.target_place_id
        target_name = CLASS_NAMES.get(target_id)

    x1, y1, x2, y2 = obj.box

    event = {
        "timestamp": round(float(timestamp), 3),
        "frame": int(frame_number),
        "activity": action,
        "action": action,
        "object": obj.name,
        "object_id": int(obj.class_id),
        "object_confidence": round(float(obj.confidence), 4),
        "target": target_name,
        "target_id": target_id,
        "location": {
            "x": round(float(obj.center[0]), 2),
            "y": round(float(obj.center[1]), 2),
        },
        "bbox": [x1, y1, x2, y2],
        "confidence": round(float(obj.confidence), 4),
        "source": "cv_rules",
    }

    event_log.write(json.dumps(event) + "\\n")
    event_log.flush()


def draw_action_banner(frame, action, object_name, target_name=None):
    height, width = frame.shape[:2]
    text = f"ACTION: {action}    OBJECT: {object_name}"
    if target_name:
        text += f"    TARGET: {target_name}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        text,
        (20, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def open_camera():
    print("Opening ASUS GlideX webcam...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    success, frame = cap.read()

    if not success or frame is None:
        cap.release()
        return None

    print("ASUS GlideX webcam connected.")
    return cap


def main():
    print("=" * 70)
    print("LIVE YOLO RULE-BASED ACTION DETECTION")
    print("=" * 70)

    print(f"Loading model: {MODEL_PATH}")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    print("YOLO loaded successfully.")
    print("Classes:", model.names)

    print(f"Loading MediaPipe Hand Landmarker: {HAND_MODEL_PATH}")
    try:
        base_options = python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(
            hand_options
        )
    except Exception as e:
        print(f"ERROR loading MediaPipe Hand Landmarker: {e}")
        return

    cap = open_camera()

    while cap is None:
        print("Could not open webcam. Retrying in 2 seconds...")
        time.sleep(2)
        cap = open_camera()

    detector = ActionDetector()

    # FSM event stream: one semantic transition per line (JSONL).
    event_log = open(FSM_EVENT_LOG_PATH, "a", encoding="utf-8")
    last_fsm_action: Dict[int, str] = {}

    print(f"FSM event log: {FSM_EVENT_LOG_PATH}")

    frame_number = 0
    failed_frames = 0
    hand_landmarker_fps = 30.0
    hand_fingertips = []

    previous_time = cv2.getTickCount()
    fps = 0.0
    cv2.namedWindow("YOLO Live Action Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live Action Detection", 800, 450)

    print()
    print("CAMERA STARTED")
    print("Q = quit")
    print("The camera will keep running until you press Q.")
    print()

    while True:
        success, frame = cap.read()

        if not success or frame is None:
            failed_frames += 1
            print(
                f"\rCamera read failed "
                f"{failed_frames}/{MAX_FAILED_FRAMES}",
                end="",
                flush=True,
            )
            time.sleep(0.03)

            if failed_frames >= MAX_FAILED_FRAMES:
                print("\nTrying to reconnect webcam...")
                cap.release()
                cv2.destroyAllWindows()

                while True:
                    time.sleep(1)
                    cap = open_camera()

                    if cap is not None:
                        failed_frames = 0
                        print("Webcam reconnected.")
                        break

                    print("Reconnect failed. Retrying...")

            continue

        failed_frames = 0
        frame_number += 1

        try:
            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU,
                verbose=False,
            )
        except Exception as e:
            print(f"\nYOLO inference error: {e}")
            continue

        detections = extract_detections(results[0])

        # MediaPipe hand landmark detection.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        timestamp_ms = int((frame_number / hand_landmarker_fps) * 1000)
        hand_result = hand_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hand_fingertips = []
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                fingertip_indices = (4, 8, 12, 16, 20)
                fingertips = [
                    (
                        int(hand_landmarks[index].x * frame.shape[1]),
                        int(hand_landmarks[index].y * frame.shape[0]),
                    )
                    for index in fingertip_indices
                ]
                hand_fingertips.extend(fingertips)

            for point in hand_fingertips:
                cv2.circle(frame, point, 6, (0, 0, 255), -1)

        hand = find_hand(detections)
        objects = find_objects(detections)
        places = find_places(detections)

        object_states = []

        for obj in objects:
            state = detector.update(
                obj, hand_fingertips, places, frame_number
            )
            object_states.append((obj, state))

        red_mouse_state = "IDLE"

        for obj, state in object_states:
            if obj.class_id == 1:   # red_computer_mouse
                red_mouse_state = state.action
                break

        record_red_mouse_state(
            frame,
            hand_result,
            frame_number,
            red_mouse_state
        )

        # Write only semantic action transitions to the FSM event stream.
        # Repeated identical states are not written every frame.
        timestamp_seconds = time.time()
        for obj, state in object_states:
            previous_action = last_fsm_action.get(obj.class_id, "IDLE")
            current_action = state.action

            if current_action != previous_action:
                place, _ = get_best_place(obj, places)
                write_fsm_event(
                    event_log,
                    timestamp_seconds,
                    frame_number,
                    obj,
                    state,
                    current_action,
                    place,
                )
                last_fsm_action[obj.class_id] = current_action

        for detection in detections:
            action = None

            for obj, state in object_states:
                if (
                    detection.class_id == obj.class_id
                    and detection.box == obj.box
                ):
                    action = state.action
                    break

            draw_detection(frame, detection, action)

        if hand_fingertips:
            for obj, state in object_states:
                distances = fingertip_box_distances(
                    hand_fingertips, obj.box
                )
                for fingertip, distance in zip(hand_fingertips, distances):
                    if distance <= CONTACT_DISTANCE:
                        cv2.line(
                            frame,
                            fingertip,
                            tuple(map(int, obj.center)),
                            (255, 255, 255),
                            2,
                        )

        if object_states:
            priority = {
                "RELEASE": 3,
                "MOVE": 2,
                "GRASP": 1,
                "IDLE": 0,
            }

            obj, state = max(
                object_states,
                key=lambda pair: priority.get(pair[1].action, 0),
            )

            draw_action_banner(
                frame, state.action, obj.name
            )

            movement = point_distance(
                state.previous_center, state.current_center
            )

            if state.baseline_area:
                size_change = abs(
                    state.current_area / state.baseline_area - 1.0
                )
            else:
                size_change = 0.0

            finger_distances = (
                fingertip_box_distances(hand_fingertips, obj.box)
                if hand_fingertips
                else []
            )
            hand_distance = min(finger_distances) if finger_distances else float("inf")

            cv2.putText(
                frame,
                f"Movement: {movement:.1f}px",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Size change: {size_change * 100:.1f}%",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"Finger distance: {hand_distance:.1f}px"
                    if hand_fingertips
                    else "Finger distance: N/A"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Grasp fingers: {sum(
                    distance <= CONTACT_DISTANCE
                    for distance in fingertip_box_distances(hand_fingertips, obj.box)
                ) if hand_fingertips else 0}/{GRASP_FINGERS_REQUIRED}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        else:
            draw_action_banner(frame, "IDLE", "No object detected")

        current_time = cv2.getTickCount()
        elapsed = (
            current_time - previous_time
        ) / cv2.getTickFrequency()
        previous_time = current_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            fps = (
                instant_fps
                if fps == 0
                else 0.9 * fps + 0.1 * instant_fps
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "LIVE",
            (frame.shape[1] - 90, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO Live Action Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("\nStopping webcam...")
    cap.release()
    event_log.close()
    hand_detector.close()
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    main()
