from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

MODEL_PATH = r"F:\sih\runs\detect\train\weights\best.pt"
CAMERA_INDEX = 1

CONFIDENCE = 0.35
IOU = 0.45

MOVEMENT_THRESHOLD = 10.0
CONTACT_DISTANCE = 100.0

PICKUP_SIZE_CHANGE = 0.08
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
    picked_up: bool = False
    placed: bool = False
    pickup_counter: int = 0
    stable_counter: int = 0
    release_counter: int = 0
    target_place_id: Optional[int] = None
    last_seen_frame: int = -1


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


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

    def update(self, obj, hand, places, frame_number):
        state = self.get_state(obj.class_id)

        state.previous_center = state.current_center
        state.current_center = obj.center
        state.last_seen_frame = frame_number

        state.previous_area = state.current_area
        state.current_area = box_area(obj.box)

        object_movement = point_distance(
            state.previous_center, state.current_center
        )
        object_moving = object_movement >= MOVEMENT_THRESHOLD

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

        hand_distance = point_distance(
            hand.center if hand else None, obj.center
        )
        in_contact = hand_distance <= CONTACT_DISTANCE

        place, place_overlap = get_best_place(obj, places)

        if not state.picked_up:
            size_changed = relative_size_change >= PICKUP_SIZE_CHANGE
            object_moved_enough = object_movement >= PICKUP_OBJECT_MOVEMENT

            pickup_signal = (
                in_contact and size_changed and object_moved_enough
            )

            if pickup_signal:
                state.pickup_counter += 1
            else:
                state.pickup_counter = max(0, state.pickup_counter - 1)

            if state.pickup_counter >= PICKUP_CONFIRM_FRAMES:
                state.picked_up = True
                state.action = "PICKUP"
                state.pickup_counter = 0

                if place is not None:
                    state.target_place_id = place.class_id

                state.baseline_count = BASELINE_FRAMES
                return state

            if in_contact:
                state.action = "GRASP"
            elif object_moving:
                state.action = "MOVE"
            else:
                state.action = "IDLE"

            return state

        if state.picked_up and not state.placed:
            if object_moving:
                state.action = "TRANSPORT"
                state.stable_counter = 0
            else:
                state.stable_counter += 1

            valid_place = (
                place is not None
                and place_overlap >= PLACE_OVERLAP_THRESHOLD
            )
            stable_enough = state.stable_counter >= PLACE_STABLE_FRAMES

            if valid_place and stable_enough:
                state.placed = True
                state.action = "PLACE"
                state.target_place_id = place.class_id
                state.stable_counter = 0
                return state

            state.action = "PLACING" if state.stable_counter > 0 else "TRANSPORT"
            return state

        if state.placed:
            release_distance = CONTACT_DISTANCE * RELEASE_DISTANCE_MULTIPLIER

            hand_separated = (
                hand is not None and hand_distance >= release_distance
            )
            object_stable = not object_moving

            if hand_separated and object_stable:
                state.release_counter += 1
            else:
                state.release_counter = max(0, state.release_counter - 1)

            if state.release_counter >= RELEASE_CONFIRM_FRAMES:
                state.action = "RELEASE"
                state.picked_up = False
                state.placed = False
                state.pickup_counter = 0
                state.stable_counter = 0
                state.release_counter = 0
                return state

            state.action = "PLACE"
            return state

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

    cap = open_camera()

    while cap is None:
        print("Could not open webcam. Retrying in 2 seconds...")
        time.sleep(2)
        cap = open_camera()

    detector = ActionDetector()
    frame_number = 0
    failed_frames = 0

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
        hand = find_hand(detections)
        objects = find_objects(detections)
        places = find_places(detections)

        object_states = []

        for obj in objects:
            state = detector.update(
                obj, hand, places, frame_number
            )
            object_states.append((obj, state))

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

        if hand is not None:
            for obj, state in object_states:
                distance = point_distance(hand.center, obj.center)

                if distance <= CONTACT_DISTANCE:
                    cv2.line(
                        frame,
                        tuple(map(int, hand.center)),
                        tuple(map(int, obj.center)),
                        (255, 255, 255),
                        2,
                    )

        if object_states:
            priority = {
                "RELEASE": 7,
                "PLACE": 6,
                "PLACING": 5,
                "TRANSPORT": 4,
                "PICKUP": 3,
                "GRASP": 2,
                "MOVE": 1,
                "IDLE": 0,
            }

            obj, state = max(
                object_states,
                key=lambda pair: priority.get(pair[1].action, 0),
            )

            target_name = (
                CLASS_NAMES.get(state.target_place_id)
                if state.target_place_id is not None
                else None
            )

            draw_action_banner(
                frame, state.action, obj.name, target_name
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

            hand_distance = (
                point_distance(hand.center, obj.center)
                if hand
                else float("inf")
            )

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
                    f"Hand distance: {hand_distance:.1f}px"
                    if hand
                    else "Hand distance: N/A"
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
                f"Pickup: {state.pickup_counter}/{PICKUP_CONFIRM_FRAMES}",
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
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    main()
