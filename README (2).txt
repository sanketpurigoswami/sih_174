from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"F:\sih\runs\detect\train\weights\best.pt"

EVENT_LOG_PATH = r"F:\sih\experiment_protocol_log.jsonl"

CAMERA_INDEX = 1

# YOLO
CONFIDENCE = 0.35
IOU = 0.45

# MediaPipe Hand Landmarker
HAND_MODEL_PATH = r"F:\sih\models\hand_landmarker.task"
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5
NUM_HANDS = 2

# ============================================================
# INSPECTION RULE
# ============================================================
# Inspection is NOT a GRASP/MOVE/RELEASE action.
#
# The current object is considered inspected when:
#
#   1. The object is detected.
#   2. At least one hand is detected.
#   3. All detected fingertip coordinates are averaged into
#      ONE hand-reference point.
#   4. The Euclidean distance between that averaged fingertip
#      point and the object center stays <= this threshold.
#   5. This condition remains true continuously for 5 seconds.
#
# IMPORTANT:
# The hand and object are ALLOWED to move together.
# Only their RELATIVE distance matters.
INSPECTION_DURATION_SECONDS = 5.0
INSPECTION_DISTANCE_THRESHOLD = 60.0  # pixels at the current camera resolution

# Optional short grace period for a single missed hand/object detection.
# Set to 0.0 for strict continuous timing.
INSPECTION_MISSING_GRACE_SECONDS = 0.15


# ============================================================
# PLACEMENT RULE
# ============================================================
# Placement is based only on the object and its assigned square.
#
# An object is correctly placed when:
#   - its bounding box overlaps the correct square sufficiently,
#   - its center is inside the correct square,
#   - and that condition remains true for this duration.
#
# No GRASP / MOVE / RELEASE logic is used.
PLACE_OVERLAP_THRESHOLD = 0.20
PLACEMENT_CONFIRM_SECONDS = 1.0


# ============================================================
# CLASS DEFINITIONS FROM THE WORKING YOLO MODEL
# ============================================================

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

# Experimental objects.
OBJECT_IDS = {0, 1, 2, 7}

# Placement surfaces.
PLACE_IDS = {3, 4, 5, 6}

HAND_ID = 9


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================
# Inspection is performed in this order.
# Change this list if the real experiment requires another order.
INSPECTION_ORDER = [
    0,  # orange_circular_cap
    1,  # red_computer_mouse
    2,  # wireless_earbuds_case
    7,  # white_box
]

# Required final locations.
OBJECT_TO_PLACE = {
    0: 3,  # object 0 -> dark blue square (place 3)
    1: 4,  # object 1 -> yellow square   (place 4)
    2: 5,  # object 2 -> pink square     (place 5)
    7: 6,  # object 7 -> green square    (place 6)
}

MAX_FAILED_FRAMES = 30


# ============================================================
# LIVE WEB UI STATE
# ============================================================

# Updated every processed frame so the Flask UI can show the current
# experiment step and a live increasing timer without changing the
# underlying protocol rules.
LIVE_PROTOCOL_STATE = {
    "phase": "IDLE",
    "status": "Waiting to start.",
    "timer": 0.0,
    "timer_duration": 5.0,
    "timer_running": False,
    "step_number": 0,
    "total_steps": len(INSPECTION_ORDER) * 2,
    "current_object_id": None,
    "current_object": None,
    "target": None,
    "inspected_count": 0,
    "placed_count": 0,
    "completed": False,
}


def _set_live_protocol_state(
    protocol,
    status_text,
    timer_value,
    distance,
    target_name,
    current_object_id,
):
    """Publish the FSM state for the browser UI."""
    global LIVE_PROTOCOL_STATE

    if protocol.phase == "INSPECTION":
        phase_label = "INSPECTION"
        step_number = protocol.inspection_index + 1
        total_steps = len(protocol.inspection_order) * 2
        timer_duration = INSPECTION_DURATION_SECONDS
        # The inspection timer is considered running ONLY after the
        # hand/object distance condition has been satisfied and the FSM
        # has actually started its continuous inspection timer.
        timer_running = protocol.inspection_timer.started_at is not None
    elif protocol.phase == "PLACEMENT":
        phase_label = "PLACEMENT"
        step_number = len(protocol.inspection_order) + len(protocol.placed) + 1
        total_steps = len(protocol.inspection_order) * 2
        timer_duration = PLACEMENT_CONFIRM_SECONDS
        timer_running = protocol.placement_timer.started_at is not None
    elif protocol.phase == "COMPLETE":
        phase_label = "COMPLETE"
        step_number = total_steps = len(protocol.inspection_order) * 2
        timer_duration = 0.0
        timer_running = False
    else:
        phase_label = protocol.phase
        step_number = 0
        total_steps = len(protocol.inspection_order) * 2
        timer_duration = INSPECTION_DURATION_SECONDS
        timer_running = False

    object_name = (
        CLASS_NAMES.get(current_object_id)
        if current_object_id is not None
        else None
    )

    timer_value = max(0.0, min(float(timer_value), float(timer_duration) if timer_duration > 0 else float(timer_value)))
    timer_started_epoch = None
    if timer_running:
        # Convert the monotonic FSM start into wall-clock seconds so the
        # browser can continue the timer smoothly between /api/status polls.
        timer_started_epoch = time.time() - timer_value

    LIVE_PROTOCOL_STATE = {
        "phase": phase_label,
        "status": status_text,
        "timer": round(timer_value, 3),
        "timer_duration": round(float(timer_duration), 3),
        "timer_running": bool(timer_running),
        "timer_started_epoch": timer_started_epoch,
        "step_number": int(step_number),
        "total_steps": int(total_steps),
        "current_object_id": current_object_id,
        "current_object": object_name,
        "target": target_name,
        "distance": None if distance is None else round(float(distance), 1),
        "inspected_count": len(protocol.inspected),
        "placed_count": len(protocol.placed),
        "completed": protocol.phase == "COMPLETE",
    }


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[float, float]


@dataclass
class ObjectTimer:
    started_at: Optional[float] = None
    last_valid_time: Optional[float] = None

    def reset(self) -> None:
        self.started_at = None
        self.last_valid_time = None

    def elapsed(self, now: float) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, now - self.started_at)


# ============================================================
# GEOMETRY
# ============================================================

def center_of_box(
    box: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def box_area(
    box: Tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = box
    return float(
        max(1, x2 - x1) *
        max(1, y2 - y1)
    )


def point_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def intersection_area(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return float(
        (x2 - x1) *
        (y2 - y1)
    )


def overlap_ratio(
    object_box: Tuple[int, int, int, int],
    place_box: Tuple[int, int, int, int],
) -> float:

    area = box_area(object_box)

    if area <= 0:
        return 0.0

    return (
        intersection_area(object_box, place_box)
        / area
    )


def point_inside_box(
    point: Tuple[float, float],
    box: Tuple[int, int, int, int],
) -> bool:

    x, y = point
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ============================================================
# YOLO DETECTION HELPERS
# ============================================================

def extract_detections(result) -> List[Detection]:

    detections: List[Detection] = []

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, confidence in zip(
        boxes,
        class_ids,
        confidences,
    ):

        confidence = float(confidence)

        if confidence < CONFIDENCE:
            continue

        class_id = int(class_id)

        x1, y1, x2, y2 = map(
            int,
            box,
        )

        bbox = (
            x1,
            y1,
            x2,
            y2,
        )

        detections.append(
            Detection(
                class_id=class_id,
                name=CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}",
                ),
                confidence=confidence,
                box=bbox,
                center=center_of_box(bbox),
            )
        )

    return detections


def best_detection_for_class(
    detections: List[Detection],
    class_id: int,
) -> Optional[Detection]:

    candidates = [
        d
        for d in detections
        if d.class_id == class_id
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda d: d.confidence,
    )


def find_places(
    detections: List[Detection],
) -> List[Detection]:

    return [
        d
        for d in detections
        if d.class_id in PLACE_IDS
    ]


def get_place_by_id(
    places: List[Detection],
    place_id: int,
) -> Optional[Detection]:

    candidates = [
        p
        for p in places
        if p.class_id == place_id
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda p: p.confidence,
    )


def get_best_place(
    obj: Detection,
    places: List[Detection],
) -> Tuple[Optional[Detection], float]:

    best_place = None
    best_overlap = 0.0

    for place in places:

        overlap = overlap_ratio(
            obj.box,
            place.box,
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_place = place

    return best_place, best_overlap


# ============================================================
# MEDIAPIPE HAND PROCESSING
# ============================================================

FINGERTIP_INDICES = (
    4,   # thumb
    8,   # index
    12,  # middle
    16,  # ring
    20,  # pinky
)


def get_all_fingertip_points(
    hand_result,
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float]]:

    points: List[Tuple[float, float]] = []

    if not hand_result.hand_landmarks:
        return points

    for hand_landmarks in hand_result.hand_landmarks:

        for index in FINGERTIP_INDICES:

            landmark = hand_landmarks[index]

            points.append(
                (
                    float(landmark.x * frame_width),
                    float(landmark.y * frame_height),
                )
            )

    return points


def average_fingertip_position(
    fingertips: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:

    if not fingertips:
        return None

    mean_x = sum(
        point[0]
        for point in fingertips
    ) / len(fingertips)

    mean_y = sum(
        point[1]
        for point in fingertips
    ) / len(fingertips)

    return (
        mean_x,
        mean_y,
    )


# ============================================================
# EXPERIMENT PROTOCOL
# ============================================================

class ExperimentProtocol:

    def __init__(
        self,
        inspection_order: List[int],
        object_to_place: Dict[int, int],
    ):

        self.inspection_order = list(
            inspection_order
        )

        self.object_to_place = dict(
            object_to_place
        )

        self.phase = "INSPECTION"

        self.inspection_index = 0

        self.inspected = set()
        self.placed = set()

        self.inspection_timer = ObjectTimer()
        self.placement_timer = ObjectTimer()

        self.last_event_text = (
            "Experiment started."
        )

    def reset(self) -> None:

        self.phase = "INSPECTION"
        self.inspection_index = 0

        self.inspected.clear()
        self.placed.clear()

        self.inspection_timer.reset()
        self.placement_timer.reset()

        self.last_event_text = (
            "Experiment restarted."
        )

    def current_inspection_object_id(
        self,
    ) -> Optional[int]:

        if (
            self.inspection_index
            >= len(self.inspection_order)
        ):
            return None

        return self.inspection_order[
            self.inspection_index
        ]

    def next_unplaced_object_id(
        self,
    ) -> Optional[int]:

        for object_id in self.inspection_order:

            if object_id not in self.placed:
                return object_id

        return None

    def update(
        self,
        detections: List[Detection],
        average_fingertip: Optional[Tuple[float, float]],
        current_time: float,
    ) -> Tuple[
        Optional[dict],
        str,
        float,
        Optional[float],
        Optional[str],
    ]:
        """
        Returns:
            event
            status text
            timer value
            hand/object distance
            target-place name
        """

        if self.phase == "INSPECTION":

            return self.update_inspection(
                detections,
                average_fingertip,
                current_time,
            )

        if self.phase == "PLACEMENT":

            return self.update_placement(
                detections,
                current_time,
            )

        return (
            None,
            "EXPERIMENT COMPLETE",
            0.0,
            None,
            None,
        )

    def update_inspection(
        self,
        detections: List[Detection],
        average_fingertip: Optional[Tuple[float, float]],
        current_time: float,
    ):

        object_id = (
            self.current_inspection_object_id()
        )

        if object_id is None:

            self.phase = "PLACEMENT"
            self.inspection_timer.reset()

            return (
                None,
                "Inspection complete. Begin placement.",
                0.0,
                None,
                None,
            )

        object_detection = best_detection_for_class(
            detections,
            object_id,
        )

        if object_detection is None:
            self.inspection_timer.reset()

            return (
                None,
                (
                    f"Inspection: waiting for "
                    f"{CLASS_NAMES[object_id]}"
                ),
                0.0,
                None,
                None,
            )

        if average_fingertip is None:
            self.inspection_timer.reset()

            return (
                None,
                (
                    f"Inspection: show your hand "
                    f"with {CLASS_NAMES[object_id]}"
                ),
                0.0,
                None,
                None,
            )

        distance = point_distance(
            average_fingertip,
            object_detection.center,
        )

        # ----------------------------------------------------
        # CORE INSPECTION RULE:
        # Only relative hand/object distance matters.
        # Both can move together.
        # ----------------------------------------------------
        if distance <= INSPECTION_DISTANCE_THRESHOLD:

            if (
                self.inspection_timer.started_at
                is None
            ):
                self.inspection_timer.started_at = (
                    current_time
                )

            self.inspection_timer.last_valid_time = (
                current_time
            )

            elapsed = (
                self.inspection_timer.elapsed(
                    current_time
                )
            )

            remaining = max(
                0.0,
                INSPECTION_DURATION_SECONDS - elapsed,
            )

            if (
                elapsed
                >= INSPECTION_DURATION_SECONDS
            ):

                self.inspected.add(
                    object_id
                )

                event = {
                    "timestamp": datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    "event": "INSPECTION_COMPLETE",
                    "object": CLASS_NAMES[
                        object_id
                    ],
                    "object_id": object_id,
                    "duration_seconds": (
                        INSPECTION_DURATION_SECONDS
                    ),
                    "hand_object_distance_px": round(
                        distance,
                        2,
                    ),
                }

                self.last_event_text = (
                    f"INSPECTION COMPLETE: "
                    f"{CLASS_NAMES[object_id]}"
                )

                self.inspection_index += 1
                self.inspection_timer.reset()

                if (
                    self.inspection_index
                    >= len(self.inspection_order)
                ):
                    self.phase = "PLACEMENT"

                next_text = (
                    "Inspection complete. Begin placement."
                    if self.phase == "PLACEMENT"
                    else (
                        f"Inspection complete: "
                        f"{CLASS_NAMES[object_id]}"
                    )
                )

                return (
                    event,
                    next_text,
                    INSPECTION_DURATION_SECONDS,
                    distance,
                    None,
                )

            return (
                None,
                (
                    f"INSPECTING "
                    f"{CLASS_NAMES[object_id]} | "
                    f"Distance: {distance:.1f}px | "
                    f"Hold: {remaining:.1f}s"
                ),
                elapsed,
                distance,
                None,
            )

        # ----------------------------------------------------
        # The relative-distance condition was broken.
        # Reset the continuous 5-second inspection timer.
        # ----------------------------------------------------
        self.inspection_timer.reset()

        return (
            None,
            (
                f"Bring hand closer to "
                f"{CLASS_NAMES[object_id]} | "
                f"Distance: {distance:.1f}px"
            ),
            0.0,
            distance,
            None,
        )

    def update_placement(
        self,
        detections: List[Detection],
        current_time: float,
    ):

        object_id = (
            self.next_unplaced_object_id()
        )

        if object_id is None:

            self.phase = "COMPLETE"

            event = {
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
                "event": "EXPERIMENT_COMPLETE",
            }

            self.last_event_text = (
                "EXPERIMENT COMPLETE"
            )

            return (
                event,
                "EXPERIMENT COMPLETE",
                0.0,
                None,
                None,
            )

        target_place_id = (
            self.object_to_place.get(
                object_id
            )
        )

        if target_place_id is None:

            self.placement_timer.reset()

            return (
                None,
                (
                    f"No placement rule for "
                    f"{CLASS_NAMES[object_id]}"
                ),
                0.0,
                None,
                None,
            )

        object_detection = best_detection_for_class(
            detections,
            object_id,
        )

        target_place = get_place_by_id(
            find_places(detections),
            target_place_id,
        )

        target_name = CLASS_NAMES[
            target_place_id
        ]

        if object_detection is None:

            self.placement_timer.reset()

            return (
                None,
                (
                    f"Placement: waiting for "
                    f"{CLASS_NAMES[object_id]} -> "
                    f"{target_name}"
                ),
                0.0,
                None,
                target_name,
            )

        if target_place is None:

            self.placement_timer.reset()

            return (
                None,
                (
                    f"Target not detected: "
                    f"{target_name}"
                ),
                0.0,
                None,
                target_name,
            )

        # Correct placement condition.
        overlap = overlap_ratio(
            object_detection.box,
            target_place.box,
        )

        center_inside_target = point_inside_box(
            object_detection.center,
            target_place.box,
        )

        correct_position = (
            overlap >= PLACE_OVERLAP_THRESHOLD
            and
            center_inside_target
        )

        # Check whether it is visibly sitting on another
        # placement square.
        best_place, best_overlap = get_best_place(
            object_detection,
            find_places(detections),
        )

        if correct_position:

            if (
                self.placement_timer.started_at
                is None
            ):
                self.placement_timer.started_at = (
                    current_time
                )

            self.placement_timer.last_valid_time = (
                current_time
            )

            elapsed = (
                self.placement_timer.elapsed(
                    current_time
                )
            )

            remaining = max(
                0.0,
                PLACEMENT_CONFIRM_SECONDS - elapsed,
            )

            if (
                elapsed
                >= PLACEMENT_CONFIRM_SECONDS
            ):

                self.placed.add(object_id)

                event = {
                    "timestamp": datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                    "event": "CORRECT_PLACEMENT",
                    "object": CLASS_NAMES[
                        object_id
                    ],
                    "object_id": object_id,
                    "target": target_name,
                    "target_id": target_place_id,
                    "overlap_ratio": round(
                        float(overlap),
                        4,
                    ),
                }

                self.last_event_text = (
                    f"CORRECT PLACEMENT: "
                    f"{CLASS_NAMES[object_id]} -> "
                    f"{target_name}"
                )

                self.placement_timer.reset()

                if (
                    len(self.placed)
                    == len(self.inspection_order)
                ):

                    self.phase = "COMPLETE"

                    complete_event = {
                        "timestamp": datetime.now().isoformat(
                            timespec="milliseconds"
                        ),
                        "event": "EXPERIMENT_COMPLETE",
                    }

                    return (
                        complete_event,
                        "EXPERIMENT COMPLETE",
                        PLACEMENT_CONFIRM_SECONDS,
                        None,
                        None,
                    )

                return (
                    event,
                    (
                        f"PLACED: "
                        f"{CLASS_NAMES[object_id]} -> "
                        f"{target_name}"
                    ),
                    elapsed,
                    None,
                    target_name,
                )

            return (
                None,
                (
                    f"CORRECT TARGET: "
                    f"{CLASS_NAMES[object_id]} -> "
                    f"{target_name} | "
                    f"Confirm: {remaining:.1f}s"
                ),
                elapsed,
                None,
                target_name,
            )

        # Wrong placement warning.
        self.placement_timer.reset()

        if (
            best_place is not None
            and
            best_place.class_id != target_place_id
            and
            best_overlap >= PLACE_OVERLAP_THRESHOLD
        ):

            return (
                None,
                (
                    f"WRONG PLACE: "
                    f"{CLASS_NAMES[object_id]} is on "
                    f"{best_place.name} | "
                    f"Required: {target_name}"
                ),
                0.0,
                None,
                target_name,
            )

        return (
            None,
            (
                f"Place {CLASS_NAMES[object_id]} on "
                f"{target_name}"
            ),
            0.0,
            None,
            target_name,
        )


# ============================================================
# DRAWING
# ============================================================

def draw_detection(
    frame,
    detection: Detection,
    status: Optional[str] = None,
) -> None:

    x1, y1, x2, y2 = detection.box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    label = (
        f"{detection.name} "
        f"{detection.confidence:.2f}"
    )

    if status:
        label += f" | {status}"

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


def draw_target_place(
    frame,
    target_place: Optional[Detection],
) -> None:

    if target_place is None:
        return

    x1, y1, x2, y2 = target_place.box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        frame,
        f"TARGET: {target_place.name}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_average_hand_point(
    frame,
    average_point: Optional[Tuple[float, float]],
) -> None:

    if average_point is None:
        return

    point = (
        int(round(average_point[0])),
        int(round(average_point[1])),
    )

    cv2.circle(
        frame,
        point,
        8,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        "AVG FINGERTIPS",
        (
            point[0] + 10,
            point[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_hand_to_object_line(
    frame,
    average_point: Optional[Tuple[float, float]],
    object_detection: Optional[Detection],
) -> None:

    if (
        average_point is None
        or object_detection is None
    ):
        return

    p1 = (
        int(round(average_point[0])),
        int(round(average_point[1])),
    )

    p2 = (
        int(round(object_detection.center[0])),
        int(round(object_detection.center[1])),
    )

    cv2.line(
        frame,
        p1,
        p2,
        (255, 255, 0),
        2,
    )


def draw_protocol_panel(
    frame,
    protocol: ExperimentProtocol,
    status_text: str,
    timer_value: float,
    distance: Optional[float],
    target_name: Optional[str],
) -> None:

    h, w = frame.shape[:2]

    panel_height = 160

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (w - 10, panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0,
        frame,
    )

    cv2.putText(
        frame,
        f"PHASE: {protocol.phase}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status_text[:110],
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if protocol.phase == "INSPECTION":

        text = (
            f"Inspection timer: "
            f"{min(timer_value, INSPECTION_DURATION_SECONDS):.1f}/"
            f"{INSPECTION_DURATION_SECONDS:.1f}s"
        )

    elif protocol.phase == "PLACEMENT":

        text = (
            f"Placement timer: "
            f"{min(timer_value, PLACEMENT_CONFIRM_SECONDS):.1f}/"
            f"{PLACEMENT_CONFIRM_SECONDS:.1f}s"
        )

    else:
        text = "Protocol finished"

    cv2.putText(
        frame,
        text,
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if distance is not None:

        cv2.putText(
            frame,
            (
                f"Avg fingertip -> object center: "
                f"{distance:.1f}px"
            ),
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        (
            f"Inspected: "
            f"{len(protocol.inspected)}/"
            f"{len(protocol.inspection_order)}"
            f"    Placed: "
            f"{len(protocol.placed)}/"
            f"{len(protocol.inspection_order)}"
        ),
        (w - 410, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if target_name:

        cv2.putText(
            frame,
            f"Required target: {target_name}",
            (w - 410, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print("Opening ASUS GlideX webcam...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF,
    )

    if not cap.isOpened():

        cap.release()
        return None

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    try:
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )
    except Exception:
        pass

    success, frame = cap.read()

    if (
        not success
        or
        frame is None
    ):

        cap.release()
        return None

    print(
        "ASUS GlideX webcam connected."
    )

    return cap


# ============================================================
# LOGGING
# ============================================================

def write_event(
    event_file,
    event: dict,
) -> None:

    event_file.write(
        json.dumps(
            event,
            ensure_ascii=False,
        )
        + "\n"
    )

    event_file.flush()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    global LIVE_PROTOCOL_STATE
    LIVE_PROTOCOL_STATE = {
        "phase": "STARTING",
        "status": "Loading YOLO and MediaPipe...",
        "timer": 0.0,
        "timer_duration": INSPECTION_DURATION_SECONDS,
        "timer_running": False,
        "step_number": 1,
        "total_steps": len(INSPECTION_ORDER) * 2,
        "current_object_id": INSPECTION_ORDER[0] if INSPECTION_ORDER else None,
        "current_object": CLASS_NAMES.get(INSPECTION_ORDER[0]) if INSPECTION_ORDER else None,
        "target": None,
        "distance": None,
        "inspected_count": 0,
        "placed_count": 0,
        "completed": False,
    }

    print("=" * 72)
    print(
        "SIH 2026 - INSPECTION + CORRECT PLACEMENT"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print(
        f"Loading YOLO model: {MODEL_PATH}"
    )

    try:

        model = YOLO(
            MODEL_PATH
        )

    except Exception as exc:

        print(
            f"ERROR loading YOLO: {exc}"
        )
        return

    print(
        "YOLO loaded successfully."
    )

    print(
        "YOLO classes:",
        model.names,
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    print(
        f"Loading hand model: {HAND_MODEL_PATH}"
    )

    try:

        base_options = (
            python.BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            )
        )

        hand_options = (
            vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=NUM_HANDS,
                min_hand_detection_confidence=(
                    HAND_MIN_DETECTION_CONFIDENCE
                ),
                min_hand_presence_confidence=(
                    HAND_MIN_PRESENCE_CONFIDENCE
                ),
                min_tracking_confidence=(
                    HAND_MIN_TRACKING_CONFIDENCE
                ),
            )
        )

        hand_detector = (
            vision.HandLandmarker.create_from_options(
                hand_options
            )
        )

    except Exception as exc:

        print(
            f"ERROR loading MediaPipe: {exc}"
        )
        return

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    protocol = ExperimentProtocol(
        inspection_order=INSPECTION_ORDER,
        object_to_place=OBJECT_TO_PLACE,
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    try:

        event_file = open(
            EVENT_LOG_PATH,
            "a",
            encoding="utf-8",
        )

    except Exception as exc:

        print(
            f"ERROR opening event log: {exc}"
        )

        hand_detector.close()
        return

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = open_camera()

    while cap is None:

        print(
            "Could not open webcam. "
            "Retrying in 2 seconds..."
        )

        time.sleep(2)

        cap = open_camera()

    # --------------------------------------------------------
    # Display / timing
    # --------------------------------------------------------

    frame_number = 0
    failed_frames = 0

    previous_tick = cv2.getTickCount()
    fps = 0.0

    cv2.namedWindow(
        "SIH 2026 - Inspection & Placement",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "SIH 2026 - Inspection & Placement",
        1100,
        700,
    )

    print()
    print("CAMERA STARTED")
    print("Q = quit")
    print("R = restart experiment")
    print()

    print(
        "INSPECTION RULE:"
    )
    print(
        f"Keep the average of all detected "
        f"fingertips within "
        f"{INSPECTION_DISTANCE_THRESHOLD:.0f}px "
        f"of the current object's center "
        f"for {INSPECTION_DURATION_SECONDS:.0f}s."
    )
    print(
        "The hand and object may move together."
    )
    print()

    print(
        "PLACEMENT RULE:"
    )

    for object_id, place_id in OBJECT_TO_PLACE.items():

        print(
            f"  Object {object_id} "
            f"({CLASS_NAMES[object_id]}) -> "
            f"Place {place_id} "
            f"({CLASS_NAMES[place_id]})"
        )

    print()

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            success, frame = cap.read()

            if (
                not success
                or
                frame is None
            ):

                failed_frames += 1

                print(
                    f"\rCamera read failed "
                    f"{failed_frames}/{MAX_FAILED_FRAMES}",
                    end="",
                    flush=True,
                )

                time.sleep(0.03)

                if (
                    failed_frames
                    >= MAX_FAILED_FRAMES
                ):

                    print(
                        "\nTrying to reconnect..."
                    )

                    cap.release()
                    cv2.destroyAllWindows()

                    while True:

                        time.sleep(1)

                        cap = open_camera()

                        if cap is not None:

                            failed_frames = 0

                            print(
                                "Webcam reconnected."
                            )

                            break

                        print(
                            "Reconnect failed. "
                            "Retrying..."
                        )

                continue

            failed_frames = 0
            frame_number += 1

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            try:

                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    iou=IOU,
                    verbose=False,
                )

            except Exception as exc:

                print(
                    f"\nYOLO inference error: "
                    f"{exc}"
                )

                continue

            detections = (
                extract_detections(
                    results[0]
                )
            )

            # ------------------------------------------------
            # MediaPipe hand landmarks
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame,
            )

            # Use a monotonically increasing timestamp
            # for the single continuous live stream.
            timestamp_ms = int(
                (frame_number / 30.0)
                * 1000
            )

            try:

                hand_result = (
                    hand_detector.detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                )

            except Exception as exc:

                print(
                    f"\nMediaPipe error: "
                    f"{exc}"
                )

                continue

            fingertips = (
                get_all_fingertip_points(
                    hand_result,
                    frame.shape[1],
                    frame.shape[0],
                )
            )

            average_fingertip = (
                average_fingertip_position(
                    fingertips
                )
            )

            # Draw every fingertip.
            for point in fingertips:

                cv2.circle(
                    frame,
                    (
                        int(round(point[0])),
                        int(round(point[1])),
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )

            # Draw average point.
            draw_average_hand_point(
                frame,
                average_fingertip,
            )

            # ------------------------------------------------
            # Protocol
            # ------------------------------------------------

            event, status_text, timer_value, distance, target_name = (
                protocol.update(
                    detections=detections,
                    average_fingertip=average_fingertip,
                    current_time=time.monotonic(),
                )
            )

            # ------------------------------------------------
            # Protocol event logging
            # ------------------------------------------------

            if event is not None:

                write_event(
                    event_file,
                    event,
                )

                print(
                    "\n"
                    + "=" * 72
                )

                print(
                    event
                )

                print(
                    "=" * 72
                )

            # ------------------------------------------------
            # Current target object
            # ------------------------------------------------

            if (
                protocol.phase
                == "INSPECTION"
            ):

                current_object_id = (
                    protocol.current_inspection_object_id()
                )

            elif (
                protocol.phase
                == "PLACEMENT"
            ):

                current_object_id = (
                    protocol.next_unplaced_object_id()
                )

            else:

                current_object_id = None

            # Publish protocol state for the browser dashboard.
            _set_live_protocol_state(
                protocol,
                status_text,
                timer_value,
                distance,
                target_name,
                current_object_id,
            )

            current_object_detection = None

            if current_object_id is not None:

                current_object_detection = (
                    best_detection_for_class(
                        detections,
                        current_object_id,
                    )
                )

            # ------------------------------------------------
            # Draw YOLO detections
            # ------------------------------------------------

            for detection in detections:

                status = None

                if (
                    detection.class_id
                    in protocol.inspected
                ):

                    status = "INSPECTED"

                if (
                    detection.class_id
                    in protocol.placed
                ):

                    status = "PLACED"

                if (
                    current_object_id is not None
                    and
                    detection.class_id
                    == current_object_id
                ):

                    if status:
                        status = (
                            f"{status} | CURRENT"
                        )
                    else:
                        status = "CURRENT"

                draw_detection(
                    frame,
                    detection,
                    status,
                )

            # ------------------------------------------------
            # Inspection-specific visual feedback
            # ------------------------------------------------

            if (
                protocol.phase
                == "INSPECTION"
            ):

                draw_hand_to_object_line(
                    frame,
                    average_fingertip,
                    current_object_detection,
                )

            # ------------------------------------------------
            # Placement-specific visual feedback
            # ------------------------------------------------

            target_place_detection = None

            if (
                protocol.phase
                == "PLACEMENT"
                and
                current_object_id is not None
            ):

                target_place_id = (
                    OBJECT_TO_PLACE.get(
                        current_object_id
                    )
                )

                if target_place_id is not None:

                    target_place_detection = (
                        get_place_by_id(
                            find_places(
                                detections
                            ),
                            target_place_id,
                        )
                    )

                    draw_target_place(
                        frame,
                        target_place_detection,
                    )

                    # Draw actual overlap information.
                    if (
                        current_object_detection
                        is not None
                        and
                        target_place_detection
                        is not None
                    ):

                        overlap = (
                            overlap_ratio(
                                current_object_detection.box,
                                target_place_detection.box,
                            )
                        )

                        cv2.putText(
                            frame,
                            (
                                f"Target overlap: "
                                f"{overlap * 100:.1f}%"
                            ),
                            (25, 190),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # ------------------------------------------------
            # Protocol panel
            # ------------------------------------------------

            draw_protocol_panel(
                frame,
                protocol,
                status_text,
                timer_value,
                distance,
                target_name,
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_tick = cv2.getTickCount()

            elapsed = (
                current_tick
                - previous_tick
            ) / cv2.getTickFrequency()

            previous_tick = current_tick

            if elapsed > 0:

                instant_fps = (
                    1.0 / elapsed
                )

                if fps == 0:
                    fps = instant_fps
                else:
                    fps = (
                        0.9 * fps
                        + 0.1 * instant_fps
                    )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "LIVE",
                (
                    frame.shape[1] - 85,
                    frame.shape[0] - 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ------------------------------------------------
            # Show
            # ------------------------------------------------

            cv2.imshow(
                "SIH 2026 - Inspection & Placement",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):

                protocol.reset()

                print(
                    "\nExperiment protocol restarted."
                )

    finally:

        LIVE_PROTOCOL_STATE = {
            **LIVE_PROTOCOL_STATE,
            "phase": "STOPPED",
            "status": "Camera stopped.",
            "timer_running": False,
        }

        print(
            "\nStopping webcam..."
        )

        cap.release()
        event_file.close()
        hand_detector.close()

        cv2.destroyAllWindows()

        print(
            "Camera stopped."
        )

        print(
            f"Experiment log: "
            f"{EVENT_LOG_PATH}"
        )


if __name__ == "__main__":
    main()
