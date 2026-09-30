from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ============================================================
# VYOMDRISHTI — INSPECTION-ONLY FSM
# ============================================================
# Single source of truth for camera + YOLO + MediaPipe + ActionDetector
# + inspection sequence + GUI state.
# Placement is intentionally disabled for this build.

MODEL_PATH = r"F:\sih\runs\detect\train\weights\best.pt"
HAND_MODEL_PATH = r"F:\sih\models\hand_landmarker.task"
EVENT_LOG_PATH = r"F:\sih\experiment_protocol_log.jsonl"

# Verified GlideX configuration.
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_MSMF

CONFIDENCE = 0.35
IOU = 0.45

# MediaPipe.
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_PRESENCE_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5
NUM_HANDS = 2

# Inspection timer.
INSPECTION_DURATION_SECONDS = 5.0

# ActionDetector grasp rule (preserved).
CONTACT_DISTANCE = 30.0
GRASP_FINGERS_REQUIRED = 4
RELEASE_DISTANCE_MULTIPLIER = 1.35
RELEASE_CONFIRM_FRAMES = 3
MOVEMENT_HISTORY_SIZE = 5
MOVE_ON_THRESHOLD = 0.04
MOVE_OFF_THRESHOLD = 0.02
BASELINE_FRAMES = 8

# BBox inspection trigger.
# Any positive intersection between the YOLO hand bbox and the current
# object's bbox is considered overlap. The timer starts when EITHER:
#   (A) ActionDetector is in GRASP/grasped latch, OR
#   (B) hand/object bboxes overlap.
BBOX_OVERLAP_PIXELS_REQUIRED = 1

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

OBJECT_IDS = {0, 1, 2, 7}
HAND_ID = 9

INSPECTION_SEQUENCE = [0, 1, 2, 7]

FRIENDLY_NAMES = {
    0: "Inspect Orange Box",
    1: "Inspect Red Mouse",
    2: "Inspect Earbuds",
    7: "Inspect White Box",
}

# ============================================================
# WEB STATE & VOICE TELEMETRY
# ============================================================

STATE_LOCK = threading.Lock()
VOICE_EVENT_LOCK = threading.Lock()
CURRENT_VOICE_EVENT_ID = 0


def trigger_server_sound(sound_type: str):
    """Play alert sounds on the host machine in a non-blocking daemon thread."""
    def _play():
        try:
            import winsound
            if sound_type == "buzzer":
                winsound.Beep(900, 180)
                time.sleep(0.06)
                winsound.Beep(700, 260)
            elif sound_type == "chime":
                winsound.Beep(587, 120)
                time.sleep(0.04)
                winsound.Beep(880, 180)
            elif sound_type == "ping":
                winsound.Beep(659, 100)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


def create_voice_event(event_type: str, text: str, tone: str = "info", step_number: Optional[int] = None, next_step_number: Optional[int] = None):
    global CURRENT_VOICE_EVENT_ID
    with VOICE_EVENT_LOCK:
        CURRENT_VOICE_EVENT_ID += 1
        ev = {
            "id": CURRENT_VOICE_EVENT_ID,
            "type": event_type,
            "text": text,
            "tone": tone,
            "step_number": step_number,
            "next_step_number": next_step_number,
            "timestamp": time.time(),
        }
    trigger_server_sound("buzzer" if tone == "warning" else ("chime" if tone in ("success", "complete") else "ping"))
    return ev


LIVE_STATE = {
    "camera": "stopped",
    "phase": "IDLE",
    "experiment_started": False,
    "completed": False,
    "status": "Press START CAMERA.",
    "action": "IDLE",
    "grasp_active": False,
    "bbox_overlap_active": False,
    "timer_trigger": "NONE",
    "timer": 0.0,
    "timer_duration": INSPECTION_DURATION_SECONDS,
    "timer_running": False,
    "timer_decaying": False,
    "timer_started_epoch": None,
    "step_number": 1,
    "total_steps": len(INSPECTION_SEQUENCE),
    "current_object_id": None,
    "current_object": None,
    "next_step_number": 2,
    "next_object_id": 1,
    "next_object": "red_computer_mouse",
    "missed_step": {
        "active": False,
        "missed_step_number": None,
        "missed_step_name": None,
        "missed_object_id": None,
        "performed_step_number": None,
        "performed_step_name": None,
        "performed_object_id": None,
        "message": "",
    },
    "voice_event": None,
    "hand_overlap_ratio": 0.0,
    "hand_overlap_pixels": 0.0,
    "inspected_count": 0,
    "frame": 0,
    "fps": 0.0,
    "sequence": [
        {"id": 0, "name": "orange_circular_cap", "status": "UPCOMING", "step_number": 1},
        {"id": 1, "name": "red_computer_mouse", "status": "UPCOMING", "step_number": 2},
        {"id": 2, "name": "wireless_earbuds_case", "status": "UPCOMING", "step_number": 3},
        {"id": 7, "name": "white_box", "status": "UPCOMING", "step_number": 4},
    ],
}


def publish_state(**updates):
    with STATE_LOCK:
        LIVE_STATE.update(updates)


def get_live_state():
    with STATE_LOCK:
        state = dict(LIVE_STATE)
        state["sequence"] = [dict(x) for x in LIVE_STATE["sequence"]]
        if isinstance(LIVE_STATE.get("missed_step"), dict):
            state["missed_step"] = dict(LIVE_STATE["missed_step"])
        if isinstance(LIVE_STATE.get("voice_event"), dict):
            state["voice_event"] = dict(LIVE_STATE["voice_event"])
        return state

# ============================================================
# DATA / GEOMETRY
# ============================================================

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
    baseline_area: Optional[float] = None
    baseline_count: int = 0
    action: str = "IDLE"
    grasped: bool = False
    picked_up: bool = False
    movement_history: deque = None
    moving: bool = False
    release_counter: int = 0

    def __post_init__(self):
        self.movement_history = deque(maxlen=MOVEMENT_HISTORY_SIZE)


def box_area(box):
    x1, y1, x2, y2 = box
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def overlap_ratio(object_box, hand_box):
    area = box_area(object_box)
    return intersection_area(object_box, hand_box) / area if area > 0 else 0.0


def normalized_movement(movement, box):
    x1, y1, x2, y2 = box
    diagonal = math.hypot(x2 - x1, y2 - y1)
    return movement / diagonal if diagonal > 0 else 0.0


def extract_detections(result):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, class_id, conf in zip(boxes, classes, confs):
        conf = float(conf)
        if conf < CONFIDENCE:
            continue
        class_id = int(class_id)
        x1, y1, x2, y2 = map(int, box)
        bbox = (x1, y1, x2, y2)
        detections.append(Detection(class_id, CLASS_NAMES.get(class_id, f"class_{class_id}"), conf, bbox, center_of_box(bbox)))
    return detections


def best_detection_for_class(detections, class_id):
    candidates = [d for d in detections if d.class_id == class_id]
    return max(candidates, key=lambda d: d.confidence) if candidates else None


def best_hand_for_object(obj, detections):
    hands = [d for d in detections if d.class_id == HAND_ID]
    if not hands:
        return None, 0.0, 0.0
    # Pick the hand with the largest overlap; if none overlaps, largest intersection is 0.
    best = max(hands, key=lambda h: intersection_area(obj.box, h.box))
    pixels = intersection_area(obj.box, best.box)
    ratio = overlap_ratio(obj.box, best.box)
    return best, ratio, pixels

# ============================================================
# MEDIAPIPE
# ============================================================

FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def get_fingertips(hand_result, width, height):
    points = []
    if not hand_result.hand_landmarks:
        return points
    for hand_landmarks in hand_result.hand_landmarks:
        for idx in FINGERTIP_INDICES:
            p = hand_landmarks[idx]
            points.append((float(p.x * width), float(p.y * height)))
    return points


def fingertip_box_distances(fingertips, box):
    x1, y1, x2, y2 = box
    distances = []
    for px, py in fingertips:
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        distances.append(math.hypot(dx, dy))
    return distances

# ============================================================
# ACTION DETECTOR
# ============================================================

class ActionDetector:
    """Original grasp logic, retained as an independent signal."""

    def __init__(self):
        self.states: Dict[int, ObjectState] = {}

    def get_state(self, class_id):
        if class_id not in self.states:
            self.states[class_id] = ObjectState(class_id)
        return self.states[class_id]

    def update(self, obj, fingertips):
        state = self.get_state(obj.class_id)
        state.previous_center = state.current_center
        state.current_center = obj.center

        movement = point_distance(state.previous_center, state.current_center)
        state.movement_history.append(normalized_movement(movement, obj.box))
        median = sorted(state.movement_history)[len(state.movement_history) // 2]
        if not state.moving and median >= MOVE_ON_THRESHOLD:
            state.moving = True
        elif state.moving and median <= MOVE_OFF_THRESHOLD:
            state.moving = False

        state.current_area = box_area(obj.box)
        if not state.picked_up and state.baseline_count < BASELINE_FRAMES:
            if state.baseline_area is None:
                state.baseline_area = state.current_area
            else:
                state.baseline_area = 0.85 * state.baseline_area + 0.15 * state.current_area
            state.baseline_count += 1

        distances = fingertip_box_distances(fingertips, obj.box) if fingertips else []
        close_fingers = sum(d <= CONTACT_DISTANCE for d in distances)
        fingers_in_contact = close_fingers >= GRASP_FINGERS_REQUIRED
        fingers_separated = (
            not fingertips
            or (bool(distances) and all(d >= CONTACT_DISTANCE * RELEASE_DISTANCE_MULTIPLIER for d in distances))
        )

        if not state.grasped:
            if fingers_in_contact:
                state.grasped = True
                state.picked_up = True
                state.action = "GRASP"
                state.release_counter = 0
                return state
            state.action = "MOVE" if state.moving else "IDLE"
            return state

        if fingers_separated and not state.moving:
            state.release_counter += 1
        else:
            state.release_counter = max(0, state.release_counter - 1)

        if state.release_counter >= RELEASE_CONFIRM_FRAMES:
            state.action = "RELEASE"
            state.grasped = False
            state.picked_up = False
            state.release_counter = 0
            return state

        state.action = "MOVE" if state.moving else "GRASP"
        return state

# ============================================================
# INSPECTION SEQUENCE — NO PLACEMENT
# ============================================================

class InspectionProtocol:
    def __init__(self):
        self.reset()

    def reset(self):
        self.started = False
        self.complete = False
        self.index = 0
        self.inspected = []
        self.current_timer = 0.0
        self.last_update_time = None
        self.last_trigger = "NONE"
        self.violation_counter = 0
        self.active_violation = None
        self.last_violation_alert_time = 0.0
        self.hold_announced_for_step = None
        self.start_announced = False

    @property
    def current_object_id(self):
        if self.complete or self.index >= len(INSPECTION_SEQUENCE):
            return None
        return INSPECTION_SEQUENCE[self.index]

    @property
    def next_object_id(self):
        if self.complete or (self.index + 1) >= len(INSPECTION_SEQUENCE):
            return None
        return INSPECTION_SEQUENCE[self.index + 1]

    @property
    def current_step_number(self):
        return min(self.index + 1, len(INSPECTION_SEQUENCE))

    @property
    def next_step_number(self):
        if (self.index + 1) < len(INSPECTION_SEQUENCE):
            return self.index + 2
        return None

    def sequence_payload(self):
        payload = []
        for i, object_id in enumerate(INSPECTION_SEQUENCE):
            step_num = i + 1
            if object_id in self.inspected:
                status = "COMPLETED"
            elif i == self.index and not self.complete:
                if self.active_violation is not None:
                    status = "MISSED"
                else:
                    status = "CURRENT"
            elif self.active_violation and (step_num == self.active_violation.get("performed_step_number")):
                status = "VIOLATION"
            else:
                status = "UPCOMING"
            payload.append({
                "id": object_id,
                "name": CLASS_NAMES[object_id],
                "status": status,
                "step_number": step_num
            })
        return payload

    def choose_step_from_detection(self, detections):
        # Experiment starts as soon as ANY experimental object is seen.
        if detections and any(d.class_id in OBJECT_IDS for d in detections):
            self.started = True

    def update(self, detections, action_states, now):
        self.choose_step_from_detection(detections)
        new_voice_event = None

        if self.last_update_time is None:
            dt = 0.033
        else:
            dt = max(0.0, min(0.3, now - self.last_update_time))
        self.last_update_time = now

        if self.started and not self.start_announced:
            self.start_announced = True
            c_name = CLASS_NAMES[INSPECTION_SEQUENCE[0]].replace("_", " ")
            n_name = CLASS_NAMES[INSPECTION_SEQUENCE[1]].replace("_", " ")
            new_voice_event = create_voice_event(
                "STEP_READY",
                f"Starting sequence. Step 1: {c_name}. Next: Step 2, {n_name}.",
                tone="info",
                step_number=1,
                next_step_number=2
            )

        if self.complete:
            self.current_timer = INSPECTION_DURATION_SECONDS
            ret = self._state(
                "INSPECTION COMPLETE — ALL 4 OBJECTS",
                INSPECTION_DURATION_SECONDS,
                False,
                "NONE",
                None,
                0.0,
                0.0,
                "IDLE",
                False,
                timer_decaying=False
            )
            if new_voice_event:
                ret["voice_event"] = new_voice_event
            return ret

        current_id = self.current_object_id
        if current_id is None:
            self.complete = True
            self.current_timer = INSPECTION_DURATION_SECONDS
            ret = self._state(
                "INSPECTION COMPLETE — ALL 4 OBJECTS",
                INSPECTION_DURATION_SECONDS,
                False,
                "NONE",
                None,
                0.0,
                0.0,
                "IDLE",
                False,
                timer_decaying=False
            )
            if new_voice_event:
                ret["voice_event"] = new_voice_event
            return ret

        current_step_num = self.current_step_number
        next_step_num = self.next_step_number

        # ------------------------------------------------------------------
        # MISSED STEP DETECTION: Check if ANY subsequent step in the sequence
        # is being manipulated/held before the current step is completed!
        # ------------------------------------------------------------------
        detected_future_violation = None
        for s_idx in range(self.index + 1, len(INSPECTION_SEQUENCE)):
            future_id = INSPECTION_SEQUENCE[s_idx]
            f_obj = best_detection_for_class(detections, future_id)
            if f_obj is not None:
                f_hand, f_ratio, f_pixels = best_hand_for_object(f_obj, detections)
                f_action_st = action_states.get(future_id)
                f_grasp = bool(f_action_st and getattr(f_action_st, "grasped", False))
                f_overlap = f_pixels >= BBOX_OVERLAP_PIXELS_REQUIRED
                if f_grasp or f_overlap:
                    detected_future_violation = {
                        "active": True,
                        "missed_step_number": current_step_num,
                        "missed_step_name": CLASS_NAMES[current_id],
                        "missed_object_id": current_id,
                        "performed_step_number": s_idx + 1,
                        "performed_step_name": CLASS_NAMES[future_id],
                        "performed_object_id": future_id,
                        "trigger": "GRASP + BBOX OVERLAP" if (f_grasp and f_overlap) else ("GRASP" if f_grasp else "BBOX OVERLAP"),
                        "message": (
                            f"Step {current_step_num} ({CLASS_NAMES[current_id].replace('_', ' ')}) was missed! "
                            f"Step {s_idx + 1} ({CLASS_NAMES[future_id].replace('_', ' ')}) is being performed."
                        )
                    }
                    break

        if detected_future_violation is not None:
            self.violation_counter += 1
            if self.violation_counter >= 3:
                self.active_violation = detected_future_violation
                # User requirement:
                # "dont abrupty make the timer to zero if something out of order is detected instead , make it decrease at twice speed only"
                self.current_timer = max(0.0, self.current_timer - (2.0 * dt))
                self.last_trigger = "SEQUENCE VIOLATION"

                # Throttle repeated voice alerts so it repeats at most every 2.8 seconds
                if (now - self.last_violation_alert_time) > 2.8:
                    self.last_violation_alert_time = now
                    v_text = (
                        f"Warning! Step {current_step_num}, {CLASS_NAMES[current_id].replace('_', ' ')}, was missed! "
                        f"Inspect Step {current_step_num} first!"
                    )
                    new_voice_event = create_voice_event(
                        "MISSED_STEP_ALERT",
                        v_text,
                        tone="warning",
                        step_number=current_step_num,
                        next_step_number=detected_future_violation["performed_step_number"]
                    )

                status = (
                    f"⚠️ MISSED STEP ALERT: STEP {current_step_num} ({CLASS_NAMES[current_id].upper()}) MISSED! "
                    f"PLEASE INSPECT STEP {current_step_num} FIRST!"
                )
                ret = self._state(
                    status,
                    self.current_timer,
                    False,
                    "SEQUENCE VIOLATION",
                    current_id,
                    0.0,
                    0.0,
                    "IDLE",
                    False,
                    missed_step=self.active_violation,
                    timer_decaying=(self.current_timer > 0.0)
                )
                if new_voice_event:
                    ret["voice_event"] = new_voice_event
                ret["event"] = {
                    "type": "SEQUENCE_VIOLATION",
                    "timestamp": datetime.now().isoformat(),
                    "details": self.active_violation
                }
                return ret
        else:
            self.violation_counter = max(0, self.violation_counter - 1)
            if self.violation_counter == 0:
                self.active_violation = None

        # ------------------------------------------------------------------
        # NORMAL STEP INSPECTION FLOW
        # ------------------------------------------------------------------
        obj = best_detection_for_class(detections, current_id)
        if obj is None:
            # If expected object is temporarily out of frame, decay timer at twice speed
            self.current_timer = max(0.0, self.current_timer - (2.0 * dt))
            self.last_trigger = "NONE"
            status = f"WAITING FOR {CLASS_NAMES[current_id].upper()} (STEP {current_step_num}/4)"
            ret = self._state(
                status,
                self.current_timer,
                False,
                "NONE",
                current_id,
                0.0,
                0.0,
                "IDLE",
                False,
                missed_step=None,
                timer_decaying=(self.current_timer > 0.0)
            )
            if new_voice_event:
                ret["voice_event"] = new_voice_event
            return ret

        hand, ratio, pixels = best_hand_for_object(obj, detections)
        action_state = action_states.get(current_id)
        action = getattr(action_state, "action", "IDLE") if action_state else "IDLE"
        grasp_active = bool(action_state and getattr(action_state, "grasped", False))
        bbox_overlap_active = pixels >= BBOX_OVERLAP_PIXELS_REQUIRED

        if grasp_active and bbox_overlap_active:
            trigger = "GRASP + BBOX OVERLAP"
        elif grasp_active:
            trigger = "GRASP"
        elif bbox_overlap_active:
            trigger = "BBOX OVERLAP"
        else:
            trigger = "NONE"

        inspection_condition = grasp_active or bbox_overlap_active

        if inspection_condition:
            self.current_timer = min(INSPECTION_DURATION_SECONDS, self.current_timer + dt)
            self.last_trigger = trigger

            if self.current_timer >= 0.15 and self.hold_announced_for_step != current_step_num:
                self.hold_announced_for_step = current_step_num
                c_clean = FRIENDLY_NAMES.get(current_id, CLASS_NAMES[current_id].replace("_", " "))
                new_voice_event = create_voice_event(
                    "STEP_HOLD",
                    f"Holding: {c_clean}.",
                    tone="info",
                    step_number=current_step_num,
                    next_step_number=next_step_num
                )

            if self.current_timer >= INSPECTION_DURATION_SECONDS:
                completed_object = current_id
                self.inspected.append(completed_object)
                self.index += 1
                self.current_timer = 0.0
                self.last_trigger = "COMPLETE"

                clean_completed = FRIENDLY_NAMES.get(completed_object, CLASS_NAMES[completed_object].replace("_", " "))

                if self.index >= len(INSPECTION_SEQUENCE):
                    self.complete = True
                    v_msg = f"{clean_completed} verified! All steps complete. Mission verified."
                    new_voice_event = create_voice_event(
                        "MISSION_COMPLETE",
                        v_msg,
                        tone="complete",
                        step_number=current_step_num,
                        next_step_number=None
                    )
                    status_str = "INSPECTION COMPLETE — ALL 4 OBJECTS"
                else:
                    new_next_id = INSPECTION_SEQUENCE[self.index]
                    clean_next = FRIENDLY_NAMES.get(new_next_id, CLASS_NAMES[new_next_id].replace("_", " "))
                    v_msg = f"{clean_completed} verified! Next: Step {self.index + 1}, {clean_next}."
                    new_voice_event = create_voice_event(
                        "STEP_VERIFIED",
                        v_msg,
                        tone="success",
                        step_number=current_step_num,
                        next_step_number=self.index + 1
                    )
                    status_str = f"INSPECTION COMPLETE — {CLASS_NAMES[completed_object].upper()}"

                ret = self._state(
                    status_str,
                    INSPECTION_DURATION_SECONDS,
                    False,
                    "COMPLETE",
                    completed_object,
                    ratio,
                    pixels,
                    action,
                    grasp_active,
                    missed_step=None,
                    timer_decaying=False
                )
                if new_voice_event:
                    ret["voice_event"] = new_voice_event
                ret["event"] = {
                    "type": "STEP_VERIFIED",
                    "step": current_step_num,
                    "object": clean_completed,
                    "next_step": self.index + 1 if not self.complete else None,
                    "timestamp": datetime.now().isoformat()
                }
                return ret

            ret = self._state(
                f"INSPECTING — {CLASS_NAMES[current_id].upper()}",
                self.current_timer,
                True,
                trigger,
                current_id,
                ratio,
                pixels,
                action,
                grasp_active,
                missed_step=None,
                timer_decaying=False
            )
            if new_voice_event:
                ret["voice_event"] = new_voice_event
            return ret

        # Not holding / released: decrease timer at twice speed (2.0 * dt)
        self.current_timer = max(0.0, self.current_timer - (2.0 * dt))
        self.last_trigger = "NONE"
        status_msg = f"WAITING FOR GRASP OR HAND BBOX OVERLAP — {CLASS_NAMES[current_id].upper()} (STEP {current_step_num}/4)"
        if self.current_timer > 0.0:
            status_msg = f"HOLD RELEASED (DECAYING 2X) — {CLASS_NAMES[current_id].upper()} (STEP {current_step_num}/4)"

        ret = self._state(
            status_msg,
            self.current_timer,
            False,
            "NONE",
            current_id,
            ratio,
            pixels,
            action,
            grasp_active,
            missed_step=None,
            timer_decaying=(self.current_timer > 0.0)
        )
        if new_voice_event:
            ret["voice_event"] = new_voice_event
        return ret

    def _state(self, status, timer, running, trigger, object_id, ratio, pixels, action, grasp, missed_step=None, timer_decaying=False):
        next_step_num = self.next_step_number
        next_id = self.next_object_id
        return {
            "status": status,
            "timer": timer,
            "timer_running": running,
            "timer_decaying": timer_decaying,
            "timer_started_epoch": (time.time() - timer) if running else None,
            "timer_duration": INSPECTION_DURATION_SECONDS,
            "timer_trigger": trigger,
            "object_id": object_id,
            "step_number": self.current_step_number,
            "next_step_number": next_step_num,
            "next_object_id": next_id,
            "next_object": CLASS_NAMES.get(next_id) if next_id is not None else None,
            "hand_overlap_ratio": ratio,
            "hand_overlap_pixels": pixels,
            "action": action,
            "grasp_active": grasp,
            "bbox_overlap_active": pixels >= BBOX_OVERLAP_PIXELS_REQUIRED,
            "missed_step": missed_step or {
                "active": False,
                "missed_step_number": None,
                "missed_step_name": None,
                "missed_object_id": None,
                "performed_step_number": None,
                "performed_step_name": None,
                "performed_object_id": None,
                "message": "",
            },
        }

# Global protocol so RESET can act without restarting the server.
PROTOCOL = InspectionProtocol()

# ============================================================
# DRAWING
# ============================================================

def draw_detection(frame, d, status=None, is_violation=False, is_missed_target=False):
    x1, y1, x2, y2 = d.box
    if is_violation:
        color = (0, 0, 255) # Red for out-of-order manipulation
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"⚠️ OUT OF ORDER: {d.name}"
    elif is_missed_target:
        color = (0, 215, 255) # Bright gold for the missed object that must be done first
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"⬅️ MUST INSPECT FIRST: {d.name}"
    else:
        color = (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.name} {d.confidence:.2f}"
        if status:
            label += f" | {status}"
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_inspection_panel(frame, p, protocol):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Compact HUD overlay taking ~12% area (well under 25%) in top-left corner
    panel_w = min(290, int(w * 0.45))
    panel_h = 115
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (140, 124, 251), 1)

    curr_num = protocol.current_step_number
    obj = (CLASS_NAMES.get(p.get("object_id"), "NONE") if p.get("object_id") is not None else "NONE").upper()
    next_num = p.get("next_step_number")
    next_obj = (p.get("next_object") or "COMPLETE").upper()

    # Line 1: Phase & Step
    cv2.putText(frame, f"STEP {curr_num}/4: {obj[:16]}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (74, 222, 154), 1, cv2.LINE_AA)

    # Line 2: Next step
    if next_num is not None:
        cv2.putText(frame, f"NEXT: S{next_num} {next_obj[:15]}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 170, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "NEXT: MISSION COMPLETE", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 170, 255), 1, cv2.LINE_AA)

    # Line 3: Trigger / Hold state
    trigger = p.get('timer_trigger', 'NONE')
    cv2.putText(frame, f"TRIGGER: {trigger[:20]}", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

    # Line 4: Timer & Decay
    timer_val = p.get('timer', 0.0)
    is_decaying = p.get('timer_decaying', False)
    timer_str = f"TIMER: {timer_val:.1f}s / {INSPECTION_DURATION_SECONDS:.1f}s"
    if is_decaying and timer_val > 0.0:
        timer_str += " (DECAY 2X)"
        timer_col = (0, 180, 255)
    elif p.get('timer_running'):
        timer_col = (74, 222, 154)
    else:
        timer_col = (255, 255, 255)
    cv2.putText(frame, timer_str, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, timer_col, 1, cv2.LINE_AA)

    # Mini Progress Bar
    prog_pct = min(1.0, max(0.0, timer_val / INSPECTION_DURATION_SECONDS))
    bar_x1, bar_y = 16, 102
    bar_w = panel_w - 16
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 4), (45, 45, 55), -1)
    if prog_pct > 0:
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + int(bar_w * prog_pct), bar_y + 4), timer_col, -1)

    # Missed Step Warning Banner (compact alert under mini panel)
    m_step = p.get("missed_step")
    if m_step and m_step.get("active"):
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (8, 8 + panel_h + 4), (8 + panel_w, 8 + panel_h + 42), (0, 0, 180), -1)
        cv2.addWeighted(alert_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (8, 8 + panel_h + 4), (8 + panel_w, 8 + panel_h + 42), (0, 0, 255), 1)
        cv2.putText(frame, f"! STEP {m_step['missed_step_number']} MISSED", (14, 8 + panel_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"INSPECT {m_step['missed_step_name'].upper()[:16]} FIRST", (14, 8 + panel_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 220), 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

# ============================================================
# CAMERA
# ============================================================

def open_camera():
    global CAMERA_INDEX
    candidates = []
    
    # Primary candidate
    candidates.append((CAMERA_INDEX, cv2.CAP_MSMF))
    candidates.append((CAMERA_INDEX, cv2.CAP_DSHOW))
    candidates.append((CAMERA_INDEX, cv2.CAP_ANY))
    
    # Fallback to alternate camera (if requested was 1, fallback to 0; if 0, fallback to 1)
    alt_idx = 0 if CAMERA_INDEX != 0 else 1
    candidates.append((alt_idx, cv2.CAP_DSHOW))
    candidates.append((alt_idx, cv2.CAP_MSMF))
    candidates.append((alt_idx, cv2.CAP_ANY))

    for idx, backend in candidates:
        try:
            backend_name = "CAP_MSMF" if backend == cv2.CAP_MSMF else ("CAP_DSHOW" if backend == cv2.CAP_DSHOW else "CAP_ANY")
            cam_label = "GlideX SharedCam" if idx == 1 else "Laptop Webcam"
            print(f"Attempting camera: index={idx} ({cam_label}), backend={backend_name}")
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                ok, frame = cap.read()
                if ok and frame is not None:
                    CAMERA_INDEX = idx
                    print(f"Camera connected successfully: {cam_label} (index={idx}), frame={frame.shape}")
                    publish_state(camera_index=idx, camera_name=cam_label)
                    return cap
                cap.release()
        except Exception as e:
            pass

    return None

# ============================================================
# MAIN
# ============================================================

def main():
    global PROTOCOL
    PROTOCOL = InspectionProtocol()
    publish_state(camera="starting", phase="INSPECTION", experiment_started=False, completed=False,
                  status="Loading YOLO + MediaPipe…", timer=0.0, timer_running=False,
                  timer_started_epoch=None, step_number=1, current_object_id=None,
                  current_object=None, action="IDLE", grasp_active=False,
                  bbox_overlap_active=False, timer_trigger="NONE", hand_overlap_ratio=0.0,
                  hand_overlap_pixels=0.0, inspected_count=0, frame=0, fps=0.0,
                  sequence=PROTOCOL.sequence_payload())

    print("=" * 72)
    print("VYOMDRISHTI — INSPECTION SEQUENCE (GRASP + BBOX OVERLAP)")
    print("=" * 72)
    print("Inspection sequence:")
    for i, object_id in enumerate(INSPECTION_SEQUENCE, 1):
        print(f"  {i}. {CLASS_NAMES[object_id]}")

    model = YOLO(MODEL_PATH)
    print("YOLO loaded successfully.")
    print("Classes:", model.names)

    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    print("MediaPipe Hand Landmarker loaded.")

    cap = open_camera()
    while cap is None:
        if cv2.waitKey(1) == ord("q"):
            print("[VYOMDRISHTI] Camera start cancelled.")
            publish_state(camera="stopped", phase="STANDBY", status="Mission stopped by user.")
            return
        publish_state(camera="waiting", phase="INSPECTION", status="Waiting for camera (GlideX or Webcam)…", timer=0.0, timer_running=False)
        print("Could not open camera. Retrying in 1 second…")
        for _ in range(10):
            if cv2.waitKey(1) == ord("q"):
                return
            time.sleep(0.1)
        cap = open_camera()

    event_file = open(EVENT_LOG_PATH, "a", encoding="utf-8")
    action_detector = ActionDetector()
    frame_number = 0
    previous_tick = cv2.getTickCount()
    fps = 0.0
    failed_frames = 0

    publish_state(camera="running", phase="INSPECTION", status="Waiting for first experimental object…",
                  timer=0.0, timer_running=False, step_number=1, total_steps=4,
                  sequence=PROTOCOL.sequence_payload())

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_frames += 1
                if failed_frames >= MAX_FAILED_FRAMES:
                    cap.release()
                    time.sleep(1)
                    cap = open_camera()
                    failed_frames = 0
                continue
            failed_frames = 0
            frame_number += 1

            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            # YOLO object + hand detection.
            results = model.predict(frame, conf=CONFIDENCE, iou=IOU, imgsz=640, verbose=False)
            detections = extract_detections(results[0])

            # MediaPipe hand landmarks are retained for the ActionDetector only.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_detector.detect_for_video(mp_image, int(frame_number * 1000 / 30.0))
            fingertips = get_fingertips(hand_result, frame.shape[1], frame.shape[0])

            action_states = {}
            for d in detections:
                if d.class_id in OBJECT_IDS:
                    action_states[d.class_id] = action_detector.update(d, fingertips)

            protocol_out = PROTOCOL.update(detections, action_states, time.monotonic())
            if protocol_out.get("event"):
                event_file.write(json.dumps(protocol_out["event"], ensure_ascii=False) + "\n")
                event_file.flush()

            current_id = protocol_out.get("object_id")
            current_obj = best_detection_for_class(detections, current_id) if current_id is not None else None

            # Draw all detections; emphasize current object and best overlapping hand.
            m_step = protocol_out.get("missed_step")
            is_viol_active = bool(m_step and m_step.get("active"))
            viol_performed_id = m_step.get("performed_object_id") if is_viol_active else None
            viol_missed_id = m_step.get("missed_object_id") if is_viol_active else None

            for d in detections:
                status = None
                is_viol = bool(is_viol_active and d.class_id == viol_performed_id)
                is_missed = bool(is_viol_active and d.class_id == viol_missed_id)
                if d.class_id in PROTOCOL.inspected:
                    status = "INSPECTED"
                elif d.class_id == current_id:
                    state = action_states.get(d.class_id)
                    status = getattr(state, "action", "IDLE") if state else "WAITING"
                draw_detection(frame, d, status, is_violation=is_viol, is_missed_target=is_missed)

            for p in fingertips:
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0,0,255), -1)

            if current_obj is not None:
                hand, ratio, _ = best_hand_for_object(current_obj, detections)
                if hand is not None:
                    cv2.rectangle(frame, (hand.box[0], hand.box[1]), (hand.box[2], hand.box[3]), (0,255,255), 3)
                    # Visual line between centers, showing which hand bbox is being evaluated.
                    cv2.line(frame, tuple(map(int, hand.center)), tuple(map(int, current_obj.center)), (0,255,255), 2)

            tick = cv2.getTickCount()
            dt = (tick - previous_tick) / cv2.getTickFrequency()
            previous_tick = tick
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            step_number = min(PROTOCOL.index + 1, 4)
            if PROTOCOL.complete:
                step_number = 4

            publish_state(
                camera="running",
                phase="COMPLETE" if PROTOCOL.complete else "INSPECTION",
                experiment_started=PROTOCOL.started,
                completed=PROTOCOL.complete,
                status=protocol_out["status"],
                action=protocol_out["action"],
                grasp_active=protocol_out["grasp_active"],
                bbox_overlap_active=protocol_out["bbox_overlap_active"],
                timer_trigger=protocol_out["timer_trigger"],
                timer=float(protocol_out["timer"]),
                timer_duration=INSPECTION_DURATION_SECONDS,
                timer_running=bool(protocol_out["timer_running"]),
                timer_decaying=bool(protocol_out.get("timer_decaying", False)),
                timer_started_epoch=protocol_out["timer_started_epoch"],
                step_number=protocol_out.get("step_number", step_number),
                total_steps=4,
                current_object_id=protocol_out["object_id"],
                current_object=CLASS_NAMES.get(protocol_out["object_id"]) if protocol_out["object_id"] is not None else None,
                next_step_number=protocol_out.get("next_step_number"),
                next_object_id=protocol_out.get("next_object_id"),
                next_object=protocol_out.get("next_object"),
                missed_step=protocol_out.get("missed_step"),
                voice_event=protocol_out.get("voice_event", LIVE_STATE.get("voice_event")),
                hand_overlap_ratio=float(protocol_out["hand_overlap_ratio"]),
                hand_overlap_pixels=float(protocol_out["hand_overlap_pixels"]),
                inspected_count=len(PROTOCOL.inspected),
                frame=frame_number,
                fps=round(fps, 1),
                sequence=PROTOCOL.sequence_payload(),
            )

            draw_inspection_panel(frame, protocol_out, PROTOCOL)
            draw_fps(frame, fps)

            # The server intercepts this and streams the processed frame.
            cv2.imshow("VYOMDRISHTI", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        event_file.close()
        hand_detector.close()
        cv2.destroyAllWindows()
        publish_state(camera="stopped", phase="STOPPED", status="Camera stopped.", timer=0.0,
                      timer_running=False, timer_started_epoch=None, action="IDLE", grasp_active=False,
                      bbox_overlap_active=False, timer_trigger="NONE")
        print("Camera stopped.")


def reset_experiment():
    global PROTOCOL
    PROTOCOL = InspectionProtocol()
    reset_event = create_voice_event(
        "RESET",
        f"Sequence reset. Step 1: {CLASS_NAMES[0].replace('_', ' ')}. Next: Step 2, {CLASS_NAMES[1].replace('_', ' ')}.",
        tone="info",
        step_number=1,
        next_step_number=2
    )
    publish_state(
        phase="INSPECTION",
        experiment_started=False,
        completed=False,
        status="Experiment reset — waiting for first experimental object…",
        action="IDLE",
        grasp_active=False,
        bbox_overlap_active=False,
        timer_trigger="NONE",
        timer=0.0,
        timer_running=False,
        timer_decaying=False,
        timer_started_epoch=None,
        step_number=1,
        current_object_id=None,
        current_object=None,
        next_step_number=2,
        next_object_id=1,
        next_object="red_computer_mouse",
        missed_step={
            "active": False,
            "missed_step_number": None,
            "missed_step_name": None,
            "missed_object_id": None,
            "performed_step_number": None,
            "performed_step_name": None,
            "performed_object_id": None,
            "message": "",
        },
        voice_event=reset_event,
        hand_overlap_ratio=0.0,
        hand_overlap_pixels=0.0,
        inspected_count=0,
        sequence=PROTOCOL.sequence_payload(),
    )


if __name__ == "__main__":
    main()
