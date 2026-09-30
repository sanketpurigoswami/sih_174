"""
detector.py — Continuous AI Detection & Mission Control FSM Engine for VIGIL.

Integrates:
  1. YOLOv11 trained weights (runs/detect/train/weights/best.pt).
  2. MediaPipe Hand Landmarker (models/hand_landmarker.task).
  3. Spatial Action FSM (GRASP, MOVE, RELEASE, PLACED).
  4. Inspection Hold Timer Protocol (5.0s grasp/overlap countdown).
  5. Placement Surface Verification (color square bounding box overlap).
  6. Continuous Standby Stream Inference + Active Mission Step Sequencer.
  7. Real-Time Telemetry Socket.IO Broadcaster.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Model paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
FALLBACK_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11n.pt")
HAND_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

# Detection tuning constants
CONFIDENCE = 0.35
IOU = 0.45
CONTACT_DISTANCE = 32.0
GRASP_FINGERS_REQUIRED = 3
PLACE_OVERLAP_THRESHOLD = 0.20
INSPECTION_DURATION_SECONDS = 5.0
ACTION_CONFIRM_FRAMES = 4
STEP_TIMEOUT_SECONDS = 180.0

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
PLACE_IDS = {3, 4, 5, 6, 7}
HAND_ID = 9

# Color mapping for HUD drawing (BGR)
COLOR_PALETTE = {
    "orange_circular_cap": (20, 140, 255),    # Vibrant Orange
    "red_computer_mouse": (40, 40, 240),      # Crimson Red
    "wireless_earbuds_case": (230, 210, 100), # Cyan / Sky
    "dark_blue_square": (200, 70, 20),        # Deep Blue
    "yellow_square": (30, 220, 240),          # Bright Yellow
    "pink_square": (180, 80, 240),            # Magenta / Pink
    "green_square": (60, 220, 80),            # Neon Green
    "white_box": (220, 220, 220),             # Crisp White
    "person": (140, 124, 251),                # Violet Accent
    "hand": (240, 160, 60),                   # Light Blue
}


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]
    center: Tuple[float, float]


def point_to_box_distance(px: float, py: float, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


def box_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def box_overlap_ratio(obj_box: Tuple[int, int, int, int], place_box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = obj_box
    area = float(max(1, x2 - x1) * max(1, y2 - y1))
    if area <= 0:
        return 0.0
    return box_intersection_area(obj_box, place_box) / area


class Detector:
    """
    Continuous detection loop + mission step evaluation engine.
    Emits real-time telemetry, step updates, timeline rows, and alerts.
    """

    def __init__(self, socketio, storage_module, video_stream):
        self.socketio = socketio
        self.storage = storage_module
        self.video = video_stream

        # Mission state
        self.running = False
        self.current_experiment: Optional[Dict[str, Any]] = None
        self.mission_id: Optional[str] = None
        self.current_step_index: int = 0
        self._step_statuses: Dict[int, str] = {}
        self._step_confidences: Dict[int, float] = {}

        # Inspection hold timer state
        self._timer_started_at: Optional[float] = None
        self._timer_duration: float = INSPECTION_DURATION_SECONDS
        self._timer_elapsed: float = 0.0
        self._timer_active: bool = False

        # Action detector persistence
        self._previous_centers: Dict[int, Tuple[float, float]] = {}
        self._confirm_counter: int = 0
        self._last_event_text: str = "IDLE"

        # Models
        self._yolo_model = None
        self._hand_detector = None
        self._models_loaded = False
        self._last_mp_timestamp_ms = 0
        self.gpu_device_name = "CPU"

        # Lock
        self._lock = threading.Lock()

        # Telemetry metrics
        self._telemetry_fps: float = 0.0
        self._last_telemetry_emit: float = 0.0

        # Background processing worker
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._continuous_detection_loop, daemon=True)
        self._worker_thread.start()

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_models(self) -> bool:
        """Loads YOLO and MediaPipe models once."""
        if self._models_loaded:
            return True

        # 1. Load YOLO
        try:
            from ultralytics import YOLO
            import torch
            if torch.cuda.is_available():
                self.gpu_device_name = f"CUDA: {torch.cuda.get_device_name(0)}"
            else:
                self.gpu_device_name = "CPU"

            path_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
            print(f"[VIGIL detector] Loading YOLO model from {path_to_use} ({self.gpu_device_name})...")
            self._yolo_model = YOLO(path_to_use)
            print("[VIGIL detector] YOLO loaded successfully.")
        except Exception as exc:
            print(f"[VIGIL detector] Warning loading YOLO: {exc}")
            self._yolo_model = None

        # 2. Load MediaPipe Hand Landmarker
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if os.path.exists(HAND_MODEL_PATH):
                print(f"[VIGIL detector] Loading MediaPipe Hand Landmarker: {HAND_MODEL_PATH}...")
                base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
                hand_options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_hand_presence_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self._hand_detector = mp_vision.HandLandmarker.create_from_options(hand_options)
                print("[VIGIL detector] MediaPipe Hand Landmarker loaded successfully.")
            else:
                print(f"[VIGIL detector] Warning: hand model not found at {HAND_MODEL_PATH}")
                self._hand_detector = None
        except Exception as exc:
            print(f"[VIGIL detector] Warning loading MediaPipe: {exc}")
            self._hand_detector = None

        self._models_loaded = True
        return True

    # ── Continuous Standby & Active Detection Loop ───────────────────────────

    def _continuous_detection_loop(self):
        """Continuously pulls frames, runs inference, renders HUD, and updates mission."""
        self._load_models()

        fps_timer = time.time()
        fps_frames = 0

        while self._worker_running:
            raw_frame = self.video.get_frame()
            if raw_frame is None or raw_frame.size == 0:
                time.sleep(0.04)
                continue

            frame_start = time.time()

            # Process frame through YOLO + MediaPipe
            analysis = self._analyze_frame(raw_frame)

            # Push annotated frame for live streaming
            if analysis.get("annotated") is not None:
                self.video.set_annotated_frame(analysis["annotated"])

            # If mission is active, evaluate step progression
            if self.running and self.current_experiment:
                self._evaluate_mission_step(analysis)

            # Measure inference FPS
            fps_frames += 1
            if time.time() - fps_timer >= 1.0:
                self._telemetry_fps = round(fps_frames / (time.time() - fps_timer), 1)
                fps_frames = 0
                fps_timer = time.time()

            # Emit live telemetry at ~10 Hz
            if time.time() - self._last_telemetry_emit >= 0.10:
                self._emit_telemetry(analysis)
                self._last_telemetry_emit = time.time()

            # Dynamic pacing
            took = time.time() - frame_start
            sleep_time = max(0.005, 0.035 - took)
            time.sleep(sleep_time)

    def process_external_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Processes an incoming frame directly from client browser webcam."""
        with self._lock:
            analysis = self._analyze_frame(frame)
            if analysis.get("annotated") is not None:
                self.video.set_annotated_frame(analysis["annotated"])

            if self.running and self.current_experiment:
                self._evaluate_mission_step(analysis)

            self._emit_telemetry(analysis)
            return analysis

    # ── Frame Analysis (YOLO + MediaPipe + FSM) ──────────────────────────────


    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Runs object detection, hand landmarking, and computes contact/overlap."""

        h, w = frame.shape[:2]
        detections: List[Detection] = []
        hand_fingertips: List[Tuple[int, int]] = []
        hand_landmarks_list: List[Any] = []
        yolo_hand_boxes: List[Tuple[int, int, int, int]] = []

        # 1. YOLO inference
        if self._yolo_model is not None:
            try:
                results = self._yolo_model.predict(
                    source=frame,
                    conf=CONFIDENCE,
                    iou=IOU,
                    verbose=False,
                )
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for box, cls_id, conf in zip(boxes, classes, confs):
                        cid = int(cls_id)
                        cname = CLASS_NAMES.get(cid, self._yolo_model.names.get(cid, f"class_{cid}"))
                        x1, y1, x2, y2 = map(int, box)
                        bbox = (x1, y1, x2, y2)
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        det = Detection(
                            class_id=cid,
                            name=cname,
                            confidence=float(conf),
                            box=bbox,
                            center=center
                        )
                        detections.append(det)
                        if cid == HAND_ID:
                            yolo_hand_boxes.append(bbox)
            except Exception as exc:
                print(f"[VIGIL detector] YOLO prediction error: {exc}")

        # 2. MediaPipe Hand Landmark inference
        if self._hand_detector is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Ensure strictly monotonic timestamps
                current_ms = int(time.time() * 1000)
                if current_ms <= self._last_mp_timestamp_ms:
                    current_ms = self._last_mp_timestamp_ms + 1
                self._last_mp_timestamp_ms = current_ms

                res = self._hand_detector.detect_for_video(mp_img, current_ms)
                if res and res.hand_landmarks:
                    fingertip_idx = (4, 8, 12, 16, 20)
                    for hand in res.hand_landmarks:
                        hand_landmarks_list.append(hand)
                        for idx in fingertip_idx:
                            lm = hand[idx]
                            hand_fingertips.append((int(lm.x * w), int(lm.y * h)))
            except Exception as exc:
                print(f"[VIGIL detector] MediaPipe error: {exc}")

        # 3. Categorize detections
        obj_detections = [d for d in detections if d.class_id in OBJECT_IDS]
        place_detections = [d for d in detections if d.class_id in PLACE_IDS]

        # 4. Action determination
        grasp_active = False
        grasped_obj_name = None
        contact_lines = []
        max_overlap_ratio = 0.0
        primary_action = "IDLE"
        current_obj_candidate = None

        if obj_detections:
            # Sort by confidence
            obj_detections.sort(key=lambda d: d.confidence, reverse=True)
            current_obj_candidate = obj_detections[0]

            for obj in obj_detections:
                # Check fingertip contacts
                fingers_in_contact = 0
                for pt in hand_fingertips:
                    dist = point_to_box_distance(pt[0], pt[1], obj.box)
                    if dist <= CONTACT_DISTANCE:
                        fingers_in_contact += 1
                        contact_lines.append((pt, (int(obj.center[0]), int(obj.center[1]))))

                # Check YOLO hand/object bbox overlap
                for hbox in yolo_hand_boxes:
                    ov = box_overlap_ratio(obj.box, hbox)
                    if ov > max_overlap_ratio:
                        max_overlap_ratio = ov

                if fingers_in_contact >= GRASP_FINGERS_REQUIRED:
                    grasp_active = True
                    grasped_obj_name = obj.name
                    current_obj_candidate = obj

                    # Check movement
                    prev = self._previous_centers.get(obj.class_id)
                    if prev:
                        movement = math.hypot(obj.center[0] - prev[0], obj.center[1] - prev[1])
                        if movement > 12.0:
                            primary_action = "MOVE"
                        else:
                            primary_action = "GRASP"
                    else:
                        primary_action = "GRASP"
                    self._previous_centers[obj.class_id] = obj.center
                    break

            # If not grasped by contact, check if high bbox overlap
            if not grasp_active and max_overlap_ratio > 0.15:
                primary_action = "GRASP"
                grasp_active = True

        # Check placement overlap
        placement_zone = None
        if obj_detections and place_detections:
            for obj in obj_detections:
                for pl in place_detections:
                    if pl.class_id == obj.class_id:
                        continue
                    ov = box_overlap_ratio(obj.box, pl.box)
                    if ov >= PLACE_OVERLAP_THRESHOLD:
                        placement_zone = pl.name
                        if not grasp_active:
                            primary_action = "PLACED"

        # 5. Render HUD Annotated Frame
        annotated = self._render_annotated_frame(
            frame,
            detections,
            hand_landmarks_list,
            hand_fingertips,
            contact_lines,
            primary_action,
            grasped_obj_name or (current_obj_candidate.name if current_obj_candidate else "NONE"),
            max_overlap_ratio,
            grasp_active,
            placement_zone
        )

        return {
            "annotated": annotated,
            "detections": detections,
            "objects": obj_detections,
            "places": place_detections,
            "action": primary_action,
            "current_object": current_obj_candidate.name if current_obj_candidate else None,
            "current_object_conf": current_obj_candidate.confidence if current_obj_candidate else 0.0,
            "grasp_active": grasp_active,
            "bbox_overlap": max_overlap_ratio,
            "target_surface": placement_zone,
            "hands_count": len(hand_landmarks_list) or len(yolo_hand_boxes),
        }

    # ── HUD Drawing ──────────────────────────────────────────────────────────

    def _render_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        hand_landmarks_list: List[Any],
        hand_fingertips: List[Tuple[int, int]],
        contact_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        action: str,
        active_obj_name: str,
        overlap_ratio: float,
        grasp_active: bool,
        placement_zone: Optional[str]
    ) -> np.ndarray:
        """Draws bounding boxes, landmarks, contact vectors, and telemetry HUD."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = COLOR_PALETTE.get(det.name, (180, 180, 180))
            is_surface = det.class_id in PLACE_IDS and det.class_id not in OBJECT_IDS

            # Bounding box
            thickness = 2 if is_surface else 3
            if is_surface:
                # Dashed/semi-transparent style for placement surface
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            conf_str = f"{det.confidence * 100:.0f}%"
            label = f"{det.name.replace('_', ' ')} · {conf_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(out, label, (x1 + 5, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0) if color == (30, 220, 240) else (255, 255, 255), 1, cv2.LINE_AA)

        # Draw MediaPipe hand landmarks
        for pt in hand_fingertips:
            cv2.circle(out, pt, 5, (0, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw contact lines
        for pt, center in contact_lines:
            cv2.line(out, pt, center, (74, 222, 154), 2, cv2.LINE_AA)

        # ── Top HUD Status Strip ──
        cv2.rectangle(out, (0, 0), (w, 42), (10, 10, 14), -1)
        cv2.line(out, (0, 42), (w, 42), (40, 40, 50), 1)

        # Mission state badge
        mission_text = f"MISSION: {self.current_experiment['name']}" if self.running and self.current_experiment else "VIGIL STANDBY"
        mission_color = (140, 124, 251) if self.running else (140, 142, 150)
        cv2.putText(out, mission_text.upper(), (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mission_color, 2, cv2.LINE_AA)

        # Action Badge (right aligned)
        act_color = {
            "GRASP": (74, 222, 154),
            "MOVE": (240, 180, 50),
            "RELEASE": (240, 120, 110),
            "PLACED": (74, 222, 154),
            "INSPECTING": (140, 124, 251),
        }.get(action, (150, 150, 150))

        action_label = f"ACTION: {action}"
        if self._timer_active and self._timer_elapsed > 0:
            action_label = f"INSPECTING: {self._timer_elapsed:.1f}/{self._timer_duration:.1f}s"
            act_color = (140, 124, 251)

        (aw, _), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        cv2.putText(out, action_label, (w - aw - 16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, act_color, 2, cv2.LINE_AA)

        # Target object indicator
        if active_obj_name != "NONE":
            obj_label = f"OBJ: {active_obj_name.replace('_', ' ')}"
            cv2.putText(out, obj_label, (w // 2 - 100, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (237, 237, 235), 1, cv2.LINE_AA)

        # Bottom HUD: Overlap & Grasp indicator
        cv2.rectangle(out, (0, h - 28), (w, h), (10, 10, 14), -1)
        telemetry_line = f"GRASP: {'YES' if grasp_active else 'NO'}  |  OVERLAP: {overlap_ratio * 100:.1f}%  |  FPS: {self._telemetry_fps}  |  {self.gpu_device_name}"
        if placement_zone:
            telemetry_line += f"  |  SURFACE: {placement_zone.replace('_', ' ')}"
        cv2.putText(out, telemetry_line, (16, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 142, 150), 1, cv2.LINE_AA)

        return out

    # ── Mission Step Evaluator ───────────────────────────────────────────────

    def _evaluate_mission_step(self, analysis: Dict[str, Any]):
        """Evaluates whether current frame satisfies the active step criteria."""
        if not self.running or not self.current_experiment:
            return

        steps = self.current_experiment.get("steps", [])
        if self.current_step_index >= len(steps):
            self._complete_mission()
            return

        current_step = steps[self.current_step_index]
        step_id = current_step["id"]
        expected_obj = current_step.get("expected_object", "").lower()
        expected_act = current_step.get("expected_action", "").upper()
        target_surface = current_step.get("target_surface", "").lower()
        duration_req = float(current_step.get("duration", INSPECTION_DURATION_SECONDS))

        action_observed = analysis["action"]
        current_obj = (analysis["current_object"] or "").lower()
        grasp_active = analysis["grasp_active"]
        bbox_overlap = analysis["bbox_overlap"]
        target_detected = (analysis["target_surface"] or "").lower()

        # Object matching
        obj_matches = (
            expected_obj in current_obj
            or current_obj in expected_obj
            or expected_obj == ""
        )

        step_fulfilled = False
        confidence = analysis["current_object_conf"] or 0.85

        # ── Step Category 1: INSPECTION / HOLD (Timed hold) ──
        if expected_act in ("INSPECT", "HOLD"):
            is_holding = (grasp_active or bbox_overlap > 0.10) and obj_matches
            if is_holding:
                if self._timer_started_at is None:
                    self._timer_started_at = time.time()
                    self._timer_duration = duration_req
                    self._timer_active = True

                self._timer_elapsed = min(self._timer_duration, time.time() - self._timer_started_at)
                progress = min(1.0, self._timer_elapsed / self._timer_duration)

                # Emit step progress
                self.socketio.emit("step_update", {
                    "step_id": step_id,
                    "status": "in_progress",
                    "confidence": round(progress, 2),
                    "label": current_step["label"],
                    "timer": round(self._timer_elapsed, 1),
                    "duration": self._timer_duration,
                })

                if self._timer_elapsed >= self._timer_duration:
                    step_fulfilled = True
                    confidence = 0.98
            else:
                # Reset timer if contact lost before completion
                if self._timer_started_at is not None:
                    self._timer_started_at = None
                    self._timer_elapsed = 0.0
                    self._timer_active = False

        # ── Step Category 2: PLACEMENT (Destination zone check) ──
        elif expected_act in ("PLACE", "RELEASE"):
            surface_matches = (
                target_surface in target_detected
                or target_detected in target_surface
                or target_surface == ""
            )
            if obj_matches and surface_matches and (action_observed in ("PLACED", "RELEASE") or not grasp_active):
                self._confirm_counter += 1
                if self._confirm_counter >= ACTION_CONFIRM_FRAMES:
                    step_fulfilled = True
            else:
                self._confirm_counter = max(0, self._confirm_counter - 1)

        # ── Step Category 3: GRASP / PICKUP ──
        elif expected_act in ("GRASP", "PICK", "PICKUP", "GRAB"):
            if obj_matches and (grasp_active or action_observed == "GRASP"):
                self._confirm_counter += 1
                if self._confirm_counter >= ACTION_CONFIRM_FRAMES:
                    step_fulfilled = True
            else:
                self._confirm_counter = max(0, self._confirm_counter - 1)

        # ── Step Category 4: MOVE / TRANSPORT ──
        elif expected_act in ("MOVE", "TRANSPORT", "CARRY", "TRANSFER"):
            if obj_matches and action_observed == "MOVE":
                self._confirm_counter += 1
                if self._confirm_counter >= ACTION_CONFIRM_FRAMES:
                    step_fulfilled = True
            else:
                self._confirm_counter = max(0, self._confirm_counter - 1)

        # Default fallback match
        else:
            if obj_matches and action_observed == expected_act:
                self._confirm_counter += 1
                if self._confirm_counter >= ACTION_CONFIRM_FRAMES:
                    step_fulfilled = True

        # If step fulfilled, complete this step and advance
        if step_fulfilled:
            self._advance_step(step_id, current_step["label"], success=True, confidence=confidence)

    def _advance_step(self, step_id: int, step_label: str, success: bool, confidence: float):
        """Marks current step as done/skipped, logs to DB and Socket, advances to next."""
        status = "done" if success else "skipped"
        self._step_statuses[step_id] = status
        self._step_confidences[step_id] = confidence

        now_iso = datetime.now(timezone.utc).isoformat()
        phase_name = self.current_experiment["name"]

        # Emit final step update
        self.socketio.emit("step_update", {
            "step_id": step_id,
            "status": status,
            "confidence": round(confidence, 2),
            "label": step_label,
            "timer": self._timer_duration,
            "duration": self._timer_duration,
        })

        # Emit timeline row
        timeline_entry = {
            "phase": phase_name,
            "step": step_id,
            "step_label": step_label,
            "success": success,
            "accuracy": round(confidence, 4),
            "start_time": now_iso,
            "end_time": now_iso,
        }
        self.socketio.emit("timeline_row", timeline_entry)

        # Save to SQLite
        self.storage.add_timeline_row(
            mission_id=self.mission_id,
            phase=phase_name,
            step=step_id,
            step_label=step_label,
            success=success,
            accuracy=confidence,
            start_time=now_iso,
            end_time=now_iso,
        )

        # Voice alert announcement
        voice_msg = f"Step {step_id} verified: {step_label}." if success else f"Step {step_id} skipped."
        self.socketio.emit("alert", {"type": "success" if success else "warning", "message": voice_msg})

        # Reset counters & timer
        self._confirm_counter = 0
        self._timer_started_at = None
        self._timer_elapsed = 0.0
        self._timer_active = False

        # Advance step index
        self.current_step_index += 1
        steps = self.current_experiment.get("steps", [])
        if self.current_step_index >= len(steps):
            self._complete_mission()
        else:
            next_step = steps[self.current_step_index]
            self._step_statuses[next_step["id"]] = "in_progress"
            self.socketio.emit("step_update", {
                "step_id": next_step["id"],
                "status": "in_progress",
                "confidence": 0.0,
                "label": next_step["label"],
            })

    def _complete_mission(self):
        """Marks the overall mission as complete."""
        self.running = False
        phase_name = self.current_experiment["name"] if self.current_experiment else "Mission"
        self.storage.set_mission_state(
            status="completed",
            phase_id=self.current_experiment["id"] if self.current_experiment else None,
            mission_id=self.mission_id,
        )
        self.socketio.emit("system_status", {
            "status": "completed",
            "label": "MISSION COMPLETE",
        })
        self.socketio.emit("alert", {
            "type": "success",
            "message": f"Mission complete: All protocol steps for {phase_name} verified successfully!",
        })

    # ── Telemetry Broadcaster ────────────────────────────────────────────────

    def _emit_telemetry(self, analysis: Dict[str, Any]):
        """Emits real-time live telemetry data over Socket.IO."""
        active_step_id = None
        if self.running and self.current_experiment:
            steps = self.current_experiment.get("steps", [])
            if self.current_step_index < len(steps):
                active_step_id = steps[self.current_step_index]["id"]

        detected_list = [
            {"name": d.name, "confidence": round(d.confidence * 100, 1), "box": d.box}
            for d in analysis.get("detections", [])
        ]

        payload = {
            "fps": self._telemetry_fps,
            "action": analysis.get("action", "IDLE"),
            "current_object": analysis.get("current_object"),
            "current_object_conf": round(analysis.get("current_object_conf", 0.0) * 100, 1),
            "grasp_active": analysis.get("grasp_active", False),
            "bbox_overlap": round(analysis.get("bbox_overlap", 0.0) * 100, 1),
            "target_surface": analysis.get("target_surface"),
            "timer_active": self._timer_active,
            "timer_elapsed": round(self._timer_elapsed, 1),
            "timer_duration": self._timer_duration,
            "timer_progress": round((self._timer_elapsed / self._timer_duration) * 100, 1) if self._timer_duration > 0 else 0,
            "hands_count": analysis.get("hands_count", 0),
            "is_running": self.running,
            "active_step_id": active_step_id,
            "detected_objects": detected_list,
        }
        self.socketio.emit("telemetry_update", payload)

    # ── Public Mission Control API ───────────────────────────────────────────

    def start(self, experiment: Dict[str, Any]) -> Tuple[bool, str]:
        """Starts a mission for the specified experiment."""
        if self.running:
            return False, "A mission is already in progress."

        self.current_experiment = experiment
        self.mission_id = f"mission-{uuid.uuid4().hex[:8]}"
        self.current_step_index = 0
        self._confirm_counter = 0
        self._timer_started_at = None
        self._timer_elapsed = 0.0
        self._timer_active = False

        steps = experiment.get("steps", [])
        self._step_statuses = {s["id"]: "pending" for s in steps}
        self._step_confidences = {s["id"]: 0.0 for s in steps}

        if steps:
            self._step_statuses[steps[0]["id"]] = "in_progress"

        self.running = True

        # Persist to DB
        self.storage.set_mission_state(
            status="running",
            phase_id=experiment["id"],
            mission_id=self.mission_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        self.socketio.emit("system_status", {"status": "running", "label": "MISSION ACTIVE"})
        if steps:
            self.socketio.emit("step_update", {
                "step_id": steps[0]["id"],
                "status": "in_progress",
                "confidence": 0.0,
                "label": steps[0]["label"],
            })

        print(f"[VIGIL detector] Mission started: {experiment['name']} (ID: {self.mission_id})")
        return True, self.mission_id

    def stop(self):
        """Stops the active mission and resets state."""
        self.running = False
        self.current_experiment = None
        self.current_step_index = 0
        self._step_statuses = {}
        self._step_confidences = {}
        self._timer_started_at = None
        self._timer_elapsed = 0.0
        self._timer_active = False

        self.storage.set_mission_state(status="idle")
        self.socketio.emit("system_status", {"status": "nominal", "label": "SYSTEM NOMINAL"})
        print("[VIGIL detector] Mission stopped.")

    def pass_step(self) -> Tuple[bool, str]:
        """Manual operator override to pass current step with success."""
        if not self.running or not self.current_experiment:
            return False, "No active mission to pass step."

        steps = self.current_experiment.get("steps", [])
        if self.current_step_index >= len(steps):
            return False, "All steps already completed."

        current_step = steps[self.current_step_index]
        self._advance_step(current_step["id"], current_step["label"], success=True, confidence=1.0)
        return True, f"Manually passed Step {current_step['id']}"

    def skip_step(self) -> Tuple[bool, str]:
        """Manual operator override to skip current step."""
        if not self.running or not self.current_experiment:
            return False, "No active mission to skip step."

        steps = self.current_experiment.get("steps", [])
        if self.current_step_index >= len(steps):
            return False, "All steps already completed."

        current_step = steps[self.current_step_index]
        self._advance_step(current_step["id"], current_step["label"], success=False, confidence=0.0)
        return True, f"Manually skipped Step {current_step['id']}"

    def get_step_statuses(self) -> Dict[int, str]:
        return self._step_statuses

    def is_running(self) -> bool:
        return self.running
