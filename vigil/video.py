"""
video.py — Universal Multi-Source Capture + MJPEG Stream Generator for VIGIL.

Supports:
  1. ASUS GlideX SharedCam (Camera 1) via cv2.CAP_MSMF.
  2. Integrated / External Webcams (Camera 0, etc.) via DSHOW / MSMF.
  3. Video File Playback (e.g., box.mp4, tester.mp4, screw.mp4) with auto-looping.
  4. Animated Synthetic Radar Placeholder fallback when no hardware is connected.

Features:
  - Thread-safe non-blocking background frame grabber.
  - Real-time measured FPS calculation.
  - Overlay toggle (AI annotations on/off).
  - Snapshot export.
  - Dynamic source switching at runtime.
"""

from __future__ import annotations

import glob
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VideoStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.annotated_lock = threading.Lock()

        # Capture backend state
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type: str = "camera"  # "camera", "video", or "placeholder"
        self.source_val: Any = 1          # camera index or file path
        self.source_name: str = "ASUS GlideX SharedCam (Index 1)"

        self.width: int = 1280
        self.height: int = 720
        self.fps: float = 0.0
        self._fps_count: int = 0
        self._fps_start_time: float = time.time()

        # Frame buffers
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._annotated_timestamp: float = 0.0
        self.show_overlay: bool = True

        self.using_placeholder: bool = False
        self._running: bool = True

        # Initial source startup
        self._init_best_source()

        # Start continuous frame capture worker
        self._capture_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._capture_thread.start()

    # ── Source Initialization ────────────────────────────────────────────────

    def _init_best_source(self):
        """Try Camera 1 (GlideX MSMF), then Camera 0, then placeholder."""
        # 1. Try GlideX (Index 1 with MSMF)
        if self._open_camera(1, cv2.CAP_MSMF):
            self.source_type = "camera"
            self.source_val = 1
            self.source_name = "Camera 1 — GlideX SharedCam (MSMF)"
            self.using_placeholder = False
            return

        # 2. Try Camera 0 with DSHOW or MSMF
        if self._open_camera(0, cv2.CAP_DSHOW) or self._open_camera(0, cv2.CAP_MSMF):
            self.source_type = "camera"
            self.source_val = 0
            self.source_name = "Camera 0 — Laptop / USB Webcam"
            self.using_placeholder = False
            return

        # 3. Check if test video files exist
        test_video = os.path.join(PROJECT_ROOT, "tester.mp4")
        if os.path.exists(test_video) and self._open_video_file(test_video):
            self.source_type = "video"
            self.source_val = "tester.mp4"
            self.source_name = "Video File — tester.mp4"
            self.using_placeholder = False
            return

        # Fallback to placeholder
        self.using_placeholder = True
        self.source_type = "placeholder"
        self.source_val = "synthetic"
        self.source_name = "Synthetic Placeholder Radar"

    def _open_camera(self, index: int, backend: int) -> bool:
        """Attempt to open camera with specified index and backend."""
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Warm-read verification
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                return False

            with self.lock:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.cap = cap
                self.height, self.width = frame.shape[:2]
                self._latest_raw_frame = frame.copy()

            backend_str = "MSMF" if backend == cv2.CAP_MSMF else "DSHOW"
            print(f"[VIGIL video] Opened Camera {index} ({backend_str}) at {self.width}x{self.height}")
            return True
        except Exception as exc:
            print(f"[VIGIL video] Error opening Camera {index}: {exc}")
            return False

    def _open_video_file(self, filepath: str) -> bool:
        """Attempt to open an MP4 video file for playback."""
        try:
            if not os.path.isabs(filepath):
                filepath = os.path.join(PROJECT_ROOT, filepath)

            if not os.path.exists(filepath):
                return False

            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                cap.release()
                return False

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                return False

            # Reset to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            with self.lock:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.cap = cap
                self.height, self.width = frame.shape[:2]
                self._latest_raw_frame = frame.copy()

            print(f"[VIGIL video] Opened Video File: {os.path.basename(filepath)} at {self.width}x{self.height}")
            return True
        except Exception as exc:
            print(f"[VIGIL video] Error opening video file {filepath}: {exc}")
            return False

    # ── Background Reader Loop ───────────────────────────────────────────────

    def _reader_loop(self):
        """Dedicated thread reading frames to keep capture buffer clean & measure FPS."""
        while self._running:
            if self.using_placeholder or self.cap is None:
                time.sleep(0.04)
                continue

            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    self.using_placeholder = True
                    continue

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    if self.source_type == "video":
                        # Loop video back to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self.cap.read()
                        if not ok or frame is None:
                            self.using_placeholder = True
                            continue
                    else:
                        self.using_placeholder = True
                        continue

                self._latest_raw_frame = frame
                self.height, self.width = frame.shape[:2]

            # Measure frame rate
            self._fps_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self.fps = round(self._fps_count / elapsed, 1)
                self._fps_count = 0
                self._fps_start_time = time.time()

            # Throttle video file playback to ~30 fps
            if self.source_type == "video":
                time.sleep(0.03)
            else:
                time.sleep(0.005)

    # ── Source Switching API ─────────────────────────────────────────────────

    def set_source(self, source_type: str, source_val: Any) -> Tuple[bool, str]:
        """Switch video source dynamically."""
        print(f"[VIGIL video] Request to switch source: {source_type} -> {source_val}")
        if source_type == "camera":
            try:
                cam_idx = int(source_val)
            except (ValueError, TypeError):
                cam_idx = 0

            # Try MSMF first for index 1, DSHOW for index 0
            backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if cam_idx == 1 else [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            opened = False
            for b in backends:
                if self._open_camera(cam_idx, b):
                    self.source_type = "camera"
                    self.source_val = cam_idx
                    b_name = "MSMF" if b == cv2.CAP_MSMF else "DSHOW"
                    self.source_name = f"Camera {cam_idx} ({b_name})"
                    self.using_placeholder = False
                    opened = True
                    break

            if opened:
                return True, f"Successfully switched to {self.source_name}"
            return False, f"Could not connect to camera {cam_idx}"

        elif source_type == "video":
            filename = str(source_val)
            filepath = filename if os.path.isabs(filename) else os.path.join(PROJECT_ROOT, filename)
            if self._open_video_file(filepath):
                self.source_type = "video"
                self.source_val = os.path.basename(filename)
                self.source_name = f"Video: {os.path.basename(filename)}"
                self.using_placeholder = False
                return True, f"Now playing video file: {os.path.basename(filename)}"
            return False, f"Could not open video file: {filename}"

        elif source_type == "placeholder":
            with self.lock:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                self.cap = None
            self.using_placeholder = True
            self.source_type = "placeholder"
            self.source_val = "synthetic"
            self.source_name = "Synthetic Placeholder Radar"
            return True, "Switched to Synthetic Placeholder feed"

        return False, f"Unknown source type: {source_type}"

    def get_available_sources(self) -> Dict[str, Any]:
        """Discover connected physical cameras and available workspace videos."""
        cameras = []

        # Check Camera 1 (ASUS GlideX)
        try:
            c1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
            if c1.isOpened():
                cameras.append({
                    "id": 1,
                    "name": "Camera 1 — ASUS GlideX SharedCam",
                    "backend": "CAP_MSMF",
                    "recommended": True
                })
                c1.release()
        except Exception:
            pass

        # Check Camera 0 (Webcam)
        try:
            c0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if c0.isOpened():
                cameras.append({
                    "id": 0,
                    "name": "Camera 0 — Integrated / USB Webcam",
                    "backend": "CAP_DSHOW",
                    "recommended": False
                })
                c0.release()
        except Exception:
            pass

        # If no cameras found via probe, add default fallback options
        if not cameras:
            cameras.append({"id": 1, "name": "Camera 1 — ASUS GlideX SharedCam", "backend": "CAP_MSMF"})
            cameras.append({"id": 0, "name": "Camera 0 — Integrated Webcam", "backend": "CAP_DSHOW"})

        # Video files in project directory
        video_files = []
        for ext in ("*.mp4", "*.avi"):
            for vpath in glob.glob(os.path.join(PROJECT_ROOT, ext)):
                vname = os.path.basename(vpath)
                size_mb = round(os.path.getsize(vpath) / (1024 * 1024), 1)
                video_files.append({"name": vname, "size_mb": size_mb})

        return {
            "current": {
                "type": self.source_type,
                "value": self.source_val,
                "name": self.source_name,
                "resolution": f"{self.width}x{self.height}",
                "fps": self.fps,
                "is_live": not self.using_placeholder,
                "overlay_active": self.show_overlay
            },
            "cameras": cameras,
            "videos": video_files,
        }

    # ── Annotated Frame Support ──────────────────────────────────────────────

    def set_annotated_frame(self, frame: np.ndarray):
        """Called by Detector to publish an annotated frame."""
        with self.annotated_lock:
            self._annotated_frame = frame.copy()
            self._annotated_timestamp = time.time()

    def _get_annotated_frame(self) -> Optional[np.ndarray]:
        with self.annotated_lock:
            # Drop annotated frame if older than 1.5 seconds to prevent stale freezes
            if self._annotated_frame is not None and (time.time() - self._annotated_timestamp < 1.5):
                return self._annotated_frame
            return None

    def toggle_overlay(self) -> bool:
        """Toggle AI bounding box and telemetry overlay on the stream."""
        self.show_overlay = not self.show_overlay
        return self.show_overlay

    # ── Frame Access ─────────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        """Return raw camera/video frame for detection processing."""
        if self.using_placeholder or self._latest_raw_frame is None:
            tick = int(time.time() * 10) % 3600
            return self._make_placeholder(tick)
        return self._latest_raw_frame.copy()

    def get_display_frame(self) -> np.ndarray:
        """Return annotated frame if overlay is active, else raw frame."""
        if self.show_overlay:
            annotated = self._get_annotated_frame()
            if annotated is not None:
                return annotated

        if self.using_placeholder or self._latest_raw_frame is None:
            tick = int(time.time() * 10) % 3600
            return self._make_placeholder(tick)

        return self._latest_raw_frame.copy()

    def is_camera_available(self) -> bool:
        return not self.using_placeholder

    # ── Placeholder Generator ────────────────────────────────────────────────

    def _make_placeholder(self, tick: int) -> np.ndarray:
        """Generate animated dark-mode radar mission control placeholder."""
        w, h = 640, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Subtle mission grid
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (20, 20, 24), 1)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (20, 20, 24), 1)

        cx, cy = w // 2, h // 2

        # Rotating radar sweep
        angle = (tick * 4) % 360
        rad = math.radians(angle)
        ex = int(cx + 170 * math.cos(rad))
        ey = int(cy + 170 * math.sin(rad))
        cv2.line(frame, (cx, cy), (ex, ey), (140, 124, 251), 2, cv2.LINE_AA)

        # Reticles
        for r in (40, 80, 120, 160):
            cv2.circle(frame, (cx, cy), r, (40, 40, 48), 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 180, cy), (cx + 180, cy), (40, 40, 48), 1)
        cv2.line(frame, (cx, cy - 180), (cx, cy + 180), (40, 40, 48), 1)

        # Header branding
        cv2.putText(frame, "VIGIL // MISSION CONTROL", (cx - 145, cy - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 124, 251), 2, cv2.LINE_AA)
        cv2.putText(frame, "AWAITING OPTICAL FEED", (cx - 110, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 142, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "Select GlideX, Webcam, or Video File from UI", (cx - 150, cy + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (95, 97, 103), 1, cv2.LINE_AA)

        # Heartbeat pulse
        if tick % 2 == 0:
            cv2.circle(frame, (cx + 160, cy - 100), 5, (74, 222, 154), -1)

        return frame

    # ── MJPEG Stream Generator ───────────────────────────────────────────────

    def generate(self):
        """Yields MJPEG frames as multipart chunks."""
        while self._running:
            frame = self.get_display_frame()
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                time.sleep(0.04)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(0.04)  # ~25 fps stream

    def get_snapshot_jpeg(self) -> bytes:
        """Returns single JPEG snapshot of current display frame."""
        frame = self.get_display_frame()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buf.tobytes() if ok else b""

    def release(self):
        self._running = False
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
