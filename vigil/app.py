"""
app.py — Flask + Flask-SocketIO Application for VIGIL Mission Control.

Serves:
  - Dashboard Single-Page Web Application.
  - Low-latency MJPEG video stream (/video_feed) and frame snapshots (/snapshot).
  - REST API for experiments, missions, camera selection, and diagnostics.
  - Real-time WebSocket bidirectional telemetry and state synchronization.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# Ensure project root and vigil directory are importable
_VIGIL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_VIGIL_DIR)
if _VIGIL_DIR not in sys.path:
    sys.path.insert(0, _VIGIL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, Response, jsonify, make_response, render_template, request
from flask_socketio import SocketIO, emit

import storage
from detector import Detector
from video import VideoStream

# ── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = "vigil-mission-control-key-2026"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize database, video streaming, and detection subsystem
storage.init_db()
video_stream = VideoStream()
detector = Detector(socketio, storage, video_stream)


# ── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Video Feed & Snapshot ────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/snapshot")
def snapshot():
    jpeg_bytes = video_stream.get_snapshot_jpeg()
    return Response(
        jpeg_bytes,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache"
        }
    )


# ── Camera & Video Source API ────────────────────────────────────────────────

@app.route("/api/cameras", methods=["GET"])
def api_get_cameras():
    """Returns available physical cameras and video test files."""
    return jsonify(video_stream.get_available_sources())


@app.route("/api/camera/select", methods=["POST"])
def api_select_camera():
    """Switches active camera or video file."""
    body = request.get_json(force=True, silent=True) or {}
    source_type = body.get("source_type", "camera")
    source_val = body.get("source_val", 1)
    ok, msg = video_stream.set_source(source_type, source_val)
    if ok:
        socketio.emit("alert", {"type": "success", "message": msg})
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.route("/api/camera/toggle_overlay", methods=["POST"])
def api_toggle_overlay():
    """Toggles AI detection bounding boxes overlay on/off."""
    active = video_stream.toggle_overlay()
    return jsonify({"ok": True, "overlay_active": active})


# ── Experiment Catalog API ───────────────────────────────────────────────────

@app.route("/api/experiments", methods=["GET"])
def api_get_experiments():
    return jsonify(storage.get_experiments())


@app.route("/api/experiments", methods=["POST"])
def api_add_experiment():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400
    ok, err = storage.add_experiment(data)
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "id": data["id"]}), 201


@app.route("/api/experiments/<phase_id>", methods=["DELETE"])
def api_delete_experiment(phase_id):
    storage.delete_experiment(phase_id)
    return jsonify({"ok": True})


# ── Mission Control API ──────────────────────────────────────────────────────

@app.route("/api/mission/start", methods=["POST"])
def api_mission_start():
    body = request.get_json(force=True, silent=True) or {}
    phase_id = body.get("phase_id")
    if not phase_id:
        return jsonify({"error": "phase_id is required"}), 400

    experiment = storage.get_experiment(phase_id)
    if not experiment:
        return jsonify({"error": f"No experiment found with id '{phase_id}'"}), 404

    ok, result = detector.start(experiment)
    if not ok:
        return jsonify({"error": result}), 409
    return jsonify({"ok": True, "mission_id": result})


@app.route("/api/mission/reset", methods=["POST"])
def api_mission_reset():
    detector.stop()
    return jsonify({"ok": True})


@app.route("/api/mission/pass_step", methods=["POST"])
def api_mission_pass_step():
    """Manual operator override to pass current step."""
    ok, msg = detector.pass_step()
    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.route("/api/mission/skip_step", methods=["POST"])
def api_mission_skip_step():
    """Manual operator override to skip current step."""
    ok, msg = detector.skip_step()
    if ok:
        return jsonify({"ok": True, "message": msg})
    return jsonify({"ok": False, "error": msg}), 400


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """Processes an incoming frame directly from client browser webcam."""
    import numpy as np
    import cv2
    if "frame" in request.files:
        file_bytes = request.files["frame"].read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            analysis = detector.process_external_frame(img)
            return jsonify({
                "ok": True,
                "action": analysis["action"],
                "grasp_active": analysis["grasp_active"],
                "bbox_overlap": round(analysis["bbox_overlap"] * 100, 1),
                "target_surface": analysis["target_surface"],
                "hands_count": analysis["hands_count"],
            })
    return jsonify({"ok": False, "error": "No frame received"}), 400


@app.route("/api/mission/state", methods=["GET"])
def api_mission_state():
    state = storage.get_mission_state()
    is_running = detector.is_running()
    state["is_running"] = is_running
    if not is_running:
        state["status"] = "idle"
        state["phase_id"] = None
        state["mission_id"] = None
    state["step_statuses"] = detector.get_step_statuses()
    return jsonify(state)



# ── Log Exports (JSON & CSV) ─────────────────────────────────────────────────

@app.route("/api/logs/download", methods=["GET"])
def api_download_logs_json():
    mission_state = storage.get_mission_state()
    mission_id = mission_state.get("mission_id")
    logs = storage.get_full_log(mission_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"vigil-mission-log-{timestamp}.json"

    payload = json.dumps({
        "mission_id": mission_id,
        "phase_id": mission_state.get("phase_id"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "timeline": logs,
    }, indent=2)

    response = make_response(payload)
    response.headers["Content-Type"] = "application/json"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@app.route("/api/logs/download/csv", methods=["GET"])
def api_download_logs_csv():
    mission_state = storage.get_mission_state()
    mission_id = mission_state.get("mission_id")
    csv_content = storage.get_timeline_csv(mission_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"vigil-mission-log-{timestamp}.csv"

    response = make_response(csv_content)
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


# ── Hardware & System Diagnostics API ────────────────────────────────────────

@app.route("/api/system/status", methods=["GET"])
def api_system_status():
    camera_ok = video_stream.is_camera_available()
    detector_ok = detector._models_loaded
    overall = "nominal" if camera_ok and detector_ok else "degraded"

    return jsonify({
        "status": overall,
        "camera": camera_ok,
        "camera_name": video_stream.source_name,
        "resolution": f"{video_stream.width}x{video_stream.height}",
        "fps": video_stream.fps or detector._telemetry_fps,
        "detector": detector_ok,
        "model_name": "YOLOv11 custom (best.pt) · 10 classes",
        "mediapipe": detector._hand_detector is not None,
        "gpu_device": detector.gpu_device_name,
        "mission": storage.get_mission_state().get("status", "idle"),
        "active_source": {
            "type": video_stream.source_type,
            "val": video_stream.source_val,
            "name": video_stream.source_name,
            "overlay": video_stream.show_overlay
        }
    })


@app.route("/api/system/diagnostics", methods=["POST"])
def api_run_diagnostics():
    """Runs instant comprehensive hardware & ML pipeline test."""
    results = {}

    # Test camera read
    raw = video_stream.get_frame()
    results["camera_read"] = {
        "status": "PASS" if raw is not None and raw.size > 0 else "FAIL",
        "shape": list(raw.shape) if raw is not None else None,
        "source": video_stream.source_name
    }

    # Test YOLO forward pass
    yolo_pass = False
    if detector._yolo_model is not None and raw is not None:
        try:
            res = detector._yolo_model.predict(source=raw, conf=0.35, verbose=False)
            yolo_pass = True
            num_boxes = len(res[0].boxes) if (res and res[0].boxes is not None) else 0
            results["yolo_inference"] = {"status": "PASS", "detections_found": num_boxes}
        except Exception as e:
            results["yolo_inference"] = {"status": "FAIL", "error": str(e)}
    else:
        results["yolo_inference"] = {"status": "FAIL", "reason": "Model or frame not available"}

    # Test SQLite
    try:
        exps = storage.get_experiments()
        results["sqlite_storage"] = {"status": "PASS", "experiments_count": len(exps)}
    except Exception as e:
        results["sqlite_storage"] = {"status": "FAIL", "error": str(e)}

    # Hardware stats
    results["compute_device"] = detector.gpu_device_name
    results["overall_health"] = "EXCELLENT" if yolo_pass else "DEGRADED"

    return jsonify(results)


# ── Socket.IO Real-Time Synchronization ──────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    """Send current state to newly connected client."""
    is_run = detector.is_running()
    status_str = "running" if is_run else "nominal"
    emit("system_status", {
        "status": status_str,
        "label": _status_label(status_str),
    })

    # Send existing timeline entries if mission was running
    state = storage.get_mission_state()
    mission_id = state.get("mission_id") if is_run else None
    if mission_id:
        rows = storage.get_timeline(mission_id)
        for row in rows:
            emit("timeline_row", row)

    # Send current step statuses ONLY if mission is actively running
    if is_run and detector.current_experiment:
        for step in detector.current_experiment["steps"]:
            sid = step["id"]
            stat = detector.get_step_statuses().get(sid, "pending")
            emit("step_update", {
                "step_id": sid,
                "status": stat,
                "confidence": detector._step_confidences.get(sid, 0.0),
                "label": step["label"],
            })


@socketio.on("client_frame")
def handle_client_frame(data):
    """Processes frame sent over WebSocket from browser webcam."""
    import base64
    import numpy as np
    import cv2
    try:
        raw_b64 = data.get("image", "")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            detector.process_external_frame(img)
    except Exception:
        pass



def _status_label(status: str) -> str:
    labels = {
        "idle": "SYSTEM NOMINAL",
        "nominal": "SYSTEM NOMINAL",
        "running": "MISSION ACTIVE",
        "completed": "MISSION COMPLETE",
        "degraded": "SYSTEM DEGRADED",
    }
    return labels.get(status, "SYSTEM NOMINAL")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  +=========================================================+")
    print("  |           VIGIL -- AI Mission Control Center            |")
    print("  +=========================================================+")
    print("  |   Local Server : http://localhost:5000                  |")
    print("  |   Architecture : YOLOv11 + MediaPipe + FSM + SocketIO   |")
    print("  +=========================================================+\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)

