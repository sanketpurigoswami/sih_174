from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file, request
from flask_cors import CORS

BASE_DIR = pathlib.Path(__file__).resolve().parent
FSM_PATH = BASE_DIR / "vyomdrishti_fsm.py"
UI_PATH = BASE_DIR / "vyomdrishti_ui.html"

spec = importlib.util.spec_from_file_location("vyomdrishti_fsm", FSM_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load FSM from {FSM_PATH}")
fsm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fsm
spec.loader.exec_module(fsm)

app = Flask(__name__)
CORS(app)

latest_lock = threading.Lock()
latest_jpeg: bytes | None = None
stop_event = threading.Event()
new_frame_event = threading.Event()
running = False
run_lock = threading.Lock()


def fake_imshow(_win, frame):
    global latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if ok:
        with latest_lock:
            latest_jpeg = buf.tobytes()
        new_frame_event.set()


def fake_waitKey(delay=1):
    if stop_event.is_set():
        return ord("q")
    return -1


cv2.imshow = fake_imshow
cv2.waitKey = fake_waitKey
cv2.namedWindow = lambda *a, **k: None
cv2.resizeWindow = lambda *a, **k: None
cv2.destroyAllWindows = lambda *a, **k: None
cv2.destroyWindow = lambda *a, **k: None

blank = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(blank, "VYOMDRISHTI", (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90,90,90), 2)
cv2.putText(blank, "Press START CAMERA", (130, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110,110,110), 1)
_, blank_buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
BLANK_JPEG = blank_buf.tobytes()

@app.get("/")
def index():
    return send_file(UI_PATH)

@app.get("/video_feed")
def video_feed():
    def generate():
        while not stop_event.is_set():
            new_frame_event.wait(timeout=0.08)
            new_frame_event.clear()
            with latest_lock:
                jpg = latest_jpeg
            frame_data = jpg if jpg is not None else BLANK_JPEG
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    )

@app.get("/snapshot")
def snapshot():
    with latest_lock:
        jpg = latest_jpeg
    return Response(jpg or BLANK_JPEG, mimetype="image/jpeg", headers={"Cache-Control":"no-store, no-cache, must-revalidate", "Pragma":"no-cache"})

@app.get("/api/status")
def status():
    state = fsm.get_live_state()
    with run_lock:
        state["server_running"] = running
    return jsonify(state)


def run_fsm(camera_index: int):
    global running, latest_jpeg
    fsm.CAMERA_INDEX = int(camera_index)
    stop_event.clear()
    with run_lock:
        running = True
    with latest_lock:
        latest_jpeg = None
    try:
        fsm.main()
    except Exception as exc:
        print(f"[VYOMDRISHTI] FSM ERROR: {exc}")
        fsm.publish_state(camera="error", phase="ERROR", status=str(exc), timer=0.0, timer_running=False)
    finally:
        with run_lock:
            running = False
        with latest_lock:
            latest_jpeg = None

@app.post("/api/start_camera")
def start_camera():
    with run_lock:
        if running:
            return jsonify({"ok": False, "msg": "Camera already running"}), 409
    data = request.get_json(silent=True) or {}
    cam = int(data.get("camera_index", 1))
    fsm.reset_experiment()
    threading.Thread(target=run_fsm, args=(cam,), daemon=True).start()
    return jsonify({"ok": True, "msg": f"Starting GlideX SharedCam on camera {cam}"})

@app.post("/api/stop_camera")
def stop_camera():
    stop_event.set()
    return jsonify({"ok": True, "msg": "Stopping camera…"})

@app.post("/api/reset_mission")
def reset_mission():
    fsm.reset_experiment()
    return jsonify({"ok": True, "msg": "Inspection sequence reset to Step 1"})

@app.post("/api/voice_test")
def voice_test():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "VyomDrishti voice telemetry operational. Ready for Step 1.")
    tone = data.get("tone", "info")
    ev = fsm.create_voice_event("TEST", msg, tone=tone)
    fsm.publish_state(voice_event=ev)
    return jsonify({"ok": True, "event": ev})

@app.post("/api/speak")
def speak():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    tone = data.get("tone", "info")
    if text:
        ev = fsm.create_voice_event("CUSTOM", text, tone=tone)
        fsm.publish_state(voice_event=ev)
        return jsonify({"ok": True, "event": ev})
    return jsonify({"ok": False, "msg": "No text provided"}), 400

@app.get("/api/log")
def get_log():
    log_file = BASE_DIR / "experiment_protocol_log.jsonl"
    entries = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
    return jsonify({"ok": True, "entries": entries[-100:]})

@app.get("/api/hardware")
def get_hardware():
    gpu_name = "CPU Host"
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "camera_index": getattr(fsm, "CAMERA_INDEX", 1),
        "model_path": str(getattr(fsm, "MODEL_PATH", "")),
        "hand_model_path": str(getattr(fsm, "HAND_MODEL_PATH", "")),
        "hold_duration": getattr(fsm, "INSPECTION_DURATION_SECONDS", 5.0),
        "contact_distance": getattr(fsm, "CONTACT_DISTANCE", 30.0),
        "confidence_threshold": getattr(fsm, "CONFIDENCE", 0.35),
        "iou_threshold": getattr(fsm, "IOU", 0.45),
        "classes": getattr(fsm, "CLASS_NAMES", {}),
        "sequence": getattr(fsm, "INSPECTION_SEQUENCE", [0, 1, 2, 7])
    })

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5051
    print(f"[VYOMDRISHTI] Web UI: http://127.0.0.1:{port}/")
    print(f"[VYOMDRISHTI] FSM: {FSM_PATH}")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
