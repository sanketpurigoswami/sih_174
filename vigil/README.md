# VIGIL — Mission Control Dashboard

> **SIH 2026 · PS26174 · Team Astramind**
>
> Mission-control dashboard for an on-board AI Human Activity Recognition
> system. Watches a fixed payload camera, tracks astronaut hand-object
> interactions against a predefined experiment step sequence, and gives
> real-time status, alerts, and logs.

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Open **http://localhost:5000** in your browser.

> **No camera?** The app falls back to an animated placeholder feed
> automatically — the full UI remains fully functional.

---

## Project Structure

```
vigil/
├── app.py                # Flask + SocketIO server, all routes
├── detector.py           # Detection pipeline (swappable/stubbable)
├── video.py              # Webcam capture + MJPEG stream generator
├── storage.py            # SQLite persistence for logs + experiment scripts
├── static/
│   ├── style.css         # Complete design system
│   └── app.js            # Socket.IO client, views, toggles, modals
├── templates/
│   └── index.html        # Single-page dashboard
├── data/
│   ├── experiments.json  # Seed catalog (Phase 1–7)
│   └── vigil.db          # SQLite database (auto-created on first run)
├── requirements.txt
└── README.md
```

---

## Plugging in Real Model Weights

The detection pipeline ships in **stub mode** — it simulates step-by-step
progress on a timer so the UI is fully demonstrable without ML dependencies.

### To switch to real inference:

1. **Install ML dependencies:**
   ```bash
   pip install mediapipe ultralytics
   ```

2. **Place YOLOv8n weights:**
   ```
   vigil/weights/yolov8n.pt
   ```
   (Download from https://github.com/ultralytics/assets/releases)

3. **Edit `detector.py`:**
   - Set `USE_REAL_MODEL = True` at the top of the file.
   - Implement `_real_process_frame(self, frame, step)`:
     - Run MediaPipe Hands on the frame to extract hand landmarks.
     - Run YOLOv8n on the frame to detect objects.
     - Match detected hand-object interactions against
       `step['expected_object']` and `step['expected_action']`.
     - Return `{ step_id, confidence, hand_object_event }`.

4. **Restart the server.** The detection loop will now call your real
   inference pipeline instead of the stub.

---

## API Endpoints

| Method | Path                   | Description                         |
|--------|------------------------|-------------------------------------|
| GET    | `/`                    | Dashboard page                      |
| GET    | `/video_feed`          | MJPEG stream                        |
| GET    | `/api/experiments`     | List all experiment scripts          |
| POST   | `/api/experiments`     | Add a new experiment script (JSON)   |
| POST   | `/api/mission/start`   | Start mission `{ phase_id }`        |
| POST   | `/api/mission/reset`   | Stop & reset current mission         |
| GET    | `/api/mission/state`   | Current mission state                |
| GET    | `/api/logs/download`   | Download mission log as JSON file    |
| GET    | `/api/system/status`   | Camera/detector/mission status       |

## Socket.IO Events (server → client)

| Event           | Payload                                                      |
|-----------------|--------------------------------------------------------------|
| `step_update`   | `{ step_id, status, confidence, label }`                     |
| `timeline_row`  | `{ phase, step, step_label, success, accuracy, start, end }` |
| `alert`         | `{ type: 'warning'|'success', message }`                     |
| `system_status` | `{ status, label }`                                         |

---

## Experiment Script Schema

```json
{
  "id": "phase-N",
  "name": "Phase N — Name",
  "steps": [
    {
      "id": 1,
      "label": "Step description",
      "expected_object": "object_id",
      "expected_action": "action_id"
    }
  ]
}
```

Upload via the **+ Script** button in the Experiment Catalog, or
`POST /api/experiments` with the JSON body.

---

## Offline-First

VIGIL runs fully offline once installed. The only external fetch is Google
Fonts (Fraunces, IBM Plex Sans, IBM Plex Mono) and the Socket.IO client
CDN. To go fully air-gapped, download these assets and serve them from
`/static/`.

---

## License

Built for SIH 2026 · Team Astramind. Internal use.
