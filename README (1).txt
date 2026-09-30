VYOMDRISHTI - UPDATED LIVE STEP TIMER GUI

Files
-----
vyomdrishti_server.py
  Flask bridge. It captures processed OpenCV frames and exposes /snapshot and
  the live FSM protocol state through /api/status.

live_inspection_placement_fsm_v2.py
  Main YOLO + MediaPipe + protocol FSM. Configured for GlideX SharedCam:
  CAMERA_INDEX = 1 and cv2.CAP_MSMF.

vyomdrishti_ui.html
  Browser GUI. The sidebar now shows:
  - STEP n / 8
  - current phase and object
  - BEING INSPECTED while the 5 s inspection condition is satisfied
  - a smoothly increasing live timer from 0.0 to 5.0 s
  - automatic advance to the next protocol step after completion
  - inspected/placed counters

LAUNCH_VYOMDRISHTI.bat
  Starts the server and opens the HTML UI using the normal `python` command.

Installation location
----------------------
Copy all four files into F:\sih. The server expects the FSM filename
`live_inspection_placement_fsm_v2.py` in F:\sih.

Run
---
1. Close Windows Camera/other camera apps that may occupy GlideX SharedCam.
2. Make sure GlideX Shared Cam is connected to the phone.
3. Double-click LAUNCH_VYOMDRISHTI.bat.
4. In the browser, leave CAM set to `1 - GlideX SharedCam`.
5. Press START CAMERA.

Protocol timer behavior
------------------------
During inspection, the browser timer increases continuously while the
existing FSM condition is satisfied. It reaches 5.0 s, the FSM marks that
object inspected, and the FSM advances to the next inspection object.
Placement uses the existing 1.0 s confirmation rule and is also shown as a
live step timer. The GUI does not replace the FSM logic; it only displays its
current state.
