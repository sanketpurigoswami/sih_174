VYOMDRISHTI — INSPECTION SEQUENCE FINAL BUILD
===============================================

This build focuses ONLY on inspection. Placement logic is disabled.

FILES
-----
vyomdrishti_fsm.py
vyomdrishti_server.py
vyomdrishti_ui.html
LAUNCH_VYOMDRISHTI.bat
README.txt

CAMERA
------
Camera 1 — ASUS GlideX SharedCam
OpenCV backend — CAP_MSMF

WEB UI
------
Start:
    cd F:\sih
    .\LAUNCH_VYOMDRISHTI.bat

Open:
    http://127.0.0.1:5051/

Do NOT open vyomdrishti_ui.html directly with file:///.

INSPECTION SEQUENCE
-------------------
1. orange_circular_cap
2. red_computer_mouse
3. wireless_earbuds_case
4. white_box

OBJECT IDENTITY
---------------
Class 0 = orange_circular_cap
Class 1 = red_computer_mouse
Class 2 = wireless_earbuds_case
Class 7 = white_box
Class 9 = hand

TIMER TRIGGER
-------------
The 5-second timer starts and continues when EITHER:
1. ActionDetector reports GRASP / its grasp latch is active, OR
2. the YOLO hand bounding box overlaps the current object's bounding box.

The GUI shows BOTH signals and the overlap percentage.

The timer resets only when BOTH signals are inactive.

AUTO START / SEQUENCE
---------------------
The experiment state becomes active automatically when an experimental object is detected.
The active inspection step always follows the fixed sequence above.
A step cannot be completed by a different object's detection.

After 5.0 seconds:
    current object -> COMPLETED
    next sequence item -> CURRENT

After the fourth object reaches 5.0 seconds:
    ALL OBJECTS INSPECTED

PLACEMENT
---------
Disabled in this build.
