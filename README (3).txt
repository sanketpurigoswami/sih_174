VYOMDRISHTI - LIVE BOUNDING-BOX INSPECTION TIMER
================================================

Camera
------
Camera 1 with OpenCV CAP_MSMF = ASUS GlideX SharedCam.

Object mapping
--------------
0 orange_circular_cap     -> 3 dark_blue_square
1 red_computer_mouse      -> 4 yellow_square
2 wireless_earbuds_case   -> 5 pink_square
7 white_box               -> 6 green_square

Inspection timer
----------------
The inspection timer starts immediately when the YOLO hand bounding box
(class 9: hand) has positive intersection area with the current object's
bounding box. The timer increases continuously from 0.0 to 5.0 seconds.
When the bounding boxes stop overlapping, the timer resets to 0.0.
MediaPipe fingertips remain visible for feedback but are not used for the
inspection trigger.

Run
---
1. Close Windows Camera/other apps using GlideX SharedCam.
2. Ensure GlideX SharedCam is connected.
3. Copy all files to F:\sih.
4. Double-click LAUNCH_VYOMDRISHTI.bat.
5. Use CAM 1 - GlideX SharedCam.
6. Press START CAMERA.
