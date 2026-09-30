@echo off
setlocal
cd /d F:\sih\vigil
title VIGIL Mission Control Center - SIH 2026
echo =========================================================
echo    VIGIL -- ON-BOARD AI MISSION CONTROL CENTER
echo    SIH 2026 - PS26174 - Team Astramind
echo =========================================================
echo.
echo URL: http://localhost:5000/
echo Models: YOLOv11 (best.pt) + MediaPipe Hand Landmarker
echo Camera Support: GlideX (Cam 1) / Webcam (Cam 0) / Video files
echo.
python app.py
pause
