@echo off
setlocal
cd /d F:\sih
title VYOMDRISHTI Server - Inspection Only
echo ==============================================
echo VYOMDRISHTI - INSPECTION SEQUENCE SERVER
 echo ==============================================
echo.
echo URL: http://127.0.0.1:5051/
echo Camera: 1 - GlideX SharedCam
echo Backend: CAP_MSMF
echo Inspection only - placement disabled
echo.
python "F:\sih\vyomdrishti_server.py" 5051
pause
