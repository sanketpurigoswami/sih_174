@echo off
title InstantHMR Live Camera Demo
cd /d "F:\InstantHMR"
echo ========================================================
echo   Launching InstantHMR (Live Camera + Complete 3D Mesh)
echo   Model: F:\hmr\checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt
echo ========================================================
echo.

C:\Users\Sanket\miniconda3\envs\instanthmr\python.exe demo.py --camera 0 --mhr-model "F:\hmr\checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt" --mesh-alpha 1.0 --mesh-wireframe --detector-stride 3 --detector-variant nano --max-persons 1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Demo exited with error code %ERRORLEVEL%.
    echo If camera failed to open, please ensure Camera Access is turned ON in Windows Settings.
    pause
)
