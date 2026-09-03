@echo off
echo ===================================================
echo   Smart CCTV 2.0 - Camera Connection Diagnostics
echo ===================================================
if "%1"=="" (
    .\.venv\Scripts\python.exe tools\check_camera.py --cam cam_01
) else (
    .\.venv\Scripts\python.exe tools\check_camera.py --cam %1
)
pause
