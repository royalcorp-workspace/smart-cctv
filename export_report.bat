@echo off
echo ===================================================
echo   Smart CCTV 2.0 - Incident Report Exporter
echo ===================================================
if "%1"=="" (
    .\.venv\Scripts\python.exe tools\export_incidents.py
) else (
    .\.venv\Scripts\python.exe tools\export_incidents.py --cam %1
)
pause
