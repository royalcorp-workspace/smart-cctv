@echo off
echo ===================================================
echo   Smart CCTV 2.0 - Data Retention Purge Runner
echo ===================================================
if "%1"=="" (
    .\.venv\Scripts\python.exe tools\purge_old_data.py --days 30
) else (
    .\.venv\Scripts\python.exe tools\purge_old_data.py --days %1
)
pause
