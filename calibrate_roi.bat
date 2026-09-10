@echo off
title Smart CCTV 2.0 - Master ROI Calibrator Hub
:MENU
cls
echo ===================================================
echo    Smart CCTV 2.0 - Multi-Camera ROI Calibrator Hub
echo ===================================================
echo.
echo   [1] Kalibrasi cam_01: Koridor Utama (192.212.160.70)
echo   [2] Kalibrasi cam_02: Area Parkir POS-2 (192.212.160.81)
echo   [3] Kalibrasi cam_03: Jalur Logistik Arah POS-1 (192.212.160.27)
echo   [4] Kalibrasi cam_04: Area Timbangan Truk (192.212.160.156)
echo   [Q] Keluar / Quit
echo.
echo ===================================================
set /p choice="Pilih kamera [1-4, Q]: "

if /i "%choice%"=="1" goto CAM01
if /i "%choice%"=="2" goto CAM02
if /i "%choice%"=="3" goto CAM03
if /i "%choice%"=="4" goto CAM04
if /i "%choice%"=="q" goto EXIT

echo [ERROR] Pilihan tidak valid! Silakan masukkan 1, 2, 3, 4, atau Q.
pause
goto MENU

:CAM01
echo.
echo Menjalankan ROI Calibrator untuk cam_01 (Koridor Utama - 192.212.160.70)...
.\.venv\Scripts\python.exe tools\roi_calibrator.py --cam cam_01
echo.
pause
goto MENU

:CAM02
echo.
echo Menjalankan ROI Calibrator untuk cam_02 (Area Parkir POS-2 - 192.212.160.81)...
.\.venv\Scripts\python.exe tools\roi_calibrator.py --cam cam_02
echo.
pause
goto MENU

:CAM03
echo.
echo Menjalankan ROI Calibrator untuk cam_03 (Jalur Logistik Arah POS-1 - 192.212.160.27)...
.\.venv\Scripts\python.exe tools\roi_calibrator.py --cam cam_03
echo.
pause
goto MENU

:CAM04
echo.
echo Menjalankan ROI Calibrator untuk cam_04 (Area Timbangan Truk - 192.212.160.156)...
.\.venv\Scripts\python.exe tools\roi_calibrator.py --cam cam_04
echo.
pause
goto MENU

:EXIT
echo.
echo Selesai. Menutup ROI Calibrator Hub.
