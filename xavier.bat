@echo off
title Xavier - System Telemetry Logger
color 0A

echo ============================================
echo           Xavier Telemetry Logger
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

:: Check and install required packages
echo Checking dependencies...
echo.

:: Check for psutil
python -c "import psutil" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing psutil...
    pip install psutil
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install psutil
        pause
        exit /b 1
    )
) else (
    echo [OK] psutil is installed
)

:: Check for nvidia-ml-py3
python -c "import pynvml" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing nvidia-ml-py3...
    pip install nvidia-ml-py3
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to install nvidia-ml-py3 - GPU monitoring may be limited
    )
) else (
    echo [OK] nvidia-ml-py3 is installed
)

:: Optional: Check for WMI (Windows temperature sensors)
python -c "import wmi" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Optional: Installing WMI for additional sensors...
    pip install wmi pywin32 >nul 2>&1
)

echo.
echo ============================================
echo    Starting Xavier...
echo ============================================
echo.
echo NOTE: For CPU temperature monitoring:
echo   1. Run LibreHardwareMonitor as Administrator
echo   2. Enable: Options -^> Remote Web Server -^> Start
echo.
echo Press Ctrl+C to stop Xavier
echo ============================================
echo.

:: Run Xavier
python telemetry_dualmode_logger.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Xavier encountered an error
    echo Check that telemetry_dualmode_logger.py is in the same folder
    pause
)

exit /b 0