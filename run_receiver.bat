@echo off
rem ==============================================================================
rem Network-Aware Compression - Windows Receiver Launch Script
rem ==============================================================================

cd /d "%~dp0"

echo ========================================================================
echo  Starting Network-Aware Compression Receiver Node (Windows)
echo ========================================================================

rem Check virtual environment
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
    echo [WARN] Virtual environment not found. Using system python.
)

rem Check model checkpoint
if not exist "models\best_unet.pt" (
    if not exist "models\best_unet.pth" (
        echo [WARN] Neither models\best_unet.pt nor models\best_unet.pth were found!
        echo Please ensure model weights are located in the models\ directory.
    )
)

echo [INFO] Launching Receiver Node with %PYTHON_EXE%...
"%PYTHON_EXE%" main_receiver.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)

