@echo off
REM PSI Bridge v3.0 Installer — Windows
REM Ghost in the Machine Labs

echo ======================================================================
echo           QUANTUM PSI BRIDGE v3.0 — INSTALLER
echo           Ghost in the Machine Labs
echo ======================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python not found. Install Python 3.8+ from python.org
        pause
        exit /b 1
    )
)
echo [OK] Python found

REM Install NumPy
pip install numpy >nul 2>&1
echo [OK] NumPy installed

REM Check Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama not found. Download from https://ollama.com/download
    echo Install Ollama, then run this installer again.
    pause
    exit /b 1
)
echo [OK] Ollama found

REM Pull model
echo Pulling gemma2:2b model...
ollama pull gemma2:2b

REM Create directories
mkdir "%USERPROFILE%\psi_bridge\locks" 2>nul
mkdir "%USERPROFILE%\psi_bridge\logs" 2>nul

REM Copy bridge
copy /Y "%~dp0psi_bridge_v3.py" "%USERPROFILE%\psi_bridge\psi_bridge.py"
copy /Y "%~dp0chat.html" "%USERPROFILE%\psi_bridge\chat.html" 2>nul

echo.
echo ======================================================================
echo   INSTALLED
echo.
echo   To start:
echo     cd %USERPROFILE%\psi_bridge
echo     python psi_bridge.py --peer ^<OTHER_DEVICE_IP^>
echo.
echo   Wait for SUBSTRATE TRANSPORT ACTIVE
echo   Then disconnect the network.
echo ======================================================================
pause
