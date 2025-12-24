@echo off
REM Quantum Data Center Web Application Startup Script (Windows)

echo =========================================
echo Quantum Data Center Control System
echo =========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo Installing requirements...
pip install -q --upgrade pip
pip install -q -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo =========================================
echo Starting FastAPI server...
echo =========================================
echo.
echo Web Interface: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo WebSocket: ws://localhost:8000/ws
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
