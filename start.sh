#!/bin/bash

# Quantum Data Center Web Application Startup Script

echo "========================================="
echo "Quantum Data Center Control System"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

echo ""
echo "========================================="
echo "Starting FastAPI server..."
echo "========================================="
echo ""
echo "🌐 Web Interface: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🔌 WebSocket: ws://localhost:8000/ws"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
