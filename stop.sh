#!/bin/bash

echo "🛑 Stopping Quantum Data Center server..."

# Kill any process on port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9
    echo "✅ Server stopped (port 8000 freed)"
else
    echo "ℹ️  No server running on port 8000"
fi
