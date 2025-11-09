#!/bin/bash

# MindSpore Backend Watchdog
# Automatically restarts the backend if it crashes

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/mindspore_misinformation_backend"
PYTHON_PATH="$SCRIPT_DIR/mindenv310/bin/python"
LOG_FILE="$BACKEND_DIR/backend.log"
PID_FILE="/tmp/mindspore_backend.pid"

echo "[WATCHDOG] Starting MindSpore Backend Watchdog..."

while true; do
    # Check if backend is running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            # Backend is running, wait before checking again
            sleep 5
            continue
        fi
    fi
    
    # Backend is not running, start it
    echo "[WATCHDOG] Backend not running. Starting..."
    
    # Kill any existing python processes on port 5000
    pkill -9 -f "python.*main.py" 2>/dev/null
    sleep 1
    
    # Start backend
    cd "$BACKEND_DIR"
    $PYTHON_PATH main.py > "$LOG_FILE" 2>&1 &
    BACKEND_PID=$!
    
    # Save PID
    echo $BACKEND_PID > "$PID_FILE"
    
    echo "[WATCHDOG] Backend started with PID $BACKEND_PID"
    
    # Wait a bit before checking again
    sleep 10
done
