#!/bin/bash
# Auto-start with Auto-Recovery - ~/start_system.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[SYSTEM] Starting MindSpore Misinformation Detection System with AUTO-RECOVERY..."
echo "[INFO] Working directory: $SCRIPT_DIR"

# Kill any existing processes
echo "[CLEANUP] Cleaning up old processes..."
lsof -ti:3000,5000 2>/dev/null | xargs -r kill -9
pkill -9 -f "watchdog.sh" 2>/dev/null
sleep 1

# Start backend watchdog (auto-restarts on crash)
echo "[BACKEND] Starting backend with watchdog (auto-recovery enabled)..."
cd "$SCRIPT_DIR"
nohup ./watchdog.sh > "$SCRIPT_DIR/watchdog.log" 2>&1 &
WATCHDOG_PID=$!

# Wait for backend to initialize
echo "[WAIT] Waiting for backend to initialize..."
sleep 5

# Start frontend in background
echo "[FRONTEND] Starting Frontend Server (port 3000)..."
cd "$SCRIPT_DIR"
nohup python3 -m http.server 3000 > "$SCRIPT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

sleep 2

# Check if both are running
if ps -p $WATCHDOG_PID > /dev/null && ps -p $FRONTEND_PID > /dev/null; then
    echo "[SUCCESS] System started successfully with AUTO-RECOVERY!"
    echo ""
    echo "[BACKEND]  http://localhost:5000 (will auto-restart on crash)"
    echo "[FRONTEND] http://localhost:3000/phone_base_model.html"
    echo ""
    echo "[FEATURES]"
    echo "   - Backend automatically restarts if it crashes"
    echo "   - Thread-safe MindSpore operations"
    echo "   - GIL conflict prevention"
    echo ""
    echo "[LOGS]"
    echo "   Backend:  tail -f ~/mindspore_misinformation_backend/backend.log"
    echo "   Watchdog: tail -f ~/watchdog.log"
    echo "   Frontend: tail -f ~/frontend.log"
    echo ""
    echo "[STOP] To stop: pkill -f 'watchdog.sh'; pkill -f 'http.server 3000'"
else
    echo "[ERROR] Failed to start system. Check logs."
    exit 1
fi
