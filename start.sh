#!/bin/bash
# No set -e: background process exits must not kill the whole script

echo "Starting KAIROS..."

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Install Python backend dependencies
echo "[1/3] Installing Python dependencies..."
cd "$ROOT"
python -m pip install -r requirements-server.txt --quiet
python -m pip install -e . --no-deps --quiet

# Start FastAPI backend
echo "[2/3] Starting backend on http://localhost:8001 ..."
python -m uvicorn kairos.server.main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!
echo "      Backend PID: $BACKEND_PID"

# Give uvicorn 2 seconds to bind before starting the frontend
sleep 2

# Start React frontend (frontend lives at project root)
echo "[3/3] Starting frontend on http://localhost:3000 ..."
cd "$ROOT"
npm install --silent
npm run dev &
FRONTEND_PID=$!
echo "      Frontend PID: $FRONTEND_PID"

echo ""
echo "============================================"
echo "  KAIROS is live."
echo ""
echo "  Dashboard : http://localhost:3000"
echo "  API docs  : http://localhost:8001/docs"
echo "  API health: http://localhost:8001/health"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both services."

# Clean up on exit
trap "echo ''; echo 'Stopping KAIROS...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'KAIROS stopped.'" EXIT

wait
