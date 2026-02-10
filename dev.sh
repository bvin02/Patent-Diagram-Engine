# Launch the Patent Diagram Generator web application for development.
#
# This starts:
#   1. FastAPI backend on port 8000
#   2. Vite dev server on port 5173 (proxies /api and /editor to backend)
#
# Usage:
#   ./dev.sh
#   open http://localhost:5173
#

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Patent Diagram Generator — Dev Server${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"

# ── Check dependencies ──────────────────────────────────────────
echo -e "\n${GREEN}[1/4]${NC} Checking Python dependencies…"
pip install -q -r "$ROOT/backend/requirements.txt"

echo -e "${GREEN}[2/4]${NC} Checking Node dependencies…"
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
    npm install
fi

# ── Start backend ───────────────────────────────────────────────
echo -e "${GREEN}[3/4]${NC} Starting FastAPI backend on :8000…"
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir "$ROOT/backend" &
BACKEND_PID=$!

# ── Start frontend dev server ──────────────────────────────────
echo -e "${GREEN}[4/4]${NC} Starting Vite dev server on :5173…"
cd "$ROOT/frontend"
npx vite &
FRONTEND_PID=$!

echo -e "\n${GREEN}✓  Ready!${NC}"
echo -e "   Frontend:  ${CYAN}http://localhost:5173${NC}"
echo -e "   Backend:   ${CYAN}http://localhost:8000${NC}"
echo -e "   Editor:    ${CYAN}http://localhost:8000/editor/method-draw/${NC}"
echo -e "\n   Press Ctrl+C to stop.\n"

# ── Cleanup on exit ─────────────────────────────────────────────
cleanup() {
    echo -e "\n${RED}Shutting down…${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait
}

trap cleanup SIGINT SIGTERM
wait
