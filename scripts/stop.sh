#!/bin/bash
# =============================================================================
# Velox V12 -- Graceful Stop
# Run: bash scripts/stop.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "  ${BOLD}Stopping Velox V12...${NC}"
echo ""

# -- Stop Docker stack ---------------------------------------------------------
if docker ps --filter "name=velox" --format "{{.Names}}" 2>/dev/null | grep -q .; then
    echo "  Stopping Docker containers..."
    docker compose down 2>&1 | sed 's/^/    /'
    echo ""
    echo "  ${GREEN}Docker stack stopped.${NC}"
else
    echo "  ${YELLOW}No Docker containers running.${NC}"
fi

# -- Stop local process (non-Docker) ------------------------------------------
PID=$(pgrep -f "python3 main.py" 2>/dev/null || true)
if [ -n "$PID" ]; then
    echo ""
    echo "  Stopping local bot process (PID $PID)..."
    kill -INT "$PID" 2>/dev/null

    # Wait up to 30 seconds for graceful shutdown
    for i in $(seq 1 30); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "  ${GREEN}Local bot stopped gracefully.${NC}"
            rm -f "$PROJECT_DIR/.bot.pid"
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "  ${YELLOW}Bot did not stop within 30s -- forcing...${NC}"
        kill -9 "$PID" 2>/dev/null
        rm -f "$PROJECT_DIR/.bot.pid"
        echo "  Bot force-stopped."
    fi
fi

echo ""
echo "  ${GREEN}Velox V12 stopped.${NC}"
echo ""
