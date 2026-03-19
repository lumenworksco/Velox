#!/bin/bash
# Show bot status: uptime, open positions, recent trades
# Works with both SQLite (local) and REST API (Docker/PostgreSQL)
cd "$(dirname "$0")"

DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:8080}"

echo "=== VELOX V10 STATUS ==="
echo

# Check if running
PID=$(pgrep -f "python3 main.py")
if [ -n "$PID" ]; then
    echo "Status:  RUNNING (PID $PID)"
    echo "Uptime:  $(ps -p "$PID" -o etime= 2>/dev/null || echo 'unknown')"
else
    echo "Status:  STOPPED (checking Docker...)"
    docker ps --filter name=velox-v10 --format "Status: {{.Status}}" 2>/dev/null || true
fi
echo

# Try REST API first (works with both SQLite and PostgreSQL)
if curl -sf "${DASHBOARD_URL}/health" > /dev/null 2>&1; then
    echo "--- Open Positions (via API) ---"
    curl -sf "${DASHBOARD_URL}/api/positions" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(none or auth required)"
    echo

    echo "--- Recent Trades (via API) ---"
    curl -sf "${DASHBOARD_URL}/api/trades?limit=5" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(none or auth required)"
    echo
elif [ -f bot.db ]; then
    # Fallback to SQLite direct access
    echo "--- Open Positions (SQLite) ---"
    sqlite3 bot.db "SELECT symbol, strategy, side, entry_price, qty, hold_type FROM open_positions;" 2>/dev/null || echo "(none)"
    echo
    echo "--- Today's Trades ---"
    TODAY=$(date +%Y-%m-%d)
    sqlite3 bot.db "SELECT COUNT(*) || ' trades today' FROM trades WHERE exit_time LIKE '${TODAY}%';" 2>/dev/null
    echo
fi

# Last log lines
if [ -f bot.log ]; then
    echo "--- Last 10 Log Lines ---"
    tail -10 bot.log
fi

echo
echo "==================================="
