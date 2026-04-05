#!/bin/bash
# =============================================================================
# Velox V12 -- Quick Status Check
# Run: bash scripts/status.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "  ${BOLD}=== Velox V12 Status ===${NC}"
echo ""

# -- Docker containers ---------------------------------------------------------
echo "  ${BOLD}Containers:${NC}"
if docker ps --filter "name=velox" --format "{{.Names}}" 2>/dev/null | grep -q .; then
    docker ps --filter "name=velox" --format "    {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null
else
    echo "    ${YELLOW}No Velox containers running${NC}"
fi
echo ""

# -- Local process (non-Docker) ------------------------------------------------
PID=$(pgrep -f "python3 main.py" 2>/dev/null || true)
if [ -n "$PID" ]; then
    UPTIME=$(ps -p "$PID" -o etime= 2>/dev/null | xargs)
    echo "  ${BOLD}Local process:${NC}"
    echo "    ${GREEN}Running${NC} (PID $PID, uptime: $UPTIME)"
    echo ""
fi

# -- Health endpoint -----------------------------------------------------------
echo "  ${BOLD}Health check:${NC}"
HEALTH=$(curl -sf http://localhost:8080/health 2>/dev/null)
if [ -n "$HEALTH" ]; then
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
else
    echo "    ${YELLOW}Dashboard not responding (http://localhost:8080/health)${NC}"
fi
echo ""

# -- Quick metrics -------------------------------------------------------------
echo "  ${BOLD}Endpoints:${NC}"
for URL in "http://localhost:8080|Dashboard" "http://localhost:9090|Prometheus" "http://localhost:3000|Grafana" "http://localhost:9093|Alertmanager"; do
    HOST="${URL%%|*}"
    NAME="${URL##*|}"
    if curl -sf --max-time 2 "$HOST" >/dev/null 2>&1; then
        printf "    ${GREEN}%-14s${NC} %s\n" "$NAME" "$HOST"
    else
        printf "    ${RED}%-14s${NC} %s (not responding)\n" "$NAME" "$HOST"
    fi
done
echo ""

# -- Disk usage ----------------------------------------------------------------
echo "  ${BOLD}Disk usage:${NC}"
if [ -d "$PROJECT_DIR/logs" ]; then
    LOGS_SIZE=$(du -sh "$PROJECT_DIR/logs" 2>/dev/null | awk '{print $1}')
    echo "    Logs:    $LOGS_SIZE"
fi
if [ -d "$PROJECT_DIR/backups" ]; then
    BACKUP_SIZE=$(du -sh "$PROJECT_DIR/backups" 2>/dev/null | awk '{print $1}')
    echo "    Backups: $BACKUP_SIZE"
fi
if [ -d "$PROJECT_DIR/data" ]; then
    DATA_SIZE=$(du -sh "$PROJECT_DIR/data" 2>/dev/null | awk '{print $1}')
    echo "    Data:    $DATA_SIZE"
fi
if [ -f "$PROJECT_DIR/bot.db" ]; then
    DB_SIZE=$(du -sh "$PROJECT_DIR/bot.db" 2>/dev/null | awk '{print $1}')
    echo "    bot.db:  $DB_SIZE"
fi
echo ""

echo "  ==============================="
echo ""
