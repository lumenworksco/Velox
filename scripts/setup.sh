#!/bin/bash
# =============================================================================
# Velox V12 -- First-Time Setup
# Run: bash scripts/setup.sh
# =============================================================================

set -euo pipefail

# ---- Colors & helpers -------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

ok()   { printf "  ${GREEN}[OK]${NC}    %s\n" "$1"; }
warn() { printf "  ${YELLOW}[WARN]${NC}  %s\n" "$1"; }
fail() { printf "  ${RED}[FAIL]${NC}  %s\n" "$1"; }
info() { printf "  ${CYAN}[INFO]${NC}  %s\n" "$1"; }
step() { printf "\n${BOLD}==> Step %s: %s${NC}\n" "$1" "$2"; }
ask()  { printf "  ${CYAN}?${NC} %s " "$1"; }

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "  ================================================================"
echo "    ${BOLD}Velox V12 Trading Bot -- First-Time Setup${NC}"
echo "  ================================================================"
echo ""
echo "  Project directory: $PROJECT_DIR"
echo ""

ERRORS=0

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================
step "1/7" "Checking prerequisites"

# -- Python 3.12+ -------------------------------------------------------------
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
        ok "Python $PY_VERSION"
    else
        fail "Python $PY_VERSION found -- 3.12+ required"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# -- pip3 ---------------------------------------------------------------------
if command -v pip3 &>/dev/null; then
    PIP_VERSION=$(pip3 --version 2>/dev/null | awk '{print $2}')
    ok "pip3 $PIP_VERSION"
elif python3 -m pip --version &>/dev/null; then
    PIP_VERSION=$(python3 -m pip --version 2>/dev/null | awk '{print $2}')
    ok "pip (via python3 -m pip) $PIP_VERSION"
else
    fail "pip3 not found"
    ERRORS=$((ERRORS + 1))
fi

# -- Docker --------------------------------------------------------------------
if command -v docker &>/dev/null; then
    DOCKER_VERSION=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
    ok "Docker $DOCKER_VERSION"

    if docker info &>/dev/null; then
        ok "Docker daemon is running"
    else
        fail "Docker is installed but the daemon is not running -- start Docker Desktop first"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Docker not found -- install Docker Desktop from https://www.docker.com/products/docker-desktop"
    ERRORS=$((ERRORS + 1))
fi

# -- docker compose (v2) ------------------------------------------------------
if docker compose version &>/dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null)
    ok "Docker Compose $COMPOSE_VERSION"
else
    fail "docker compose not available -- Docker Desktop includes it by default"
    ERRORS=$((ERRORS + 1))
fi

# -- Required ports ------------------------------------------------------------
PORTS=(8080 9090 3000 9093 5432)
PORTS_OK=true
for PORT in "${PORTS[@]}"; do
    if lsof -i ":$PORT" -sTCP:LISTEN &>/dev/null; then
        fail "Port $PORT is already in use"
        PORTS_OK=false
        ERRORS=$((ERRORS + 1))
    fi
done
if $PORTS_OK; then
    ok "Required ports available (8080, 9090, 3000, 9093, 5432)"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    fail "$ERRORS prerequisite(s) failed. Fix the issues above and re-run this script."
    exit 1
fi

ok "All prerequisites passed"

# =============================================================================
# Step 2: Install Python dependencies
# =============================================================================
step "2/7" "Installing Python dependencies"

if [ ! -f requirements.txt ]; then
    fail "requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

info "Running pip3 install -r requirements.txt ..."
if pip3 install -r requirements.txt 2>&1 | tail -5; then
    ok "Python dependencies installed"
else
    fail "pip install failed -- check the output above"
    exit 1
fi

# =============================================================================
# Step 3: Create .env from template
# =============================================================================
step "3/7" "Configuring environment (.env)"

if [ -f .env ]; then
    warn ".env already exists -- skipping creation"
    info "To reconfigure, delete .env and re-run this script"
else
    if [ ! -f .env.example ]; then
        fail ".env.example not found -- cannot create .env"
        exit 1
    fi

    # Start with the template
    cp .env.example .env
    info "Created .env from .env.example"

    echo ""

    # ---- Alpaca credentials --------------------------------------------------
    echo "  ${BOLD}Alpaca API credentials${NC} (get keys at https://app.alpaca.markets)"
    echo ""
    ask "Alpaca API Key:"
    read -r ALPACA_KEY
    ask "Alpaca API Secret:"
    read -rs ALPACA_SECRET
    echo ""

    if [ -z "$ALPACA_KEY" ] || [ -z "$ALPACA_SECRET" ]; then
        fail "Alpaca API key and secret are required"
        rm -f .env
        exit 1
    fi

    # Write Alpaca credentials
    sed -i.bak "s|^ALPACA_API_KEY=.*|ALPACA_API_KEY=$ALPACA_KEY|" .env
    sed -i.bak "s|^ALPACA_API_SECRET=.*|ALPACA_API_SECRET=$ALPACA_SECRET|" .env

    # ---- Paper vs Live -------------------------------------------------------
    echo ""
    ask "Use LIVE trading? (y/N -- default is paper):"
    read -r LIVE_MODE
    if [[ "$LIVE_MODE" =~ ^[Yy] ]]; then
        sed -i.bak "s|^ALPACA_LIVE=.*|ALPACA_LIVE=true|" .env
        warn "LIVE TRADING enabled -- real money will be at risk"
    else
        sed -i.bak "s|^ALPACA_LIVE=.*|ALPACA_LIVE=false|" .env
        ok "Paper trading mode selected"
    fi

    # ---- Telegram notifications ----------------------------------------------
    echo ""
    ask "Enable Telegram notifications? (y/N):"
    read -r USE_TELEGRAM
    if [[ "$USE_TELEGRAM" =~ ^[Yy] ]]; then
        ask "Telegram Bot Token:"
        read -r TG_TOKEN
        ask "Telegram Chat ID:"
        read -r TG_CHAT_ID
        if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
            sed -i.bak "s|^TELEGRAM_ENABLED=.*|TELEGRAM_ENABLED=true|" .env
            sed -i.bak "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TG_TOKEN|" .env
            sed -i.bak "s|^TELEGRAM_CHAT_ID=.*|TELEGRAM_CHAT_ID=$TG_CHAT_ID|" .env
            ok "Telegram notifications enabled"
        else
            warn "Missing token or chat ID -- Telegram left disabled"
        fi
    else
        info "Telegram notifications skipped"
    fi

    # ---- FRED API key --------------------------------------------------------
    echo ""
    ask "Do you have a FRED API key? (free at https://fred.stlouisfed.org/docs/api/api_key.html) (y/N):"
    read -r USE_FRED
    if [[ "$USE_FRED" =~ ^[Yy] ]]; then
        ask "FRED API Key:"
        read -r FRED_KEY
        if [ -n "$FRED_KEY" ]; then
            # Uncomment and set the FRED key
            sed -i.bak "s|^# FRED_API_KEY=.*|FRED_API_KEY=$FRED_KEY|" .env
            ok "FRED API key configured"
        else
            warn "Empty key -- FRED left unconfigured"
        fi
    else
        info "FRED API key skipped (macro signals will use fallback data)"
    fi

    # ---- Database & Grafana passwords ----------------------------------------
    echo ""
    info "Generating secure passwords for PostgreSQL and Grafana..."
    PG_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    GF_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")

    sed -i.bak "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$PG_PASSWORD|" .env
    sed -i.bak "s|^GRAFANA_ADMIN_PASSWORD=.*|GRAFANA_ADMIN_PASSWORD=$GF_PASSWORD|" .env
    # Update the DATABASE_URL to match the new password
    sed -i.bak "s|^DATABASE_URL=.*|DATABASE_URL=postgresql://velox:${PG_PASSWORD}@postgres:5432/velox|" .env

    ok "Secure passwords generated and written to .env"
    info "Grafana admin password: $GF_PASSWORD  (save this somewhere safe)"

    # Clean up sed backup files
    rm -f .env.bak

    echo ""
    ok ".env configuration complete"
fi

# =============================================================================
# Step 4: Initialize database
# =============================================================================
step "4/7" "Initializing local database"

info "Running database init and migrations..."
if python3 -c "import database; database.init_db(); database.run_migrations()" 2>&1; then
    ok "Database initialized and migrations applied"
else
    warn "Database initialization had issues (this is OK if using Docker PostgreSQL exclusively)"
fi

# =============================================================================
# Step 5: Verify Alpaca connection
# =============================================================================
step "5/7" "Verifying Alpaca API connection"

# Source .env for the check
set -a
source .env 2>/dev/null || true
set +a

ALPACA_CHECK=$(python3 -c "
import os, sys
try:
    from alpaca.trading.client import TradingClient
    key = os.getenv('ALPACA_API_KEY', '')
    secret = os.getenv('ALPACA_API_SECRET', '')
    live = os.getenv('ALPACA_LIVE', 'false').lower() == 'true'
    if not key or not secret or key == 'your_api_key_here':
        print('SKIP: Alpaca credentials not configured')
        sys.exit(0)
    client = TradingClient(key, secret, paper=not live)
    account = client.get_account()
    mode = 'LIVE' if live else 'PAPER'
    print(f'OK: Connected to Alpaca ({mode})')
    print(f'    Account status: {account.status}')
    print(f'    Equity:         \${float(account.equity):,.2f}')
    print(f'    Buying power:   \${float(account.buying_power):,.2f}')
    print(f'    Day trades:     {account.daytrade_count}/3')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" 2>&1) || true

if echo "$ALPACA_CHECK" | grep -q "^OK:"; then
    while IFS= read -r line; do
        if [[ "$line" == OK:* ]]; then
            ok "${line#OK: }"
        else
            info "$line"
        fi
    done <<< "$ALPACA_CHECK"
elif echo "$ALPACA_CHECK" | grep -q "^SKIP:"; then
    warn "${ALPACA_CHECK#SKIP: }"
else
    fail "Alpaca connection failed"
    echo "    $ALPACA_CHECK"
    warn "You can fix your credentials in .env and re-run this step later"
fi

# =============================================================================
# Step 6: Start Docker stack
# =============================================================================
step "6/7" "Starting Docker stack"

ask "Start the Docker stack now? (Y/n):"
read -r START_DOCKER
if [[ "$START_DOCKER" =~ ^[Nn] ]]; then
    info "Docker stack not started. Run 'docker compose up -d' when ready."
else
    info "Building and starting containers (this may take a few minutes on first run)..."
    echo ""

    if docker compose up -d --build 2>&1; then
        echo ""
        ok "Docker stack started"

        # Wait for health checks
        info "Waiting for services to become healthy..."
        TIMEOUT=90
        ELAPSED=0
        while [ $ELAPSED -lt $TIMEOUT ]; do
            HEALTHY=$(docker ps --filter "name=velox" --filter "health=healthy" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')
            TOTAL=$(docker ps --filter "name=velox" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$TOTAL" -gt 0 ] && [ "$HEALTHY" -eq "$TOTAL" ]; then
                break
            fi
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            printf "    ... %ds / %ds (healthy: %s/%s)\r" "$ELAPSED" "$TIMEOUT" "$HEALTHY" "$TOTAL"
        done
        echo ""

        # Show container status
        echo ""
        info "Container status:"
        docker ps --filter "name=velox" --format "    {{.Names}}\t{{.Status}}" 2>/dev/null
        echo ""
    else
        fail "Docker Compose failed -- check the output above"
        warn "You can retry with: docker compose up -d"
    fi
fi

# =============================================================================
# Step 7: Summary
# =============================================================================
step "7/7" "Setup complete"

echo ""
echo "  ================================================================"
echo "    ${BOLD}${GREEN}Velox V12 is ready${NC}"
echo "  ================================================================"
echo ""
echo "  ${BOLD}Endpoints:${NC}"
echo "    Dashboard:    ${CYAN}http://localhost:8080${NC}"
echo "    Prometheus:   ${CYAN}http://localhost:9090${NC}"
echo "    Grafana:      ${CYAN}http://localhost:3000${NC}  (admin / see .env for password)"
echo "    Alertmanager: ${CYAN}http://localhost:9093${NC}"
echo ""
echo "  ${BOLD}Common commands:${NC}"
echo "    View logs:     docker compose logs -f velox"
echo "    Check status:  bash scripts/status.sh"
echo "    Stop all:      bash scripts/stop.sh"
echo "    Restart:       docker compose restart velox"
echo ""
echo "  ${BOLD}Important files:${NC}"
echo "    Configuration: .env"
echo "    Bot logs:      logs/ and bot.log"
echo "    Backups:       backups/"
echo ""
if [ -f docs/runbooks ]; then
    echo "  ${BOLD}Documentation:${NC}"
    echo "    Runbooks:      docs/runbooks/"
    echo ""
fi
echo "  For detailed setup instructions, see the README."
echo ""
echo "  ================================================================"
echo ""
