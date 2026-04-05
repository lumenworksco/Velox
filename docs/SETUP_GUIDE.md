# Velox V12 -- Setup and Operations Guide

Version 12.0 | Last updated: 2026-04-05

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (5 Minutes)](#2-quick-start-5-minutes)
3. [Configuration Guide](#3-configuration-guide)
4. [Daily Operations](#4-daily-operations)
5. [Parameter Tuning](#5-parameter-tuning)
6. [Architecture Overview](#6-architecture-overview)
7. [Troubleshooting](#7-troubleshooting)
8. [FAQ](#8-faq)

---

## 1. Prerequisites

### Operating System

| OS | Minimum Version | Notes |
|---|---|---|
| macOS | 13 (Ventura) or later | Apple Silicon (M1+) and Intel both supported |
| Ubuntu | 22.04 LTS or later | Recommended for VPS/cloud deployments |
| Debian | 12 (Bookworm) or later | Lightweight alternative to Ubuntu |
| Other Linux | Kernel 5.10+ with glibc 2.31+ | Any distro that runs Docker |

Windows is not directly supported. Use WSL2 with Ubuntu 22.04 if running on Windows.

### Required Software

- **Docker Desktop** (macOS/Windows) or **Docker Engine + Docker Compose** (Linux)
  - Docker Engine 24.0 or later
  - Docker Compose v2.20 or later
  - At least 4 GB of RAM allocated to Docker
- **Python 3.12 or later** -- only needed if you plan to run outside Docker or retrain the ML model
- **Git** -- for cloning the repository (if distributed via Git)

### Required Accounts

1. **Alpaca Brokerage Account**
   - Sign up at [https://app.alpaca.markets](https://app.alpaca.markets)
   - Both paper trading and live trading accounts are available
   - Paper trading is enabled by default -- no real money is at risk until you explicitly switch to live mode
   - You will need your API Key and API Secret from the Alpaca dashboard

### Optional (Free) API Keys

These keys are not required but improve signal quality:

| Service | Purpose | How to Get |
|---|---|---|
| FRED (Federal Reserve) | Macroeconomic data for regime detection and macro surprise signals | [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) -- free registration |

All other data sources (market data via Alpaca, sentiment via FinBERT, earnings data via yfinance) are either included with your Alpaca account or run locally at no cost.

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 10 GB free | 20 GB free (for logs, backups, and historical data) |
| Network | Stable broadband | Low-latency connection to US East Coast |

---

## 2. Quick Start (5 Minutes)

### Step 1: Extract the Code

If you received a zip file:

```bash
unzip velox-v12.zip
cd trading_bot
```

If you received a Git repository:

```bash
git clone <repository-url>
cd trading_bot
```

### Step 2: Create Your Environment File

```bash
cp .env.example .env
```

### Step 3: Get Your Alpaca API Keys

1. Log in to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Navigate to the **Paper Trading** section (left sidebar)
3. Click **API Keys** and then **Generate New Key**
4. Copy both the **API Key** and **Secret Key** -- the secret is shown only once

### Step 4: Configure Your .env File

Open `.env` in any text editor and fill in the required values:

```bash
# Required -- paste your Alpaca keys here
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_LIVE=false

# Required -- set secure passwords for the database and Grafana
POSTGRES_PASSWORD=choose_a_strong_password_here
GRAFANA_ADMIN_PASSWORD=choose_a_strong_password_here
```

Leave `ALPACA_LIVE=false` for now. This runs the bot in paper trading mode using simulated money.

### Step 5: Start the Bot

```bash
docker compose up -d
```

This starts all six services:

| Service | Description | Port |
|---|---|---|
| velox | The trading bot | 8080 |
| postgres | PostgreSQL database | 5432 |
| prometheus | Metrics collection | 9090 |
| grafana | Metrics dashboards | 3000 |
| alertmanager | Alert routing | 9093 |
| backup | Automated daily backups | -- |

First startup takes 2-3 minutes while Docker builds the image and downloads dependencies.

### Step 6: Verify Everything Is Running

```bash
docker compose ps
```

All services should show `Up (healthy)` or `Up`. Then open the dashboard:

- **Dashboard**: [http://localhost:8080](http://localhost:8080)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (log in with `admin` / your `GRAFANA_ADMIN_PASSWORD`)

### Step 7: Check the Logs

```bash
docker compose logs -f velox
```

You should see startup messages confirming:
- Database initialized and migrations applied
- Alpaca API connectivity verified
- Strategies loaded
- Waiting for market open (if outside trading hours)

Press `Ctrl+C` to stop following logs.

**Congratulations -- Velox is now running in paper trading mode.** Let it run for at least one full trading day to see it generate signals and execute trades.

---

## 3. Configuration Guide

All configuration is done through two files:
- **`.env`** -- credentials, feature flags, and infrastructure settings (requires restart)
- **`config/strategies.yaml`** and **`config/risk.yaml`** -- strategy and risk parameters (hot-reloadable; changes take effect without restarting)

### 3.1 Switching from Paper to Live Trading

**WARNING: Live trading uses real money. Only switch to live mode after you have thoroughly tested the bot in paper mode and are comfortable with its behavior.**

1. Log in to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Switch to your **Live Trading** account
3. Generate new API keys for the live account (live and paper keys are different)
4. Update your `.env` file:

```bash
ALPACA_API_KEY=your_live_api_key
ALPACA_API_SECRET=your_live_api_secret
ALPACA_LIVE=true
```

5. Restart the bot:

```bash
docker compose down
docker compose up -d
```

To switch back to paper trading, set `ALPACA_LIVE=false` and use your paper API keys.

### 3.2 Adjusting Strategy Allocations

Strategy allocations control how much of your portfolio each strategy can use. Edit `config/strategies.yaml`:

```yaml
allocations:
  STAT_MR:      0.25    # Statistical Mean Reversion (25%)
  VWAP:         0.13    # VWAP Mean Reversion (13%)
  KALMAN_PAIRS: 0.27    # Kalman Pairs Trading (27%)
  ORB:          0.12    # Opening Range Breakout (12%)
  MICRO_MOM:    0.05    # Micro Momentum (5%)
  PEAD:         0.18    # Post-Earnings Drift (18%)
```

**Rules:**
- Allocations must sum to 1.00 (100%)
- Changes take effect on the next scan cycle (within 2 minutes) -- no restart needed
- The adaptive allocation system will adjust these dynamically based on performance, so these values serve as baseline starting points

### 3.3 Adjusting Risk Parameters

Edit `config/risk.yaml` to change risk settings. Key parameters:

```yaml
position_sizing:
  risk_per_trade_pct:    0.008    # Risk 0.8% of portfolio per trade
  max_position_pct:      0.08     # Max 8% of portfolio in one position
  max_positions:         12       # Max 12 simultaneous positions
  max_portfolio_deploy:  0.55     # Max 55% of portfolio deployed at once
```

**Recommended ranges:**

| Parameter | Conservative | Default | Aggressive |
|---|---|---|---|
| `risk_per_trade_pct` | 0.003 | 0.008 | 0.015 |
| `max_position_pct` | 0.04 | 0.08 | 0.12 |
| `max_positions` | 6 | 12 | 20 |
| `max_portfolio_deploy` | 0.35 | 0.55 | 0.75 |

Changes to `risk.yaml` take effect without restarting.

### 3.4 Enabling or Disabling Strategies

To disable a specific strategy, set its allocation to `0.00` in `config/strategies.yaml` and redistribute the weight to other strategies. For strategies with an explicit enabled flag, you can also set them in `.env`:

```bash
# In .env (requires restart)
PEAD_PRE_EARNINGS_ENABLED=false    # Disable pre-earnings exploitation
```

Or set `enabled: false` in the relevant strategy section of `config/strategies.yaml`:

```yaml
orb:
  enabled: false    # Disable Opening Range Breakout
pead:
  enabled: false    # Disable Post-Earnings Drift
```

### 3.5 Configuring Telegram Alerts

Velox can send real-time trade alerts, daily summaries, and risk warnings to Telegram.

1. Open Telegram and message **@BotFather**
2. Send `/newbot` and follow the prompts to create a bot
3. Copy the **bot token** (looks like `123456789:ABCdefGhIJKlmNoPQRsTUVwxYZ`)
4. Send any message to your new bot, then visit:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
5. Find your `chat_id` in the JSON response (a numeric ID)
6. Update your `.env` file:

```bash
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxYZ
TELEGRAM_CHAT_ID=987654321
```

7. Restart the bot:

```bash
docker compose restart velox
```

You will receive notifications for:
- Trade opened (symbol, strategy, entry price, size, TP/SL levels)
- Trade closed (symbol, P&L, exit reason, hold time)
- Daily performance summary
- Risk warnings (circuit breaker activations, VIX halts)

### 3.6 Configuring the FRED API Key

The FRED API key enables macroeconomic data integration (yield curve, unemployment, GDP surprise signals).

1. Register at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Confirm your email and copy your API key
3. Add to `.env`:

```bash
FRED_API_KEY=your_fred_api_key_here
```

4. Restart the bot:

```bash
docker compose restart velox
```

Without a FRED key, the bot operates normally but skips macro-based signals.

---

## 4. Daily Operations

### 4.1 Checking Bot Status

**Quick check from the command line:**

```bash
docker compose ps
```

All services should show `Up (healthy)`.

**Detailed status:**

```bash
./status.sh
```

This shows the process status, open positions, recent trades, and the last 10 log lines.

**Via the dashboard API:**

```bash
curl http://localhost:8080/health
```

A healthy response returns HTTP 200.

### 4.2 Viewing the Dashboard

Open [http://localhost:8080](http://localhost:8080) in your browser. The dashboard shows:

- Current portfolio value and daily P&L
- Open positions with entry prices, unrealized P&L, and strategy labels
- Recent trade history with exit reasons and realized P&L
- Signal log (generated, acted on, and skipped signals with reasons)
- Risk state (circuit breaker tier, VIX level, portfolio heat)
- Strategy health metrics

If you configured dashboard authentication (see `.env.example`), you will need to log in first.

### 4.3 Checking Grafana Metrics

Open [http://localhost:3000](http://localhost:3000) and log in with:
- Username: `admin`
- Password: your `GRAFANA_ADMIN_PASSWORD` from `.env`

Grafana provides time-series visualizations of:
- Portfolio equity curve
- Per-strategy P&L breakdown
- Signal generation rates
- Execution latency and slippage
- VIX levels and risk scalar
- Circuit breaker activations

Prometheus (the underlying metrics store) is accessible at [http://localhost:9090](http://localhost:9090) for ad-hoc queries.

### 4.4 Viewing Trade History

**Via the dashboard:**

Navigate to the Trades section at [http://localhost:8080](http://localhost:8080).

**Via the API:**

```bash
# Last 20 trades
curl http://localhost:8080/api/trades?limit=20

# Trades for a specific symbol
curl http://localhost:8080/api/trades?symbol=AAPL
```

**Via SQLite directly (if running locally without Docker):**

```bash
sqlite3 bot.db "SELECT symbol, strategy, side, entry_price, exit_price, pnl, exit_reason, exit_time FROM trades ORDER BY exit_time DESC LIMIT 20;"
```

### 4.5 Generating Performance Reports

The bot automatically generates a daily end-of-day summary at 4:15 PM ET, which includes:
- Total trades executed
- Win rate and average P&L
- Sharpe ratio (rolling)
- Strategy-level breakdown

If Telegram is configured, this summary is sent as a notification. The data is also persisted to the `daily_snapshots` table for historical analysis.

### 4.6 What to Do if the Bot Crashes

1. **Check the logs:**

```bash
docker compose logs --tail=100 velox
```

2. **Check for common causes:**
   - Network connectivity issues (Alpaca API unreachable)
   - Invalid API keys (rotated or expired)
   - Database corruption (rare)
   - Out of memory (check Docker resource limits)

3. **Restart the bot:**

```bash
docker compose restart velox
```

The bot is designed to recover gracefully from crashes. On startup, it:
- Reconnects to Alpaca and verifies credentials
- Syncs its internal state with broker positions
- Resumes monitoring open positions
- Continues normal operation

### 4.7 Restarting the Entire Stack

```bash
# Graceful restart (preserves data)
docker compose down
docker compose up -d

# Force rebuild (after code changes)
docker compose down
docker compose up -d --build
```

To restart only the trading bot without touching the database or monitoring stack:

```bash
docker compose restart velox
```

---

## 5. Parameter Tuning

### 5.1 Strategy Allocations Explained

| Strategy | Default | Description |
|---|---|---|
| **STAT_MR** | 25% | Statistical Mean Reversion -- identifies stocks that have deviated from their statistical mean using z-scores, RSI, and Hurst exponent filtering. Best in range-bound markets. |
| **VWAP** | 13% | VWAP Mean Reversion -- trades deviations from the Volume-Weighted Average Price using Ornstein-Uhlenbeck z-scores. Intraday focus. |
| **KALMAN_PAIRS** | 27% | Kalman Pairs Trading -- trades cointegrated stock pairs using a Kalman filter to track the dynamic hedge ratio. Market-neutral. |
| **ORB** | 12% | Opening Range Breakout -- trades breakouts from the first 30-minute range with volume and ADX confirmation. Morning-only strategy. |
| **MICRO_MOM** | 5% | Micro Momentum -- captures momentum in high-beta stocks following SPY volume spikes. Very short-term (under 20 minutes). |
| **PEAD** | 18% | Post-Earnings Announcement Drift -- trades the academically documented tendency for stocks to continue drifting in the direction of an earnings surprise. Multi-day holds. |

### 5.2 Risk Parameters Explained

| Parameter | Default | Range | Description |
|---|---|---|---|
| `risk_per_trade_pct` | 0.008 | 0.003 -- 0.015 | Percentage of portfolio risked on each trade. Controls position sizing via the risk/reward calculation. |
| `max_position_pct` | 0.08 | 0.04 -- 0.12 | Maximum percentage of portfolio that a single position can represent. |
| `max_positions` | 12 | 4 -- 20 | Maximum number of simultaneous open positions across all strategies. |
| `max_portfolio_deploy` | 0.55 | 0.30 -- 0.80 | Maximum percentage of total portfolio value that can be deployed in positions at any time. The remainder stays in cash. |
| `daily_loss_halt` | -2.5% | -1.0% -- -4.0% | If daily P&L drops below this threshold, all new entries are blocked for the rest of the day. |
| `gain_lock_pct` | 1.5% | 0.5% -- 3.0% | When daily gains reach this level, position sizes are reduced to lock in profits. |
| `vol_target_daily` | 1.0% | 0.5% -- 2.0% | Target daily portfolio volatility. Positions are scaled up/down to match this target. |
| `vix_halt_threshold` | 40 | 30 -- 50 | VIX level above which all new position entries are halted. |
| `beta_max_abs` | 0.3 | 0.1 -- 0.5 | Maximum absolute portfolio beta. The bot will hedge to keep beta within this range. |
| `portfolio_heat_max` | 0.60 | 0.30 -- 0.80 | Maximum portfolio heat (aggregate risk across all positions). |

### 5.3 Circuit Breaker Tiers

The bot uses a four-tier circuit breaker system that progressively reduces risk as daily losses mount:

| Tier | Trigger | Action |
|---|---|---|
| **Yellow** | -1.0% daily P&L | New position sizes reduced by 50% |
| **Orange** | -2.0% daily P&L | All new entries blocked; existing positions managed only |
| **Red** | -3.0% daily P&L | All day-trade positions closed |
| **Black** | -4.0% daily P&L | Kill switch -- ALL positions closed immediately |

These thresholds can be adjusted in `config/risk.yaml` under `circuit_breaker`.

### 5.4 Complete Parameter Reference

The following table lists all key parameters with their defaults and safe ranges.

**Position Sizing:**

| Parameter | Config Key | Default | Safe Range |
|---|---|---|---|
| Risk per trade | `risk_per_trade_pct` | 0.8% | 0.3% -- 1.5% |
| Max position size | `max_position_pct` | 8% | 4% -- 12% |
| Min position value | `min_position_value` | $100 | $50 -- $500 |
| Max open positions | `max_positions` | 12 | 4 -- 20 |
| Max capital deployed | `max_portfolio_deploy` | 55% | 30% -- 80% |

**Volatility and Regime:**

| Parameter | Config Key | Default | Safe Range |
|---|---|---|---|
| Daily vol target | `vol_target_daily` | 1.0% | 0.5% -- 2.0% |
| Vol target max | `vol_target_max` | 1.5% | 1.0% -- 3.0% |
| Vol scalar floor | `vol_scalar_min` | 0.3 | 0.1 -- 0.5 |
| Vol scalar ceiling | `vol_scalar_max` | 1.5 | 1.0 -- 2.0 |
| Bearish size cut | `bearish_size_cut` | 40% | 20% -- 60% |

**Kelly Criterion:**

| Parameter | Config Key | Default | Safe Range |
|---|---|---|---|
| Kelly enabled | `kelly.enabled` | true | true/false |
| Min trades for Kelly | `kelly.min_trades` | 30 | 20 -- 50 |
| Kelly fraction | `kelly.fraction_mult` | 0.5 (half-Kelly) | 0.25 -- 0.75 |
| Kelly min risk | `kelly.min_risk` | 0.3% | 0.1% -- 0.5% |
| Kelly max risk | `kelly.max_risk` | 2.0% | 1.0% -- 3.0% |

**Daily P&L Controls:**

| Parameter | Config Key | Default | Safe Range |
|---|---|---|---|
| Daily loss halt | `daily_pnl.loss_halt` | -2.5% | -1.0% -- -4.0% |
| Gain lock trigger | `daily_pnl.gain_lock_pct` | 1.5% | 0.5% -- 3.0% |
| Loss halt trigger | `daily_pnl.loss_halt_pct` | -1.0% | -0.5% -- -2.0% |

**Short Selling (disabled by default):**

| Parameter | Config Key | Default | Safe Range |
|---|---|---|---|
| Short enabled | `ALLOW_SHORT` in .env | false | true/false |
| Short size multiplier | `short.size_multiplier` | 0.75 | 0.50 -- 1.00 |
| Short hard stop | `short.hard_stop_pct` | 4.0% | 2.0% -- 6.0% |

### 5.5 What NOT to Change

The following parameters are carefully tuned and should not be modified without understanding the full system:

- **HMM regime detection parameters** (`hmm.*`) -- trained on years of market data; altering these can cause incorrect regime classification
- **Kalman filter parameters** (`kalman_delta`, `kalman_obs_noise`) -- mathematically derived; incorrect values break the pairs tracking
- **Slippage model coefficients** (`slippage.*`) -- calibrated against real execution data
- **Monte Carlo simulation count** (`monte_carlo.simulations`) -- lowering this degrades tail-risk estimates
- **Correlation thresholds** (`portfolio_heat.cluster_correlation_threshold`, `correlation_threshold`) -- prevent concentrated positions; loosening these increases correlation risk
- **Circuit breaker thresholds** -- these are your last line of defense; do not widen them without serious consideration
- **Feature flags for experimental features** (TFT, Alpha Agents, RL Execution) -- these are disabled by default because they are not production-ready

---

## 6. Architecture Overview

### 6.1 High-Level System Diagram

```
                          +------------------------------------------+
                          |           Velox V12 Trading Bot           |
                          +------------------------------------------+
                          |                                          |
  Alpaca API  <---------> |   main.py (Orchestrator Loop)            |
  (Market Data            |     |                                    |
   + Orders)              |     +-- engine/scanner.py                |
                          |     |     Scans 6 strategies every 2min  |
                          |     |                                    |
                          |     +-- engine/signal_processor.py       |
                          |     |     Filters, sizes, submits orders |
                          |     |                                    |
                          |     +-- engine/exit_processor.py         |
                          |     |     Manages TP/SL/trailing/time    |
                          |     |                                    |
                          |     +-- engine/exit_orchestrator.py      |
                          |     |     Advanced exits (profit tiers,  |
                          |     |     dead signal, scale-out)        |
                          |     |                                    |
                          |     +-- engine/broker_sync.py            |
                          |     |     Syncs state with Alpaca        |
                          |     |                                    |
                          |     +-- engine/daily_tasks.py            |
                          |           EOD close, resets, backups     |
                          |                                          |
                          +--+--+--+--+--+--+--+--+-----------------+
                             |  |  |  |  |  |  |  |
            +----------------+  |  |  |  |  |  |  +----------------+
            |                   |  |  |  |  |  |                    |
    +-------v------+  +--------v--v--v--v--v--v--------+  +---------v--------+
    |  Strategies  |  |       Risk Management          |  |   Data Layer     |
    |              |  |                                 |  |                  |
    | StatMR       |  | risk_manager.py   (sizing)      |  | data.py (Alpaca) |
    | VWAP         |  | vol_targeting.py  (vol target)  |  | yfinance (VIX)   |
    | KalmanPairs  |  | circuit_breaker.py (4-tier CB)  |  | FinBERT (NLP)    |
    | ORB          |  | kelly.py          (Kelly crit)  |  | FRED (macro)     |
    | MicroMom     |  | daily_pnl_lock.py (P&L locks)   |  | EDGAR (filings)  |
    | PEAD         |  | beta_neutralizer.py (hedging)   |  |                  |
    +--------------+  | portfolio_heat.py  (correlation) |  +------------------+
                      | pdt_tracker.py    (PDT rule)    |
                      +---------------------------------+
                                     |
                          +----------v-----------+
                          |      Persistence     |
                          |  SQLite (bot.db)     |
                          |  PostgreSQL (Docker)  |
                          +----------------------+
                                     |
               +---------------------+---------------------+
               |                     |                     |
      +--------v--------+  +--------v--------+  +---------v--------+
      |   Dashboard     |  |   Prometheus    |  |     Grafana      |
      | :8080 (FastAPI) |  | :9090 (metrics) |  | :3000 (charts)   |
      +-----------------+  +-----------------+  +------------------+
```

### 6.2 Signal Flow Pipeline

1. **Scanner** (`engine/scanner.py`) -- Every 2 minutes during market hours, all six strategies are scanned for signals across the symbol universe (~120 stocks and ETFs).

2. **Signal Generation** -- Each strategy independently produces `Signal` objects containing: symbol, direction (long/short), confidence score, strategy name, suggested take-profit and stop-loss levels.

3. **Signal Processing** (`engine/signal_processor.py`) -- Signals pass through a multi-stage filter:
   - **Earnings filter** -- skip symbols with earnings within 2 days
   - **Correlation check** -- skip if too correlated (>92%) with an existing position
   - **Data quality gate** -- reject if data is stale or anomalous
   - **ML scoring** -- 4-model ensemble (LightGBM, XGBoost, CatBoost, RandomForest) scores the signal; low-scoring signals are rejected
   - **NLP sentiment** -- FinBERT sentiment analysis adjusts confidence
   - **Transaction cost model** -- reject if expected return does not cover estimated costs
   - **PDT protection** -- block if the trade would violate the Pattern Day Trader rule

4. **Position Sizing** -- Surviving signals are sized using:
   - Risk-per-trade percentage (0.8% default)
   - Volatility targeting (scale to 1% daily vol)
   - Kelly criterion (half-Kelly based on rolling win rate)
   - VIX risk scalar (reduce in high-volatility regimes)
   - Regime adjustment (40% size cut in bearish regimes)
   - Portfolio heat check (block if total heat exceeds limit)

5. **Order Submission** -- Sized orders are submitted to Alpaca as bracket orders (entry + take-profit + stop-loss) via the execution module with smart routing and adaptive TWAP.

### 6.3 Risk Management

Risk management operates at multiple levels:

**Pre-Trade:**
- Position sizing limits (max 8% per position, max 55% deployed)
- Correlation limiter (reject positions correlated >92% with existing holdings)
- Portfolio heat monitor (cap aggregate risk at 60%)
- Sector concentration limit (max 30% in any sector)
- Transaction cost filter (reject negative-EV trades)

**Intraday:**
- Four-tier circuit breaker (Yellow/Orange/Red/Black)
- Daily P&L lock (reduce size after +1.5% gains, halt after -1.0% losses)
- VIX-based scaling (reduce positions as VIX rises; halt above VIX 40)
- Beta neutralization (keep portfolio beta between -0.3 and +0.3)
- Per-symbol daily loss cap ($200 max loss per symbol per day)
- Re-entry cooldown (30-minute block after a stop-loss)

**Portfolio Level:**
- Monte Carlo tail-risk simulation (10,000 paths, 21-day horizon)
- Conditional Value at Risk (CVaR) limit at -8%
- Adaptive allocation (Sortino-weighted rebalancing across strategies)
- Black-Litterman portfolio optimization

### 6.4 Exit Management

Exits are managed by the Exit Orchestrator (`engine/exit_orchestrator.py`) with multiple mechanisms:

- **Take-Profit** -- bracket order TP set at entry
- **Stop-Loss** -- bracket order SL set at entry
- **ATR Trailing Stop** -- once in profit by 0.5x ATR, the stop trails at 1.0-2.5x ATR (varies by strategy)
- **Time Stop** -- positions are closed after a strategy-specific maximum hold time
- **RSI Momentum Exit** -- close profitable positions if RSI exceeds the exit threshold
- **Volatility Expansion Exit** -- close losing positions if ATR expands beyond 2.5x the entry ATR
- **Breakeven Stop** -- after the first partial profit take, stop-loss moves to entry + 0.1%
- **EOD Close** -- intraday strategies are closed before market close, with eligible positions optionally held overnight

### 6.5 Database Schema Overview

Velox uses SQLite (`bot.db`) for local state and PostgreSQL (in Docker) for production persistence. Key tables:

| Table | Purpose |
|---|---|
| `trades` | Completed trade records (symbol, strategy, entry/exit prices, P&L, exit reason, commission) |
| `signals` | All generated signals (acted on or skipped, with skip reasons) |
| `open_positions` | Currently open positions with TP/SL levels, partial exit tracking, overnight hold flags |
| `daily_snapshots` | End-of-day portfolio snapshots (value, cash, P&L, win rate, Sharpe) |
| `audit_log` | Compliance audit trail (order submissions, risk decisions, circuit breaker events) |
| `execution_analytics` | Order fill analysis (slippage, latency, fill quality) |
| `event_log` | System events (strategy scans, errors, restarts) |
| `schema_version` | Database migration tracking |

The schema auto-migrates on startup. No manual database management is required.

---

## 7. Troubleshooting

### 7.1 Common Errors and Fixes

**"ALPACA_API_KEY environment variable is required"**

Your `.env` file is missing the Alpaca API key, or the key is empty. Open `.env` and verify that `ALPACA_API_KEY` and `ALPACA_API_SECRET` are set correctly with no leading or trailing spaces.

**"Set POSTGRES_PASSWORD in .env"**

The Docker Compose file requires `POSTGRES_PASSWORD` to be set. Add a strong password to your `.env` file.

**Bot starts but makes no trades**

- Check if it is outside market hours (9:30 AM -- 4:00 PM ET, weekdays only). The bot only trades during market hours.
- Check if the circuit breaker is active: `curl http://localhost:8080/api/risk`
- Check if there are signals being generated but filtered: `curl http://localhost:8080/api/signals`
- Verify your Alpaca account has sufficient buying power.

**"Connection refused" when accessing the dashboard**

- Verify the bot is running: `docker compose ps`
- Check for startup errors: `docker compose logs velox`
- Ensure port 8080 is not in use by another application.

**Alpaca API errors (403 Forbidden)**

- Your API keys may be expired or incorrect. Regenerate them at [https://app.alpaca.markets](https://app.alpaca.markets).
- If using live keys with `ALPACA_LIVE=false` (or vice versa), the keys will not work. Paper and live keys are separate.

**High memory usage**

The default Docker memory limit is 2 GB for the bot. If you see out-of-memory errors:

```bash
# Check memory usage
docker stats velox-v12

# Increase limit in docker-compose.yml if needed
# Under velox > deploy > resources > limits > memory
```

### 7.2 How to Check Logs

**Docker logs (recommended):**

```bash
# Follow live logs
docker compose logs -f velox

# Last 200 lines
docker compose logs --tail=200 velox

# Filter for errors only
docker compose logs velox 2>&1 | grep -i error
```

**Log file (if running locally):**

```bash
tail -f bot.log
```

**Structured logging** is enabled by default in Docker mode (`STRUCTURED_LOGGING=true`), producing JSON-formatted log lines for machine parsing.

### 7.3 How to Reset the Database

**WARNING: This deletes all trade history, signals, and position data. Back up first.**

```bash
# Stop the bot
docker compose down

# Remove the SQLite database
rm bot.db

# Remove the PostgreSQL volume (Docker)
docker volume rm trading_bot_pgdata

# Restart -- fresh database will be created automatically
docker compose up -d
```

To back up before resetting:

```bash
# SQLite backup
cp bot.db bot.db.backup

# PostgreSQL backup
docker compose exec postgres pg_dump -U velox velox > backup.sql
```

### 7.4 How to Retrain the ML Model

The ML model can be retrained to incorporate recent market data:

```bash
# If running locally with Python installed:
cd trading_bot
python3 scripts/train_ml_model.py --days 252 --optimize --trials 30

# If running in Docker:
docker compose exec velox python scripts/train_ml_model.py --days 252 --optimize --trials 30
```

**Options:**
- `--days 252` -- use the last 252 trading days (1 year) of data for training
- `--optimize` -- run Optuna hyperparameter optimization
- `--trials 30` -- number of Optuna optimization trials (more trials = better model, slower training)

The trained model is saved to `models/` and is automatically detected on the next startup. Training takes 10-30 minutes depending on hardware and the number of optimization trials.

Retraining is recommended every 1-3 months to keep the model adapted to current market conditions.

---

## 8. FAQ

### How much capital do I need?

**Minimum: $25,000.** This is required by the SEC's Pattern Day Trader (PDT) rule, which applies to accounts that execute 4 or more day trades within 5 business days. Velox has built-in PDT protection that tracks your day trades and blocks new entries if you would exceed the limit, but maintaining $25,000+ avoids the restriction entirely.

If your account is below $25,000, the bot will still work but will be limited to 3 day trades per rolling 5-day period. Swing strategies (PEAD, Kalman Pairs) are less affected since they hold positions for multiple days.

### Can I run this on a VPS?

**Yes.** Any Linux VPS with Docker support will work. Recommended providers:

- DigitalOcean (4 GB Droplet, ~$24/month)
- Hetzner (CPX21, ~$8/month)
- AWS Lightsail (4 GB, ~$20/month)
- Any provider with a US East Coast data center (for lower latency to Alpaca)

Use the same Docker setup instructions. For production VPS deployments, also consider:
- Setting up automatic OS updates
- Configuring a firewall (allow only ports you need)
- Setting up Telegram alerts so you are notified of issues remotely

### How much does it cost to run?

**The software itself has zero recurring costs:**
- Alpaca is commission-free for US stock trading
- FRED API is free
- FinBERT NLP sentiment runs locally (no API calls)
- All other data sources are free or included with Alpaca

**Infrastructure costs** depend on where you run it:
- On your own machine: $0 (just electricity)
- On a VPS: $8-24/month depending on provider
- Cloud (AWS/GCP): $15-40/month depending on instance size

### What are the risks?

**Market risk.** This is a fully automated trading system that buys and sells real securities. Markets can move against any position, and no algorithm guarantees profits. Historical performance does not predict future results.

**Technical risk.** Software bugs, API outages, network failures, or broker issues can cause unexpected behavior. The bot includes multiple safety mechanisms (circuit breakers, daily loss halts, kill switch), but no system is immune to technical failures.

**Model risk.** The ML model and strategy parameters are optimized on historical data. Market conditions change, and strategies that worked in the past may underperform in the future. Regular monitoring and periodic retraining are essential.

**Regulatory risk.** Automated trading is legal in the US for personal accounts, but you are responsible for compliance with all applicable regulations, including tax reporting on gains and losses.

**Recommendations:**
- Always start with paper trading and run for at least 2-4 weeks before going live
- Never invest money you cannot afford to lose
- Monitor the bot daily, especially during the first few weeks of live trading
- Keep the circuit breaker thresholds at their defaults or tighter

### Can I modify the strategies?

**Yes.** All source code is included and fully editable. Strategy files are located in the `strategies/` directory:

| File | Strategy |
|---|---|
| `strategies/stat_mean_reversion.py` | Statistical Mean Reversion |
| `strategies/vwap.py` | VWAP Mean Reversion |
| `strategies/kalman_pairs.py` | Kalman Pairs Trading |
| `strategies/orb_v2.py` | Opening Range Breakout |
| `strategies/micro_momentum.py` | Micro Momentum |
| `strategies/pead.py` | Post-Earnings Announcement Drift |

All strategies extend the `strategies/base.py` base class. To add a new strategy:

1. Create a new file in `strategies/`
2. Inherit from the `Signal` base class
3. Implement the `scan()` and `check_exit()` methods
4. Register the strategy in `main.py` and add it to `STRATEGY_ALLOCATIONS` in `config/settings.py`
5. Rebuild the Docker image: `docker compose up -d --build`

### How do I update to a new version?

When a new version is released:

1. Back up your `.env` file and database
2. Replace the `trading_bot/` directory with the new version
3. Restore your `.env` file
4. Rebuild and restart:

```bash
docker compose down
docker compose up -d --build
```

Database migrations run automatically on startup. Your trade history and configuration are preserved.

### Can I run multiple instances?

Running multiple instances against the same brokerage account is not recommended and can cause conflicting trades, duplicate orders, and inconsistent position tracking. If you want to test different configurations, use separate Alpaca paper accounts with different API keys.

---

*This document covers Velox V12. For questions not addressed here, consult the source code comments or contact the developer.*
