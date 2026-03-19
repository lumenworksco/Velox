# Velox V10 — Autonomous Algorithmic Trading System

![CI](https://github.com/lumenworksco/Velox-Trader/actions/workflows/ci.yml/badge.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade autonomous equity trading system built on the Alpaca API. Features 6 diversified strategies, a full order management system, tiered circuit breaker, real-time VaR monitoring, structured logging, and a Docker production stack with PostgreSQL, Prometheus, and Grafana.

**Philosophy:** Consistent returns over big wins. Target 0.3-0.8% per trade, 65-75% win rate, 15-25 trades/day across 6 active strategies.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/lumenworksco/Velox-Trader.git
cd Velox-Trader
cp .env.example .env
# Edit .env with your Alpaca API keys
docker compose up -d
```

Services:
- **Velox Dashboard**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Option 2: Local

```bash
pip install -r requirements.txt
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-secret-key"
python main.py
```

---

## Strategies

| Strategy | Allocation | Type | Hold | Description |
|---|---|---|---|---|
| **StatMeanReversion** | 50% | Mean Reversion | Intraday | OU process z-score entries with Hurst/ADF filtering on 2-min bars |
| **VWAP v2 Hybrid** | 20% | Mean Reversion | Intraday | VWAP + OU z-score dual confirmation with bid-ask spread filter |
| **KalmanPairsTrader** | 20% | Market-Neutral | Multi-day | Kalman filter dynamic hedge ratios on cointegrated sector pairs |
| **ORB v2** | 5% | Breakout | Day | Opening range breakout with gap/range quality filters (10:00-11:30 AM) |
| **IntradayMicroMomentum** | 5% | Event-Driven | 8 min | SPY volume spike detection, high-beta stock scalps |
| **PEAD** | Optional | Event-Driven | Multi-day | Post-earnings announcement drift |

### How It Works

1. **9:00 AM** -- Dynamic universe selection filters 127+ symbols by volume, market cap, and regime
2. **9:00 AM** -- StatMR builds mean-reversion universe via Hurst exponent, ADF stationarity, and OU half-life
3. **9:30 AM** -- 2-minute scan cycle begins: all strategies scan for signals every 120 seconds
4. **Signal pipeline** -- Each signal passes through: transaction cost filter, VaR check, correlation limiter, news sentiment, optional LLM scoring
5. **OMS** -- Orders tracked through 7-state lifecycle (PENDING -> SUBMITTED -> FILLED/CANCELLED/REJECTED)
6. **Every 15 min** -- Beta neutralizer checks portfolio beta and hedges with SPY if |beta| > 0.3
7. **Weekly** -- KalmanPairs selects top 15 cointegrated pairs; walk-forward validator checks OOS Sharpe
8. **EOD** -- End-of-day close routine with overnight hold selection

---

## Architecture

```
trading_bot/
  main.py                    # Thin orchestrator (~1500 lines, down from 2300)
  config.py                  # All configuration and strategy parameters
  data.py                    # Alpaca market data (REST + WebSocket)
  execution.py               # Order routing, TWAP splitting, bracket orders
  database.py                # SQLite persistence layer

  engine/                    # V10 engine package (extracted from main.py)
    startup.py               # Module initialization and startup checks
    signal_processor.py      # Full signal pipeline with OMS + cost filter + VaR
    scanner.py               # Strategy scan orchestration
    daily_tasks.py           # Daily reset, weekly tasks, EOD close
    broker_sync.py           # Position reconciliation with broker
    exit_processor.py        # Strategy exits + WebSocket exits
    events.py                # Pub/sub event bus
    metrics.py               # Prometheus counters/gauges/histograms
    logging_config.py        # structlog dev/prod configuration

  oms/                       # Order Management System
    order.py                 # Order dataclass with 7-state machine
    order_manager.py         # Thread-safe registry with idempotency keys
    kill_switch.py           # Emergency halt: cancel all + close all
    transaction_cost.py      # Pre-trade cost estimation (spread + slippage)

  strategies/
    base.py                  # Signal dataclass, shared types
    regime.py                # SPY HMM market regime detection
    stat_mean_reversion.py   # OU z-score mean reversion (50%)
    vwap.py                  # VWAP + OU hybrid entries (20%)
    kalman_pairs.py          # Kalman filter pairs trading (20%)
    orb_v2.py                # Opening range breakout v2 (5%)
    micro_momentum.py        # SPY vol spike micro momentum (5%)
    pead.py                  # Post-earnings announcement drift
    dynamic_universe.py      # Regime-adaptive universe selection

  risk/
    risk_manager.py          # Trade tracking, position limits
    circuit_breaker.py       # V10 tiered circuit breaker (4 tiers)
    var_monitor.py           # Parametric + Historical + Monte Carlo VaR
    correlation_limiter.py   # Eigenvalue-based effective bets + sector limits
    vol_targeting.py         # Volatility-targeted position sizing
    daily_pnl_lock.py        # P&L lock states (NORMAL/GAIN_LOCK/LOSS_HALT)
    beta_neutralizer.py      # Portfolio beta monitoring + SPY hedging

  db/                        # SQLAlchemy database abstraction
    __init__.py              # Dual-backend engine (SQLite/PostgreSQL)
    models.py                # 16 table definitions
    migrations/              # Alembic migration scripts

  auth/                      # Dashboard authentication
    jwt_auth.py              # JWT token create/verify

  analytics/                 # Performance metrics and statistical tools
    performance.py           # Sharpe, Sortino, drawdown, attribution
    ou_tools.py              # Ornstein-Uhlenbeck parameter fitting
    hurst.py                 # Hurst exponent (R/S analysis)
    consistency_score.py     # Consistency score (0-100)

  web_dashboard.py           # FastAPI dashboard with Apple-style UI
  monitoring/
    prometheus.yml           # Prometheus scrape configuration

  tests/                     # 911 unit tests
  Dockerfile
  docker-compose.yml         # Production stack: Velox + PostgreSQL + Prometheus + Grafana
```

---

## V10 Features

### Order Management System (OMS)
Full order lifecycle tracking with 7-state machine (PENDING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED / CANCELLED / REJECTED / EXPIRED). Thread-safe registry with idempotency keys prevents duplicate orders.

### Tiered Circuit Breaker
Progressive risk reduction based on daily P&L:
- **Yellow** (-1%): Reduce new position sizes by 50%
- **Orange** (-2%): Stop all new entries
- **Red** (-3%): Close day trades, keep swing positions
- **Black** (-4%): Kill switch -- close everything

### VaR Monitor
Real-time portfolio Value-at-Risk using three methods:
- Parametric VaR (95% and 99%)
- Historical simulation
- Monte Carlo (10,000 paths)

### Correlation Limiter
Prevents concentration risk via eigenvalue-based effective bets calculation and sector Herfindahl index monitoring.

### Transaction Cost Filter
Pre-trade expected value check: rejects signals where estimated costs (spread + slippage + commission) exceed expected profit.

### Structured Logging
JSON-formatted logs in production (structlog), human-readable in development. All events include correlation IDs for tracing.

### Prometheus Metrics
Exposed at `/metrics` on port 8080:
- `velox_open_positions` -- Current position count
- `velox_daily_pnl` -- Running daily P&L
- `velox_order_latency` -- Order submission to fill latency
- `velox_signal_count` -- Signals generated per strategy
- `velox_circuit_breaker_state` -- Current circuit breaker tier

### Web Dashboard
Apple-style frosted glass UI at http://localhost:8080 with:
- Live equity and P&L from Alpaca account
- Open positions with real-time unrealized P&L
- Trade log filterable by strategy
- Signal filter analysis and exit reason breakdown
- OMS status, circuit breaker state, kill switch controls
- Auto-refresh every 30 seconds

### Dynamic Universe
Daily symbol selection at 9 AM based on volume, market cap, and current market regime. Adapts universe size and composition to volatility conditions.

---

## Risk Engine

| Component | Description |
|---|---|
| **Tiered Circuit Breaker** | 4-tier progressive risk reduction (-1% to -4%) |
| **VaR Monitor** | Parametric + Historical + Monte Carlo VaR at 95%/99% |
| **Correlation Limiter** | Eigenvalue effective bets + sector concentration limits |
| **Volatility Targeting** | Scales position sizes so daily portfolio vol = 1% target |
| **Daily P&L Lock** | GAIN_LOCK at +1.5% (30% sizing), LOSS_HALT at -1.0% (stops new trades) |
| **Beta Neutralization** | Monitors portfolio beta, hedges with SPY when \|beta\| > 0.3 |
| **Transaction Cost Filter** | Rejects negative expected-value trades before submission |
| **Kill Switch** | Emergency halt: cancels all orders, closes all positions |
| **TWAP Execution** | Orders > $2,000 split into 5 time-weighted slices |

---

## Docker Production Stack

```bash
docker compose up -d              # Start all services
docker compose logs -f velox      # Follow trading bot logs
docker compose exec velox python main.py --diagnose  # Run diagnostic
```

| Service | Port | Description |
|---|---|---|
| **velox** | 8080 | Trading bot + web dashboard |
| **postgres** | 5432 | PostgreSQL 16 database |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Monitoring dashboards (admin/admin) |

Environment variables are configured in `.env` (see `.env.example`).

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `ALPACA_LIVE` | `false` | Paper vs live trading |
| `MAX_POSITIONS` | `12` | Maximum concurrent positions |
| `SCAN_INTERVAL_SEC` | `120` | Seconds between strategy scans |
| `RISK_PER_TRADE_PCT` | `0.8%` | Max risk per trade |
| `VOL_TARGET_DAILY` | `1.0%` | Daily portfolio volatility target |
| `DATABASE_URL` | `sqlite:///bot.db` | Database connection (PostgreSQL supported) |
| `STRUCTURED_LOGGING` | `false` | Enable JSON structured logging |
| `WEB_DASHBOARD_ENABLED` | `true` | Enable web dashboard on port 8080 |
| `WATCHDOG_ENABLED` | `false` | Enable position watchdog |
| `NEWS_SENTIMENT_ENABLED` | `true` | Enable Alpaca news sentiment filter |
| `LLM_SCORING_ENABLED` | `false` | Enable Claude Haiku signal scoring |
| `ADAPTIVE_EXITS_ENABLED` | `true` | Enable VIX-aware adaptive exits |
| `WALK_FORWARD_ENABLED` | `true` | Enable weekly walk-forward validation |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram trade alerts |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific modules
pytest tests/test_v10_oms.py -v
pytest tests/test_v10_events.py -v
pytest tests/test_v10_phase4.py -v
```

911 tests covering all strategies, risk modules, OMS, event bus, circuit breaker, and analytics.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check (Docker/monitoring) |
| `GET /api/stats` | Live performance stats from Alpaca + analytics |
| `GET /api/positions` | Current open positions (live broker data) |
| `GET /api/trades` | Trade history (filterable by strategy) |
| `GET /api/portfolio_history` | Daily portfolio snapshots |
| `GET /api/signals` | Signal log by date |
| `GET /api/signal_stats` | Signal skip reason breakdown |
| `GET /api/trade_analysis` | Exit reason analysis |
| `GET /api/risk-state` | Risk engine state (vol scalar, beta, P&L lock) |
| `GET /api/strategy_health` | Per-strategy health metrics |
| `GET /api/v10/oms` | OMS order status and history |
| `GET /api/v10/circuit_breaker` | Circuit breaker state and history |
| `POST /api/v10/kill_switch/activate` | Activate emergency kill switch |

---

## Version History

| Version | Date | Focus |
|---|---|---|
| V1 | 2025-01 | ORB + VWAP strategies, basic risk management |
| V2 | 2025-05 | Momentum strategy, WebSocket monitoring |
| V3 | 2025-09 | ML signal filter, short selling, dynamic allocation |
| V4 | 2026-03 | Sector rotation, pairs trading, MTF, news filter |
| V5 | 2026-03 | EMA scalping, shadow mode, advanced exits |
| V6 | 2026-03 | Complete rebuild: statistical mean reversion, vol targeting |
| V7 | 2026-03 | 5-strategy diversification, news sentiment, LLM scoring, adaptive exits |
| V8 | 2026-03 | Bug fixes, thread safety, dead code removal |
| V9 | 2026-03 | Engine decomposition, PostgreSQL, OMS skeleton |
| **V10** | **2026-03** | **Production-grade: tiered circuit breaker, VaR monitor, correlation limiter, event bus, structured logging, Prometheus metrics, Docker stack, Apple-style dashboard** |

---

## Risk Warning

> **This is experimental software. Use at your own risk. Past performance is not indicative of future results.** Trading equities involves substantial risk of loss. This software is provided for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this software.

---

## License

[MIT](LICENSE)
