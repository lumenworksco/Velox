# Velox V6 — Autonomous Algorithmic Trading System

![CI](https://github.com/lumenworksco/Velox-Trader/actions/workflows/ci.yml/badge.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An autonomous equity trading system built on the Alpaca API, targeting consistent low-variance returns through statistical mean reversion. Features volatility-targeted position sizing, daily P&L locking, beta neutralization, and a real-time web dashboard.

**Philosophy:** V6 abandons momentum-chasing (V1-V5) in favor of mean reversion. Targets 0.3-0.8% per trade, 65-75% win rate, 15-30 trades/day, and 0.5-1.5% daily P&L.

---

## Quick Start

```bash
git clone https://github.com/lumenworksco/Velox-Trader.git
cd Velox-Trader
pip install -r requirements.txt
```

Set your environment variables:

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
export ALPACA_LIVE=false                  # true for live trading
export TELEGRAM_ENABLED=false              # optional
export TELEGRAM_BOT_TOKEN=""              # from @BotFather
export TELEGRAM_CHAT_ID=""                # your chat ID
```

Run:

```bash
python main.py
```

---

## Strategies

| Strategy | Allocation | Type | Hold | Description |
|---|---|---|---|---|
| **StatMeanReversion** | 60% | Mean Reversion | Intraday | OU process z-score entries with Hurst/ADF filtering |
| **KalmanPairsTrader** | 25% | Market-Neutral | Multi-day | Kalman filter dynamic hedge ratios on cointegrated pairs |
| **IntradayMicroMomentum** | 15% | Event-Driven | 8 min | SPY volume spike detection, high-beta stock scalps |

### How It Works

1. **9:00 AM** — StatMR filters 128 symbols by Hurst exponent (<0.52), ADF stationarity, and OU half-life (1-48h) to build a universe of ~40 mean-reverting candidates
2. **9:30 AM** — 2-minute scan cycle begins: all three strategies scan for signals every 120 seconds
3. **Ongoing** — Z-score based entries (|z| > 1.5) and exits (|z| < 0.2), with partial exits at |z| < 0.5 and stops at |z| > 2.5
4. **Every 15 min** — Beta neutralizer checks portfolio beta and hedges with SPY if |beta| > 0.3
5. **Weekly (Monday)** — KalmanPairs selects top 15 cointegrated pairs within sector groups

---

## Architecture

```
trading_bot/
  main.py                  # Entry point, 2-min scan loop orchestrator
  config.py                # All configuration and strategy parameters
  data.py                  # Alpaca market data (REST + WebSocket)
  execution.py             # Order routing, TWAP splitting, bracket orders
  database.py              # SQLite persistence (trades, signals, OU params)
  dashboard.py             # Rich terminal dashboard
  web_dashboard.py         # FastAPI web dashboard (port 8080)
  position_monitor.py      # WebSocket real-time position monitoring
  notifications.py         # Telegram trade alerts
  earnings.py              # Earnings calendar filter
  correlation.py           # Correlation-based position conflict filter
  strategies/
    base.py                # Signal dataclass, shared types
    regime.py              # SPY EMA market regime detection
    stat_mean_reversion.py # OU z-score mean reversion (60%)
    kalman_pairs.py        # Kalman filter pairs trading (25%)
    micro_momentum.py      # SPY vol spike micro momentum (15%)
    archive/               # V1-V5 strategies (preserved)
  risk/
    risk_manager.py        # Trade tracking, circuit breaker, position limits
    vol_targeting.py       # Volatility-targeted position sizing
    daily_pnl_lock.py      # P&L lock states (NORMAL/GAIN_LOCK/LOSS_HALT)
    beta_neutralizer.py    # Portfolio beta monitoring + SPY hedging
  analytics/
    ou_tools.py            # Ornstein-Uhlenbeck parameter fitting
    hurst.py               # Hurst exponent (R/S analysis)
    consistency_score.py   # Consistency score (0-100)
  tests/                   # 108 unit tests
  Dockerfile
  docker-compose.yml
```

---

## Risk Engine

| Component | Description |
|---|---|
| **Volatility Targeting** | Scales position sizes so daily portfolio vol = 1% target. Blends VIX (30%), portfolio ATR (40%), rolling P&L std (30%) |
| **Daily P&L Lock** | GAIN_LOCK at +1.5% (reduces to 30% sizing), LOSS_HALT at -1.0% (stops all new trades) |
| **Beta Neutralization** | Monitors dollar-weighted portfolio beta, hedges with SPY when \|beta\| > 0.3 |
| **Circuit Breaker** | Hard stop at -2.5% daily loss, closes all positions |
| **TWAP Execution** | Orders > $2,000 split into 5 time-weighted slices (60s apart) |

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `ALPACA_LIVE` | `false` | Paper vs live trading |
| `MAX_POSITIONS` | `12` | Maximum concurrent positions |
| `MAX_DAILY_LOSS` | `-2.5%` | Circuit breaker threshold |
| `SCAN_INTERVAL_SEC` | `120` | Seconds between strategy scans |
| `RISK_PER_TRADE_PCT` | `0.8%` | Max risk per trade |
| `VOL_TARGET_DAILY` | `1.0%` | Daily portfolio volatility target |
| `PNL_GAIN_LOCK_PCT` | `+1.5%` | P&L threshold to enter GAIN_LOCK |
| `PNL_LOSS_HALT_PCT` | `-1.0%` | P&L threshold to enter LOSS_HALT |
| `MR_ZSCORE_ENTRY` | `1.5` | Mean reversion z-score entry threshold |
| `MR_ZSCORE_EXIT` | `0.2` | Mean reversion z-score full exit |
| `MR_ZSCORE_STOP` | `2.5` | Mean reversion z-score stop loss |
| `MR_HURST_MAX` | `0.52` | Maximum Hurst exponent for MR universe |
| `PAIRS_ZSCORE_ENTRY` | `2.0` | Pairs z-score entry threshold |
| `BETA_MAX_ABS` | `0.3` | Max absolute portfolio beta before hedging |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram trade alerts |
| `WEBSOCKET_MONITORING` | `true` | Enable WebSocket position monitoring |

---

## Web Dashboard

Access at `http://localhost:8080` when running. Features:

- **Equity curve** — 60-day portfolio value chart
- **Risk state** — Vol scalar, portfolio beta, P&L lock status, consistency score
- **Strategy allocation** — MR 60% / PAIRS 25% / MICRO 15% with open trade counts
- **Trade log** — Filterable by strategy with P&L breakdown
- **Signal analysis** — Filter skip reasons and exit reason breakdown
- **Shadow trades** — Paper trade tracking for strategy evaluation

API endpoints: `/api/stats`, `/api/trades`, `/api/positions`, `/api/portfolio_history`, `/api/consistency`, `/api/risk-state`, `/api/signal_stats`, `/api/shadow_trades`, `/api/trade_analysis`

---

## Docker

```bash
docker-compose up -d
```

The bot runs in a container with automatic restarts and health checks. Set environment variables in `.env`.

---

## Testing

```bash
# Run all V6 tests
pytest tests/ --ignore=tests/test_risk.py --ignore=tests/test_strategies.py \
  --ignore=tests/test_exit_manager.py --ignore=tests/test_news_filter.py -v

# Run specific test modules
pytest tests/test_stat_mean_reversion.py -v
pytest tests/test_kalman_pairs.py -v
pytest tests/test_vol_targeting.py -v
```

108 tests covering all V6 strategies, risk modules, and analytics.

---

## Version History

| Version | Focus |
|---|---|
| V1 | ORB + VWAP strategies, basic risk management |
| V2 | Momentum strategy, WebSocket monitoring |
| V3 | ML signal filter, short selling, dynamic allocation |
| V4 | Sector rotation, pairs trading, MTF confirmation, news filter |
| V5 | EMA scalping, shadow mode, advanced exits |
| **V6** | **Complete rebuild: statistical mean reversion, volatility targeting, P&L locking, beta neutralization** |

---

## Risk Warning

> **This is experimental software. Use at your own risk. Past performance is not indicative of future results.** Trading equities involves substantial risk of loss. This software is provided for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this software.

---

## License

[MIT](LICENSE)
