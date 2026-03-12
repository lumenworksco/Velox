# Algo Trading Bot V4

<!-- Badges -->
![CI](https://github.com/YOUR_USERNAME/trading-bot/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/trading-bot/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/trading-bot)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An automated equity trading bot built on the Alpaca API, featuring 6 strategies, ML-based signal filtering, real-time WebSocket monitoring, and a web dashboard. Supports both paper and live trading with comprehensive risk management.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/trading-bot.git
cd trading-bot
pip install -r requirements.txt
```

Set your environment variables:

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
export ALPACA_LIVE=false          # true for live trading
export TELEGRAM_TOKEN=""          # optional
export TELEGRAM_CHAT_ID=""        # optional
```

Run:

```bash
python main.py
```

---

## Strategies

| Strategy | Type | Hold | Description |
|---|---|---|---|
| ORB | Breakout | Day | Opening range breakout with 3:1 R/R |
| VWAP | Mean Reversion | Day | VWAP band bounce with time stop |
| Momentum | Trend | Swing | Multi-day momentum continuation |
| Gap & Go | Breakout | Day | Pre-market gap continuation |
| Sector Rotation | Momentum | Swing | Sector ETF relative strength |
| Pairs Trading | Market-Neutral | Swing | Cointegration-based pairs |

---

## Architecture

```
trading_bot/
  main.py            # Entry point, orchestrator loop
  config.py          # All configuration flags and parameters
  data.py            # Market data fetching and caching
  strategies/        # Strategy implementations (ORB, VWAP, Momentum, etc.)
  execution.py       # Order routing, bracket orders, position management
  risk.py            # Risk manager, circuit breaker, position sizing
  dashboard.py       # Rich terminal dashboard + web dashboard
  models/            # ML signal filter model artifacts
  tests/             # Test suite
  requirements.txt
  Dockerfile
  docker-compose.yml
```

---

## Features

### V3
- ML-based signal filter for trade quality scoring
- WebSocket real-time price monitoring
- Dynamic capital allocation across strategies
- Short selling support
- Telegram alerts for fills, errors, and daily P&L
- Web dashboard with live positions and equity curve

### V4
- Multi-timeframe (MTF) confirmation for entries
- VIX-based risk scaling (reduce size in high-vol regimes)
- News sentiment filter via API integration
- Sector rotation strategy with ETF relative strength
- Pairs trading with cointegration detection
- Advanced exits: scaled take-profits, trailing stops, RSI-based exits, volatility-based exits
- Docker deployment
- Comprehensive test suite with CI/CD

---

## Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `ALPACA_LIVE` | `false` | Paper vs live trading |
| `MAX_POSITIONS` | `10` | Maximum concurrent positions |
| `MAX_DAILY_LOSS` | `-2.5%` | Circuit breaker threshold |
| `ORB_ENABLED` | `true` | Enable ORB strategy |
| `VWAP_ENABLED` | `true` | Enable VWAP strategy |
| `MOMENTUM_ENABLED` | `true` | Enable Momentum strategy |
| `GAP_GO_ENABLED` | `true` | Enable Gap & Go strategy |
| `SECTOR_ROTATION_ENABLED` | `false` | Enable Sector Rotation |
| `PAIRS_ENABLED` | `false` | Enable Pairs Trading |
| `ML_FILTER` | `true` | Enable ML signal filter |
| `VIX_SCALING` | `true` | Scale risk by VIX level |
| `NEWS_FILTER` | `false` | Enable news sentiment filter |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram alerts |
| `WEB_DASHBOARD` | `false` | Enable web dashboard |
| `SCAN_INTERVAL` | `60` | Seconds between scans |

---

## Docker

```bash
docker-compose up -d
```

The bot runs in a container with automatic restarts. Set environment variables in `.env` or `docker-compose.yml`.

---

## Risk Warning

> **This is experimental software. Use at your own risk. Past performance is not indicative of future results.** Trading equities involves substantial risk of loss. This software is provided for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this software.

---

## License

[MIT](LICENSE)
