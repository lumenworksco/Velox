# Changelog

All notable changes to this project will be documented in this file.

## [4.0.0] - 2026-03-13

### Added
- Multi-timeframe (MTF) confirmation for trade entries
- VIX-based risk scaling to reduce exposure in high-volatility regimes
- News sentiment filter via API integration
- Sector rotation strategy using ETF relative strength
- Pairs trading strategy with cointegration detection
- Advanced exit types: scaled take-profits, trailing stops, RSI-based exits, volatility-based exits
- Docker and docker-compose deployment
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions and Codecov

## [3.0.0] - 2025-09-01

### Added
- ML-based signal filter for trade quality scoring
- Dynamic capital allocation across strategies
- WebSocket real-time price monitoring
- Short selling support
- Gap & Go strategy (pre-market gap continuation)
- Relative strength scanning
- Telegram alerts for fills, errors, and daily P&L summaries
- Web dashboard with live positions and equity curve
- Auto-optimization of strategy parameters via walk-forward analysis

## [2.0.0] - 2025-05-01

### Added
- Momentum strategy for multi-day trend following
- SQLite database for trade logging and analytics
- Backtesting engine with historical data replay
- Earnings date filter to avoid holding through reports
- Correlation filter to limit exposure to correlated positions

## [1.0.0] - 2025-01-01

### Added
- Opening Range Breakout (ORB) strategy with 3:1 R/R
- VWAP Mean Reversion strategy with 45-minute time stop
- Bracket order execution (entry + take-profit + stop-loss)
- Market regime detection via SPY 20-day EMA
- Rich terminal dashboard with live display
- State persistence via state.json
- Circuit breaker at -2.5% daily loss
- Paper/live mode switching via ALPACA_LIVE environment variable
- 50 hardcoded liquid symbols
