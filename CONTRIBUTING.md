# Contributing

Thanks for your interest in contributing to Velox.

## Adding a New Strategy

1. Create a new file in `strategies/` (e.g., `strategies/my_strategy.py`).

2. Implement the required interface:

   ```python
   class MyStrategy:
       def __init__(self):
           pass

       def scan(self, symbols: list[str], data: dict) -> list[dict]:
           """
           Scan symbols and return a list of trade signals.

           Each signal dict must contain:
             - symbol: str
             - side: "buy" | "sell"
             - entry: float
             - stop_loss: float
             - take_profit: float
             - strategy: str (strategy name for routing)
           """
           signals = []
           # Your logic here
           return signals
   ```

3. Add configuration flags in `config.py`:

   ```python
   MY_STRATEGY_ENABLED = os.getenv("MY_STRATEGY_ENABLED", "false").lower() == "true"
   ```

4. Register the strategy in `engine/startup.py` inside `initialize_strategies()`.

5. Write tests in `tests/test_my_strategy.py`.

## Project Structure

- **`engine/`** -- Core trading engine (startup, scanning, signal processing, exits, events)
- **`oms/`** -- Order Management System (order lifecycle, kill switch, cost model)
- **`risk/`** -- Risk modules (circuit breaker, VaR, correlation, vol targeting, beta)
- **`strategies/`** -- Trading strategies (StatMR, VWAP, Pairs, ORB, MicroMom, PEAD)
- **`db/`** -- SQLAlchemy database abstraction and Alembic migrations
- **`auth/`** -- JWT authentication for the web dashboard
- **`analytics/`** -- Performance metrics, OU tools, Hurst exponent, consistency scoring
- **`tests/`** -- 911 unit tests

## Pull Request Requirements

- All existing tests must pass (`pytest tests/ -v`).
- New features must be gated behind config flags (disabled by default).
- Include tests for any new strategy or module.
- Update `CHANGELOG.md` with your changes.
- Signals must pass through the full pipeline in `engine/signal_processor.py` (cost filter, VaR, correlation limiter).

## Code Style

- Formatter/linter: [Ruff](https://docs.astral.sh/ruff/)
- Use type hints on all function signatures.
- Run before submitting:

  ```bash
  ruff check .
  ruff format .
  ```

## Docker Development

```bash
docker compose up -d --build velox   # Rebuild and restart bot
docker compose logs -f velox         # Follow logs
docker compose exec velox python main.py --diagnose  # Run diagnostics
```

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (redact any API keys)
