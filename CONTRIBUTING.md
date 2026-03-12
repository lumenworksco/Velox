# Contributing

Thanks for your interest in contributing to the Algo Trading Bot.

## Adding a New Strategy

1. Create a new file in `strategies/` (e.g., `strategies/my_strategy.py`).

2. Implement the required interface:

   ```python
   class MyStrategy:
       def __init__(self, config: dict):
           self.config = config

       def scan(self, symbols: list[str], data: dict) -> list[dict]:
           """
           Scan symbols and return a list of trade signals.

           Each signal dict must contain:
             - symbol: str
             - side: "buy" | "sell"
             - entry: float
             - stop_loss: float
             - take_profit: float
             - strategy: str (strategy name)
           """
           signals = []
           # Your logic here
           return signals
   ```

3. Add configuration flags in `config.py`:

   ```python
   MY_STRATEGY_ENABLED = os.getenv("MY_STRATEGY_ENABLED", "false").lower() == "true"
   ```

4. Register the strategy in `strategies/__init__.py` so the orchestrator picks it up.

5. Write tests in `tests/test_my_strategy.py`.

## Pull Request Requirements

- All existing tests must pass (`pytest tests/ -v`).
- New features must be gated behind config flags (disabled by default).
- Include tests for any new strategy or module.
- Update `CHANGELOG.md` with your changes.

## Code Style

- Formatter/linter: [Ruff](https://docs.astral.sh/ruff/)
- Use type hints on all function signatures.
- Run before submitting:

  ```bash
  ruff check .
  ruff format .
  ```

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (redact any API keys)
