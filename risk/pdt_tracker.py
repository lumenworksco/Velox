"""Pattern Day Trader (PDT) rule enforcement.

FINRA Rule 4210: Accounts with < $25,000 equity cannot make more than
3 day trades in a rolling 5-business-day window.

A "day trade" is opening and closing the same security on the same day.
"""

import logging
from collections import deque
from datetime import datetime, date, timedelta

import config

logger = logging.getLogger(__name__)


class PDTTracker:
    """Track day trades and enforce PDT limits."""

    PDT_EQUITY_THRESHOLD = 25_000.0
    MAX_DAY_TRADES = 3  # In 5-business-day rolling window

    def __init__(self):
        # Each entry: (date, symbol) — completed round-trip same-day trades
        self._day_trades: deque[tuple[date, str]] = deque()
        self._enabled = getattr(config, 'PDT_ENFORCEMENT_ENABLED', True)

    def record_day_trade(self, symbol: str, trade_date: date | None = None):
        """Record a completed same-day round trip."""
        d = trade_date or datetime.now(config.ET).date()
        self._day_trades.append((d, symbol))
        self._prune()
        count = self.day_trade_count
        logger.info(f"PDT: Day trade recorded for {symbol}. Count: {count}/{self.MAX_DAY_TRADES}")
        if count >= self.MAX_DAY_TRADES:
            logger.warning(f"PDT: At limit ({count}/{self.MAX_DAY_TRADES}). New day trades BLOCKED.")

    def can_day_trade(self, equity: float) -> bool:
        """Check if a new day trade is allowed.

        Returns True if:
        - PDT enforcement is disabled, OR
        - Account equity >= $25,000, OR
        - Day trade count in last 5 business days < 3
        """
        if not self._enabled:
            return True
        if equity >= self.PDT_EQUITY_THRESHOLD:
            return True
        self._prune()
        return self.day_trade_count < self.MAX_DAY_TRADES

    @property
    def day_trade_count(self) -> int:
        """Number of day trades in the rolling 5-business-day window."""
        self._prune()
        return len(self._day_trades)

    @property
    def remaining_day_trades(self) -> int:
        """How many day trades are still available."""
        return max(0, self.MAX_DAY_TRADES - self.day_trade_count)

    def _prune(self):
        """Remove day trades older than 5 business days."""
        cutoff = self._business_days_ago(5)
        while self._day_trades and self._day_trades[0][0] < cutoff:
            self._day_trades.popleft()

    @staticmethod
    def _business_days_ago(n: int) -> date:
        """Get the date N business days ago."""
        d = datetime.now(config.ET).date()
        count = 0
        while count < n:
            d -= timedelta(days=1)
            if d.weekday() < 5:  # Mon-Fri
                count += 1
        return d

    @property
    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "day_trade_count": self.day_trade_count,
            "remaining": self.remaining_day_trades,
            "at_limit": self.day_trade_count >= self.MAX_DAY_TRADES,
        }
