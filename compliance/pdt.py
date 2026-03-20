"""COMPLY-004: Pattern Day Trader (PDT) Compliance — rolling 5-business-day tracking.

FINRA Rule 4210: An account with < $25,000 equity is limited to 3 day trades
in any rolling 5-business-day window.  Exceeding this triggers a 90-day
restriction.

This module:
  - Tracks day trades in a rolling 5-business-day window
  - Checks whether a new day trade is allowed given current equity
  - Handles mid-day equity drops below $25k
  - Is fully thread-safe
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDT_EQUITY_THRESHOLD = 25_000.00
PDT_MAX_DAY_TRADES = 3
PDT_ROLLING_WINDOW_DAYS = 5  # business days


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DayTradeRecord:
    """A single day-trade record."""

    symbol: str
    trade_date: date
    timestamp: str = ""
    side: str = ""
    qty: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0


@dataclass
class PDTStatus:
    """Current PDT compliance status."""

    can_day_trade: bool
    remaining_day_trades: int
    reason: str
    equity: float = 0.0
    day_trades_in_window: int = 0
    window_start: date = field(default_factory=date.today)
    window_end: date = field(default_factory=date.today)
    is_pdt_restricted: bool = False


# ---------------------------------------------------------------------------
# PDTCompliance
# ---------------------------------------------------------------------------

class PDTCompliance:
    """Pattern Day Trader compliance tracker with rolling 5-business-day window.

    Thread-safe.  Maintains a persistent record of day trades for rolling
    window calculations.

    Usage:
        pdt = PDTCompliance()

        # Before opening a day trade
        allowed, remaining, reason = pdt.can_day_trade(equity=24_500.00)
        if allowed:
            execute_trade()
            pdt.record_day_trade("AAPL")

        # Check status
        status = pdt.get_status(equity=24_500.00)
    """

    def __init__(self, log_path: str | None = None, alert_callback=None):
        """
        Args:
            log_path: Path for persistent day-trade log (JSONL).
            alert_callback: Optional callable(level, message, source) for alerts.
        """
        self._log_path = Path(log_path or "pdt_day_trades.jsonl")
        self._alert_callback = alert_callback
        self._lock = threading.RLock()  # reentrant for nested calls

        # In-memory day trade records
        self._day_trades: list[DayTradeRecord] = []

        # Track whether PDT restriction has been triggered
        self._pdt_restricted = False
        self._restriction_start: Optional[date] = None

        # Load persisted records
        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_day_trade(self, equity: float) -> tuple[bool, int, str]:
        """Check whether a new day trade is allowed.

        Args:
            equity: Current account equity in dollars.

        Returns:
            Tuple of (allowed: bool, remaining: int, reason: str).
            - allowed: True if a day trade can be placed.
            - remaining: Number of day trades remaining in the window.
            - reason: Human-readable explanation.
        """
        with self._lock:
            return self._check_inner(equity)

    def record_day_trade(self, symbol: str, trade_date: date | None = None,
                         side: str = "", qty: float = 0.0,
                         entry_price: float = 0.0, exit_price: float = 0.0):
        """Record a completed day trade.

        A day trade is defined as opening and closing a position in the same
        security on the same trading day.

        Args:
            symbol: Ticker symbol.
            trade_date: Date of the trade. Defaults to today (ET).
            side: "buy" or "sell".
            qty: Number of shares.
            entry_price: Entry fill price.
            exit_price: Exit fill price.
        """
        with self._lock:
            self._record_inner(symbol, trade_date, side, qty,
                               entry_price, exit_price)

    def get_status(self, equity: float) -> PDTStatus:
        """Get comprehensive PDT compliance status.

        Args:
            equity: Current account equity.

        Returns:
            PDTStatus dataclass with full details.
        """
        with self._lock:
            allowed, remaining, reason = self._check_inner(equity)
            window_start, window_end = self._get_rolling_window()
            trades_in_window = self._count_trades_in_window(window_start, window_end)

            return PDTStatus(
                can_day_trade=allowed,
                remaining_day_trades=remaining,
                reason=reason,
                equity=equity,
                day_trades_in_window=trades_in_window,
                window_start=window_start,
                window_end=window_end,
                is_pdt_restricted=self._pdt_restricted,
            )

    def get_trades_in_window(self) -> list[DayTradeRecord]:
        """Return all day trades in the current rolling window."""
        with self._lock:
            window_start, window_end = self._get_rolling_window()
            return [
                t for t in self._day_trades
                if window_start <= t.trade_date <= window_end
            ]

    def reset_restriction(self):
        """Manually clear a PDT restriction (e.g. after 90-day cool-off)."""
        with self._lock:
            self._pdt_restricted = False
            self._restriction_start = None
            logger.info("PDT: Restriction manually cleared")

    # ------------------------------------------------------------------
    # Internal — core logic
    # ------------------------------------------------------------------

    def _check_inner(self, equity: float) -> tuple[bool, int, str]:
        """Core PDT check logic."""
        # Accounts with >= $25k equity are exempt from PDT limits
        if equity >= PDT_EQUITY_THRESHOLD:
            return True, 999, "Equity above $25k — PDT rule does not apply"

        # Check if PDT-restricted
        if self._pdt_restricted:
            return (
                False, 0,
                f"PDT restriction active since {self._restriction_start}. "
                f"No day trades allowed until restriction is cleared."
            )

        # Count day trades in rolling window
        window_start, window_end = self._get_rolling_window()
        trades_in_window = self._count_trades_in_window(window_start, window_end)
        remaining = max(0, PDT_MAX_DAY_TRADES - trades_in_window)

        if remaining <= 0:
            return (
                False, 0,
                f"PDT limit reached: {trades_in_window} day trades in rolling "
                f"5-business-day window ({window_start} to {window_end}). "
                f"Equity ${equity:,.2f} is below $25k threshold."
            )

        # Allow but warn if close to limit
        if remaining == 1:
            reason = (
                f"CAUTION: Last day trade available. "
                f"{trades_in_window}/{PDT_MAX_DAY_TRADES} used in window. "
                f"Equity ${equity:,.2f} is below $25k."
            )
        else:
            reason = (
                f"{remaining} day trades remaining in window. "
                f"{trades_in_window}/{PDT_MAX_DAY_TRADES} used. "
                f"Equity ${equity:,.2f} is below $25k."
            )

        return True, remaining, reason

    def _record_inner(self, symbol: str, trade_date: date | None,
                      side: str, qty: float,
                      entry_price: float, exit_price: float):
        """Core recording logic."""
        if trade_date is None:
            trade_date = datetime.now(config.ET).date()

        now = datetime.now(config.ET)

        record = DayTradeRecord(
            symbol=symbol,
            trade_date=trade_date,
            timestamp=now.isoformat(),
            side=side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
        )

        self._day_trades.append(record)

        # Persist
        self._write_record(record)

        # Check if this triggers PDT violation
        window_start, window_end = self._get_rolling_window()
        trades_in_window = self._count_trades_in_window(window_start, window_end)

        logger.info(
            f"PDT: Recorded day trade {symbol} on {trade_date} "
            f"({trades_in_window}/{PDT_MAX_DAY_TRADES} in window)"
        )

        if trades_in_window > PDT_MAX_DAY_TRADES:
            self._pdt_restricted = True
            self._restriction_start = trade_date
            msg = (
                f"PDT VIOLATION: {trades_in_window} day trades in "
                f"5-business-day window (limit: {PDT_MAX_DAY_TRADES}). "
                f"Account may be restricted."
            )
            logger.critical(msg)

            if self._alert_callback:
                try:
                    self._alert_callback("EMERGENCY", msg, "pdt_compliance")
                except Exception as e:
                    logger.error(f"PDT alert callback failed: {e}")

        elif trades_in_window == PDT_MAX_DAY_TRADES:
            msg = (
                f"PDT WARNING: {trades_in_window}/{PDT_MAX_DAY_TRADES} day trades "
                f"used. No more day trades allowed in this window."
            )
            logger.warning(msg)

            if self._alert_callback:
                try:
                    self._alert_callback("WARNING", msg, "pdt_compliance")
                except Exception as e:
                    logger.error(f"PDT alert callback failed: {e}")

        # Prune old records (keep last 60 business days)
        cutoff = trade_date - timedelta(days=90)
        self._day_trades = [t for t in self._day_trades if t.trade_date >= cutoff]

    # ------------------------------------------------------------------
    # Internal — rolling window
    # ------------------------------------------------------------------

    def _get_rolling_window(self) -> tuple[date, date]:
        """Calculate the current 5-business-day rolling window.

        Returns (window_start, window_end) where window_end is today.
        """
        today = datetime.now(config.ET).date()
        window_end = today

        # Walk back 5 business days
        business_days_back = 0
        current = today
        while business_days_back < PDT_ROLLING_WINDOW_DAYS:
            current -= timedelta(days=1)
            if self._is_business_day(current):
                business_days_back += 1

        window_start = current
        return window_start, window_end

    @staticmethod
    def _is_business_day(d: date) -> bool:
        """Check if a date is a business day (Mon-Fri, no major holidays).

        Uses a simplified check: weekdays only.  For production, integrate
        with a market calendar (e.g. exchange_calendars or pandas_market_calendars).
        """
        return d.weekday() < 5  # Monday=0, Friday=4

    def _count_trades_in_window(self, window_start: date, window_end: date) -> int:
        """Count day trades within the rolling window."""
        return sum(
            1 for t in self._day_trades
            if window_start <= t.trade_date <= window_end
        )

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _write_record(self, record: DayTradeRecord):
        """Append a day-trade record to the log file."""
        try:
            entry = {
                "symbol": record.symbol,
                "trade_date": record.trade_date.isoformat(),
                "timestamp": record.timestamp,
                "side": record.side,
                "qty": record.qty,
                "entry_price": record.entry_price,
                "exit_price": record.exit_price,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"PDT: Failed to write trade record: {e}")

    def _load_history(self):
        """Load persisted day-trade records from log file."""
        if not self._log_path.exists():
            return

        try:
            cutoff = datetime.now(config.ET).date() - timedelta(days=90)
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        trade_date = date.fromisoformat(entry["trade_date"])
                        if trade_date < cutoff:
                            continue
                        self._day_trades.append(DayTradeRecord(
                            symbol=entry.get("symbol", ""),
                            trade_date=trade_date,
                            timestamp=entry.get("timestamp", ""),
                            side=entry.get("side", ""),
                            qty=float(entry.get("qty", 0)),
                            entry_price=float(entry.get("entry_price", 0)),
                            exit_price=float(entry.get("exit_price", 0)),
                        ))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug(f"PDT: Skipping malformed log line: {e}")

            logger.info(f"PDT: Loaded {len(self._day_trades)} day-trade records from history")
        except Exception as e:
            logger.error(f"PDT: Failed to load history: {e}")
