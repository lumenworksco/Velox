"""COMPLY-002: Market Manipulation Detection — self-surveillance for wash trading, spoofing, etc.

Monitors own trading activity for patterns that could be flagged as manipulation:
  - Wash trading: buy + sell of same symbol within seconds
  - Spoofing: rapid order placement and cancellation
  - Marking the close: concentrated trading in final 5 minutes
  - Layering: multiple limit orders at different prices cancelled before fill

Runs daily check on all trades and produces a SurveillanceReport.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SurveillanceFlag:
    """A single surveillance flag for a potential issue."""

    flag_type: str  # "wash_trade", "spoofing", "marking_close", "layering"
    severity: str   # "low", "medium", "high"
    symbol: str
    timestamp: str
    description: str
    trades_involved: list[dict] = field(default_factory=list)


@dataclass
class SurveillanceReport:
    """Daily surveillance report."""

    date: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(config.ET))
    total_trades_checked: int = 0
    flags: list[SurveillanceFlag] = field(default_factory=list)
    wash_trade_count: int = 0
    spoofing_count: int = 0
    marking_close_count: int = 0
    layering_count: int = 0
    clean: bool = True
    errors: list[str] = field(default_factory=list)

    @property
    def has_flags(self) -> bool:
        return len(self.flags) > 0

    @property
    def high_severity_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "high")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Wash trade: buy+sell within this many seconds
WASH_TRADE_WINDOW_SEC = 60

# Marking the close: trades in last N minutes before close
MARKING_CLOSE_MINUTES = 5
MARKING_CLOSE_CONCENTRATION_PCT = 0.30  # flag if > 30% of daily volume in last 5 min

# Spoofing: order placed and cancelled within N seconds
SPOOFING_CANCEL_WINDOW_SEC = 5
SPOOFING_MIN_CANCELS = 3  # flag if >= 3 rapid cancels for same symbol

# Layering: multiple orders at different prices
LAYERING_MIN_ORDERS = 4  # flag if >= 4 same-side orders at different prices


# ---------------------------------------------------------------------------
# SelfSurveillance
# ---------------------------------------------------------------------------

class SelfSurveillance:
    """Monitors own trading for potential market manipulation patterns.

    Usage:
        surveillance = SelfSurveillance()

        # Daily check (call after market close)
        report = surveillance.run_daily_check(today_trades)

        # Real-time check on individual trade
        is_wash = surveillance.check_wash_trade(trade)
    """

    def __init__(self, alert_callback=None):
        """
        Args:
            alert_callback: Optional callable(level, message, source) for alerts.
        """
        self._alert_callback = alert_callback
        self._report_history: list[SurveillanceReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_wash_trade(self, trades: list[dict] | dict) -> list[SurveillanceFlag]:
        """Check for wash trade patterns in a list of trades.

        A wash trade is a buy and sell of the same symbol within a very short
        time window (default 60s), resulting in no meaningful change in position.

        Also accepts a single trade dict for backwards compatibility (returns
        a list with zero or one flag).

        Args:
            trades: List of trade dicts, or a single trade dict.

        Returns:
            List of SurveillanceFlag objects for any wash trades detected.
        """
        try:
            if isinstance(trades, dict):
                # Single-trade legacy path
                is_wash = self._check_wash_single(trades)
                if is_wash:
                    return [SurveillanceFlag(
                        flag_type="wash_trade",
                        severity="high",
                        symbol=trades.get("symbol", ""),
                        timestamp=str(trades.get("entry_time", "")),
                        description="Single trade flagged as wash trade",
                        trades_involved=[trades],
                    )]
                return []
            return self._detect_wash_trades(trades)
        except Exception as e:
            logger.error(f"SelfSurveillance.check_wash_trade failed: {e}")
            return []

    def check_marking_close(self, trades: list[dict]) -> list[SurveillanceFlag]:
        """Check for marking-the-close patterns.

        Flags symbols where a disproportionate share of trading activity
        occurs in the last 5 minutes before market close.

        Args:
            trades: List of trade dicts for the day.

        Returns:
            List of SurveillanceFlag objects for any flagged symbols.
        """
        try:
            return self._detect_marking_close(trades)
        except Exception as e:
            logger.error(f"SelfSurveillance.check_marking_close failed: {e}")
            return []

    def check_layering(self, orders: list[dict]) -> list[SurveillanceFlag]:
        """Check for layering patterns in order history.

        Layering is placing multiple orders at different prices on the same
        side, then cancelling them before fill.

        Args:
            orders: List of order dicts with keys: symbol, side, price,
                    order_type, status, timestamp.

        Returns:
            List of SurveillanceFlag objects for any layering detected.
        """
        try:
            return self._detect_layering_from_orders(orders)
        except Exception as e:
            logger.error(f"SelfSurveillance.check_layering failed: {e}")
            return []

    def run_daily_check(self, trades: list[dict],
                        orders: list[dict] | None = None) -> SurveillanceReport:
        """Run all surveillance checks on a day's trades.

        Args:
            trades: List of trade dicts with keys:
                symbol, side, qty, entry_time, exit_time, entry_price, exit_price,
                strategy, order_id (optional).
            orders: Optional list of order dicts for spoofing/layering checks.
                    If None, attempts to load from database.

        Returns:
            SurveillanceReport with any flags found.

        Never raises.
        """
        try:
            return self._run_daily_inner(trades, orders)
        except Exception as e:
            logger.error(f"SelfSurveillance.run_daily_check failed: {e}")
            report = SurveillanceReport(
                date=datetime.now(config.ET).strftime("%Y-%m-%d"),
                errors=[str(e)],
                clean=False,
            )
            return report

    @property
    def report_history(self) -> list[SurveillanceReport]:
        return list(self._report_history)

    # ------------------------------------------------------------------
    # Internal — daily check
    # ------------------------------------------------------------------

    def _run_daily_inner(self, trades: list[dict],
                         orders: list[dict] | None = None) -> SurveillanceReport:
        """Core daily surveillance logic."""
        today = datetime.now(config.ET).strftime("%Y-%m-%d")
        report = SurveillanceReport(date=today, total_trades_checked=len(trades))

        if not trades:
            self._report_history.append(report)
            return report

        # 1. Wash trading
        wash_flags = self._detect_wash_trades(trades)
        report.flags.extend(wash_flags)
        report.wash_trade_count = len(wash_flags)

        # 2. Marking the close
        close_flags = self._detect_marking_close(trades)
        report.flags.extend(close_flags)
        report.marking_close_count = len(close_flags)

        # 3. Spoofing (requires order history)
        spoof_flags = self._detect_spoofing(trades)
        report.flags.extend(spoof_flags)
        report.spoofing_count = len(spoof_flags)

        # 4. Layering (use explicit orders if provided, else DB)
        if orders is not None:
            layer_flags = self._detect_layering_from_orders(orders)
        else:
            layer_flags = self._detect_layering(trades)
        report.flags.extend(layer_flags)
        report.layering_count = len(layer_flags)

        report.clean = not report.has_flags

        # Store history (keep last 90 days)
        self._report_history.append(report)
        if len(self._report_history) > 90:
            self._report_history = self._report_history[-90:]

        # Alert on high severity
        if report.high_severity_count > 0:
            self._send_alert(report)

        # Log summary
        if report.clean:
            logger.info(f"SURVEILLANCE: Daily check clean — {len(trades)} trades checked")
        else:
            logger.warning(
                f"SURVEILLANCE: {len(report.flags)} flags found — "
                f"wash={report.wash_trade_count}, close={report.marking_close_count}, "
                f"spoof={report.spoofing_count}, layer={report.layering_count}"
            )

        return report

    # ------------------------------------------------------------------
    # Internal — wash trade detection
    # ------------------------------------------------------------------

    def _detect_wash_trades(self, trades: list[dict]) -> list[SurveillanceFlag]:
        """Detect potential wash trades: same symbol buy+sell within seconds."""
        flags = []

        # Group trades by symbol
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            by_symbol[t.get("symbol", "")].append(t)

        for symbol, symbol_trades in by_symbol.items():
            if len(symbol_trades) < 2:
                continue

            # Sort by time
            sorted_trades = sorted(
                symbol_trades,
                key=lambda t: str(t.get("entry_time", "") or t.get("exit_time", ""))
            )

            for i in range(len(sorted_trades) - 1):
                t1 = sorted_trades[i]
                t2 = sorted_trades[i + 1]

                if self._is_wash_pair(t1, t2):
                    severity = "high" if abs(t1.get("pnl", 0) or 0) < 1.0 else "medium"
                    flags.append(SurveillanceFlag(
                        flag_type="wash_trade",
                        severity=severity,
                        symbol=symbol,
                        timestamp=str(t1.get("entry_time", "")),
                        description=(
                            f"Potential wash trade: {t1.get('side', '?')} then "
                            f"{t2.get('side', '?')} within {WASH_TRADE_WINDOW_SEC}s"
                        ),
                        trades_involved=[t1, t2],
                    ))

        return flags

    def _is_wash_pair(self, t1: dict, t2: dict) -> bool:
        """Check if two trades form a wash pair."""
        # Must be opposite sides
        side1 = (t1.get("side", "") or "").lower()
        side2 = (t2.get("side", "") or "").lower()
        if side1 == side2:
            return False

        # Must be within time window
        time1 = self._parse_time(t1.get("exit_time") or t1.get("entry_time"))
        time2 = self._parse_time(t2.get("entry_time") or t2.get("exit_time"))
        if time1 is None or time2 is None:
            return False

        delta = abs((time2 - time1).total_seconds())
        return delta <= WASH_TRADE_WINDOW_SEC

    def _check_wash_single(self, trade: dict) -> bool:
        """Check a single trade against recent trades for wash pattern."""
        try:
            import database
            conn = database._get_conn()

            symbol = trade.get("symbol", "")
            trade_time = trade.get("entry_time") or trade.get("exit_time", "")

            # Look for opposite-side trades on same symbol within window
            rows = conn.execute(
                "SELECT side, entry_time, exit_time FROM trades "
                "WHERE symbol = ? ORDER BY exit_time DESC LIMIT 5",
                (symbol,),
            ).fetchall()

            trade_side = (trade.get("side", "") or "").lower()
            for row in rows:
                row_side = (row[0] or "").lower()
                if row_side == trade_side:
                    continue

                row_time = self._parse_time(row[2] or row[1])
                trade_dt = self._parse_time(trade_time)
                if row_time and trade_dt:
                    delta = abs((trade_dt - row_time).total_seconds())
                    if delta <= WASH_TRADE_WINDOW_SEC:
                        return True

            return False
        except Exception as e:
            logger.debug(f"Wash trade check error: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal — marking the close
    # ------------------------------------------------------------------

    def _detect_marking_close(self, trades: list[dict]) -> list[SurveillanceFlag]:
        """Detect concentrated trading near market close."""
        flags = []

        close_time = config.MARKET_CLOSE  # 16:00 ET
        cutoff_time = time(
            close_time.hour,
            close_time.minute - MARKING_CLOSE_MINUTES
        )

        # Group by symbol
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            by_symbol[t.get("symbol", "")].append(t)

        for symbol, symbol_trades in by_symbol.items():
            total_volume = sum(abs(t.get("qty", 0) or 0) for t in symbol_trades)
            if total_volume == 0:
                continue

            # Volume in last N minutes
            close_volume = 0
            close_trades = []
            for t in symbol_trades:
                trade_time = self._extract_time_of_day(
                    t.get("entry_time") or t.get("exit_time")
                )
                if trade_time and trade_time >= cutoff_time:
                    close_volume += abs(t.get("qty", 0) or 0)
                    close_trades.append(t)

            concentration = close_volume / total_volume if total_volume > 0 else 0
            if concentration > MARKING_CLOSE_CONCENTRATION_PCT and len(close_trades) >= 2:
                flags.append(SurveillanceFlag(
                    flag_type="marking_close",
                    severity="high" if concentration > 0.5 else "medium",
                    symbol=symbol,
                    timestamp=str(close_trades[0].get("entry_time", "")),
                    description=(
                        f"{concentration:.0%} of daily volume ({close_volume}/{total_volume} shares) "
                        f"in last {MARKING_CLOSE_MINUTES} minutes before close"
                    ),
                    trades_involved=close_trades,
                ))

        return flags

    def _check_marking_close_inner(self, trades: list[dict]) -> list[str]:
        """Return list of symbols flagged for marking-the-close."""
        flags = self._detect_marking_close(trades)
        return [f.symbol for f in flags]

    # ------------------------------------------------------------------
    # Internal — spoofing detection
    # ------------------------------------------------------------------

    def _detect_spoofing(self, trades: list[dict]) -> list[SurveillanceFlag]:
        """Detect rapid order placement and cancellation (spoofing).

        Requires order cancellation data. Falls back gracefully if unavailable.
        """
        flags = []
        try:
            import database
            conn = database._get_conn()

            # Check if we have order tracking
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "order_history" not in tables:
                return flags

            # Look for symbols with many rapid cancellations
            today = datetime.now(config.ET).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT symbol, COUNT(*) as cancel_count "
                "FROM order_history "
                "WHERE status = 'cancelled' AND date(timestamp) = ? "
                "GROUP BY symbol "
                "HAVING cancel_count >= ?",
                (today, SPOOFING_MIN_CANCELS),
            ).fetchall()

            for row in rows:
                symbol, count = row[0], row[1]
                # Check if cancellations were rapid (within window)
                cancel_rows = conn.execute(
                    "SELECT timestamp FROM order_history "
                    "WHERE symbol = ? AND status = 'cancelled' AND date(timestamp) = ? "
                    "ORDER BY timestamp",
                    (symbol, today),
                ).fetchall()

                rapid_count = 0
                timestamps = [self._parse_time(r[0]) for r in cancel_rows]
                timestamps = [t for t in timestamps if t is not None]

                for i in range(1, len(timestamps)):
                    if (timestamps[i] - timestamps[i - 1]).total_seconds() <= SPOOFING_CANCEL_WINDOW_SEC:
                        rapid_count += 1

                if rapid_count >= SPOOFING_MIN_CANCELS - 1:
                    flags.append(SurveillanceFlag(
                        flag_type="spoofing",
                        severity="high",
                        symbol=symbol,
                        timestamp=str(timestamps[0]) if timestamps else today,
                        description=(
                            f"{count} orders cancelled, {rapid_count} rapid cancellations "
                            f"within {SPOOFING_CANCEL_WINDOW_SEC}s window"
                        ),
                    ))

        except Exception as e:
            logger.debug(f"Spoofing detection error: {e}")

        return flags

    # ------------------------------------------------------------------
    # Internal — layering detection
    # ------------------------------------------------------------------

    def _detect_layering(self, trades: list[dict]) -> list[SurveillanceFlag]:
        """Detect layering: multiple same-side limit orders at different prices."""
        flags = []
        try:
            import database
            conn = database._get_conn()

            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "order_history" not in tables:
                return flags

            today = datetime.now(config.ET).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT symbol, side, COUNT(DISTINCT price) as price_levels, "
                "COUNT(*) as order_count "
                "FROM order_history "
                "WHERE order_type = 'limit' AND date(timestamp) = ? "
                "AND status IN ('cancelled', 'expired') "
                "GROUP BY symbol, side "
                "HAVING price_levels >= ?",
                (today, LAYERING_MIN_ORDERS),
            ).fetchall()

            for row in rows:
                symbol, side, levels, count = row[0], row[1], row[2], row[3]
                flags.append(SurveillanceFlag(
                    flag_type="layering",
                    severity="high" if levels >= 6 else "medium",
                    symbol=symbol,
                    timestamp=today,
                    description=(
                        f"{count} {side} limit orders at {levels} different price levels "
                        f"(all cancelled/expired)"
                    ),
                ))

        except Exception as e:
            logger.debug(f"Layering detection error: {e}")

        return flags

    # ------------------------------------------------------------------
    # Internal — layering from explicit order list
    # ------------------------------------------------------------------

    def _detect_layering_from_orders(self, orders: list[dict]) -> list[SurveillanceFlag]:
        """Detect layering from an explicit list of order dicts.

        Args:
            orders: List of dicts with keys: symbol, side, price, order_type,
                    status, timestamp.
        """
        flags = []

        # Filter to cancelled/expired limit orders
        cancelled = [
            o for o in orders
            if (o.get("order_type", "") or "").lower() == "limit"
            and (o.get("status", "") or "").lower() in ("cancelled", "expired")
        ]

        # Group by (symbol, side)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for o in cancelled:
            key = (o.get("symbol", ""), (o.get("side", "") or "").lower())
            groups[key].append(o)

        for (symbol, side), group_orders in groups.items():
            # Count distinct price levels
            prices = set()
            for o in group_orders:
                price = o.get("price")
                if price is not None:
                    prices.add(float(price))

            if len(prices) >= LAYERING_MIN_ORDERS:
                flags.append(SurveillanceFlag(
                    flag_type="layering",
                    severity="high" if len(prices) >= 6 else "medium",
                    symbol=symbol,
                    timestamp=str(group_orders[0].get("timestamp", "")),
                    description=(
                        f"{len(group_orders)} {side} limit orders at {len(prices)} "
                        f"different price levels (all cancelled/expired)"
                    ),
                    trades_involved=group_orders,
                ))

        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time(time_str) -> Optional[datetime]:
        """Parse a datetime string to a datetime object."""
        if time_str is None:
            return None
        if isinstance(time_str, datetime):
            return time_str
        try:
            # ISO format
            return datetime.fromisoformat(str(time_str))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_time_of_day(time_str) -> Optional[time]:
        """Extract just the time-of-day component."""
        if time_str is None:
            return None
        try:
            dt = datetime.fromisoformat(str(time_str))
            return dt.time()
        except (ValueError, TypeError):
            return None

    def _send_alert(self, report: SurveillanceReport):
        """Send alert for high-severity surveillance flags."""
        if not self._alert_callback:
            return

        try:
            high_flags = [f for f in report.flags if f.severity == "high"]
            msg = (
                f"Surveillance alert: {len(high_flags)} high-severity flags detected\n"
                + "\n".join(f"  - {f.flag_type}: {f.symbol} — {f.description}"
                           for f in high_flags[:5])
            )
            self._alert_callback("CRITICAL", msg, "surveillance")
        except Exception as e:
            logger.error(f"Surveillance alert failed: {e}")
