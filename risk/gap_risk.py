"""RISK-006: Overnight gap risk management.

Manages exposure to overnight price gaps by:
1. Tracking historical gap distributions per symbol
2. Computing gap risk (expected loss from overnight gap)
3. Enforcing max 30% overnight portfolio exposure
4. Auto-closing non-swing day-trade positions by 3:55 PM ET

Gap = (open_price - prev_close) / prev_close

Historical gap data is used to estimate the tail risk of holding
positions overnight. High-gap-risk symbols get reduced overnight exposure.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# Limits
MAX_OVERNIGHT_EXPOSURE_PCT = 0.30   # Max 30% of portfolio held overnight
CLOSE_BY_TIME = time(15, 55)        # Close day-trade positions by 3:55 PM ET
GAP_LOOKBACK_DAYS = 252             # 1 year of historical gaps

# Risk thresholds
HIGH_GAP_RISK_THRESHOLD = 0.02      # Symbols with avg |gap| > 2% are high-risk
MAX_SINGLE_OVERNIGHT_PCT = 0.08     # Max 8% portfolio in any single overnight hold


@dataclass
class GapStats:
    """Historical gap statistics for a symbol."""
    symbol: str
    mean_gap: float = 0.0          # Mean gap (signed)
    mean_abs_gap: float = 0.0      # Mean absolute gap
    std_gap: float = 0.0           # Std dev of gaps
    max_negative_gap: float = 0.0  # Worst historical gap (most negative)
    max_positive_gap: float = 0.0  # Best historical gap
    gap_95_pct: float = 0.0        # 95th percentile of |gap| (tail risk)
    sample_count: int = 0
    last_updated: datetime | None = None


@dataclass
class GapRiskResult:
    """Gap risk assessment for a single position."""
    symbol: str
    position_size_dollars: float
    gap_risk_dollars: float        # Expected gap loss (95th percentile)
    gap_risk_pct: float            # As fraction of portfolio
    is_high_risk: bool = False
    recommended_action: str = ""   # "hold", "reduce", "close"


class GapRiskManager:
    """Manages overnight gap risk across the portfolio.

    Usage:
        gap_mgr = GapRiskManager()
        gap_mgr.update_gap_stats("TSLA", gap_history)
        risk = gap_mgr.compute_gap_risk("TSLA", 10000.0, portfolio_equity=100000.0)

        # At 3:50 PM, check which positions to close
        to_close = gap_mgr.get_positions_to_close(open_trades, portfolio_equity)
    """

    def __init__(
        self,
        max_overnight_pct: float = MAX_OVERNIGHT_EXPOSURE_PCT,
        close_by: time = CLOSE_BY_TIME,
        max_single_pct: float = MAX_SINGLE_OVERNIGHT_PCT,
    ):
        self._max_overnight_pct = max_overnight_pct
        self._close_by = close_by
        self._max_single_pct = max_single_pct

        # symbol -> GapStats
        self._gap_stats: dict[str, GapStats] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Gap statistics
    # ------------------------------------------------------------------

    def update_gap_stats(self, symbol: str, gap_returns: list[float] | np.ndarray):
        """Update gap statistics for a symbol from historical gap data.

        Args:
            symbol: Ticker symbol
            gap_returns: List of historical gap returns (open/prev_close - 1)
        """
        if len(gap_returns) < 5:
            logger.debug(f"Gap stats {symbol}: insufficient data ({len(gap_returns)} gaps)")
            return

        gaps = np.array(gap_returns, dtype=float)
        abs_gaps = np.abs(gaps)

        stats = GapStats(
            symbol=symbol,
            mean_gap=float(np.mean(gaps)),
            mean_abs_gap=float(np.mean(abs_gaps)),
            std_gap=float(np.std(gaps)),
            max_negative_gap=float(np.min(gaps)),
            max_positive_gap=float(np.max(gaps)),
            gap_95_pct=float(np.percentile(abs_gaps, 95)),
            sample_count=len(gaps),
            last_updated=datetime.now(config.ET),
        )

        with self._lock:
            self._gap_stats[symbol] = stats

        logger.debug(
            f"Gap stats {symbol}: mean={stats.mean_gap:.4f} "
            f"std={stats.std_gap:.4f} max_neg={stats.max_negative_gap:.4f} "
            f"95pct={stats.gap_95_pct:.4f} (n={stats.sample_count})"
        )

    def update_gap_stats_from_ohlc(
        self, symbol: str, opens: np.ndarray, closes: np.ndarray
    ):
        """Compute and update gap stats from OHLC arrays.

        Args:
            symbol: Ticker symbol
            opens: Array of open prices (index-aligned with closes)
            closes: Array of close prices (previous day close at index i
                    corresponds to open at index i+1)
        """
        if len(opens) < 10 or len(closes) < 10:
            return
        # Gap = open[i] / close[i-1] - 1
        prev_close = closes[:-1]
        next_open = opens[1:]
        mask = prev_close > 0
        gap_returns = (next_open[mask] / prev_close[mask]) - 1.0
        self.update_gap_stats(symbol, gap_returns)

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    def compute_gap_risk(
        self,
        symbol: str,
        position_size: float,
        portfolio_equity: float = 100_000.0,
    ) -> float:
        """Compute the dollar gap risk for a position.

        Uses the 95th percentile of historical |gap| as the expected
        worst-case gap move. Returns the estimated loss in dollars.

        Args:
            symbol: Ticker symbol
            position_size: Dollar notional of the position (positive)
            portfolio_equity: Total portfolio equity

        Returns:
            Expected gap loss in dollars (positive = loss)
        """
        with self._lock:
            stats = self._gap_stats.get(symbol)

        if stats is None or stats.sample_count < 5:
            # Default assumption: 2% gap risk for unknown symbols
            default_gap = 0.02
            return abs(position_size) * default_gap

        # Use 95th percentile of absolute gaps
        gap_risk = abs(position_size) * stats.gap_95_pct
        return gap_risk

    def assess_position(
        self,
        symbol: str,
        position_size: float,
        portfolio_equity: float,
    ) -> GapRiskResult:
        """Full gap risk assessment for a position.

        Returns recommendation: hold, reduce, or close before overnight.
        """
        gap_risk = self.compute_gap_risk(symbol, position_size, portfolio_equity)
        gap_risk_pct = gap_risk / portfolio_equity if portfolio_equity > 0 else 0.0

        with self._lock:
            stats = self._gap_stats.get(symbol)

        is_high_risk = False
        if stats and stats.mean_abs_gap > HIGH_GAP_RISK_THRESHOLD:
            is_high_risk = True

        # Determine recommendation
        position_pct = abs(position_size) / portfolio_equity if portfolio_equity > 0 else 0.0

        if is_high_risk and position_pct > self._max_single_pct:
            action = "reduce"
        elif is_high_risk and gap_risk_pct > 0.02:
            action = "close"
        elif position_pct > self._max_single_pct:
            action = "reduce"
        else:
            action = "hold"

        return GapRiskResult(
            symbol=symbol,
            position_size_dollars=abs(position_size),
            gap_risk_dollars=gap_risk,
            gap_risk_pct=gap_risk_pct,
            is_high_risk=is_high_risk,
            recommended_action=action,
        )

    # ------------------------------------------------------------------
    # Simple convenience methods
    # ------------------------------------------------------------------

    def should_close_before_eod(
        self,
        trade: dict,
        current_time: datetime | None = None,
    ) -> bool:
        """Check whether a non-swing trade should be closed before end of day.

        Closes any trade that is not marked as a swing/overnight hold after
        3:55 PM ET.

        Args:
            trade: Trade dict or object with at least 'hold_type' attribute.
                Trades with hold_type != 'swing' are candidates for closure.
            current_time: Current time (defaults to now in ET).

        Returns:
            True if the trade should be closed before EOD.
        """
        if current_time is None:
            current_time = datetime.now(config.ET)

        if current_time.time() < self._close_by:
            return False

        hold_type = (
            getattr(trade, "hold_type", None)
            or (trade.get("hold_type", "day") if isinstance(trade, dict) else "day")
        )
        return hold_type != "swing"

    def get_overnight_sizing_multiplier(self, symbol: str) -> float:
        """Get a sizing multiplier for overnight holds based on gap risk.

        High-gap-risk symbols receive reduced overnight sizing.

        Args:
            symbol: Ticker symbol.

        Returns:
            Multiplier in [0.25, 1.0]. 1.0 = normal sizing, lower = reduce.
        """
        with self._lock:
            stats = self._gap_stats.get(symbol)

        if stats is None or stats.sample_count < 5:
            # Unknown symbol: conservative 50% sizing
            return 0.5

        # Scale down based on average absolute gap relative to threshold
        # At threshold (2%): 0.5x, at 2x threshold (4%): 0.25x
        if stats.mean_abs_gap <= HIGH_GAP_RISK_THRESHOLD * 0.5:
            return 1.0
        elif stats.mean_abs_gap >= HIGH_GAP_RISK_THRESHOLD * 2.0:
            return 0.25
        else:
            # Linear interpolation from 1.0 at half-threshold to 0.25 at 2x threshold
            ratio = (stats.mean_abs_gap - HIGH_GAP_RISK_THRESHOLD * 0.5) / (
                HIGH_GAP_RISK_THRESHOLD * 1.5
            )
            return max(0.25, 1.0 - 0.75 * ratio)

    def get_max_overnight_exposure(self, equity: float) -> float:
        """Get the maximum dollar amount allowed for overnight exposure.

        Args:
            equity: Total portfolio equity.

        Returns:
            Maximum overnight exposure in dollars (30% of portfolio by default).
        """
        return equity * self._max_overnight_pct

    # ------------------------------------------------------------------
    # Portfolio-level overnight management
    # ------------------------------------------------------------------

    def get_positions_to_close(
        self,
        open_trades: dict[str, Any],
        portfolio_equity: float,
        now: datetime | None = None,
    ) -> list[str]:
        """Determine which positions should be closed before overnight.

        Closes all non-swing day-trade positions. If total overnight
        exposure would exceed the limit, also identifies swing positions
        to reduce (highest gap-risk first).

        Args:
            open_trades: symbol -> TradeRecord (or dict with hold_type, side, qty, entry_price)
            portfolio_equity: Total equity
            now: Current time (defaults to now)

        Returns:
            List of symbols that should be closed
        """
        if now is None:
            now = datetime.now(config.ET)

        current_time = now.time()
        if current_time < self._close_by:
            return []  # Not yet time

        to_close = []
        overnight_candidates = []

        for symbol, trade in open_trades.items():
            hold_type = getattr(trade, "hold_type", None) or (
                trade.get("hold_type", "day") if isinstance(trade, dict) else "day"
            )
            strategy = getattr(trade, "strategy", None) or (
                trade.get("strategy", "") if isinstance(trade, dict) else ""
            )

            # Close all day-trade positions
            if hold_type == "day":
                # Check if strategy is eligible for overnight hold
                if strategy not in config.OVERNIGHT_ELIGIBLE_STRATEGIES:
                    to_close.append(symbol)
                    continue

            # Track swing/overnight positions for exposure check
            qty = getattr(trade, "qty", 0) or (
                trade.get("qty", 0) if isinstance(trade, dict) else 0
            )
            price = getattr(trade, "entry_price", 0) or (
                trade.get("entry_price", 0) if isinstance(trade, dict) else 0
            )
            notional = qty * price
            gap_risk = self.compute_gap_risk(symbol, notional, portfolio_equity)

            overnight_candidates.append({
                "symbol": symbol,
                "notional": notional,
                "gap_risk": gap_risk,
                "hold_type": hold_type,
            })

        # Check total overnight exposure
        if portfolio_equity > 0:
            remaining = [c for c in overnight_candidates if c["symbol"] not in to_close]
            total_overnight = sum(c["notional"] for c in remaining)
            max_overnight = portfolio_equity * self._max_overnight_pct

            if total_overnight > max_overnight:
                # Sort by gap risk (highest first) and close until under limit
                remaining.sort(key=lambda c: c["gap_risk"], reverse=True)
                for candidate in remaining:
                    if total_overnight <= max_overnight:
                        break
                    to_close.append(candidate["symbol"])
                    total_overnight -= candidate["notional"]
                    logger.info(
                        f"Gap risk: closing {candidate['symbol']} to reduce "
                        f"overnight exposure (gap_risk=${candidate['gap_risk']:.0f})"
                    )

        if to_close:
            logger.info(
                f"Gap risk manager: {len(to_close)} positions to close before overnight: "
                f"{', '.join(to_close)}"
            )

        return to_close

    def check_overnight_capacity(
        self,
        current_overnight_notional: float,
        new_position_notional: float,
        portfolio_equity: float,
    ) -> tuple[bool, str]:
        """Check if adding a new overnight position would exceed the limit.

        Args:
            current_overnight_notional: Total current overnight exposure ($)
            new_position_notional: New position notional ($)
            portfolio_equity: Total portfolio equity

        Returns:
            (allowed, reason)
        """
        if portfolio_equity <= 0:
            return False, "No portfolio equity"

        total = current_overnight_notional + new_position_notional
        limit = portfolio_equity * self._max_overnight_pct

        if total > limit:
            return False, (
                f"Overnight exposure would be {total / portfolio_equity:.1%} "
                f"(limit: {self._max_overnight_pct:.0%})"
            )

        return True, ""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "tracked_symbols": len(self._gap_stats),
                "max_overnight_pct": f"{self._max_overnight_pct:.0%}",
                "close_by": self._close_by.strftime("%H:%M"),
                "high_risk_symbols": [
                    s for s, stats in self._gap_stats.items()
                    if stats.mean_abs_gap > HIGH_GAP_RISK_THRESHOLD
                ],
            }
