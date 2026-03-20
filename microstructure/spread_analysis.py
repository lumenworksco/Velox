"""MICRO-004: Effective Spread and Adverse Selection Analysis.

Computes execution quality metrics:
- Effective spread: actual cost of crossing the spread
- Realized spread: market-maker profit after price impact
- Price impact / adverse selection: how much price moves against the trade

Usage:
    analyzer = SpreadAnalyzer(impact_horizon_sec=300)
    analyzer.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20,
                          timestamp=datetime.now(timezone.utc))
    # ... 5 minutes later, call update_midpoint to record future midpoint
    analyzer.update_midpoint("AAPL", future_midpoint=150.25,
                             original_timestamp=original_ts)

    eff_spread = analyzer.get_effective_spread("AAPL")
    adverse = analyzer.get_adverse_selection("AAPL")
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _TradeRecord:
    """Single trade for spread analysis."""
    symbol: str
    trade_price: float
    bid: float
    ask: float
    midpoint: float
    effective_spread: float
    timestamp: datetime
    side: str                          # inferred: "buy" or "sell"
    future_midpoint: Optional[float] = None
    realized_spread: Optional[float] = None
    price_impact: Optional[float] = None


class SpreadAnalyzer:
    """Effective spread, realized spread, and adverse selection tracker.

    Metrics:
        Effective spread = 2 * |trade_price - midpoint|
        Price impact = sign * (future_midpoint - midpoint)
        Realized spread = effective_spread - 2 * price_impact

    Where sign is +1 for buys, -1 for sells, and future_midpoint is the
    midpoint some configurable horizon (default 5 minutes) after the trade.

    Args:
        impact_horizon_sec: Seconds after trade to measure price impact
            (default 300 = 5 minutes).
        max_history: Maximum trade records per symbol.
    """

    def __init__(
        self,
        impact_horizon_sec: int = 300,
        max_history: int = 5000,
    ) -> None:
        self._impact_horizon = timedelta(seconds=impact_horizon_sec)
        self._max_history = max_history
        self._trades: dict[str, deque[_TradeRecord]] = {}
        self._pending_impact: dict[str, list[_TradeRecord]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Trade recording
    # ------------------------------------------------------------------ #

    def record_trade(
        self,
        symbol: str,
        trade_price: float,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
        side: Optional[str] = None,
    ) -> float:
        """Record a trade execution for spread analysis.

        If side is not provided, it is inferred: trade at or above midpoint
        is classified as a buy; below midpoint is a sell.

        Args:
            symbol: Ticker symbol.
            trade_price: Execution price.
            bid: Best bid at time of trade.
            ask: Best ask at time of trade.
            timestamp: Trade time. Defaults to now (UTC).
            side: "buy" or "sell". Inferred if not provided.

        Returns:
            The effective spread for this trade.
        """
        if bid <= 0 or ask <= 0 or trade_price <= 0:
            logger.warning(
                "Invalid trade data: price=%.4f bid=%.4f ask=%.4f",
                trade_price, bid, ask,
            )
            return 0.0

        midpoint = (bid + ask) / 2
        effective_spread = 2.0 * abs(trade_price - midpoint)

        # Infer side
        if side is None:
            side = "buy" if trade_price >= midpoint else "sell"

        ts = timestamp or datetime.now(timezone.utc)

        record = _TradeRecord(
            symbol=symbol,
            trade_price=trade_price,
            bid=bid,
            ask=ask,
            midpoint=midpoint,
            effective_spread=effective_spread,
            timestamp=ts,
            side=side.lower(),
        )

        with self._lock:
            if symbol not in self._trades:
                self._trades[symbol] = deque(maxlen=self._max_history)
            self._trades[symbol].append(record)

            # Track for future midpoint update
            if symbol not in self._pending_impact:
                self._pending_impact[symbol] = []
            self._pending_impact[symbol].append(record)

        return effective_spread

    def update_midpoint(
        self,
        symbol: str,
        future_midpoint: float,
        current_time: Optional[datetime] = None,
    ) -> int:
        """Update pending trades with the future midpoint for price impact.

        Call this periodically (e.g. every minute) with the current midpoint.
        Trades older than impact_horizon_sec will have their price impact
        and realized spread computed.

        Args:
            symbol: Ticker symbol.
            future_midpoint: Current midpoint price.
            current_time: Current time. Defaults to now (UTC).

        Returns:
            Number of trades updated.
        """
        now = current_time or datetime.now(timezone.utc)

        with self._lock:
            if symbol not in self._pending_impact:
                return 0

            updated = 0
            remaining = []

            for record in self._pending_impact[symbol]:
                elapsed = now - record.timestamp
                if elapsed >= self._impact_horizon:
                    sign = 1.0 if record.side == "buy" else -1.0
                    record.future_midpoint = future_midpoint
                    record.price_impact = sign * (future_midpoint - record.midpoint)
                    record.realized_spread = record.effective_spread - 2.0 * record.price_impact
                    updated += 1
                else:
                    remaining.append(record)

            self._pending_impact[symbol] = remaining

        if updated > 0:
            logger.debug(
                "Updated %d trades for %s with future midpoint %.4f",
                updated, symbol, future_midpoint,
            )

        return updated

    # ------------------------------------------------------------------ #
    #  Spread queries
    # ------------------------------------------------------------------ #

    def get_effective_spread(self, symbol: str) -> float:
        """Average effective spread for a symbol.

        Returns:
            Average effective spread in price units.
            Returns 0.0 if no trades recorded.
        """
        with self._lock:
            if symbol not in self._trades or not self._trades[symbol]:
                return 0.0
            spreads = [t.effective_spread for t in self._trades[symbol]]
        return float(np.mean(spreads))

    def get_effective_spread_bps(self, symbol: str) -> float:
        """Average effective spread in basis points relative to midpoint.

        Returns:
            Spread in basis points (1 bp = 0.01%).
        """
        with self._lock:
            if symbol not in self._trades or not self._trades[symbol]:
                return 0.0
            bps_values = []
            for t in self._trades[symbol]:
                if t.midpoint > 0:
                    bps_values.append((t.effective_spread / t.midpoint) * 10_000)
        return float(np.mean(bps_values)) if bps_values else 0.0

    def get_adverse_selection(self, symbol: str) -> float:
        """Average price impact (adverse selection component) for a symbol.

        Positive value means the trade was adversely selected (price moved
        against the liquidity provider, in favor of the trade initiator).

        Returns:
            Average price impact in price units.
            Returns 0.0 if insufficient data.
        """
        with self._lock:
            if symbol not in self._trades:
                return 0.0
            impacts = [
                t.price_impact for t in self._trades[symbol]
                if t.price_impact is not None
            ]
        return float(np.mean(impacts)) if impacts else 0.0

    def get_realized_spread(self, symbol: str) -> float:
        """Average realized spread (market-maker profit) for a symbol.

        Realized spread = effective spread - 2 * price impact.
        Low or negative realized spread means the market maker is losing
        money, indicating high adverse selection.

        Returns:
            Average realized spread in price units.
        """
        with self._lock:
            if symbol not in self._trades:
                return 0.0
            realized = [
                t.realized_spread for t in self._trades[symbol]
                if t.realized_spread is not None
            ]
        return float(np.mean(realized)) if realized else 0.0

    def get_execution_quality_score(self, symbol: str) -> float:
        """Composite execution quality score.

        Combines effective spread (lower is better) and adverse selection
        (lower is better for us as takers) into a single 0-1 score.

        Returns:
            Float in [0.0, 1.0]. Higher = better execution quality.
        """
        eff_bps = self.get_effective_spread_bps(symbol)
        adverse = self.get_adverse_selection(symbol)

        # Score components (higher is better)
        # Effective spread: 10 bps = 0.5 score, 0 bps = 1.0, 20+ bps = 0.0
        spread_score = max(0.0, 1.0 - eff_bps / 20.0)

        # Adverse selection: positive impact (price moves in our favor) = good
        # Normalize: -0.10 = bad (0.0), 0 = neutral (0.5), +0.10 = good (1.0)
        with self._lock:
            has_trades = symbol in self._trades and bool(self._trades[symbol])
            avg_price = float(np.mean([t.midpoint for t in self._trades[symbol]])) if has_trades else 0.0
        if has_trades:
            if avg_price > 0 and adverse != 0:
                impact_pct = adverse / avg_price
                impact_score = min(1.0, max(0.0, 0.5 + impact_pct * 50))
            else:
                impact_score = 0.5
        else:
            impact_score = 0.5

        return 0.6 * spread_score + 0.4 * impact_score

    # ------------------------------------------------------------------ #
    #  Housekeeping
    # ------------------------------------------------------------------ #

    @property
    def tracked_symbols(self) -> list[str]:
        """List of symbols with recorded trades."""
        with self._lock:
            return list(self._trades.keys())

    def get_trade_count(self, symbol: str) -> int:
        """Number of recorded trades for a symbol."""
        with self._lock:
            if symbol not in self._trades:
                return 0
            return len(self._trades[symbol])

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear data for a symbol or all symbols."""
        with self._lock:
            if symbol is not None:
                self._trades.pop(symbol, None)
                self._pending_impact.pop(symbol, None)
            else:
                self._trades.clear()
                self._pending_impact.clear()
        logger.debug("SpreadAnalyzer reset (symbol=%s)", symbol)

    def __repr__(self) -> str:
        with self._lock:
            total_trades = sum(len(t) for t in self._trades.values())
            n_symbols = len(self._trades)
        return (
            f"SpreadAnalyzer(symbols={n_symbols}, "
            f"total_trades={total_trades})"
        )
