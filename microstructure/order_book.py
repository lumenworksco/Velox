"""MICRO-002: Order Book Imbalance Analysis.

Tracks top-of-book bid/ask imbalance to gauge short-term directional
pressure. Positive imbalance (more bids) suggests buying pressure;
negative imbalance (more asks) suggests selling pressure.

Usage:
    analyzer = OrderBookAnalyzer(rolling_window=20)
    analyzer.update_quote(bid=150.10, ask=150.15, bid_size=500, ask_size=200)
    imbalance = analyzer.get_imbalance()       # -1 to +1
    smoothed = analyzer.get_rolling_imbalance() # smoothed version

Best used for entry timing on momentum/ORB strategies: enter longs when
imbalance is positive and trending up.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _QuoteSnapshot:
    """Single top-of-book quote record."""
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    imbalance: float
    timestamp: datetime


class OrderBookAnalyzer:
    """Top-of-book order imbalance tracker.

    Computes bid/ask size imbalance at the best bid and ask,
    with optional rolling smoothing for noise reduction.

    Args:
        rolling_window: Number of quotes to average for smoothed imbalance.
        max_history: Maximum quote snapshots to retain in memory.
    """

    def __init__(
        self,
        rolling_window: int = 20,
        max_history: int = 1000,
    ) -> None:
        if rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {rolling_window}")

        self._rolling_window = rolling_window
        self._history: deque[_QuoteSnapshot] = deque(maxlen=max_history)
        self._per_symbol: dict[str, deque[_QuoteSnapshot]] = {}
        self._current_imbalance: float = 0.0

    # ------------------------------------------------------------------ #
    #  Quote ingestion
    # ------------------------------------------------------------------ #

    def update_quote(
        self,
        bid: float,
        ask: float,
        bid_size: int,
        ask_size: int,
        symbol: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """Record a new top-of-book quote and compute imbalance.

        Args:
            bid: Best bid price.
            ask: Best ask price.
            bid_size: Size at best bid.
            ask_size: Size at best ask.
            symbol: Optional symbol for per-symbol tracking.
            timestamp: Quote timestamp. Defaults to now (UTC).

        Returns:
            Raw imbalance value in [-1.0, 1.0].
        """
        if bid <= 0 or ask <= 0:
            logger.warning("Invalid quote: bid=%.4f ask=%.4f", bid, ask)
            return self._current_imbalance

        if bid_size < 0 or ask_size < 0:
            logger.warning("Negative size: bid_size=%d ask_size=%d", bid_size, ask_size)
            return self._current_imbalance

        total_size = bid_size + ask_size
        if total_size == 0:
            imbalance = 0.0
        else:
            imbalance = (bid_size - ask_size) / total_size

        ts = timestamp or datetime.now(timezone.utc)
        snapshot = _QuoteSnapshot(
            bid=bid, ask=ask,
            bid_size=bid_size, ask_size=ask_size,
            imbalance=imbalance, timestamp=ts,
        )

        self._history.append(snapshot)
        self._current_imbalance = imbalance

        # Per-symbol tracking
        if symbol is not None:
            if symbol not in self._per_symbol:
                self._per_symbol[symbol] = deque(maxlen=self._history.maxlen)
            self._per_symbol[symbol].append(snapshot)

        return imbalance

    # ------------------------------------------------------------------ #
    #  Imbalance queries
    # ------------------------------------------------------------------ #

    def get_imbalance(self, symbol: Optional[str] = None) -> float:
        """Get the most recent raw imbalance.

        Args:
            symbol: If provided, returns imbalance for that symbol.

        Returns:
            Float in [-1.0, 1.0]. Positive = more bids (buying pressure).
        """
        history = self._get_history(symbol)
        if not history:
            return 0.0
        return history[-1].imbalance

    def get_rolling_imbalance(self, symbol: Optional[str] = None) -> float:
        """Get the rolling average imbalance for smoothing.

        Args:
            symbol: If provided, computes rolling for that symbol.

        Returns:
            Float in [-1.0, 1.0]. Smoothed over rolling_window quotes.
        """
        history = self._get_history(symbol)
        if not history:
            return 0.0

        window = list(history)[-self._rolling_window:]
        return float(np.mean([s.imbalance for s in window]))

    def get_imbalance_trend(self, symbol: Optional[str] = None) -> float:
        """Get the trend direction of imbalance (slope).

        Fits a simple linear regression over the rolling window to detect
        whether imbalance is increasing (positive) or decreasing (negative).

        Returns:
            Slope of imbalance trend. Positive = strengthening buy pressure.
        """
        history = self._get_history(symbol)
        if len(history) < 3:
            return 0.0

        window = list(history)[-self._rolling_window:]
        if len(window) < 3:
            return 0.0

        imbalances = np.array([s.imbalance for s in window])
        x = np.arange(len(imbalances))
        coeffs = np.polyfit(x, imbalances, 1)
        return float(coeffs[0])

    def get_spread(self, symbol: Optional[str] = None) -> float:
        """Get the most recent quoted spread as a fraction of midpoint.

        Returns:
            Spread / midpoint. E.g. 0.001 = 10 bps.
        """
        history = self._get_history(symbol)
        if not history:
            return 0.0

        latest = history[-1]
        midpoint = (latest.bid + latest.ask) / 2
        if midpoint <= 0:
            return 0.0
        return (latest.ask - latest.bid) / midpoint

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _get_history(self, symbol: Optional[str]) -> deque:
        """Return the correct history deque for a symbol or global."""
        if symbol is not None and symbol in self._per_symbol:
            return self._per_symbol[symbol]
        return self._history

    @property
    def quote_count(self) -> int:
        """Total number of quotes recorded (global)."""
        return len(self._history)

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear history for a symbol or all symbols."""
        if symbol is not None:
            self._per_symbol.pop(symbol, None)
        else:
            self._history.clear()
            self._per_symbol.clear()
            self._current_imbalance = 0.0
        logger.debug("OrderBookAnalyzer reset (symbol=%s)", symbol)

    def __repr__(self) -> str:
        return (
            f"OrderBookAnalyzer(imbalance={self._current_imbalance:+.3f}, "
            f"quotes={len(self._history)}, symbols={len(self._per_symbol)})"
        )
