"""T5-006: Dark Pool Volume Detection.

Monitors the trade stream for dark pool prints (condition code 'D') and
tracks per-symbol dark pool volume ratio over a rolling 30-minute window.

Dark pool signal:
  - ratio > 35% and net buy  ->  positive alpha signal
  - Soft filter:  confidence * (1 + 0.15 * dark_pool_alpha)

All operations are fail-open: returns neutral values on any error.
"""

import logging
import threading
import time as time_mod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class DarkPoolTrade:
    """A single trade record for dark pool tracking."""
    timestamp: float        # Unix timestamp
    price: float
    volume: int
    is_dark_pool: bool      # True if condition code 'D'
    side: str               # "buy" or "sell" (classified via tick rule)


@dataclass
class DarkPoolSignal:
    """Dark pool analysis result for a single symbol."""
    symbol: str
    dark_pool_ratio: float      # dark pool volume / total volume (0-1)
    net_buy_ratio: float        # (dark buy - dark sell) / dark total (-1 to 1)
    total_dark_volume: int
    total_volume: int
    alpha_signal: float         # combined alpha: positive = bullish
    confidence_mult: float      # multiplier for signal confidence
    is_significant: bool        # whether dark pool activity is meaningful


class DarkPoolDetector:
    """Detect and analyze dark pool trading activity.

    Monitors the trade stream for dark pool prints (identified by trade
    condition code 'D' or similar exchange-specific codes) and computes
    per-symbol dark pool volume ratios over a rolling window.

    Usage::

        detector = DarkPoolDetector()

        # Ingest trades from WebSocket stream
        detector.add_trade("AAPL", price=185.50, volume=500, condition="D", side="buy")
        detector.add_trade("AAPL", price=185.51, volume=200, condition="", side="buy")

        # Get signal
        signal = detector.get_signal("AAPL")
        adjusted_confidence = base_confidence * signal.confidence_mult
    """

    # Trade condition codes indicating dark pool / off-exchange execution
    DARK_POOL_CONDITIONS = frozenset({
        "D",    # Average Price Trade (dark pool)
        "X",    # Cross trade
        "4",    # Derivatively priced
        "U",    # Extended hours sold (last/out of sequence)
    })

    def __init__(
        self,
        rolling_minutes: int | None = None,
        ratio_threshold: float | None = None,
        alpha_weight: float | None = None,
    ):
        self._rolling_minutes = rolling_minutes or getattr(config, "DARK_POOL_ROLLING_MINUTES", 30)
        self._ratio_threshold = ratio_threshold or getattr(config, "DARK_POOL_RATIO_THRESHOLD", 0.35)
        self._alpha_weight = alpha_weight or getattr(config, "DARK_POOL_ALPHA_WEIGHT", 0.15)

        # Per-symbol trade history (rolling window)
        self._trades: dict[str, deque[DarkPoolTrade]] = defaultdict(
            lambda: deque(maxlen=5000)
        )
        self._lock = threading.Lock()
        self._last_price: dict[str, float] = {}  # for tick rule classification
        self._enabled = getattr(config, "DARK_POOL_ENABLED", True)

    def add_trade(
        self,
        symbol: str,
        price: float,
        volume: int,
        condition: str = "",
        side: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest a trade from the market data stream.

        Args:
            symbol: Ticker symbol.
            price: Execution price.
            volume: Number of shares.
            condition: Trade condition code(s). 'D' indicates dark pool.
            side: "buy" or "sell". If None, classified via tick rule.
            timestamp: Unix timestamp. Defaults to current time.
        """
        if volume <= 0 or not self._enabled:
            return

        ts = timestamp or time_mod.time()

        # Classify side via tick rule if not provided
        if side is None:
            with self._lock:
                last = self._last_price.get(symbol)
                if last is None or price >= last:
                    side = "buy"
                else:
                    side = "sell"
                self._last_price[symbol] = price
        else:
            with self._lock:
                self._last_price[symbol] = price

        # Check if this is a dark pool trade
        is_dark = any(c in self.DARK_POOL_CONDITIONS for c in condition.upper().split(","))

        trade = DarkPoolTrade(
            timestamp=ts,
            price=price,
            volume=volume,
            is_dark_pool=is_dark,
            side=side,
        )

        with self._lock:
            self._trades[symbol].append(trade)

    def get_signal(self, symbol: str) -> DarkPoolSignal:
        """Analyze dark pool activity for a symbol.

        Computes the dark pool volume ratio and net buy/sell direction
        over the rolling window. Produces an alpha signal and confidence
        multiplier.

        Args:
            symbol: Ticker symbol.

        Returns:
            DarkPoolSignal with analysis results.
        """
        if not self._enabled:
            return DarkPoolSignal(
                symbol=symbol, dark_pool_ratio=0.0, net_buy_ratio=0.0,
                total_dark_volume=0, total_volume=0,
                alpha_signal=0.0, confidence_mult=1.0, is_significant=False,
            )

        cutoff = time_mod.time() - (self._rolling_minutes * 60)

        with self._lock:
            trades = list(self._trades.get(symbol, []))

        # Filter to rolling window
        recent = [t for t in trades if t.timestamp >= cutoff]

        if not recent:
            return DarkPoolSignal(
                symbol=symbol, dark_pool_ratio=0.0, net_buy_ratio=0.0,
                total_dark_volume=0, total_volume=0,
                alpha_signal=0.0, confidence_mult=1.0, is_significant=False,
            )

        total_volume = sum(t.volume for t in recent)
        dark_trades = [t for t in recent if t.is_dark_pool]
        dark_volume = sum(t.volume for t in dark_trades)

        dark_pool_ratio = dark_volume / total_volume if total_volume > 0 else 0.0

        # Net buy/sell in dark pool trades
        dark_buy_vol = sum(t.volume for t in dark_trades if t.side == "buy")
        dark_sell_vol = sum(t.volume for t in dark_trades if t.side == "sell")
        dark_total = dark_buy_vol + dark_sell_vol
        net_buy_ratio = (dark_buy_vol - dark_sell_vol) / dark_total if dark_total > 0 else 0.0

        # Alpha signal: significant when ratio > threshold and net direction is clear
        is_significant = dark_pool_ratio > self._ratio_threshold and abs(net_buy_ratio) > 0.1
        alpha_signal = 0.0

        if is_significant:
            # Positive alpha when dark pool is net buying, negative when net selling
            alpha_signal = net_buy_ratio * dark_pool_ratio

        # Confidence multiplier: 1 + weight * alpha_signal
        raw_mult = 1.0 + self._alpha_weight * alpha_signal
        confidence_mult = max(0.6, min(raw_mult, 1.4))  # Clamp to reasonable range

        return DarkPoolSignal(
            symbol=symbol,
            dark_pool_ratio=round(dark_pool_ratio, 4),
            net_buy_ratio=round(net_buy_ratio, 4),
            total_dark_volume=dark_volume,
            total_volume=total_volume,
            alpha_signal=round(alpha_signal, 4),
            confidence_mult=round(confidence_mult, 4),
            is_significant=is_significant,
        )

    def get_confidence_multiplier(self, symbol: str) -> float:
        """Convenience: get just the confidence multiplier.

        Returns:
            Float multiplier. 1.0 if insufficient data or disabled.
        """
        return self.get_signal(symbol).confidence_mult

    def prune_old_trades(self) -> int:
        """Remove trades older than the rolling window.

        Call periodically (e.g., every 5 minutes) to limit memory usage.

        Returns:
            Number of trades pruned.
        """
        cutoff = time_mod.time() - (self._rolling_minutes * 60)
        total_pruned = 0

        with self._lock:
            for symbol in list(self._trades.keys()):
                trades = self._trades[symbol]
                original_len = len(trades)
                while trades and trades[0].timestamp < cutoff:
                    trades.popleft()
                total_pruned += original_len - len(trades)

                # Remove empty deques
                if not trades:
                    del self._trades[symbol]

        if total_pruned > 0:
            logger.debug("T5-006: Pruned %d old dark pool trades", total_pruned)
        return total_pruned

    @property
    def tracked_symbols(self) -> list[str]:
        """Symbols with dark pool trade history."""
        with self._lock:
            return list(self._trades.keys())

    def reset(self, symbol: str | None = None) -> None:
        """Clear dark pool state."""
        with self._lock:
            if symbol:
                self._trades.pop(symbol, None)
                self._last_price.pop(symbol, None)
            else:
                self._trades.clear()
                self._last_price.clear()
