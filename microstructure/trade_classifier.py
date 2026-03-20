"""MICRO-003: Trade Size Classification and Institutional Flow Analysis.

Classifies individual trades by size into retail, mixed, and institutional
categories, then tracks net institutional flow direction per symbol.

Usage:
    classifier = TradeClassifier()
    trade_type = classifier.classify_trade(size=5000, price=150.0)
    # -> TradeType.INSTITUTIONAL

    classifier.record_trade("AAPL", size=5000, price=150.0, side="buy")
    classifier.record_trade("AAPL", size=50, price=150.05, side="sell")
    flow = classifier.get_institutional_flow("AAPL")
    # -> positive float: institutional net buying

Signal: institutional buying + retail selling -> bullish divergence.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """Trade size classification."""
    SMALL = "small"           # < 100 shares (retail)
    MEDIUM = "medium"         # 100-1000 shares (mixed)
    LARGE = "large"           # > 1000 shares or > $50k notional (institutional)


@dataclass
class _ClassifiedTrade:
    """Recorded trade with classification metadata."""
    symbol: str
    size: int
    price: float
    notional: float
    side: str               # "buy" or "sell"
    trade_type: TradeType
    timestamp: datetime


class TradeClassifier:
    """Classify trades by size and track institutional flow.

    Thresholds:
        - Small (retail): < 100 shares
        - Medium (mixed): 100-1000 shares
        - Large (institutional): > 1000 shares OR notional > $50,000

    Args:
        small_threshold: Max shares for a "small" trade (default 100).
        large_threshold: Min shares for a "large" trade (default 1000).
        large_notional_threshold: Min notional value for "large" (default $50,000).
        max_history: Maximum trades to keep per symbol.
    """

    def __init__(
        self,
        small_threshold: int = 100,
        large_threshold: int = 1000,
        large_notional_threshold: float = 50_000.0,
        max_history: int = 5000,
    ) -> None:
        self._small_threshold = small_threshold
        self._large_threshold = large_threshold
        self._large_notional_threshold = large_notional_threshold
        self._max_history = max_history

        self._trades: dict[str, deque[_ClassifiedTrade]] = {}
        self._global_trades: deque[_ClassifiedTrade] = deque(maxlen=max_history)

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify_trade(self, size: int, price: float) -> TradeType:
        """Classify a single trade by size.

        Args:
            size: Number of shares.
            price: Trade price (used for notional value check).

        Returns:
            TradeType enum value.
        """
        if size <= 0 or price <= 0:
            return TradeType.SMALL

        notional = size * price

        if size >= self._large_threshold or notional >= self._large_notional_threshold:
            return TradeType.LARGE
        elif size < self._small_threshold:
            return TradeType.SMALL
        else:
            return TradeType.MEDIUM

    # ------------------------------------------------------------------ #
    #  Trade recording
    # ------------------------------------------------------------------ #

    def record_trade(
        self,
        symbol: str,
        size: int,
        price: float,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> TradeType:
        """Record and classify a trade for flow tracking.

        Args:
            symbol: Ticker symbol.
            size: Number of shares.
            price: Execution price.
            side: "buy" or "sell".
            timestamp: Trade time. Defaults to now (UTC).

        Returns:
            The TradeType classification.
        """
        if size <= 0 or price <= 0:
            return TradeType.SMALL

        trade_type = self.classify_trade(size, price)
        ts = timestamp or datetime.now(timezone.utc)

        classified = _ClassifiedTrade(
            symbol=symbol,
            size=size,
            price=price,
            notional=size * price,
            side=side.lower(),
            trade_type=trade_type,
            timestamp=ts,
        )

        if symbol not in self._trades:
            self._trades[symbol] = deque(maxlen=self._max_history)
        self._trades[symbol].append(classified)
        self._global_trades.append(classified)

        return trade_type

    # ------------------------------------------------------------------ #
    #  Flow analysis
    # ------------------------------------------------------------------ #

    def get_institutional_flow(self, symbol: str) -> float:
        """Get net institutional flow direction for a symbol.

        Computes (large_buy_volume - large_sell_volume) / total_large_volume.

        Args:
            symbol: Ticker symbol.

        Returns:
            Float in [-1.0, 1.0]. Positive = net institutional buying.
            Returns 0.0 if no institutional trades recorded.
        """
        if symbol not in self._trades:
            return 0.0

        large_buy_vol = 0
        large_sell_vol = 0

        for t in self._trades[symbol]:
            if t.trade_type == TradeType.LARGE:
                if t.side == "buy":
                    large_buy_vol += t.size
                else:
                    large_sell_vol += t.size

        total = large_buy_vol + large_sell_vol
        if total == 0:
            return 0.0

        return (large_buy_vol - large_sell_vol) / total

    def get_retail_flow(self, symbol: str) -> float:
        """Get net retail flow direction for a symbol.

        Same as institutional but for small trades.

        Returns:
            Float in [-1.0, 1.0]. Positive = net retail buying.
        """
        if symbol not in self._trades:
            return 0.0

        small_buy_vol = 0
        small_sell_vol = 0

        for t in self._trades[symbol]:
            if t.trade_type == TradeType.SMALL:
                if t.side == "buy":
                    small_buy_vol += t.size
                else:
                    small_sell_vol += t.size

        total = small_buy_vol + small_sell_vol
        if total == 0:
            return 0.0

        return (small_buy_vol - small_sell_vol) / total

    def get_flow_divergence(self, symbol: str) -> float:
        """Detect divergence between institutional and retail flow.

        Positive divergence = institutions buying while retail sells (bullish).
        Negative divergence = institutions selling while retail buys (bearish).

        Returns:
            Float in [-2.0, 2.0]. Magnitude indicates conviction.
        """
        inst_flow = self.get_institutional_flow(symbol)
        retail_flow = self.get_retail_flow(symbol)
        return inst_flow - retail_flow

    def get_volume_breakdown(self, symbol: str) -> dict[str, int]:
        """Get volume breakdown by trade type for a symbol.

        Returns:
            Dict with keys: small_volume, medium_volume, large_volume, total_volume.
        """
        result = {
            "small_volume": 0,
            "medium_volume": 0,
            "large_volume": 0,
            "total_volume": 0,
        }

        if symbol not in self._trades:
            return result

        for t in self._trades[symbol]:
            if t.trade_type == TradeType.SMALL:
                result["small_volume"] += t.size
            elif t.trade_type == TradeType.MEDIUM:
                result["medium_volume"] += t.size
            else:
                result["large_volume"] += t.size
            result["total_volume"] += t.size

        return result

    def get_institutional_participation_rate(self, symbol: str) -> float:
        """Fraction of total volume from institutional-size trades.

        Returns:
            Float in [0.0, 1.0]. Higher = more institutional activity.
        """
        breakdown = self.get_volume_breakdown(symbol)
        total = breakdown["total_volume"]
        if total == 0:
            return 0.0
        return breakdown["large_volume"] / total

    # ------------------------------------------------------------------ #
    #  Housekeeping
    # ------------------------------------------------------------------ #

    @property
    def tracked_symbols(self) -> list[str]:
        """List of symbols with recorded trades."""
        return list(self._trades.keys())

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear trade history for a symbol or all symbols."""
        if symbol is not None:
            self._trades.pop(symbol, None)
        else:
            self._trades.clear()
            self._global_trades.clear()
        logger.debug("TradeClassifier reset (symbol=%s)", symbol)

    def __repr__(self) -> str:
        return (
            f"TradeClassifier(symbols={len(self._trades)}, "
            f"total_trades={len(self._global_trades)})"
        )
