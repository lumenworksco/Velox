"""MICRO-001: Volume-Synchronized Probability of Informed Trading (VPIN).

VPIN measures the probability that informed traders are active in a market.
High VPIN values (>0.7) indicate toxic order flow and elevated risk of
adverse selection. Use as a filter, sizing modifier, and regime indicator.

Usage:
    vpin = VPIN(bucket_volume=10_000, n_buckets=50)
    vpin.add_trade(price=150.25, volume=500, side="buy")
    vpin.add_trade(price=150.20, volume=300, side="sell")
    current_vpin = vpin.compute_vpin()
    # Returns float 0.0-1.0; >0.7 means widen stops, reduce sizing

References:
    Easley, Lopez de Prado, O'Hara (2012) — "Flow Toxicity and Liquidity
    in a High-Frequency World"
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _Trade:
    """Single trade record for VPIN bucketing."""
    price: float
    volume: int
    side: str  # "buy" or "sell"


@dataclass
class _Bucket:
    """Volume-equal bucket for VPIN computation."""
    buy_volume: int = 0
    sell_volume: int = 0
    total_volume: int = 0


class VPIN:
    """Volume-Synchronized Probability of Informed Trading.

    Buckets trades into volume-equal bins, classifies each trade as
    buyer- or seller-initiated using the tick rule when side is unknown,
    and computes VPIN as the average absolute order imbalance across
    the most recent n_buckets.

    Args:
        bucket_volume: Target volume per bucket (e.g. 10,000 shares).
        n_buckets: Number of recent buckets to average over.
        high_vpin_threshold: VPIN level considered "high" (default 0.7).
    """

    def __init__(
        self,
        bucket_volume: int = 10_000,
        n_buckets: int = 50,
        high_vpin_threshold: float = 0.7,
    ) -> None:
        if bucket_volume <= 0:
            raise ValueError(f"bucket_volume must be positive, got {bucket_volume}")
        if n_buckets <= 0:
            raise ValueError(f"n_buckets must be positive, got {n_buckets}")

        self._bucket_volume = bucket_volume
        self._n_buckets = n_buckets
        self._high_vpin_threshold = high_vpin_threshold

        self._completed_buckets: deque[_Bucket] = deque(maxlen=n_buckets)
        self._current_bucket = _Bucket()
        self._last_price: Optional[float] = None
        self._total_trades_processed: int = 0

    # ------------------------------------------------------------------ #
    #  Trade ingestion
    # ------------------------------------------------------------------ #

    def add_trade(
        self,
        price: float,
        volume: int,
        side: Optional[str] = None,
    ) -> None:
        """Add a trade and bucket it.

        Args:
            price: Trade execution price.
            volume: Number of shares.
            side: "buy" or "sell". If None, the tick rule is applied:
                  uptick or zero-uptick -> buy; downtick -> sell.
        """
        if volume <= 0:
            return

        classified_side = side
        if classified_side is None:
            classified_side = self._classify_tick_rule(price)

        self._last_price = price
        self._total_trades_processed += 1

        remaining = volume
        while remaining > 0:
            space = self._bucket_volume - self._current_bucket.total_volume
            fill = min(remaining, space)

            if classified_side == "buy":
                self._current_bucket.buy_volume += fill
            else:
                self._current_bucket.sell_volume += fill
            self._current_bucket.total_volume += fill
            remaining -= fill

            if self._current_bucket.total_volume >= self._bucket_volume:
                self._completed_buckets.append(self._current_bucket)
                self._current_bucket = _Bucket()

    def _classify_tick_rule(self, price: float) -> str:
        """Classify trade direction using the tick rule.

        Uptick (price > last) -> buyer-initiated.
        Downtick (price < last) -> seller-initiated.
        Zero tick -> same as previous classification (default buy).
        """
        if self._last_price is None or price >= self._last_price:
            return "buy"
        return "sell"

    # ------------------------------------------------------------------ #
    #  VPIN computation
    # ------------------------------------------------------------------ #

    def compute_vpin(self, trades: Optional[list[dict]] = None) -> float:
        """Compute current VPIN value.

        If ``trades`` is provided, each dict must have keys: price, volume,
        and optionally side. Trades are ingested before computing.

        Returns:
            Float in [0.0, 1.0]. Higher values indicate more informed trading.
            Returns 0.0 if insufficient buckets are available.
        """
        if trades is not None:
            for t in trades:
                self.add_trade(
                    price=t["price"],
                    volume=t["volume"],
                    side=t.get("side"),
                )

        if len(self._completed_buckets) == 0:
            return 0.0

        # VPIN = mean(|V_buy - V_sell| / V_bucket) across recent buckets
        imbalances = []
        for bucket in self._completed_buckets:
            if bucket.total_volume > 0:
                imbalance = abs(bucket.buy_volume - bucket.sell_volume) / bucket.total_volume
                imbalances.append(imbalance)

        if not imbalances:
            return 0.0

        vpin = float(np.mean(imbalances))
        return min(max(vpin, 0.0), 1.0)

    # ------------------------------------------------------------------ #
    #  Convenience methods
    # ------------------------------------------------------------------ #

    def is_toxic(self) -> bool:
        """Return True if current VPIN exceeds the high threshold."""
        return self.compute_vpin() >= self._high_vpin_threshold

    def get_sizing_modifier(self) -> float:
        """Return a position sizing multiplier based on VPIN.

        Returns:
            1.0 when VPIN is low (safe), scaling down to 0.3 when VPIN
            is at or above the high threshold.
        """
        vpin = self.compute_vpin()
        if vpin < 0.3:
            return 1.0
        elif vpin >= self._high_vpin_threshold:
            return 0.3
        else:
            # Linear interpolation: 1.0 at 0.3, 0.3 at threshold
            slope = (1.0 - 0.3) / (0.3 - self._high_vpin_threshold)
            return max(0.3, 1.0 + slope * (vpin - 0.3))

    @property
    def completed_bucket_count(self) -> int:
        """Number of fully filled buckets available."""
        return len(self._completed_buckets)

    @property
    def total_trades(self) -> int:
        """Total number of trades processed."""
        return self._total_trades_processed

    def reset(self) -> None:
        """Clear all state."""
        self._completed_buckets.clear()
        self._current_bucket = _Bucket()
        self._last_price = None
        self._total_trades_processed = 0
        logger.debug("VPIN state reset")

    @classmethod
    def from_adv(cls, adv: float, n_buckets: int = 50, toxic_threshold: float = 0.35) -> "VPIN":
        """Create a VPIN instance with bucket volume = 1/50 x ADV.

        Per Easley, Lopez de Prado, O'Hara (2012), the optimal bucket
        volume V* = ADV / 50 ensures ~50 buckets per day.

        Args:
            adv: Average daily volume in shares.
            n_buckets: Number of rolling buckets (default 50).
            toxic_threshold: VPIN level to flag as TOXIC_FLOW (default 0.35).

        Returns:
            Configured VPIN instance.
        """
        bucket_vol = max(1, int(adv / 50))
        return cls(bucket_volume=bucket_vol, n_buckets=n_buckets, high_vpin_threshold=toxic_threshold)

    def estimate_toxicity(self, trades: list[dict] | None = None) -> dict:
        """Estimate order-flow toxicity using VPIN.

        Implements the full ELO (2012) toxicity estimation:
        1. Ingest any new trades.
        2. Compute VPIN over rolling n buckets.
        3. Flag as TOXIC_FLOW if VPIN > 0.35.

        Args:
            trades: Optional list of trade dicts with keys: price, volume, side (optional).

        Returns:
            Dict with keys:
                vpin (float): Current VPIN value [0, 1].
                is_toxic (bool): True if VPIN > toxic threshold (0.35).
                label (str): "TOXIC_FLOW" or "NORMAL".
                buckets_filled (int): Number of completed buckets.
                sizing_modifier (float): Position sizing multiplier [0.3, 1.0].
        """
        if trades is not None:
            for t in trades:
                self.add_trade(
                    price=t["price"],
                    volume=t["volume"],
                    side=t.get("side"),
                )

        vpin_val = self.compute_vpin()
        toxic = vpin_val >= self._high_vpin_threshold

        return {
            "vpin": round(vpin_val, 4),
            "is_toxic": toxic,
            "label": "TOXIC_FLOW" if toxic else "NORMAL",
            "buckets_filled": len(self._completed_buckets),
            "sizing_modifier": round(self.get_sizing_modifier(), 3),
        }

    def __repr__(self) -> str:
        vpin = self.compute_vpin()
        return (
            f"VPIN(vpin={vpin:.3f}, buckets={len(self._completed_buckets)}/"
            f"{self._n_buckets}, toxic={self.is_toxic()})"
        )
