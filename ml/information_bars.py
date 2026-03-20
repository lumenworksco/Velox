"""LPRADO-003: Information-Driven Bars — event-based sampling.

Implements Tick Imbalance Bars (TIB), Volume Imbalance Bars (VIB), and
Dollar Imbalance Bars (DIB) from Lopez de Prado's *Advances in Financial
Machine Learning* (Chapter 2).

Standard time bars (1-min, 5-min) sample at fixed intervals regardless of
market activity, producing bars with wildly varying information content.
Information-driven bars sample when a statistically significant imbalance
is detected, producing bars with roughly equal information content.

Imbalance bars trigger a new bar when the cumulative signed imbalance
(tick direction, volume, or dollar volume) exceeds a dynamic threshold
based on an EWMA of past bar sizes.

Usage:
    gen = InformationBarGenerator(ewma_span=100)
    tib = gen.generate_tick_imbalance_bars(tick_df)
    vib = gen.generate_volume_imbalance_bars(tick_df)
    dib = gen.generate_dollar_imbalance_bars(tick_df)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class _BarState:
    """Mutable state for bar construction."""
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0
    dollar_volume: float = 0.0
    n_ticks: int = 0
    start_time: Optional[pd.Timestamp] = None
    cumulative_imbalance: float = 0.0


class InformationBarGenerator:
    """Generate information-driven bars from tick-level data.

    Supports three bar types based on different imbalance measures:
    - Tick Imbalance Bars (TIB): signed tick direction (+1/-1)
    - Volume Imbalance Bars (VIB): signed volume
    - Dollar Imbalance Bars (DIB): signed dollar volume (price * volume)

    All three use the same core algorithm: accumulate signed imbalance,
    trigger a new bar when |cumulative_imbalance| exceeds a dynamic
    EWMA-based threshold.

    Input tick data must be a DataFrame with columns:
        - timestamp: datetime (or index)
        - price: float (trade price)
        - volume: float (trade size, can be 1 for pure tick data)
    """

    def __init__(
        self,
        ewma_span: int = 100,
        initial_threshold: Optional[float] = None,
    ):
        """
        Args:
            ewma_span: Span for EWMA of expected bar sizes and imbalances.
                       Larger values produce more stable thresholds.
            initial_threshold: Starting threshold before EWMA has data.
                               If None, auto-calibrated from first 100 ticks.
        """
        self.ewma_span = ewma_span
        self.initial_threshold = initial_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_tick_imbalance_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Generate Tick Imbalance Bars (TIB).

        Samples a new bar when the cumulative tick direction imbalance
        exceeds a dynamic threshold. Tick direction is +1 if price >= prev
        price, -1 otherwise (tick rule).

        Args:
            ticks: DataFrame with 'price', 'volume' columns and datetime index
                   or 'timestamp' column. Must be sorted chronologically.

        Returns:
            DataFrame with OHLCV columns indexed by bar start time.
        """
        return self._generate_imbalance_bars(ticks, bar_type="tick")

    def generate_volume_imbalance_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Generate Volume Imbalance Bars (VIB).

        Samples a new bar when the cumulative signed volume imbalance
        exceeds a dynamic threshold. Sign is determined by the tick rule.

        Args:
            ticks: DataFrame with 'price', 'volume' columns.

        Returns:
            DataFrame with OHLCV columns indexed by bar start time.
        """
        return self._generate_imbalance_bars(ticks, bar_type="volume")

    def generate_dollar_imbalance_bars(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Generate Dollar Imbalance Bars (DIB).

        Samples a new bar when the cumulative signed dollar volume
        (price * volume) imbalance exceeds a dynamic threshold.

        Args:
            ticks: DataFrame with 'price', 'volume' columns.

        Returns:
            DataFrame with OHLCV columns indexed by bar start time.
        """
        return self._generate_imbalance_bars(ticks, bar_type="dollar")

    # ------------------------------------------------------------------
    # Core engine
    # ------------------------------------------------------------------

    def _generate_imbalance_bars(
        self,
        ticks: pd.DataFrame,
        bar_type: str,
    ) -> pd.DataFrame:
        """Core imbalance bar generation engine.

        Args:
            ticks: Tick-level DataFrame.
            bar_type: One of "tick", "volume", "dollar".

        Returns:
            DataFrame of completed bars.
        """
        ticks = self._validate_ticks(ticks)
        if ticks.empty:
            return self._empty_bars_df()

        prices = ticks["price"].values.astype(np.float64)
        volumes = ticks["volume"].values.astype(np.float64)
        timestamps = ticks.index

        n = len(prices)

        # Compute tick directions using the tick rule
        directions = self._compute_tick_directions(prices)

        # Compute signed imbalance per tick based on bar type
        if bar_type == "tick":
            imbalances = directions.astype(np.float64)
        elif bar_type == "volume":
            imbalances = directions * volumes
        elif bar_type == "dollar":
            imbalances = directions * volumes * prices
        else:
            raise ValueError(f"Unknown bar_type: {bar_type}")

        # Auto-calibrate initial threshold if not provided
        threshold = self.initial_threshold
        if threshold is None:
            # Use the expected absolute imbalance over first 100 ticks
            calibration_n = min(100, n)
            threshold = float(np.abs(imbalances[:calibration_n]).mean() * calibration_n)
            if threshold < 1e-10:
                threshold = 1.0
            logger.debug(
                f"Auto-calibrated {bar_type} imbalance threshold: {threshold:.4f}"
            )

        # EWMA state
        alpha = 2.0 / (self.ewma_span + 1)
        ewma_imbalance = threshold  # EWMA of |cumulative imbalance at bar close|
        ewma_ticks = 100.0          # EWMA of ticks per bar

        # Bar construction
        bars: list[dict] = []
        state = _BarState()
        state.start_time = timestamps[0]

        for i in range(n):
            p = prices[i]
            v = volumes[i]
            t = timestamps[i]

            # Update OHLCV
            if state.n_ticks == 0:
                state.open = p
                state.high = p
                state.low = p
                state.start_time = t
            else:
                state.high = max(state.high, p)
                state.low = min(state.low, p)

            state.close = p
            state.volume += v
            state.dollar_volume += p * v
            state.n_ticks += 1
            state.cumulative_imbalance += imbalances[i]

            # Check if imbalance exceeds threshold
            expected_imbalance = ewma_ticks * abs(
                imbalances[max(0, i - int(ewma_ticks)): i + 1].mean()
                if state.n_ticks > 1 else imbalances[i]
            )
            dynamic_threshold = max(expected_imbalance, threshold * 0.1)

            if abs(state.cumulative_imbalance) >= dynamic_threshold and state.n_ticks >= 2:
                # Emit bar
                bars.append({
                    "timestamp": state.start_time,
                    "open": state.open,
                    "high": state.high,
                    "low": state.low,
                    "close": state.close,
                    "volume": state.volume,
                    "dollar_volume": state.dollar_volume,
                    "n_ticks": state.n_ticks,
                    "imbalance": state.cumulative_imbalance,
                })

                # Update EWMA
                ewma_imbalance = (
                    alpha * abs(state.cumulative_imbalance)
                    + (1 - alpha) * ewma_imbalance
                )
                ewma_ticks = (
                    alpha * state.n_ticks + (1 - alpha) * ewma_ticks
                )

                # Reset state
                state = _BarState()

        # Don't emit incomplete final bar — data may be streaming
        if not bars:
            logger.debug(f"No {bar_type} imbalance bars generated (threshold too high?)")
            return self._empty_bars_df()

        result = pd.DataFrame(bars)
        result.set_index("timestamp", inplace=True)
        result.index.name = "timestamp"

        logger.info(
            f"Generated {len(result)} {bar_type} imbalance bars from {n} ticks "
            f"(avg {n / len(result):.0f} ticks/bar)"
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tick_directions(prices: np.ndarray) -> np.ndarray:
        """Compute tick direction using the tick rule.

        +1 if price >= previous price (uptick or zero-tick following uptick)
        -1 if price < previous price (downtick)

        The first tick is assigned +1 by convention.
        """
        n = len(prices)
        directions = np.ones(n, dtype=np.int8)

        for i in range(1, n):
            if prices[i] < prices[i - 1]:
                directions[i] = -1
            elif prices[i] == prices[i - 1]:
                # Carry forward previous direction (zero-tick rule)
                directions[i] = directions[i - 1]
            # else: uptick, already +1

        return directions

    @staticmethod
    def _validate_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize tick DataFrame.

        Ensures 'price' and 'volume' columns exist, index is datetime.
        """
        if ticks.empty:
            logger.warning("Empty tick DataFrame provided")
            return ticks

        # Handle 'timestamp' column vs datetime index
        df = ticks.copy()
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        if "price" not in df.columns:
            raise ValueError("Tick DataFrame must have a 'price' column")

        if "volume" not in df.columns:
            # Default to 1 tick per observation (pure tick data)
            df["volume"] = 1.0

        # Remove rows with invalid prices
        df = df[df["price"] > 0].copy()

        return df

    @staticmethod
    def _empty_bars_df() -> pd.DataFrame:
        """Return an empty DataFrame with the expected bar schema."""
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume",
                      "dollar_volume", "n_ticks", "imbalance"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )
