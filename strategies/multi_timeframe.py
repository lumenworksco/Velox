"""Multi-Timeframe Filter (STRAT-004).

Scores signal alignment across 2-minute, 15-minute, and daily timeframes.
Used as a confirmation overlay to adjust signal confidence before execution.

Alignment scoring:
    All three timeframes aligned  -> 1.0
    Two timeframes aligned        -> 0.7
    One timeframe aligned         -> 0.4
    None aligned                  -> 0.0 (signal rejected)

Trend detection per timeframe:
    - EMA crossover (fast/slow) for direction
    - RSI for overbought/oversold confirmation
    - Price position relative to VWAP (intraday) or 20-period EMA (daily)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import Signal

logger = logging.getLogger(__name__)

# Alignment score thresholds
SCORE_ALL_ALIGNED = 1.0
SCORE_TWO_ALIGNED = 0.7
SCORE_ONE_ALIGNED = 0.4
SCORE_NONE_ALIGNED = 0.0

# Minimum confidence to pass filter
MIN_CONFIDENCE_PASS = 0.3

# EMA periods per timeframe
EMA_FAST = {
    "2min": 8,
    "15min": 8,
    "daily": 10,
}

EMA_SLOW = {
    "2min": 21,
    "15min": 21,
    "daily": 50,
}

# RSI settings
RSI_PERIOD = 14
RSI_BULLISH_THRESHOLD = 45   # RSI above this confirms bullish bias
RSI_BEARISH_THRESHOLD = 55   # RSI below this confirms bearish bias

# Minimum bars required per timeframe
MIN_BARS = {
    "2min": 30,
    "15min": 30,
    "daily": 60,
}


@dataclass
class TimeframeSignal:
    """Directional bias from a single timeframe."""
    timeframe: str
    direction: int       # +1 bullish, -1 bearish, 0 neutral
    strength: float      # 0.0 to 1.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    rsi: float = 50.0
    price_vs_anchor: float = 0.0  # % above/below VWAP or EMA


@dataclass
class AlignmentResult:
    """Result of multi-timeframe alignment scoring."""
    score: float                           # 0.0 to 1.0
    aligned_count: int                     # 0 to 3
    timeframe_signals: Dict[str, TimeframeSignal] = field(default_factory=dict)
    direction_consensus: int = 0           # +1, -1, or 0


class MultiTimeframeFilter:
    """Multi-Timeframe confirmation filter for signal quality.

    Evaluates directional agreement across 2-min, 15-min, and daily bars.
    When timeframes agree, signal confidence is boosted; when they disagree,
    confidence is reduced or the signal is filtered out entirely.

    Usage:
        mtf = MultiTimeframeFilter()
        score = mtf.score_signal(signal, bars_2min, bars_15min, bars_daily)
        filtered = mtf.filter_signals(signals, market_data)
    """

    def __init__(self, min_confidence: float = MIN_CONFIDENCE_PASS):
        self.min_confidence = min_confidence
        self._cache: Dict[str, Tuple[datetime, AlignmentResult]] = {}
        self._cache_ttl_sec = 120  # Cache alignment for 2 minutes

    def score_signal(
        self,
        signal: Signal,
        bars_2min: Optional[pd.DataFrame],
        bars_15min: Optional[pd.DataFrame],
        bars_daily: Optional[pd.DataFrame],
    ) -> float:
        """Score a signal's alignment across three timeframes.

        Args:
            signal: The trading signal to evaluate.
            bars_2min: 2-minute OHLCV bars for the signal's symbol.
            bars_15min: 15-minute OHLCV bars for the signal's symbol.
            bars_daily: Daily OHLCV bars for the signal's symbol.

        Returns:
            Alignment score from 0.0 (no alignment) to 1.0 (full alignment).
        """
        desired_direction = 1 if signal.side == "buy" else -1

        tf_signals: Dict[str, TimeframeSignal] = {}

        # Analyze each timeframe
        for label, bars, timeframe in [
            ("2min", bars_2min, "2min"),
            ("15min", bars_15min, "15min"),
            ("daily", bars_daily, "daily"),
        ]:
            tf_sig = self._analyze_timeframe(bars, timeframe)
            if tf_sig is not None:
                tf_signals[label] = tf_sig

        if not tf_signals:
            return SCORE_NONE_ALIGNED

        # Count how many timeframes agree with the signal direction
        aligned_count = sum(
            1 for tf in tf_signals.values()
            if tf.direction == desired_direction
        )
        total_available = len(tf_signals)

        # Score based on alignment ratio
        if total_available == 3:
            if aligned_count == 3:
                score = SCORE_ALL_ALIGNED
            elif aligned_count == 2:
                score = SCORE_TWO_ALIGNED
            elif aligned_count == 1:
                score = SCORE_ONE_ALIGNED
            else:
                score = SCORE_NONE_ALIGNED
        elif total_available == 2:
            # Two timeframes available: scale proportionally
            if aligned_count == 2:
                score = SCORE_ALL_ALIGNED * 0.9   # Slight discount
            elif aligned_count == 1:
                score = SCORE_TWO_ALIGNED * 0.9
            else:
                score = SCORE_NONE_ALIGNED
        else:
            # Only one timeframe available: limited signal
            score = SCORE_ONE_ALIGNED if aligned_count == 1 else SCORE_NONE_ALIGNED

        # Weighted strength bonus: stronger agreement boosts the score
        if tf_signals and aligned_count > 0:
            avg_strength = np.mean([
                tf.strength for tf in tf_signals.values()
                if tf.direction == desired_direction
            ])
            score = score * (0.8 + 0.2 * avg_strength)

        return round(min(score, 1.0), 4)

    def filter_signals(
        self,
        signals: List[Signal],
        market_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> List[Signal]:
        """Filter and adjust signals based on multi-timeframe alignment.

        Adjusts each signal's confidence by blending with the alignment score.
        Signals with resulting confidence below min_confidence are removed.

        Args:
            signals: List of raw signals from strategies.
            market_data: Dict mapping symbol -> {"2min": df, "15min": df, "daily": df}.

        Returns:
            Filtered list of signals with adjusted confidence values.
        """
        filtered: List[Signal] = []

        for signal in signals:
            symbol_data = market_data.get(signal.symbol, {})
            bars_2min = symbol_data.get("2min")
            bars_15min = symbol_data.get("15min")
            bars_daily = symbol_data.get("daily")

            alignment_score = self.score_signal(
                signal, bars_2min, bars_15min, bars_daily
            )

            # Blend original confidence with alignment score
            # 60% original confidence, 40% alignment
            adjusted_confidence = (
                0.6 * signal.confidence + 0.4 * alignment_score
            )

            if adjusted_confidence < self.min_confidence:
                logger.debug(
                    f"MTF filter rejected {signal.symbol} {signal.side}: "
                    f"alignment={alignment_score:.2f}, "
                    f"adjusted_conf={adjusted_confidence:.2f} < {self.min_confidence}"
                )
                continue

            # Create new signal with adjusted confidence and metadata
            adjusted_signal = Signal(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                entry_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                reason=signal.reason,
                hold_type=signal.hold_type,
                pair_id=signal.pair_id,
                confidence=round(adjusted_confidence, 4),
                metadata={
                    **signal.metadata,
                    "mtf_alignment_score": alignment_score,
                    "mtf_original_confidence": signal.confidence,
                },
                timestamp=signal.timestamp,
                pair_symbol=signal.pair_symbol,
            )
            filtered.append(adjusted_signal)

        logger.info(
            f"MTF filter: {len(filtered)}/{len(signals)} signals passed"
        )
        return filtered

    def _analyze_timeframe(
        self, bars: Optional[pd.DataFrame], timeframe: str
    ) -> Optional[TimeframeSignal]:
        """Analyze a single timeframe for directional bias.

        Uses EMA crossover, RSI, and price-anchor relationship to
        determine bullish (+1), bearish (-1), or neutral (0) bias.
        """
        if bars is None or bars.empty:
            return None

        min_required = MIN_BARS.get(timeframe, 30)
        if len(bars) < min_required:
            return None

        close = bars["close"].astype(float)

        # EMA crossover
        fast_period = EMA_FAST[timeframe]
        slow_period = EMA_SLOW[timeframe]
        ema_fast = close.ewm(span=fast_period, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=slow_period, adjust=False).mean().iloc[-1]

        # RSI
        rsi = self._compute_rsi(close, RSI_PERIOD)

        # Price vs anchor (VWAP for intraday, EMA-20 for daily)
        current_price = close.iloc[-1]
        if timeframe == "daily":
            anchor = close.ewm(span=20, adjust=False).mean().iloc[-1]
        else:
            anchor = self._compute_vwap(bars)

        price_vs_anchor = (current_price - anchor) / anchor if anchor > 0 else 0.0

        # Determine direction via voting
        votes = 0

        # EMA crossover vote
        if ema_fast > ema_slow:
            votes += 1
        elif ema_fast < ema_slow:
            votes -= 1

        # RSI vote
        if rsi > RSI_BULLISH_THRESHOLD:
            votes += 1
        elif rsi < RSI_BEARISH_THRESHOLD:
            votes -= 1

        # Price vs anchor vote
        if price_vs_anchor > 0.001:
            votes += 1
        elif price_vs_anchor < -0.001:
            votes -= 1

        # Classify direction
        if votes >= 2:
            direction = 1
        elif votes <= -2:
            direction = -1
        else:
            direction = 0

        # Strength: how decisive the signals are
        ema_spread = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
        rsi_extremity = abs(rsi - 50) / 50
        strength = min(1.0, (ema_spread * 20 + rsi_extremity + abs(price_vs_anchor) * 10) / 3)

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=round(strength, 4),
            ema_fast=round(ema_fast, 4),
            ema_slow=round(ema_slow, 4),
            rsi=round(rsi, 2),
            price_vs_anchor=round(price_vs_anchor, 6),
        )

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> float:
        """Compute RSI for a price series."""
        if len(close) < period + 1:
            return 50.0

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_vwap(bars: pd.DataFrame) -> float:
        """Compute VWAP from intraday OHLCV bars."""
        if "volume" not in bars.columns or bars["volume"].sum() == 0:
            return bars["close"].mean()

        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        cumulative_vp = (typical_price * bars["volume"]).sum()
        cumulative_vol = bars["volume"].sum()

        return cumulative_vp / cumulative_vol if cumulative_vol > 0 else bars["close"].iloc[-1]
