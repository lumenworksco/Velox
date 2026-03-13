"""Multi-Timeframe Confirmation — confirms signals against higher timeframe trend."""

import logging
import time as _time
from datetime import datetime, timedelta

import pandas_ta as ta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_bars

logger = logging.getLogger(__name__)

# Strategy -> higher timeframe mapping
_HIGHER_TF = {
    "ORB":             TimeFrame(15, TimeFrameUnit.Minute),
    "VWAP":            TimeFrame(5, TimeFrameUnit.Minute),
    "GAP_GO":          TimeFrame(1, TimeFrameUnit.Hour),
    "MOMENTUM":        TimeFrame.Day,
}

# (V5: per-strategy MTF toggle moved to config.MTF_ENABLED_FOR)

# Lookback durations to ensure we get at least 30 bars
_LOOKBACK = {
    "ORB":       timedelta(days=5),    # 15-min bars -> 5 trading days
    "VWAP":      timedelta(days=2),    # 5-min bars  -> 2 trading days
    "GAP_GO":    timedelta(days=10),   # 1-hour bars -> 10 trading days
    "MOMENTUM":  timedelta(days=60),   # daily bars  -> 60 calendar days
}


class MultiTimeframeConfirmation:
    """Confirms trading signals against higher timeframe trend direction."""

    def __init__(self):
        self._cache: dict[str, tuple[float, bool]] = {}  # key -> (timestamp, result)

    def confirm(self, signal: dict, now: datetime) -> bool:
        """Check if a signal aligns with the higher timeframe trend.

        Args:
            signal: dict with at least 'symbol', 'strategy', and 'side' keys.
            now: current datetime (timezone-aware).

        Returns:
            True if the signal is confirmed (or confirmation is disabled/unavailable).
        """
        # 1. Check feature flag
        if not config.MTF_CONFIRMATION_ENABLED:
            return True

        strategy = signal.get("strategy", "")
        symbol = signal.get("symbol", "")
        side = signal.get("side", "buy").lower()

        # 2. Check per-strategy MTF toggle
        if not config.MTF_ENABLED_FOR.get(strategy, True):
            return True

        # 3. Look up higher timeframe
        higher_tf = _HIGHER_TF.get(strategy)
        if higher_tf is None:
            logger.warning("MTF: unknown strategy '%s', allowing signal", strategy)
            return True

        # 4. Check cache
        cache_key = f"{symbol}:{strategy}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_ts, cached_result = cached
            if (_time.time() - cached_ts) < config.MTF_CACHE_SECONDS:
                logger.debug("MTF: cache hit for %s (%s)", cache_key, cached_result)
                return cached_result

        # 5. Fetch higher-TF bars and compute indicators
        try:
            lookback = _LOOKBACK[strategy]
            start = now - lookback
            bars = get_bars(symbol, higher_tf, start=start, end=now)

            if bars is None or len(bars) < 21:
                logger.warning(
                    "MTF: insufficient bars for %s on %s (%d), allowing signal",
                    symbol, strategy, 0 if bars is None else len(bars),
                )
                return True

            close = bars["close"]

            # 6. Compute EMAs
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()

            # 7. Compute RSI(14) via pandas_ta
            rsi_series = ta.rsi(close, length=14)
            if rsi_series is None or rsi_series.empty:
                logger.warning("MTF: RSI calculation failed for %s, allowing signal", symbol)
                return True

            latest_ema9 = ema9.iloc[-1]
            latest_ema21 = ema21.iloc[-1]
            latest_rsi = rsi_series.iloc[-1]

            # 8. Confirm based on side
            if side == "buy":
                confirmed = latest_ema9 > latest_ema21 and latest_rsi > 40
            else:  # sell / short
                confirmed = latest_ema9 < latest_ema21 and latest_rsi < 60

            # 9. Cache the result
            self._cache[cache_key] = (_time.time(), confirmed)

            logger.info(
                "MTF: %s %s %s — ema9=%.2f ema21=%.2f rsi=%.1f → %s",
                side, symbol, strategy,
                latest_ema9, latest_ema21, latest_rsi,
                "CONFIRMED" if confirmed else "REJECTED",
            )
            return confirmed

        except Exception:
            # 10. Fail-open on any error
            logger.exception("MTF: error confirming %s %s, allowing signal", symbol, strategy)
            return True


# Module-level singleton
mtf_confirmer = MultiTimeframeConfirmation()
