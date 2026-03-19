"""V8: Multi-timeframe confluence filter.

Checks alignment across daily, hourly, and 15-min timeframes.
- Breakout strategies (ORB, MICRO_MOM): need trend alignment (confluence >= 0.66)
- Mean reversion strategies (STAT_MR, VWAP): want dislocation (confluence <= 0.33)
"""

import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Cache to avoid refetching (thread-safe)
_mtf_lock = threading.Lock()
_mtf_cache: dict[str, tuple[float, datetime]] = {}


def _compute_ema(prices: pd.Series, period: int) -> float | None:
    """Compute EMA and return last value."""
    if len(prices) < period:
        return None
    ema = prices.ewm(span=period, adjust=False).mean()
    return float(ema.iloc[-1])


def get_mtf_confluence(symbol: str, side: str, now: datetime | None = None) -> float:
    """Get multi-timeframe confluence score for a symbol and trade direction.

    Checks:
    1. Daily: Price vs EMA20
    2. Hourly: Price vs EMA20
    3. 15-min: Price vs EMA9

    Args:
        symbol: Ticker symbol
        side: 'buy' or 'sell'
        now: Current time (for caching)

    Returns:
        Confluence score 0.0-1.0 (proportion of timeframes aligned with trade direction)
    """
    if not config.MTF_CONFLUENCE_ENABLED:
        return 0.5  # Neutral — don't filter

    # Check cache
    cache_key = f"{symbol}_{side}"
    with _mtf_lock:
        if now and cache_key in _mtf_cache:
            cached_score, cached_time = _mtf_cache[cache_key]
            if (now - cached_time).total_seconds() < config.MTF_CACHE_SECONDS:
                return cached_score

    aligned = 0
    total = 0

    try:
        from data import get_daily_bars, get_intraday_bars
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        # 1. Daily: Price vs EMA20
        try:
            daily_bars = get_daily_bars(symbol, days=30)
            if daily_bars is not None and not daily_bars.empty and len(daily_bars) >= 20:
                ema20 = _compute_ema(daily_bars["close"], 20)
                current = float(daily_bars["close"].iloc[-1])
                if ema20 is not None:
                    total += 1
                    if side == "buy" and current > ema20:
                        aligned += 1
                    elif side == "sell" and current < ema20:
                        aligned += 1
        except Exception as e:
            logger.debug(f"MTF daily check failed for {symbol}: {e}")

        # 2. Hourly: Price vs EMA20
        try:
            if now:
                start = now - timedelta(days=5)
                hourly_bars = get_intraday_bars(symbol, TimeFrame.Hour, start=start, end=now)
            else:
                hourly_bars = None

            if hourly_bars is not None and not hourly_bars.empty and len(hourly_bars) >= 20:
                ema20 = _compute_ema(hourly_bars["close"], 20)
                current = float(hourly_bars["close"].iloc[-1])
                if ema20 is not None:
                    total += 1
                    if side == "buy" and current > ema20:
                        aligned += 1
                    elif side == "sell" and current < ema20:
                        aligned += 1
        except Exception as e:
            logger.debug(f"MTF hourly check failed for {symbol}: {e}")

        # 3. 15-min: Price vs EMA9
        try:
            if now:
                start = now - timedelta(days=2)
                min15_bars = get_intraday_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), start=start, end=now)
            else:
                min15_bars = None

            if min15_bars is not None and not min15_bars.empty and len(min15_bars) >= 9:
                ema9 = _compute_ema(min15_bars["close"], 9)
                current = float(min15_bars["close"].iloc[-1])
                if ema9 is not None:
                    total += 1
                    if side == "buy" and current > ema9:
                        aligned += 1
                    elif side == "sell" and current < ema9:
                        aligned += 1
        except Exception as e:
            logger.debug(f"MTF 15-min check failed for {symbol}: {e}")

    except ImportError as e:
        logger.debug(f"MTF confluence import failed: {e}")
        return 0.5

    score = aligned / total if total > 0 else 0.5

    # Cache result
    if now:
        with _mtf_lock:
            _mtf_cache[cache_key] = (score, now)

    return score


def check_mtf_filter(symbol: str, strategy: str, side: str,
                     now: datetime | None = None) -> tuple[bool, str]:
    """Check if a trade passes the MTF confluence filter.

    Returns (allowed, reason).
    """
    if not config.MTF_CONFLUENCE_ENABLED:
        return True, ""

    confluence = get_mtf_confluence(symbol, side, now)

    # Breakout strategies need trend alignment
    if strategy in ("ORB", "MICRO_MOM"):
        if confluence < config.MTF_MIN_CONFLUENCE_BREAKOUT:
            return False, f"mtf_confluence={confluence:.2f}<{config.MTF_MIN_CONFLUENCE_BREAKOUT}"
        return True, ""

    # Mean reversion strategies want dislocation
    if strategy in ("STAT_MR", "VWAP"):
        if confluence > config.MTF_MAX_CONFLUENCE_MEANREV:
            return False, f"mtf_confluence={confluence:.2f}>{config.MTF_MAX_CONFLUENCE_MEANREV}"
        return True, ""

    # Other strategies — no filter
    return True, ""


def clear_cache():
    """Clear the MTF cache."""
    global _mtf_cache
    with _mtf_lock:
        _mtf_cache = {}
