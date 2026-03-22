"""T4-003: Indicator computation cache — avoids recomputing RSI, ATR, VWAP, Hurst every cycle.

IndicatorCache is keyed on (symbol, bar_hash) where bar_hash is derived from
the last bar's timestamp. When new bar data arrives (timestamp changes), the
cached indicators for that symbol are automatically invalidated.

Cached indicators: RSI(14), ATR(14), VWAP, Hurst, OU halflife, consistency score.
"""

import logging
import threading
import time as _time
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 500  # Max symbol entries


def _bar_hash(bars: pd.DataFrame) -> str:
    """Compute a hash key from bar data.

    Uses the last bar's index (timestamp) and close price for fast invalidation.
    When a new bar arrives, the timestamp changes and the cache misses.
    """
    if bars.empty:
        return "empty"
    last_idx = bars.index[-1]
    last_close = bars["close"].iloc[-1] if "close" in bars.columns else 0
    return f"{last_idx}_{last_close:.4f}_{len(bars)}"


class IndicatorCache:
    """Thread-safe indicator computation cache.

    Usage:
        from analytics.indicator_cache import indicator_cache

        # Check cache before computing
        rsi = indicator_cache.get("AAPL", bars, "rsi_14")
        if rsi is None:
            rsi = compute_rsi(bars, 14)
            indicator_cache.put("AAPL", bars, "rsi_14", rsi)
    """

    _instance: Optional["IndicatorCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self, max_size: int = _MAX_CACHE_SIZE):
        # _cache: {symbol: {bar_hash: {indicator_name: value}}}
        self._cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info("T4-003: IndicatorCache initialized (max_size=%d)", max_size)

    @classmethod
    def instance(cls) -> "IndicatorCache":
        """Get or create the singleton IndicatorCache."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(self, symbol: str, bars: pd.DataFrame, indicator: str) -> Any | None:
        """Get a cached indicator value.

        Returns None on cache miss (indicator not cached or bars have changed).
        """
        bh = _bar_hash(bars)
        with self._lock:
            sym_cache = self._cache.get(symbol)
            if sym_cache is None:
                self._misses += 1
                return None

            hash_cache = sym_cache.get(bh)
            if hash_cache is None:
                self._misses += 1
                return None

            value = hash_cache.get(indicator)
            if value is None:
                self._misses += 1
                return None

            self._hits += 1
            return value

    def put(self, symbol: str, bars: pd.DataFrame, indicator: str, value: Any):
        """Cache an indicator value.

        Automatically invalidates stale entries (different bar_hash).
        """
        bh = _bar_hash(bars)
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}

            sym_cache = self._cache[symbol]

            # If bar_hash changed, the old indicators are stale — clear them
            if bh not in sym_cache:
                sym_cache.clear()
                sym_cache[bh] = {}

            sym_cache[bh][indicator] = value

            # Evict if too many symbols
            if len(self._cache) > self._max_size:
                self._evict_oldest()

    def get_or_compute(self, symbol: str, bars: pd.DataFrame,
                       indicator: str, compute_fn, *args, **kwargs) -> Any:
        """Get cached value or compute and cache it.

        Args:
            symbol: Ticker symbol.
            bars: Bar DataFrame (used for cache key).
            indicator: Indicator name (e.g., "rsi_14", "atr_14").
            compute_fn: Callable that returns the indicator value.
            *args, **kwargs: Passed to compute_fn.

        Returns:
            The indicator value (from cache or freshly computed).
        """
        cached = self.get(symbol, bars, indicator)
        if cached is not None:
            return cached

        try:
            value = compute_fn(*args, **kwargs)
            if value is not None:
                self.put(symbol, bars, indicator, value)
            return value
        except Exception as e:
            logger.debug("T4-003: Compute failed for %s/%s: %s", symbol, indicator, e)
            return None

    def invalidate(self, symbol: str | None = None):
        """Invalidate cache entries. If symbol is None, clear all."""
        with self._lock:
            if symbol is None:
                self._cache.clear()
            elif symbol in self._cache:
                del self._cache[symbol]

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total_indicators = sum(
                len(indicators)
                for sym_cache in self._cache.values()
                for indicators in sym_cache.values()
            )
            total = self._hits + self._misses
            return {
                "symbols_cached": len(self._cache),
                "total_indicators": total_indicators,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
            }

    def _evict_oldest(self):
        """Remove the oldest symbol entry. Must hold _lock."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]


# Module-level singleton accessor
indicator_cache = IndicatorCache.instance()
