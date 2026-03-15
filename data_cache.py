"""V8: Data caching layer for bar data.

Caches bar data by symbol+timeframe to avoid refetching from Alpaca
every 2-minute scan cycle for 130+ symbols.
"""

import logging
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

import pandas as pd

import config

logger = logging.getLogger(__name__)


class BarCache:
    """Thread-safe LRU cache for bar data.

    Caches bars by (symbol, timeframe) key with configurable TTL per timeframe.
    Supports delta fetching — only requests bars newer than last cached bar.
    """

    # TTL per timeframe in seconds
    DEFAULT_TTL = {
        "1Min": 60,
        "2Min": 60,
        "5Min": 120,
        "15Min": 300,
        "1Hour": 600,
        "1Day": 3600,
    }

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame | None = None,
                 fetch_fn=None, **fetch_kwargs) -> pd.DataFrame | None:
        """Get cached bars or fetch fresh ones.

        Args:
            symbol: Ticker symbol
            timeframe: String representation (e.g., "1Min", "5Min", "1Day")
            bars: If provided, store these bars directly (bypass fetch)
            fetch_fn: Callable to fetch bars if cache miss. Signature: fetch_fn(**fetch_kwargs) -> DataFrame
            **fetch_kwargs: Passed to fetch_fn

        Returns:
            DataFrame of bars, or None if unavailable
        """
        key = (symbol, timeframe)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                ttl = self.DEFAULT_TTL.get(timeframe, 120)
                age = (datetime.now() - entry["last_fetch"]).total_seconds()

                if age < ttl:
                    # Cache hit — move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return entry["bars"].copy() if entry["bars"] is not None else None

        # Cache miss or stale
        self._misses += 1

        if bars is not None:
            # Caller provided bars directly
            result = bars
        elif fetch_fn is not None:
            try:
                result = fetch_fn(**fetch_kwargs)
            except Exception as e:
                logger.debug(f"Cache fetch failed for {symbol}/{timeframe}: {e}")
                # Return stale data if available
                with self._lock:
                    if key in self._cache:
                        return self._cache[key]["bars"].copy()
                return None
        else:
            return None

        # Store in cache
        with self._lock:
            self._cache[key] = {
                "bars": result,
                "last_fetch": datetime.now(),
            }
            self._cache.move_to_end(key)

            # Evict oldest entries if over max size
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return result.copy() if result is not None else None

    def put(self, symbol: str, timeframe: str, bars: pd.DataFrame):
        """Directly store bars in cache."""
        key = (symbol, timeframe)
        with self._lock:
            self._cache[key] = {
                "bars": bars,
                "last_fetch": datetime.now(),
            }
            self._cache.move_to_end(key)

            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, symbol: str, timeframe: str | None = None):
        """Remove cached data for a symbol (optionally for specific timeframe)."""
        with self._lock:
            if timeframe:
                self._cache.pop((symbol, timeframe), None)
            else:
                keys_to_remove = [k for k in self._cache if k[0] == symbol]
                for k in keys_to_remove:
                    del self._cache[k]

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "symbols_cached": len(set(k[0] for k in self._cache)),
            }


# Module-level singleton
_bar_cache: BarCache | None = None


def get_bar_cache() -> BarCache:
    """Get or create the global bar cache singleton."""
    global _bar_cache
    if _bar_cache is None:
        _bar_cache = BarCache(max_size=500)
    return _bar_cache
