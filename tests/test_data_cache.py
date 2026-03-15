"""Tests for V8 data caching layer."""

import time
import threading

import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class TestBarCache:
    """Tests for BarCache class."""

    def _make_cache(self, max_size=500):
        from data_cache import BarCache
        return BarCache(max_size=max_size)

    def _make_bars(self, n=10, symbol="AAPL"):
        """Create a minimal DataFrame resembling bar data."""
        dates = pd.date_range("2026-03-13 09:30", periods=n, freq="1min", tz=ET)
        return pd.DataFrame({
            "open": np.random.uniform(148, 152, n),
            "high": np.random.uniform(150, 154, n),
            "low": np.random.uniform(146, 150, n),
            "close": np.random.uniform(148, 152, n),
            "volume": np.random.randint(1000, 100000, n),
        }, index=dates)

    def test_put_and_get(self):
        """Basic put and get operations."""
        cache = self._make_cache()
        bars = self._make_bars()

        cache.put("AAPL", "1Min", bars)
        result = cache.get_bars("AAPL", "1Min")

        assert result is not None
        assert len(result) == len(bars)

    def test_cache_miss(self):
        """Get on empty cache returns None."""
        cache = self._make_cache()
        result = cache.get_bars("AAPL", "1Min")
        assert result is None

    def test_cache_hit_increments(self):
        """Hit counter increases on cache hits."""
        cache = self._make_cache()
        bars = self._make_bars()

        cache.put("AAPL", "1Min", bars)
        cache.get_bars("AAPL", "1Min")
        cache.get_bars("AAPL", "1Min")

        stats = cache.cache_stats()
        assert stats["hits"] == 2

    def test_cache_returns_copy(self):
        """Returned DataFrame should be a copy, not the original."""
        cache = self._make_cache()
        bars = self._make_bars()

        cache.put("AAPL", "1Min", bars)
        result1 = cache.get_bars("AAPL", "1Min")
        result2 = cache.get_bars("AAPL", "1Min")

        assert result1 is not result2  # Different objects

    def test_lru_eviction(self):
        """Oldest entries should be evicted when cache exceeds max size."""
        cache = self._make_cache(max_size=3)

        for sym in ["AAPL", "MSFT", "NVDA", "TSLA"]:
            cache.put(sym, "1Min", self._make_bars())

        stats = cache.cache_stats()
        assert stats["size"] == 3

        # AAPL should be evicted (oldest)
        result = cache.get_bars("AAPL", "1Min")
        assert result is None

    def test_invalidate_symbol(self):
        """Invalidate should remove all timeframes for a symbol."""
        cache = self._make_cache()
        cache.put("AAPL", "1Min", self._make_bars())
        cache.put("AAPL", "5Min", self._make_bars())

        cache.invalidate("AAPL")

        assert cache.get_bars("AAPL", "1Min") is None
        assert cache.get_bars("AAPL", "5Min") is None

    def test_invalidate_specific_timeframe(self):
        """Invalidate with timeframe should only remove that entry."""
        cache = self._make_cache()
        cache.put("AAPL", "1Min", self._make_bars())
        cache.put("AAPL", "5Min", self._make_bars())

        cache.invalidate("AAPL", "1Min")

        assert cache.get_bars("AAPL", "1Min") is None
        assert cache.get_bars("AAPL", "5Min") is not None

    def test_clear(self):
        """Clear should empty the cache and reset stats."""
        cache = self._make_cache()
        cache.put("AAPL", "1Min", self._make_bars())
        cache.put("MSFT", "1Min", self._make_bars())

        cache.clear()

        stats = cache.cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_stats(self):
        """Stats should report correct values."""
        cache = self._make_cache()
        bars = self._make_bars()

        cache.put("AAPL", "1Min", bars)
        cache.put("AAPL", "5Min", bars)
        cache.put("MSFT", "1Min", bars)

        cache.get_bars("AAPL", "1Min")  # hit
        cache.get_bars("NVDA", "1Min")  # miss

        stats = cache.cache_stats()
        assert stats["size"] == 3
        assert stats["symbols_cached"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_fetch_fn_called_on_miss(self):
        """fetch_fn should be called on cache miss."""
        cache = self._make_cache()
        bars = self._make_bars()

        call_count = 0
        def mock_fetch(**kwargs):
            nonlocal call_count
            call_count += 1
            return bars

        result = cache.get_bars("AAPL", "1Min", fetch_fn=mock_fetch)
        assert result is not None
        assert call_count == 1

        # Second call should use cache
        result2 = cache.get_bars("AAPL", "1Min", fetch_fn=mock_fetch)
        assert result2 is not None
        assert call_count == 1  # Not called again

    def test_fetch_fn_failure_returns_stale(self):
        """On fetch failure, stale cache data should be returned."""
        cache = self._make_cache()
        bars = self._make_bars()
        cache.put("AAPL", "1Min", bars)

        # Manually expire the entry
        cache._cache[("AAPL", "1Min")]["last_fetch"] = datetime(2020, 1, 1)

        def failing_fetch(**kwargs):
            raise ConnectionError("API down")

        result = cache.get_bars("AAPL", "1Min", fetch_fn=failing_fetch)
        assert result is not None  # Returns stale data

    def test_thread_safety(self):
        """Cache should be safe for concurrent access."""
        cache = self._make_cache()
        errors = []

        def writer(sym):
            try:
                for _ in range(50):
                    cache.put(sym, "1Min", self._make_bars(5))
            except Exception as e:
                errors.append(e)

        def reader(sym):
            try:
                for _ in range(50):
                    cache.get_bars(sym, "1Min")
            except Exception as e:
                errors.append(e)

        threads = []
        for sym in ["AAPL", "MSFT", "NVDA"]:
            threads.append(threading.Thread(target=writer, args=(sym,)))
            threads.append(threading.Thread(target=reader, args=(sym,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_different_timeframes_separate(self):
        """Same symbol with different timeframes should be separate cache entries."""
        cache = self._make_cache()
        bars_1m = self._make_bars(5)
        bars_5m = self._make_bars(10)

        cache.put("AAPL", "1Min", bars_1m)
        cache.put("AAPL", "5Min", bars_5m)

        result_1m = cache.get_bars("AAPL", "1Min")
        result_5m = cache.get_bars("AAPL", "5Min")

        assert len(result_1m) == 5
        assert len(result_5m) == 10


class TestBarCacheSingleton:
    """Test the module-level singleton."""

    def test_get_bar_cache_returns_same_instance(self):
        """get_bar_cache should return the same instance."""
        import data_cache
        # Reset singleton for test
        data_cache._bar_cache = None

        cache1 = data_cache.get_bar_cache()
        cache2 = data_cache.get_bar_cache()

        assert cache1 is cache2

        # Cleanup
        data_cache._bar_cache = None
