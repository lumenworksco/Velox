"""Tests for V8 data quality checks."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class TestCheckBarQuality:

    def _make_bars(self, n=100, start_price=100.0):
        dates = pd.date_range("2026-03-13 09:30", periods=n, freq="1min", tz=ET)
        close = start_price + np.cumsum(np.random.normal(0, 0.1, n))
        return pd.DataFrame({
            "open": close - 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(1000, 100000, n),
        }, index=dates)

    def test_clean_data(self):
        from data_quality import check_bar_quality
        bars = self._make_bars()
        result = check_bar_quality(bars, "AAPL")
        assert result.is_clean
        assert len(result.issues) == 0

    def test_empty_data(self):
        from data_quality import check_bar_quality
        result = check_bar_quality(pd.DataFrame(), "AAPL")
        assert not result.is_clean

    def test_none_data(self):
        from data_quality import check_bar_quality
        result = check_bar_quality(None, "AAPL")
        assert not result.is_clean

    def test_insufficient_bars(self):
        from data_quality import check_bar_quality
        bars = self._make_bars(n=10)
        result = check_bar_quality(bars, "AAPL", min_bars=50)
        assert not result.is_clean
        assert any("Only 10 bars" in i for i in result.issues)

    def test_price_anomaly(self):
        from data_quality import check_bar_quality
        bars = self._make_bars()
        bars.iloc[50, bars.columns.get_loc("close")] = 200.0  # 100% jump
        result = check_bar_quality(bars, "AAPL")
        assert not result.is_clean
        assert any("Price anomaly" in i for i in result.issues)

    def test_zero_volume_halt(self):
        from data_quality import check_bar_quality
        bars = self._make_bars()
        bars.iloc[10:30, bars.columns.get_loc("volume")] = 0  # 20% zero volume
        result = check_bar_quality(bars, "AAPL")
        assert not result.is_clean
        assert any("zero-volume" in i for i in result.issues)

    def test_ohlc_consistency(self):
        from data_quality import check_bar_quality
        bars = self._make_bars()
        # Make high lower than close (invalid)
        bars.iloc[5, bars.columns.get_loc("high")] = bars.iloc[5]["close"] - 5
        result = check_bar_quality(bars, "AAPL")
        assert not result.is_clean
        assert any("OHLC inconsistency" in i for i in result.issues)

    def test_duplicate_timestamps(self):
        from data_quality import check_bar_quality
        bars = self._make_bars()
        # Add duplicate timestamp
        bars.index = bars.index[:99].append(bars.index[98:99])
        result = check_bar_quality(bars, "AAPL")
        assert not result.is_clean
        assert any("duplicate" in i for i in result.issues)

    def test_add_issue_method(self):
        from data_quality import DataQualityResult
        r = DataQualityResult(is_clean=True)
        r.add_issue("test issue")
        assert not r.is_clean
        assert "test issue" in r.issues
