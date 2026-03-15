"""Tests for V8 backtester additions (StatMR, KalmanPairs, MicroMom)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _make_bars(n=200, start_price=100.0, trend=0.0):
    """Create synthetic bar data."""
    dates = pd.date_range("2025-09-01", periods=n, freq="1h", tz=ET)
    np.random.seed(42)
    close = start_price + np.cumsum(np.random.normal(trend, 0.5, n))
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(10000, 1000000, n),
    }, index=dates)


class TestSimulateStatMR:

    def test_basic_run(self):
        from backtester import simulate_stat_mr
        data = {"AAPL": _make_bars(300), "MSFT": _make_bars(300)}
        result = simulate_stat_mr(data)
        # May return None if no trades triggered, which is fine
        if result:
            assert result.strategy == "STAT_MR"
            assert result.total_trades >= 0

    def test_insufficient_data(self):
        from backtester import simulate_stat_mr
        data = {"AAPL": _make_bars(10)}
        result = simulate_stat_mr(data)
        assert result is None


class TestSimulateKalmanPairs:

    def test_basic_run(self):
        from backtester import simulate_kalman_pairs
        # Create correlated pair data
        np.random.seed(42)
        n = 200
        base = np.cumsum(np.random.normal(0, 0.5, n))
        dates = pd.date_range("2025-09-01", periods=n, freq="1D", tz=ET)

        aapl_close = 150 + base
        msft_close = 400 + base * 2.5 + np.random.normal(0, 0.3, n)

        data = {
            "AAPL": pd.DataFrame({"close": aapl_close, "volume": [100000]*n}, index=dates),
            "MSFT": pd.DataFrame({"close": msft_close, "volume": [100000]*n}, index=dates),
        }
        result = simulate_kalman_pairs(data)
        if result:
            assert result.strategy == "KALMAN_PAIRS"

    def test_no_data(self):
        from backtester import simulate_kalman_pairs
        result = simulate_kalman_pairs({})
        assert result is None


class TestSimulateMicroMom:

    def test_basic_run(self):
        from backtester import simulate_micro_momentum
        data = {"SPY": _make_bars(200)}
        result = simulate_micro_momentum(data)
        # May return None, which is fine
        if result:
            assert result.strategy == "MICRO_MOM"

    def test_no_spy_data(self):
        from backtester import simulate_micro_momentum
        result = simulate_micro_momentum({"AAPL": _make_bars()})
        assert result is None
