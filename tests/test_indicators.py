"""Tests for analytics/indicators.py — shared technical indicator functions."""

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame suitable for indicator functions."""
    assert len(highs) == len(lows) == len(closes) == len(volumes)
    return pd.DataFrame({
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _uniform_bars(n: int = 5, price: float = 100.0, volume: float = 1_000.0) -> pd.DataFrame:
    """Return n bars where high=price+0.5, low=price-0.5, close=price, vol=volume.

    Typical price = (h+l+c)/3 = (100.5 + 99.5 + 100) / 3 = 100.0 exactly.
    VWAP = (typical * vol * n) / (vol * n) = 100.0
    """
    h = price + 0.5
    lo = price - 0.5
    return _make_bars(
        highs=[h] * n,
        lows=[lo] * n,
        closes=[price] * n,
        volumes=[volume] * n,
    )


# ---------------------------------------------------------------------------
# compute_vwap
# ---------------------------------------------------------------------------

class TestComputeVwap:
    """Tests for analytics.indicators.compute_vwap."""

    def test_returns_float_for_valid_bars(self):
        """compute_vwap returns a float for well-formed bars."""
        from analytics.indicators import compute_vwap
        bars = _uniform_bars(n=10)
        result = compute_vwap(bars)
        assert isinstance(result, float)

    def test_known_value_uniform_bars(self):
        """VWAP equals the typical price when all bars are identical."""
        from analytics.indicators import compute_vwap
        bars = _uniform_bars(n=5, price=200.0, volume=500.0)
        # typical = (200.5 + 199.5 + 200) / 3 = 200.0
        result = compute_vwap(bars)
        assert result == pytest.approx(200.0, abs=1e-9)

    def test_volume_weighted(self):
        """Bars with higher volume pull VWAP toward their typical price."""
        from analytics.indicators import compute_vwap
        # Bar 1: typical=100, vol=1  → contribution 100
        # Bar 2: typical=200, vol=9  → contribution 1800
        # VWAP = (100 + 1800) / (1 + 9) = 190.0
        bars = _make_bars(
            highs=[100.5, 200.5],
            lows=[99.5,  199.5],
            closes=[100.0, 200.0],
            volumes=[1.0,   9.0],
        )
        result = compute_vwap(bars)
        assert result == pytest.approx(190.0, abs=1e-9)

    def test_returns_none_for_empty_dataframe(self):
        """compute_vwap returns None for an empty DataFrame."""
        from analytics.indicators import compute_vwap
        result = compute_vwap(pd.DataFrame())
        assert result is None

    def test_returns_none_for_all_zero_volume(self):
        """compute_vwap returns None when all volumes are 0."""
        from analytics.indicators import compute_vwap
        bars = _make_bars(
            highs=[101.0, 102.0],
            lows=[99.0,  98.0],
            closes=[100.0, 100.0],
            volumes=[0.0,   0.0],
        )
        result = compute_vwap(bars)
        assert result is None

    def test_returns_none_when_volume_column_missing(self):
        """compute_vwap returns None when the 'volume' column is absent."""
        from analytics.indicators import compute_vwap
        bars = pd.DataFrame({"high": [101.0], "low": [99.0], "close": [100.0]})
        result = compute_vwap(bars)
        assert result is None

    def test_single_bar(self):
        """With a single bar, VWAP equals the typical price of that bar."""
        from analytics.indicators import compute_vwap
        bars = _make_bars(
            highs=[102.0], lows=[98.0], closes=[100.0], volumes=[1000.0]
        )
        # typical = (102 + 98 + 100) / 3 = 100.0
        result = compute_vwap(bars)
        assert result == pytest.approx(100.0, abs=1e-9)

    def test_result_is_finite(self):
        """VWAP result must be a finite number (no NaN, no Inf)."""
        from analytics.indicators import compute_vwap
        bars = _uniform_bars(n=20, price=350.0, volume=2000.0)
        result = compute_vwap(bars)
        assert result is not None
        assert math.isfinite(result)

    def test_vwap_between_low_and_high(self):
        """VWAP must fall between the overall session low and high."""
        from analytics.indicators import compute_vwap
        import random
        random.seed(42)
        highs  = [100.0 + random.uniform(0, 5) for _ in range(30)]
        lows   = [100.0 - random.uniform(0, 5) for _ in range(30)]
        closes = [100.0 + random.uniform(-2, 2) for _ in range(30)]
        vols   = [random.uniform(500, 5000) for _ in range(30)]
        bars = _make_bars(highs, lows, closes, vols)
        result = compute_vwap(bars)
        assert result is not None
        assert min(lows) <= result <= max(highs)


# ---------------------------------------------------------------------------
# compute_vwap_bands
# ---------------------------------------------------------------------------

class TestComputeVwapBands:
    """Tests for analytics.indicators.compute_vwap_bands."""

    def test_returns_three_tuple_for_valid_bars(self):
        """compute_vwap_bands returns a 3-tuple of floats."""
        from analytics.indicators import compute_vwap_bands
        bars = _uniform_bars(n=10, price=150.0)
        result = compute_vwap_bands(bars)
        assert result is not None
        assert len(result) == 3
        vwap, upper, lower = result
        assert isinstance(vwap, float)
        assert isinstance(upper, float)
        assert isinstance(lower, float)

    def test_upper_greater_than_vwap_greater_than_lower(self):
        """upper_band > vwap > lower_band for typical bars with non-zero variance."""
        from analytics.indicators import compute_vwap_bands
        # Use bars with price variation so variance > 0
        bars = _make_bars(
            highs=[102.0, 105.0, 108.0, 103.0, 101.0],
            lows=[98.0,  100.0, 104.0, 99.0,  97.0],
            closes=[100.0, 102.0, 106.0, 101.0, 99.0],
            volumes=[1000.0] * 5,
        )
        result = compute_vwap_bands(bars, std_mult=2.0)
        assert result is not None
        vwap, upper, lower = result
        assert upper > vwap
        assert vwap > lower

    def test_std_mult_zero_returns_equal_bands(self):
        """With std_mult=0, upper == vwap == lower."""
        from analytics.indicators import compute_vwap_bands
        bars = _uniform_bars(n=10, price=100.0)
        result = compute_vwap_bands(bars, std_mult=0.0)
        assert result is not None
        vwap, upper, lower = result
        assert upper == pytest.approx(vwap, abs=1e-9)
        assert lower == pytest.approx(vwap, abs=1e-9)

    def test_returns_none_for_empty_dataframe(self):
        """compute_vwap_bands returns None for an empty DataFrame."""
        from analytics.indicators import compute_vwap_bands
        result = compute_vwap_bands(pd.DataFrame())
        assert result is None

    def test_returns_none_for_all_zero_volume(self):
        """compute_vwap_bands returns None when all volume is zero."""
        from analytics.indicators import compute_vwap_bands
        bars = _make_bars(
            highs=[101.0, 102.0],
            lows=[99.0,  98.0],
            closes=[100.0, 100.0],
            volumes=[0.0,   0.0],
        )
        result = compute_vwap_bands(bars)
        assert result is None

    def test_returns_none_when_volume_column_missing(self):
        """compute_vwap_bands returns None when 'volume' column is absent."""
        from analytics.indicators import compute_vwap_bands
        bars = pd.DataFrame({"high": [101.0], "low": [99.0], "close": [100.0]})
        result = compute_vwap_bands(bars)
        assert result is None

    def test_vwap_consistent_with_compute_vwap(self):
        """The vwap from compute_vwap_bands matches compute_vwap."""
        from analytics.indicators import compute_vwap, compute_vwap_bands
        bars = _make_bars(
            highs=[103.0, 104.0, 102.0, 105.0],
            lows=[99.0,  100.0, 98.0,  101.0],
            closes=[101.0, 102.0, 100.0, 103.0],
            volumes=[800.0, 1200.0, 600.0, 900.0],
        )
        vwap_standalone = compute_vwap(bars)
        bands_result = compute_vwap_bands(bars, std_mult=2.0)
        assert bands_result is not None
        vwap_from_bands = bands_result[0]
        assert vwap_from_bands == pytest.approx(vwap_standalone, rel=1e-9)

    def test_wider_multiplier_widens_bands(self):
        """A larger std_mult produces wider bands."""
        from analytics.indicators import compute_vwap_bands
        bars = _make_bars(
            highs=[105.0, 108.0, 103.0],
            lows=[95.0,  99.0,  97.0],
            closes=[100.0, 103.0, 100.0],
            volumes=[1000.0] * 3,
        )
        r1 = compute_vwap_bands(bars, std_mult=1.0)
        r2 = compute_vwap_bands(bars, std_mult=2.0)
        assert r1 is not None and r2 is not None
        _, upper1, lower1 = r1
        _, upper2, lower2 = r2
        assert upper2 >= upper1
        assert lower2 <= lower1

    def test_all_results_are_finite(self):
        """All three returned values must be finite numbers."""
        from analytics.indicators import compute_vwap_bands
        bars = _uniform_bars(n=15, price=500.0, volume=10_000.0)
        result = compute_vwap_bands(bars, std_mult=2.0)
        assert result is not None
        for val in result:
            assert math.isfinite(val), f"Non-finite value returned: {val}"

    def test_single_bar_uniform_produces_equal_bands(self):
        """A single bar where typical price is constant → std_dev=0 → equal bands."""
        from analytics.indicators import compute_vwap_bands
        bars = _make_bars(
            highs=[102.0], lows=[98.0], closes=[100.0], volumes=[1000.0]
        )
        result = compute_vwap_bands(bars, std_mult=2.0)
        assert result is not None
        vwap, upper, lower = result
        # With only one point, variance=0, so std_dev=0 and all three are equal
        assert upper == pytest.approx(vwap, abs=1e-9)
        assert lower == pytest.approx(vwap, abs=1e-9)
