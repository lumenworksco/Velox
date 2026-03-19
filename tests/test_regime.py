"""Tests for strategies/regime.py — MarketRegime detection."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spy_bars(n: int, close_prices: list[float]) -> pd.DataFrame:
    """Return a minimal DataFrame that looks like what get_daily_bars returns."""
    assert len(close_prices) == n
    return pd.DataFrame({"close": close_prices}, index=range(n))


def _make_regime():
    """Return a fresh MarketRegime instance."""
    from strategies.regime import MarketRegime
    return MarketRegime()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketRegimeInit:
    """Tests for MarketRegime initial state."""

    def test_initial_regime_is_unknown(self):
        """A freshly created MarketRegime has regime='UNKNOWN'."""
        regime = _make_regime()
        assert regime.regime == "UNKNOWN"

    def test_initial_last_check_is_none(self):
        """last_check is None before any update call."""
        regime = _make_regime()
        assert regime.last_check is None

    def test_initial_prices_are_zero(self):
        """spy_price and spy_ema are 0.0 before any data fetch."""
        regime = _make_regime()
        assert regime.spy_price == 0.0
        assert regime.spy_ema == 0.0


class TestMarketRegimeUpdate:
    """Tests for MarketRegime.update() — regime classification."""

    def test_returns_bullish_when_price_above_ema(self):
        """Regime is BULLISH when the last SPY close is above the 20-day EMA."""
        # Build 25 bars of steadily rising prices so EMA < last close
        closes = list(range(100, 125))          # 100..124 — EMA ≈ 112, last=124
        df = _make_spy_bars(25, closes)

        with patch("strategies.regime.get_daily_bars", return_value=df), \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result == "BULLISH"
        assert regime.regime == "BULLISH"
        assert regime.spy_price > regime.spy_ema

    def test_returns_bearish_when_price_below_ema(self):
        """Regime is BEARISH when the last SPY close is below the 20-day EMA."""
        # Build 25 bars of steadily falling prices so EMA > last close
        closes = list(range(124, 99, -1))       # 124..100 — EMA ≈ 112, last=100
        df = _make_spy_bars(25, closes)

        with patch("strategies.regime.get_daily_bars", return_value=df), \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result == "BEARISH"
        assert regime.regime == "BEARISH"
        assert regime.spy_price < regime.spy_ema

    def test_last_check_is_set_after_update(self):
        """last_check is populated after a successful update."""
        closes = list(range(100, 125))
        df = _make_spy_bars(25, closes)
        now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            regime.update(now)

        assert regime.last_check == now

    def test_regime_cached_within_interval(self):
        """update() does NOT re-fetch data if called within REGIME_CHECK_INTERVAL_MIN."""
        closes = list(range(100, 125))
        df = _make_spy_bars(25, closes)
        now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)

        with patch("strategies.regime.get_daily_bars", return_value=df) as mock_fetch, \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime = _make_regime()
            regime.update(now)
            # Call again 1 minute later — should use cache, no second fetch
            regime.update(now + timedelta(minutes=1))

        # get_daily_bars should only be called once
        assert mock_fetch.call_count == 1

    def test_regime_refreshed_after_interval(self):
        """update() re-fetches after the cache interval has elapsed."""
        closes = list(range(100, 125))
        df = _make_spy_bars(25, closes)
        now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)

        with patch("strategies.regime.get_daily_bars", return_value=df) as mock_fetch, \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime = _make_regime()
            regime.update(now)
            # Call again well after the interval
            regime.update(now + timedelta(hours=2))

        assert mock_fetch.call_count == 2

    def test_returns_unknown_on_empty_dataframe(self):
        """V10: Empty DataFrame falls back to UNKNOWN (conservative default)."""
        with patch("strategies.regime.get_daily_bars", return_value=pd.DataFrame()):
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result == "UNKNOWN"

    def test_returns_unknown_on_insufficient_bars(self):
        """V10: Fewer bars than EMA period falls back to UNKNOWN."""
        # Only 5 bars, but EMA period is 20
        df = _make_spy_bars(5, [500.0] * 5)

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result == "UNKNOWN"

    def test_returns_unknown_on_data_fetch_exception(self):
        """V10: If get_daily_bars raises, regime stays UNKNOWN (conservative)."""
        with patch("strategies.regime.get_daily_bars", side_effect=RuntimeError("network error")):
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result == "UNKNOWN"

    def test_preserves_previous_regime_on_exception(self):
        """If regime was previously set and then an error occurs, it is preserved."""
        closes = list(range(100, 125))
        df = _make_spy_bars(25, closes)
        now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)

        # First successful update → BULLISH
        with patch("strategies.regime.get_daily_bars", return_value=df), \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime = _make_regime()
            regime.update(now)

        assert regime.regime == "BULLISH"

        # Second update fails — should retain BULLISH (not flip to UNKNOWN)
        with patch("strategies.regime.get_daily_bars", side_effect=RuntimeError("timeout")), \
             patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 30
            mock_config.REGIME_EMA_PERIOD = 20
            regime.last_check = None   # Force cache expiry
            result = regime.update(now + timedelta(hours=1))

        assert result == "BULLISH"

    def test_valid_regime_values(self):
        """update() always returns one of the valid regime strings."""
        valid_regimes = {"BULLISH", "BEARISH", "UNKNOWN"}
        closes = list(range(100, 125))
        df = _make_spy_bars(25, closes)

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            result = regime.update(datetime(2026, 3, 15, 10, 0, tzinfo=ET))

        assert result in valid_regimes


class TestIsSpyPositiveToday:
    """Tests for MarketRegime.is_spy_positive_today()."""

    def test_positive_when_close_rises(self):
        """Returns True when today's close is higher than yesterday's."""
        df = _make_spy_bars(2, [500.0, 505.0])  # yesterday=500, today=505

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            assert regime.is_spy_positive_today() == True  # noqa: E712 — numpy bool compat

    def test_negative_when_close_falls(self):
        """Returns False when today's close is lower than yesterday's."""
        df = _make_spy_bars(2, [505.0, 500.0])  # yesterday=505, today=500

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            assert regime.is_spy_positive_today() == False  # noqa: E712 — numpy bool compat

    def test_returns_false_on_empty_dataframe(self):
        """Returns False when data is empty."""
        with patch("strategies.regime.get_daily_bars", return_value=pd.DataFrame()):
            regime = _make_regime()
            assert regime.is_spy_positive_today() is False

    def test_returns_false_on_single_bar(self):
        """Returns False when there is only one bar (cannot compare)."""
        df = _make_spy_bars(1, [500.0])

        with patch("strategies.regime.get_daily_bars", return_value=df):
            regime = _make_regime()
            assert regime.is_spy_positive_today() is False

    def test_returns_false_on_exception(self):
        """Returns False gracefully when get_daily_bars raises."""
        with patch("strategies.regime.get_daily_bars", side_effect=Exception("error")):
            regime = _make_regime()
            assert regime.is_spy_positive_today() is False
