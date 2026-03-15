"""Tests for replay.py — MarketReplay, ReplayResult, ComparisonResult."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


def _make_bars_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame mimicking Alpaca minute bars."""
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    return df


# ============================================================
# Dataclass creation tests
# ============================================================

class TestReplayResult:
    """Test ReplayResult dataclass."""

    def test_default_creation(self):
        from replay import ReplayResult
        r = ReplayResult(date="2026-03-10")
        assert r.date == "2026-03-10"
        assert r.signals == []
        assert r.trades == []
        assert r.total_pnl == 0.0
        assert r.win_rate == 0.0
        assert r.sharpe == 0.0
        assert r.config_used == {}

    def test_creation_with_values(self):
        from replay import ReplayResult
        r = ReplayResult(
            date="2026-03-10",
            signals=[{"symbol": "AAPL"}],
            trades=[{"pnl": 50.0}],
            total_pnl=50.0,
            win_rate=1.0,
            sharpe=2.5,
            config_used={"ORB_ENABLED": False},
        )
        assert r.total_pnl == 50.0
        assert r.win_rate == 1.0
        assert len(r.signals) == 1
        assert r.config_used["ORB_ENABLED"] is False


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_default_creation(self):
        from replay import ComparisonResult
        c = ComparisonResult(date="2026-03-10")
        assert c.date == "2026-03-10"
        assert c.delta_pnl == 0.0
        assert c.delta_sharpe == 0.0
        assert c.delta_win_rate == 0.0

    def test_creation_with_results(self):
        from replay import ReplayResult, ComparisonResult
        ra = ReplayResult(date="2026-03-10", total_pnl=100.0, sharpe=1.5, win_rate=0.6)
        rb = ReplayResult(date="2026-03-10", total_pnl=50.0, sharpe=0.8, win_rate=0.4)
        c = ComparisonResult(
            date="2026-03-10",
            result_a=ra,
            result_b=rb,
            delta_pnl=50.0,
            delta_sharpe=0.7,
            delta_win_rate=0.2,
        )
        assert c.delta_pnl == 50.0
        assert c.delta_sharpe == pytest.approx(0.7)
        assert c.delta_win_rate == pytest.approx(0.2)


# ============================================================
# MarketReplay tests
# ============================================================

class TestMarketReplay:
    """Test MarketReplay with mocked data."""

    def _make_replay(self):
        from replay import MarketReplay
        return MarketReplay()

    def test_replay_day_no_data(self):
        """Replay with no data returns empty result."""
        replay = self._make_replay()
        with patch("replay.MarketReplay._load_bars", return_value={}):
            result = replay.replay_day("2026-03-10")
        assert result.date == "2026-03-10"
        assert result.signals == []
        assert result.trades == []
        assert result.total_pnl == 0.0

    def test_replay_day_with_mock_bars(self):
        """Replay with mocked bar data generates signals and trades."""
        replay = self._make_replay()
        # Build proper minute-bar DataFrames spanning the ORB window and beyond
        rows = []
        base = datetime(2026, 3, 10, 9, 30, tzinfo=ET)
        price = 150.0
        for i in range(390):  # 6.5 hours of 1-min bars
            t = base + pd.Timedelta(minutes=i)
            # Create a slow uptrend for ORB breakout
            p = price + i * 0.01
            rows.append({
                "timestamp": t,
                "open": p - 0.05,
                "high": p + 0.10,
                "low": p - 0.10,
                "close": p,
                "volume": 50000.0,
            })
        mock_bars = {"AAPL": _make_bars_df(rows)}
        with patch.object(replay, "_load_bars", return_value=mock_bars):
            result = replay.replay_day("2026-03-10")
        assert result.date == "2026-03-10"
        assert isinstance(result.signals, list)
        assert isinstance(result.trades, list)

    def test_replay_day_config_overrides(self):
        """Config overrides are applied and restored."""
        import config as cfg
        replay = self._make_replay()
        original_value = cfg.ORB_ENABLED

        with patch.object(replay, "_load_bars", return_value={}):
            result = replay.replay_day("2026-03-10", {"ORB_ENABLED": not original_value})

        # Config should be restored
        assert cfg.ORB_ENABLED == original_value
        assert result.config_used == {"ORB_ENABLED": not original_value}

    def test_replay_day_disabled(self):
        """Replay returns empty when REPLAY_ENABLED is False."""
        import config as cfg
        replay = self._make_replay()
        original = getattr(cfg, "REPLAY_ENABLED", True)
        try:
            cfg.REPLAY_ENABLED = False
            result = replay.replay_day("2026-03-10")
            assert result.signals == []
            assert result.trades == []
        finally:
            cfg.REPLAY_ENABLED = original

    def test_replay_fail_open_on_signal_error(self):
        """Signal generation errors don't crash the replay."""
        replay = self._make_replay()
        # Provide valid DataFrame bars so the scan loop runs
        rows = [{"timestamp": datetime(2026, 3, 10, 10, 0, tzinfo=ET),
                 "open": 150.0, "high": 152.0, "low": 149.0,
                 "close": 151.0, "volume": 500000.0}]
        mock_bars = {"AAPL": _make_bars_df(rows)}
        with patch.object(replay, "_load_bars", return_value=mock_bars), \
             patch.object(replay, "_generate_signals", side_effect=Exception("boom")):
            result = replay.replay_day("2026-03-10")
        assert result.date == "2026-03-10"

    def test_replay_fail_open_on_load_error(self):
        """Load errors don't crash the replay."""
        replay = self._make_replay()
        with patch.object(replay, "_load_bars", side_effect=Exception("db error")):
            result = replay.replay_day("2026-03-10")
        assert result.date == "2026-03-10"
        assert result.total_pnl == 0.0

    def test_compare_configs(self):
        """compare_configs runs both configs and computes deltas."""
        from replay import ReplayResult
        replay = self._make_replay()

        result_a = ReplayResult(date="2026-03-10", total_pnl=100.0, sharpe=1.5, win_rate=0.6)
        result_b = ReplayResult(date="2026-03-10", total_pnl=50.0, sharpe=0.8, win_rate=0.4)

        with patch.object(replay, "replay_day", side_effect=[result_a, result_b]):
            comparison = replay.compare_configs(
                "2026-03-10",
                {"ORB_ENABLED": True},
                {"ORB_ENABLED": False},
            )

        assert comparison.delta_pnl == pytest.approx(50.0)
        assert comparison.delta_sharpe == pytest.approx(0.7)
        assert comparison.delta_win_rate == pytest.approx(0.2)

    def test_compare_configs_fail_open(self):
        """compare_configs doesn't crash on replay errors."""
        replay = self._make_replay()
        with patch.object(replay, "replay_day", side_effect=Exception("crash")):
            comparison = replay.compare_configs("2026-03-10", {}, {})
        assert comparison.date == "2026-03-10"
        assert comparison.delta_pnl == 0.0


# ============================================================
# Helper method tests
# ============================================================

class TestReplayHelpers:
    """Test internal helper methods."""

    def _make_replay(self):
        from replay import MarketReplay
        return MarketReplay()

    def test_calc_win_rate_empty(self):
        from replay import MarketReplay
        assert MarketReplay._calc_win_rate([]) == 0.0

    def test_calc_win_rate(self):
        from replay import MarketReplay
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        assert MarketReplay._calc_win_rate(trades) == pytest.approx(2 / 3)

    def test_calc_sharpe_empty(self):
        from replay import MarketReplay
        assert MarketReplay._calc_sharpe([]) == 0.0

    def test_calc_sharpe_single(self):
        from replay import MarketReplay
        assert MarketReplay._calc_sharpe([{"pnl": 10.0}]) == 0.0

    def test_calc_sharpe_multiple(self):
        from replay import MarketReplay
        trades = [{"pnl": 10.0}, {"pnl": 10.0}]  # zero variance
        assert MarketReplay._calc_sharpe(trades) == 0.0

        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        sharpe = MarketReplay._calc_sharpe(trades)
        assert sharpe > 0  # positive mean, some variance

    def test_calc_pnl_buy(self):
        from replay import MarketReplay, _SimTrade
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=100.0, take_profit=110.0, stop_loss=95.0,
            exit_price=110.0,
        )
        assert MarketReplay._calc_pnl(trade) == 10.0

    def test_calc_pnl_sell(self):
        from replay import MarketReplay, _SimTrade
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="sell",
            entry_price=100.0, take_profit=90.0, stop_loss=105.0,
            exit_price=90.0,
        )
        assert MarketReplay._calc_pnl(trade) == 10.0

    def test_calc_pnl_no_exit(self):
        from replay import MarketReplay, _SimTrade
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=100.0, take_profit=110.0, stop_loss=95.0,
        )
        assert MarketReplay._calc_pnl(trade) == 0.0

    def test_generate_scan_times(self):
        replay = self._make_replay()
        times = replay._generate_scan_times("2026-03-10")
        assert len(times) > 0
        assert times[0].hour == 9
        assert times[0].minute == 30
        assert times[-1].hour <= 16

    def test_generate_scan_times_invalid_date(self):
        replay = self._make_replay()
        times = replay._generate_scan_times("not-a-date")
        assert times == []

    def test_apply_and_restore_overrides(self):
        import config as cfg
        replay = self._make_replay()
        original = cfg.ORB_ENABLED

        originals = replay._apply_overrides({"ORB_ENABLED": not original})
        assert cfg.ORB_ENABLED == (not original)

        replay._restore_overrides(originals)
        assert cfg.ORB_ENABLED == original

    def test_apply_overrides_unknown_key(self):
        replay = self._make_replay()
        originals = replay._apply_overrides({"NONEXISTENT_KEY_12345": True})
        assert originals == {}

    def test_trade_to_dict(self):
        from replay import MarketReplay, _SimTrade
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, take_profit=155.0, stop_loss=148.0,
            entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
            exit_price=155.0,
            exit_time=datetime(2026, 3, 10, 11, 0, tzinfo=ET),
            exit_reason="take_profit",
            pnl=5.0,
        )
        d = MarketReplay._trade_to_dict(trade)
        assert d["symbol"] == "AAPL"
        assert d["pnl"] == 5.0
        assert d["exit_reason"] == "take_profit"
        assert "entry_time" in d
        assert "exit_time" in d

    def test_check_exits_take_profit_buy(self):
        """TP hit when bar high >= take_profit."""
        from replay import MarketReplay, _SimTrade
        replay = MarketReplay()
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=100.0, take_profit=105.0, stop_loss=95.0,
            entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
        )
        open_trades = {"AAPL": trade}
        bars = {"AAPL": _make_bars_df([{
            "timestamp": datetime(2026, 3, 10, 10, 30, tzinfo=ET),
            "open": 104.0, "high": 106.0, "low": 103.0,
            "close": 105.5, "volume": 100000.0,
        }])}
        sim_time = datetime(2026, 3, 10, 10, 30, tzinfo=ET)

        closed = replay._check_exits(sim_time, open_trades, bars)
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"

    def test_check_exits_stop_loss_buy(self):
        """SL hit when bar low <= stop_loss."""
        from replay import MarketReplay, _SimTrade
        replay = MarketReplay()
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=100.0, take_profit=105.0, stop_loss=95.0,
            entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
        )
        open_trades = {"AAPL": trade}
        bars = {"AAPL": _make_bars_df([{
            "timestamp": datetime(2026, 3, 10, 10, 30, tzinfo=ET),
            "open": 96.0, "high": 97.0, "low": 94.0,
            "close": 94.5, "volume": 100000.0,
        }])}
        sim_time = datetime(2026, 3, 10, 10, 30, tzinfo=ET)

        closed = replay._check_exits(sim_time, open_trades, bars)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"

    def test_check_exits_no_trigger(self):
        """No exit when price stays between SL and TP."""
        from replay import MarketReplay, _SimTrade
        replay = MarketReplay()
        trade = _SimTrade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=100.0, take_profit=105.0, stop_loss=95.0,
            entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
        )
        open_trades = {"AAPL": trade}
        bars = {"AAPL": _make_bars_df([{
            "timestamp": datetime(2026, 3, 10, 10, 30, tzinfo=ET),
            "open": 100.5, "high": 102.0, "low": 99.0,
            "close": 101.0, "volume": 100000.0,
        }])}
        sim_time = datetime(2026, 3, 10, 10, 30, tzinfo=ET)

        closed = replay._check_exits(sim_time, open_trades, bars)
        assert len(closed) == 0

    def test_compute_vwap(self):
        """VWAP computation returns (vwap, upper, lower)."""
        from replay import MarketReplay
        df = _make_bars_df([
            {"timestamp": datetime(2026, 3, 10, 10, 0, tzinfo=ET),
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10000.0},
            {"timestamp": datetime(2026, 3, 10, 10, 1, tzinfo=ET),
             "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 15000.0},
            {"timestamp": datetime(2026, 3, 10, 10, 2, tzinfo=ET),
             "open": 101.0, "high": 101.5, "low": 99.5, "close": 100.0, "volume": 12000.0},
        ])
        result = MarketReplay._compute_vwap(df)
        assert result is not None
        vwap, upper, lower = result
        assert lower < vwap < upper
        assert vwap > 0

    def test_compute_vwap_empty(self):
        from replay import MarketReplay
        assert MarketReplay._compute_vwap(pd.DataFrame()) is None

    def test_get_bar_at_time(self):
        """Returns (high, low, close) for nearest bar."""
        from replay import MarketReplay
        replay = MarketReplay()
        bars = {"AAPL": _make_bars_df([
            {"timestamp": datetime(2026, 3, 10, 10, 0, tzinfo=ET),
             "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 50000.0},
        ])}
        result = replay._get_bar_at_time(
            "AAPL", datetime(2026, 3, 10, 10, 0, tzinfo=ET), bars
        )
        assert result is not None
        high, low, close = result
        assert high == 102.0
        assert low == 99.0
        assert close == 101.0

    def test_get_bar_at_time_no_symbol(self):
        from replay import MarketReplay
        replay = MarketReplay()
        assert replay._get_bar_at_time("AAPL", datetime(2026, 3, 10, 10, 0, tzinfo=ET), {}) is None

    def test_bars_up_to(self):
        """Filters bars to only those before sim_time."""
        from replay import MarketReplay
        replay = MarketReplay()
        df = _make_bars_df([
            {"timestamp": datetime(2026, 3, 10, 10, 0, tzinfo=ET),
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10000.0},
            {"timestamp": datetime(2026, 3, 10, 11, 0, tzinfo=ET),
             "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 15000.0},
            {"timestamp": datetime(2026, 3, 10, 12, 0, tzinfo=ET),
             "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 12000.0},
        ])
        result = replay._bars_up_to(df, datetime(2026, 3, 10, 11, 0, tzinfo=ET))
        assert len(result) <= 2  # 10:00 and 11:00, not 12:00

    def test_orb_ranges(self):
        """ORB range computed from 9:30-10:00 bars."""
        from replay import MarketReplay
        replay = MarketReplay()
        rows = []
        for i in range(30):  # 9:30 - 9:59
            rows.append({
                "timestamp": datetime(2026, 3, 10, 9, 30 + i, tzinfo=ET),
                "open": 100.0, "high": 102.0 + i * 0.01,
                "low": 98.0 - i * 0.01, "close": 100.0 + i * 0.01,
                "volume": 50000.0,
            })
        bars = {"AAPL": _make_bars_df(rows)}
        orb = replay._compute_orb_ranges(bars)
        assert "AAPL" in orb
        assert orb["AAPL"]["high"] > orb["AAPL"]["low"]
        assert orb["AAPL"]["volume"] > 0
