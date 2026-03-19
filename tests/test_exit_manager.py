"""Tests for exit_manager.py — scaled TP, trailing stops, RSI/volatility exits."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from conftest import _make_trade, ET


class TestExitManagerDisabled:
    def test_disabled_returns_empty(self, override_config):
        """When ADVANCED_EXITS_ENABLED=False, check_exits returns []."""
        with override_config(ADVANCED_EXITS_ENABLED=False):
            from exit_manager import ExitManager
            from risk import RiskManager
            em = ExitManager()
            rm = RiskManager(current_equity=100_000)
            now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
            actions = em.check_exits(rm, now)
            assert actions == []


class TestScaledTP:
    def test_scaled_tp_level_1(self, override_config):
        """Triggers partial exit at 1/3 of target range."""
        with override_config(ADVANCED_EXITS_ENABLED=True, SCALED_TP_ENABLED=True,
                             BREAKEVEN_STOP_ENABLED=True):
            from exit_manager import ExitManager
            from risk import RiskManager

            em = ExitManager()
            rm = RiskManager(current_equity=100_000, starting_equity=100_000)

            # entry=100, tp=106, range=6, 1/3=102
            trade = _make_trade(symbol="TEST", entry_price=100.0, qty=30,
                                take_profit=106.0, stop_loss=97.0, side="buy")
            rm.register_trade(trade)

            now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)

            mock_snap = MagicMock()
            mock_snap.latest_trade.price = 102.0

            with patch("data.get_snapshot", return_value=mock_snap), \
                 patch("execution.get_trading_client") as mock_client:
                mock_client.return_value.close_position.return_value = True
                actions = em.check_exits(rm, now)

            assert len(actions) == 1
            assert actions[0]["action"] == "partial_tp_1"
            assert actions[0]["symbol"] == "TEST"

    def test_scaled_tp_level_2_moves_breakeven(self, override_config):
        """At 2/3 target, stop moves to breakeven."""
        with override_config(ADVANCED_EXITS_ENABLED=True, SCALED_TP_ENABLED=True,
                             BREAKEVEN_STOP_ENABLED=True):
            from exit_manager import ExitManager
            from risk import RiskManager

            em = ExitManager()
            rm = RiskManager(current_equity=100_000, starting_equity=100_000)

            # entry=100, tp=106, range=6, 2/3=104
            trade = _make_trade(symbol="TEST", entry_price=100.0, qty=20,
                                take_profit=106.0, stop_loss=97.0, side="buy",
                                partial_exits=1)
            rm.register_trade(trade)

            now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)

            mock_snap = MagicMock()
            mock_snap.latest_trade.price = 104.0

            with patch("data.get_snapshot", return_value=mock_snap), \
                 patch("execution.get_trading_client") as mock_client:
                mock_client.return_value.close_position.return_value = True
                actions = em.check_exits(rm, now)

            assert len(actions) == 1
            assert actions[0]["action"] == "partial_tp_2"
            # Stop should have moved to breakeven
            assert rm.open_trades["TEST"].stop_loss == pytest.approx(100.1, abs=0.2)


class TestTrailingStop:
    def test_trailing_stop_updates_highest(self, override_config):
        """ATR trailing stop triggers when price drops below trail."""
        with override_config(ADVANCED_EXITS_ENABLED=True, SCALED_TP_ENABLED=False,
                             ATR_TRAILING_ENABLED=True,
                             ATR_TRAIL_MULT={"STAT_MR": 2.0},
                             ATR_TRAIL_ACTIVATION=0.5,
                             TRAILING_STOP_PCT=0.015, RSI_EXIT_THRESHOLD=80):
            from exit_manager import ExitManager
            from risk import RiskManager

            em = ExitManager()
            rm = RiskManager(current_equity=100_000, starting_equity=100_000)

            # Use ATR trailing: entry_atr=1.0, mult=2.0 → trail = 2.0
            # highest=110, trail stop = 110 - 2.0 = 108.0
            trade = _make_trade(symbol="NVDA", strategy="STAT_MR",
                                entry_price=100.0, qty=10, side="buy",
                                take_profit=115.0, stop_loss=95.0,
                                highest_price_seen=110.0, entry_atr=1.0)
            rm.register_trade(trade)

            now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)

            # Price at 107.5 is below ATR trail of 108.0
            mock_snap = MagicMock()
            mock_snap.latest_trade.price = 107.5

            with patch("data.get_snapshot", return_value=mock_snap):
                actions = em.check_exits(rm, now)

            assert len(actions) == 1
            assert actions[0]["action"] == "atr_trailing_stop"


class TestVolatilityExit:
    def test_volatility_exit_atr_expansion(self, override_config):
        """ATR > 2x entry ATR triggers exit on losing trade."""
        with override_config(ADVANCED_EXITS_ENABLED=True, SCALED_TP_ENABLED=False,
                             ATR_EXPANSION_MULT=2.0, RSI_EXIT_THRESHOLD=80,
                             TRAILING_STOP_PCT=0.015):
            from exit_manager import ExitManager
            from risk import RiskManager
            import pandas as pd

            em = ExitManager()
            rm = RiskManager(current_equity=100_000, starting_equity=100_000)

            # Losing trade with known entry ATR
            trade = _make_trade(symbol="TEST", strategy="ORB",
                                entry_price=100.0, qty=10, side="buy",
                                take_profit=106.0, stop_loss=97.0,
                                entry_atr=0.5)
            rm.register_trade(trade)

            now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)

            # Price below entry (losing), ATR has expanded
            mock_snap = MagicMock()
            mock_snap.latest_trade.price = 99.0

            mock_bars = pd.DataFrame({
                "high": [101.0] * 30,
                "low": [98.0] * 30,
                "close": [99.0] * 30,
            })
            mock_atr = pd.Series([1.5] * 30)  # 1.5 > 0.5 * 2.0 = 1.0

            with patch("data.get_snapshot", return_value=mock_snap), \
                 patch("exit_manager.get_intraday_bars", return_value=mock_bars), \
                 patch("exit_manager.ta.atr", return_value=mock_atr), \
                 patch("exit_manager.ta.rsi", return_value=pd.Series([50.0] * 30)):
                actions = em.check_exits(rm, now)

            assert len(actions) == 1
            assert actions[0]["action"] == "volatility_expansion"
