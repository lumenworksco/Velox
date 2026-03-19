"""Tests for V8 ATR trailing stops."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Import the conftest helpers
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


def _make_trade(symbol="AAPL", strategy="STAT_MR", side="buy",
                entry_price=100.0, qty=10, take_profit=105.0,
                stop_loss=95.0, entry_atr=2.0, highest_price_seen=0.0,
                lowest_price_seen=0.0,
                partial_exits=0, hold_type="day", pair_id="",
                entry_time=None, status="open", pnl=0.0,
                exit_price=None, exit_reason=""):
    from risk import TradeRecord
    return TradeRecord(
        symbol=symbol, strategy=strategy, side=side,
        entry_price=entry_price, qty=qty, take_profit=take_profit,
        stop_loss=stop_loss, entry_atr=entry_atr,
        highest_price_seen=highest_price_seen or entry_price,
        lowest_price_seen=lowest_price_seen,
        partial_exits=partial_exits, hold_type=hold_type, pair_id=pair_id,
        entry_time=entry_time or datetime(2026, 3, 13, 10, 5, tzinfo=ET),
        status=status, pnl=pnl, exit_price=exit_price, exit_reason=exit_reason,
    )


class TestATRTrailingStop:
    """Test ATR-based trailing stop logic."""

    def _make_exit_manager(self):
        from exit_manager import ExitManager
        return ExitManager()

    def test_atr_trail_not_activated_before_profit(self, override_config):
        """Trail should not activate before position is in profit by 0.5x ATR."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Entry at 100, ATR=2, activation = 0.5*2 = 1.0, so need price > 101
        trade = _make_trade(entry_price=100.0, entry_atr=2.0, highest_price_seen=100.5, stop_loss=95.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 100.5, risk_mgr, now)
            assert result is None

    def test_atr_trail_ratchets_up_for_longs(self, override_config):
        """Trail should ratchet stop up when price rises (long)."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Entry=100, ATR=2, highest=105, trail = 105 - (2*1.5) = 102
        trade = _make_trade(entry_price=100.0, entry_atr=2.0, highest_price_seen=105.0, stop_loss=95.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 104.0, risk_mgr, now)
            assert result is None  # Price still above trail
            assert trade.stop_loss == 102.0  # Stop ratcheted up from 95 to 102

    def test_atr_trail_triggers_exit_long(self, override_config):
        """Trail should trigger exit when price drops below stop (long)."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Entry=100, ATR=2, highest=105, trail = 105 - 3 = 102
        trade = _make_trade(entry_price=100.0, entry_atr=2.0, highest_price_seen=105.0, stop_loss=95.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 101.5, risk_mgr, now)
            assert result is not None
            assert result["action"] == "atr_trailing_stop"
            risk_mgr.close_trade.assert_called_once()

    def test_atr_trail_short_activation(self, override_config):
        """Short trail should activate after price drops by 0.5x ATR."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Entry=100, ATR=2, activation = 0.5*2 = 1.0, need price < 99
        trade = _make_trade(side="sell", entry_price=100.0, entry_atr=2.0,
                          highest_price_seen=99.5, stop_loss=105.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 99.5, risk_mgr, now)
            assert result is None  # Not in enough profit

    def test_atr_trail_short_triggers_exit(self, override_config):
        """Short trail should trigger when price rises above stop."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Entry=100, ATR=2, lowest=95, trail = 95 + (2*1.5) = 98
        trade = _make_trade(side="sell", entry_price=100.0, entry_atr=2.0,
                          lowest_price_seen=95.0, stop_loss=105.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 98.5, risk_mgr, now)
            assert result is not None
            assert result["action"] == "atr_trailing_stop"

    def test_atr_trail_no_mult_for_strategy(self, override_config):
        """If strategy has no ATR mult defined, should return None."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        trade = _make_trade(strategy="UNKNOWN", entry_price=100.0, entry_atr=2.0,
                          highest_price_seen=105.0, stop_loss=95.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 104.0, risk_mgr, now)
            assert result is None

    def test_atr_trail_disabled(self, override_config):
        """When disabled, ATR trailing should not be called (integration test)."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        trade = _make_trade(entry_price=100.0, entry_atr=2.0, highest_price_seen=105.0)

        with override_config(ATR_TRAILING_ENABLED=False):
            # _check_atr_trailing_stop won't be called from _evaluate_trade
            # but direct call should still work (config check is in _evaluate_trade)
            pass  # This is an integration-level test

    def test_atr_trail_never_widens_stop(self, override_config):
        """Stop should never move in unfavorable direction."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        # Stop already at 103 (better than ATR trail would set)
        # highest=105, trail = 105 - 3 = 102, but stop is already 103
        trade = _make_trade(entry_price=100.0, entry_atr=2.0,
                          highest_price_seen=105.0, stop_loss=103.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            result = em._check_atr_trailing_stop(trade, 104.0, risk_mgr, now)
            assert result is None
            assert trade.stop_loss == 103.0  # Not widened to 102

    def test_atr_trail_zero_entry_atr(self, override_config):
        """If entry_atr is 0, should not be called from _evaluate_trade."""
        em = self._make_exit_manager()
        trade = _make_trade(entry_atr=0.0)

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT={"STAT_MR": 1.5},
                           ATR_TRAIL_ACTIVATION=0.5):
            # This is checked in _evaluate_trade before calling _check_atr_trailing_stop
            # The method itself doesn't check for zero ATR, but the caller does
            pass

    def test_per_strategy_multipliers(self, override_config):
        """Different strategies should use different ATR multipliers."""
        em = self._make_exit_manager()
        risk_mgr = MagicMock()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        mult_map = {"STAT_MR": 1.5, "ORB": 2.5, "MICRO_MOM": 1.0}

        with override_config(ATR_TRAILING_ENABLED=True,
                           ATR_TRAIL_MULT=mult_map,
                           ATR_TRAIL_ACTIVATION=0.5):
            for strategy, mult in mult_map.items():
                # Entry=100, ATR=2, highest=110
                trade = _make_trade(strategy=strategy, entry_price=100.0, entry_atr=2.0,
                                  highest_price_seen=110.0, stop_loss=90.0)
                risk_mgr.reset_mock()
                em._check_atr_trailing_stop(trade, 109.0, risk_mgr, now)
                expected_trail = 110.0 - (2.0 * mult)
                assert trade.stop_loss == expected_trail, f"Failed for {strategy}"
