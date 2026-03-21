"""Tests for strategies.overnight — OvernightManager and OvernightDecision."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from conftest import _make_trade

ET = ZoneInfo("America/New_York")


# ===================================================================
# Helpers
# ===================================================================


def _build_open_trades(**kwargs):
    """Build an open_trades dict from keyword args: symbol -> trade_kwargs."""
    trades = {}
    for symbol, trade_kwargs in kwargs.items():
        trade_kwargs.setdefault("symbol", symbol)
        trade_kwargs.setdefault("status", "open")
        trades[symbol] = _make_trade(**trade_kwargs)
    return trades


def _profitable_trade(symbol, strategy, pnl_pct=0.01):
    """Create a trade in profit by the given pct.  entry=100, qty=100."""
    entry = 100.0
    qty = 100
    pnl = entry * qty * pnl_pct  # e.g. 0.01 -> $100
    return {
        "symbol": symbol,
        "strategy": strategy,
        "entry_price": entry,
        "qty": qty,
        "pnl": pnl,
        "side": "buy",
        "take_profit": entry * 1.05,
        "stop_loss": entry * 0.97,
    }


# ===================================================================
# Tests: select_overnight_holds
# ===================================================================


class TestEligibleStrategies:
    """Only PEAD, STAT_MR, KALMAN_PAIRS are eligible for overnight holds."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_eligible_strategy_can_hold(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert len(decisions) == 1
        assert decisions[0].action == "hold"
        assert decisions[0].symbol == "AAPL"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_stat_mr_eligible(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            MSFT=_profitable_trade("MSFT", "STAT_MR"),
        )
        decisions = mgr.select_overnight_holds(trades)
        hold_decisions = [d for d in decisions if d.action == "hold"]
        assert len(hold_decisions) == 1

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_kalman_pairs_eligible(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            NVDA=_profitable_trade("NVDA", "KALMAN_PAIRS"),
        )
        decisions = mgr.select_overnight_holds(trades)
        hold_decisions = [d for d in decisions if d.action == "hold"]
        assert len(hold_decisions) == 1

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_ineligible_strategy_orb_always_closes(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "ORB"),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert len(decisions) == 1
        assert decisions[0].action == "close"
        assert decisions[0].reason == "ineligible_strategy"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_ineligible_strategy_vwap_always_closes(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "VWAP"),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "close"
        assert decisions[0].reason == "ineligible_strategy"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_ineligible_strategy_micro_mom_closes(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "MICRO_MOM"),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "close"


class TestProfitThreshold:
    """Position must exceed OVERNIGHT_MIN_PROFIT_PCT to hold."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_profitable_position_holds(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        # 1% profit, well above 0.3% threshold
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.01),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "hold"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_below_min_profit_closes(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        # 0.1% profit, below 0.3% threshold
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.001),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "close"
        assert decisions[0].reason == "insufficient_profit"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_negative_pnl_closes(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=-0.02),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "close"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_exactly_at_threshold_holds(self, _mock_earnings):
        """At exactly the threshold, position qualifies to hold."""
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        # Exactly 0.3% — at threshold, should hold
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.003),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "hold"


class TestMaxPositions:
    """Max OVERNIGHT_MAX_POSITIONS (4) can be held overnight."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_respects_max_positions(self, _mock_earnings, override_config):
        from strategies.overnight import OvernightManager

        with override_config(OVERNIGHT_MAX_POSITIONS=2):
            mgr = OvernightManager()
            trades = _build_open_trades(
                AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.02),
                MSFT=_profitable_trade("MSFT", "STAT_MR", pnl_pct=0.015),
                NVDA=_profitable_trade("NVDA", "KALMAN_PAIRS", pnl_pct=0.01),
            )
            decisions = mgr.select_overnight_holds(trades)
            holds = [d for d in decisions if d.action == "hold"]
            closes = [d for d in decisions if d.action == "close"]
            assert len(holds) == 2
            assert len(closes) == 1

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_most_profitable_kept(self, _mock_earnings, override_config):
        """When exceeding max, the most profitable are kept."""
        from strategies.overnight import OvernightManager

        with override_config(OVERNIGHT_MAX_POSITIONS=1):
            mgr = OvernightManager()
            trades = _build_open_trades(
                AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.005),
                MSFT=_profitable_trade("MSFT", "PEAD", pnl_pct=0.03),
            )
            decisions = mgr.select_overnight_holds(trades)
            holds = [d for d in decisions if d.action == "hold"]
            assert len(holds) == 1
            assert holds[0].symbol == "MSFT"  # Most profitable


class TestFlightToSafety:
    """Flight-to-safety blocks all overnight holds."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_flight_to_safety_blocks_holds(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        decisions = mgr.select_overnight_holds(
            trades,
            cross_asset_signals={"flight_to_safety": True},
        )
        assert decisions[0].action == "close"
        assert decisions[0].reason == "flight_to_safety"

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_no_flight_allows_holds(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        decisions = mgr.select_overnight_holds(
            trades,
            cross_asset_signals={"flight_to_safety": False},
        )
        assert decisions[0].action == "hold"


class TestSizeReduction:
    """Held positions have size_reduction = OVERNIGHT_SIZE_REDUCTION."""

    @patch("strategies.overnight._gap_risk_manager", None)
    @patch("earnings.has_earnings_soon", return_value=False)
    def test_hold_has_correct_size_reduction(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        decisions = mgr.select_overnight_holds(trades)
        hold = [d for d in decisions if d.action == "hold"][0]
        assert hold.size_reduction == 0.40  # default OVERNIGHT_SIZE_REDUCTION

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_close_has_full_size_reduction(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "ORB"),  # ineligible
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].size_reduction == 1.0


class TestEarningsBlock:
    """Earnings within EARNINGS_FILTER_DAYS blocks overnight hold."""

    @patch("strategies.overnight.has_earnings_soon", return_value=True)
    def test_earnings_blocks_overnight_hold(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        decisions = mgr.select_overnight_holds(trades)
        assert decisions[0].action == "close"
        assert decisions[0].reason == "earnings_soon"


class TestDisabled:
    """When OVERNIGHT_HOLD_ENABLED = False, all positions close."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_disabled_closes_all(self, _mock_earnings, override_config):
        from strategies.overnight import OvernightManager

        with override_config(OVERNIGHT_HOLD_ENABLED=False):
            mgr = OvernightManager()
            trades = _build_open_trades(
                AAPL=_profitable_trade("AAPL", "PEAD"),
                MSFT=_profitable_trade("MSFT", "STAT_MR"),
            )
            decisions = mgr.select_overnight_holds(trades)
            assert all(d.action == "close" for d in decisions)
            assert all(d.reason == "overnight_disabled" for d in decisions)
            assert len(decisions) == 2


class TestEmptyTrades:
    """Empty open_trades returns empty decisions."""

    def test_empty_trades_returns_empty(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        decisions = mgr.select_overnight_holds({})
        assert decisions == []


# ===================================================================
# Tests: morning_gap_check
# ===================================================================


class TestMorningGapCheck:
    """morning_gap_check evaluates overnight positions at market open."""

    def test_adverse_gap_long_triggers_exit(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "AAPL": {
                "entry_price": 100.0,
                "side": "buy",
                "qty": 50,
                "strategy": "PEAD",
                "stop_loss": 97.0,
            },
        }
        # Gapped down 2% (> 1% threshold)
        prices = {"AAPL": 98.0}
        actions = mgr.morning_gap_check(positions, prices)
        assert len(actions) == 1
        assert actions[0]["action"] == "exit"
        assert actions[0]["reason"] == "overnight_gap_stop"

    def test_adverse_gap_short_triggers_exit(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "TSLA": {
                "entry_price": 200.0,
                "side": "sell",
                "qty": 25,
                "strategy": "STAT_MR",
                "stop_loss": 206.0,
            },
        }
        # Gapped up 2% (adverse for short)
        prices = {"TSLA": 204.0}
        actions = mgr.morning_gap_check(positions, prices)
        assert len(actions) == 1
        assert actions[0]["action"] == "exit"

    def test_favorable_gap_long_updates_stop(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "AAPL": {
                "entry_price": 100.0,
                "side": "buy",
                "qty": 50,
                "strategy": "PEAD",
                "stop_loss": 97.0,
            },
        }
        # Gapped up 2% (favorable for long)
        prices = {"AAPL": 102.0}
        actions = mgr.morning_gap_check(positions, prices)
        assert len(actions) == 1
        assert actions[0]["action"] == "update_stop"
        assert actions[0]["reason"] == "overnight_gap_ok"
        assert "new_stop" in actions[0]

    def test_favorable_gap_short_updates_stop(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "TSLA": {
                "entry_price": 200.0,
                "side": "sell",
                "qty": 25,
                "strategy": "STAT_MR",
                "stop_loss": 206.0,
            },
        }
        # Gapped down 2% (favorable for short)
        prices = {"TSLA": 196.0}
        actions = mgr.morning_gap_check(positions, prices)
        assert len(actions) == 1
        assert actions[0]["action"] == "update_stop"

    def test_small_adverse_gap_below_threshold_updates_stop(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "AAPL": {
                "entry_price": 100.0,
                "side": "buy",
                "qty": 50,
                "strategy": "PEAD",
                "stop_loss": 97.0,
            },
        }
        # Gapped down only 0.5% (below 1% threshold)
        prices = {"AAPL": 99.5}
        actions = mgr.morning_gap_check(positions, prices)
        assert actions[0]["action"] == "update_stop"

    def test_missing_price_skipped(self):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        positions = {
            "AAPL": {
                "entry_price": 100.0,
                "side": "buy",
                "qty": 50,
                "strategy": "PEAD",
                "stop_loss": 97.0,
            },
        }
        # No price for AAPL
        actions = mgr.morning_gap_check(positions, {})
        assert actions == []


# ===================================================================
# Tests: reset_daily and get_overnight_positions
# ===================================================================


class TestResetDaily:
    """reset_daily clears overnight state."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_reset_clears_positions(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        # Build overnight positions via select
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        mgr.select_overnight_holds(trades)
        assert len(mgr.get_overnight_positions()) > 0

        mgr.reset_daily()
        assert mgr.get_overnight_positions() == {}


class TestGetOvernightPositions:
    """get_overnight_positions returns a copy of internal state."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_returns_dict_copy(self, _mock_earnings):
        from strategies.overnight import OvernightManager

        mgr = OvernightManager()
        trades = _build_open_trades(
            AAPL=_profitable_trade("AAPL", "PEAD"),
        )
        mgr.select_overnight_holds(trades)
        positions = mgr.get_overnight_positions()
        # Mutating the returned dict should not affect internal state
        positions.clear()
        assert len(mgr.get_overnight_positions()) == 1


# ===================================================================
# Tests: mixed scenario
# ===================================================================


class TestMixedScenario:
    """Multiple positions with varying eligibility."""

    @patch("earnings.has_earnings_soon", return_value=False)
    def test_mixed_strategies_correct_decisions(self, _mock_earnings, override_config):
        from strategies.overnight import OvernightManager

        with override_config(OVERNIGHT_MAX_POSITIONS=2):
            mgr = OvernightManager()
            trades = _build_open_trades(
                AAPL=_profitable_trade("AAPL", "PEAD", pnl_pct=0.02),
                MSFT=_profitable_trade("MSFT", "ORB", pnl_pct=0.03),    # ineligible
                NVDA=_profitable_trade("NVDA", "STAT_MR", pnl_pct=0.01),
                TSLA=_profitable_trade("TSLA", "VWAP", pnl_pct=0.05),   # ineligible
            )
            decisions = mgr.select_overnight_holds(trades)
            holds = {d.symbol for d in decisions if d.action == "hold"}
            closes = {d.symbol for d in decisions if d.action == "close"}
            # PEAD and STAT_MR are eligible and profitable
            assert holds == {"AAPL", "NVDA"}
            # ORB and VWAP are ineligible
            assert "MSFT" in closes
            assert "TSLA" in closes
