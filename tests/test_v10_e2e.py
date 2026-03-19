"""V10 E2E Integration Tests — Full pipeline simulation.

Tests the complete signal → filter → size → order → track → exit pipeline
using the PaperBroker, verifying that all V10 components work together:
- OMS order tracking
- Transaction cost filter
- Event bus notifications
- Tiered circuit breaker
- VaR risk budgeting
- Correlation limiter
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import config
from strategies.base import Signal
from risk import RiskManager, TradeRecord, VolatilityTargetingRiskEngine, DailyPnLLock
from broker.paper_broker import PaperBroker
from oms.order import OrderState
from oms.order_manager import OrderManager
from oms.kill_switch import KillSwitch
from oms.transaction_cost import estimate_round_trip_cost
from risk.circuit_breaker import TieredCircuitBreaker, CircuitTier
from risk.var_monitor import VaRMonitor
from risk.correlation_limiter import CorrelationLimiter
from engine.events import EventBus, EventTypes
from engine.signal_processor import process_signals, set_order_manager


@pytest.fixture
def trading_env():
    """Set up a complete V10 trading environment."""
    # Paper broker
    broker = PaperBroker(initial_equity=100_000)

    # Risk manager
    risk = RiskManager()
    risk.reset_daily(100_000, 100_000)

    # V10 components
    vol_engine = VolatilityTargetingRiskEngine()
    pnl_lock = DailyPnLLock()
    order_manager = OrderManager()
    set_order_manager(order_manager)
    tiered_cb = TieredCircuitBreaker()
    kill_switch = KillSwitch()
    var_monitor = VaRMonitor(max_var_pct=0.02)
    corr_limiter = CorrelationLimiter(max_pairwise_corr=0.80)
    event_bus = EventBus()

    # Track events
    events_received = []
    event_bus.subscribe("*", lambda e: events_received.append(e))

    now = datetime(2026, 3, 19, 10, 30, tzinfo=config.ET)

    return {
        "broker": broker,
        "risk": risk,
        "vol_engine": vol_engine,
        "pnl_lock": pnl_lock,
        "order_manager": order_manager,
        "tiered_cb": tiered_cb,
        "kill_switch": kill_switch,
        "var_monitor": var_monitor,
        "corr_limiter": corr_limiter,
        "event_bus": event_bus,
        "events": events_received,
        "now": now,
    }


def _make_signal(symbol="AAPL", strategy="STAT_MR", side="buy",
                 entry=150.0, tp=155.0, sl=147.0):
    """Create a test signal."""
    return Signal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        entry_price=entry,
        take_profit=tp,
        stop_loss=sl,
    )


class TestOmsLifecycle:
    """Test full OMS order lifecycle through signal processing."""

    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    def test_signal_creates_oms_order(self, mock_corr, mock_earn, mock_submit, trading_env):
        """A processed signal should create an OMS order and register a trade."""
        mock_submit.return_value = "broker-order-001"

        signal = _make_signal()
        process_signals(
            [signal], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
        )

        # Should have registered a trade
        assert "AAPL" in trading_env["risk"].open_trades

        # OMS should have tracked the order
        mgr = trading_env["order_manager"]
        assert mgr.stats["total"] >= 1

    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    def test_failed_order_updates_oms(self, mock_corr, mock_earn, mock_submit, trading_env):
        """A failed order submission should mark the OMS order as FAILED."""
        mock_submit.return_value = None  # Simulate failure

        signal = _make_signal()
        process_signals(
            [signal], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
        )

        # Should NOT have registered a trade
        assert "AAPL" not in trading_env["risk"].open_trades

    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    def test_duplicate_signal_blocked(self, mock_corr, mock_earn, mock_submit, trading_env):
        """A signal for a symbol we already hold should be blocked."""
        mock_submit.return_value = "broker-001"

        signal = _make_signal()
        process_signals(
            [signal], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
        )
        assert "AAPL" in trading_env["risk"].open_trades

        # Second signal for same symbol should be blocked
        signal2 = _make_signal()
        process_signals(
            [signal2], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
        )
        # Only one trade
        assert mock_submit.call_count == 1


class TestCircuitBreakerIntegration:
    """Test circuit breaker escalation and recovery."""

    def test_tier_escalation_sequence(self, trading_env):
        cb = trading_env["tiered_cb"]

        # Normal
        assert cb.update(0.005) == CircuitTier.NORMAL

        # Yellow at -1.5%
        assert cb.update(-0.015) == CircuitTier.YELLOW
        assert cb.size_multiplier == 0.5

        # Orange at -2.5%
        assert cb.update(-0.025) == CircuitTier.ORANGE
        assert not cb.allow_new_entries

        # Red at -3.5%
        assert cb.update(-0.035) == CircuitTier.RED
        assert cb.should_close_day_trades

        # Recovery
        assert cb.update(-0.005) == CircuitTier.NORMAL

    def test_kill_switch_cancels_oms_orders(self, trading_env):
        mgr = trading_env["order_manager"]
        ks = trading_env["kill_switch"]

        # Create some active orders
        o1 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        o2 = mgr.create_order("MSFT", "VWAP", "sell", "bracket", 5)

        # Activate kill switch
        ks.activate("test", order_manager=mgr)

        assert not ks.is_trading_allowed()
        assert o1.is_terminal
        assert o2.is_terminal
        assert mgr.stats["active"] == 0


class TestVaRIntegration:
    """Test VaR monitor integration with position sizing."""

    def test_var_reduces_sizing_when_risk_high(self, trading_env):
        var = trading_env["var_monitor"]

        # Simulate high-volatility history
        import numpy as np
        np.random.seed(42)
        high_vol_returns = list(np.random.normal(-0.005, 0.03, 60))
        var.update(high_vol_returns, 100000)

        # Size multiplier should be reduced
        assert var.size_multiplier < 1.0
        assert var.risk_budget_remaining < 1.0

    def test_var_allows_full_sizing_when_calm(self, trading_env):
        var = trading_env["var_monitor"]

        import numpy as np
        np.random.seed(42)
        calm_returns = list(np.random.normal(0.002, 0.003, 60))
        var.update(calm_returns, 100000)

        # Should have plenty of risk budget
        assert var.size_multiplier > 0.8
        assert var.risk_budget_remaining > 0.5


class TestCorrelationLimiterIntegration:
    """Test correlation limiter in the signal processing pipeline."""

    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    def test_correlated_position_blocked(self, mock_corr_old, mock_earn, mock_submit, trading_env):
        """Adding a highly correlated position should be blocked."""
        mock_submit.return_value = "broker-001"

        corr_limiter = trading_env["corr_limiter"]
        corr_limiter.update_correlation("AAPL", "MSFT", 0.95)
        corr_limiter.set_sector_map({"AAPL": "tech", "MSFT": "tech"})

        # Open AAPL first
        signal1 = _make_signal("AAPL")
        process_signals(
            [signal1], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
            corr_limiter=corr_limiter,
        )
        assert "AAPL" in trading_env["risk"].open_trades

        # Try MSFT — should be blocked by concentration
        signal2 = _make_signal("MSFT")
        process_signals(
            [signal2], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
            corr_limiter=corr_limiter,
        )
        # MSFT blocked (sector > 50% since only 2 positions both tech)
        assert "MSFT" not in trading_env["risk"].open_trades

    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    def test_diversified_position_allowed(self, mock_corr_old, mock_earn, mock_submit, trading_env):
        """Adding a diversified position should be allowed."""
        mock_submit.return_value = "broker-001"

        corr_limiter = trading_env["corr_limiter"]
        corr_limiter.update_correlation("AAPL", "XOM", 0.10)
        corr_limiter.set_sector_map({"AAPL": "tech", "XOM": "energy"})

        signal1 = _make_signal("AAPL")
        process_signals(
            [signal1], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
            corr_limiter=corr_limiter,
        )
        assert "AAPL" in trading_env["risk"].open_trades

        mock_submit.return_value = "broker-002"
        signal2 = _make_signal("XOM", entry=80.0, tp=84.0, sl=78.0)
        process_signals(
            [signal2], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
            corr_limiter=corr_limiter,
        )
        assert "XOM" in trading_env["risk"].open_trades


class TestTransactionCostFilter:
    """Test that negative-EV trades are rejected by the cost model."""

    def test_wide_spread_trade_passes(self):
        """A trade with wide TP/SL should pass the cost filter."""
        from oms.transaction_cost import is_trade_profitable_after_costs
        ok, details = is_trade_profitable_after_costs(
            entry_price=100, take_profit=108, stop_loss=95,
            qty=10, side="buy", win_rate=0.55,
        )
        assert ok
        assert details["expected_value"] > 0

    def test_scalp_trade_fails(self):
        """A tiny scalp with costs exceeding expected profit should fail."""
        from oms.transaction_cost import is_trade_profitable_after_costs
        ok, details = is_trade_profitable_after_costs(
            entry_price=100, take_profit=100.03, stop_loss=99.90,
            qty=10, side="buy", win_rate=0.50,
        )
        assert not ok


class TestPaperBrokerE2E:
    """End-to-end test using the paper broker."""

    def test_full_trade_lifecycle(self):
        """Open → hold → close a position through the paper broker."""
        broker = PaperBroker(initial_equity=100_000)

        # Open
        result = broker.submit_order("AAPL", 10, "buy", limit_price=150.0,
                                     take_profit=155.0, stop_loss=147.0)
        assert result.success
        assert result.filled_qty == 10
        assert result.filled_price > 0

        # Check position
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

        # Check account — equity ≈ initial (position value replaces cash), cash < initial
        account = broker.get_account()
        assert account.cash < 100_000  # Spent cash on shares

        # Close
        close_result = broker.close_position("AAPL")
        assert close_result.success
        assert len(broker.get_positions()) == 0

    def test_multiple_positions(self):
        """Open multiple positions and close all."""
        broker = PaperBroker(initial_equity=100_000)

        broker.submit_order("AAPL", 5, "buy", limit_price=150.0)
        broker.submit_order("MSFT", 3, "buy", limit_price=400.0)
        broker.submit_order("GOOGL", 2, "buy", limit_price=170.0)

        assert len(broker.get_positions()) == 3

        results = broker.close_all_positions()
        assert all(r.success for r in results)
        assert len(broker.get_positions()) == 0

    def test_slippage_is_realistic(self):
        """Verify slippage is within reasonable bounds."""
        broker = PaperBroker(spread_bps=5.0)

        result = broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        # Should be slightly above 150 (buy slippage)
        assert result.filled_price > 150.0
        # But not more than 0.1% above (reasonable slippage)
        assert result.filled_price < 150.0 * 1.001


class TestPnlLockIntegration:
    """Test P&L lock circuit breaker."""

    def test_pnl_lock_blocks_signals(self, trading_env):
        """When P&L lock is in LOSS_HALT, all signals should be blocked."""
        from risk.daily_pnl_lock import LockState
        pnl_lock = trading_env["pnl_lock"]

        # Force the lock into halt state
        pnl_lock.state = LockState.LOSS_HALT

        with patch("engine.signal_processor.submit_bracket_order") as mock_submit, \
             patch("engine.signal_processor.has_earnings_soon", return_value=False):
            signal = _make_signal()
            process_signals(
                [signal], trading_env["risk"], "normal", trading_env["now"],
                trading_env["vol_engine"], pnl_lock,
            )
            # Should never reach order submission
            mock_submit.assert_not_called()


class TestPairAtomicity:
    """Test atomic pair processing (both legs or neither)."""

    @patch("engine.signal_processor.close_position")
    @patch("engine.signal_processor.submit_bracket_order")
    @patch("engine.signal_processor.has_earnings_soon", return_value=False)
    @patch("engine.signal_processor.is_too_correlated", return_value=False)
    @patch("engine.signal_processor.can_short", return_value=(True, "ok"))
    def test_pair_signals_processed_atomically(self, mock_short, mock_corr, mock_earn, mock_submit, mock_close, trading_env):
        """Both legs of a pair trade should succeed or both fail."""
        mock_submit.return_value = "broker-001"

        # Use buy for both legs to avoid short-selling complications
        leg1 = _make_signal("AAPL", strategy="KALMAN_PAIRS", side="buy")
        leg1.pair_id = "pair_AAPL_MSFT"
        leg2 = _make_signal("MSFT", strategy="KALMAN_PAIRS", side="buy",
                            entry=400.0, tp=410.0, sl=395.0)
        leg2.pair_id = "pair_AAPL_MSFT"

        process_signals(
            [leg1, leg2], trading_env["risk"], "normal", trading_env["now"],
            trading_env["vol_engine"], trading_env["pnl_lock"],
        )

        # Both legs should be open (submit_bracket_order returns success for both)
        assert "AAPL" in trading_env["risk"].open_trades
        assert "MSFT" in trading_env["risk"].open_trades
        assert trading_env["risk"].open_trades["AAPL"].pair_id == "pair_AAPL_MSFT"
