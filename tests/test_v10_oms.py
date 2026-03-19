"""Tests for V10 OMS — Order Management System."""

import pytest
from datetime import datetime

from oms.order import Order, OrderState
from oms.order_manager import OrderManager
from oms.kill_switch import KillSwitch
from oms.transaction_cost import estimate_round_trip_cost, is_trade_profitable_after_costs


class TestOrder:
    """Test Order dataclass and state machine."""

    def test_order_creation_defaults(self):
        order = Order(symbol="AAPL", strategy="STAT_MR", side="buy", qty=10)
        assert order.state == OrderState.PENDING
        assert order.symbol == "AAPL"
        assert order.is_active
        assert not order.is_terminal
        assert order.oms_id  # auto-generated

    def test_valid_transitions(self):
        order = Order()
        assert order.transition(OrderState.SUBMITTED)
        assert order.state == OrderState.SUBMITTED
        assert order.submitted_at is not None

        assert order.transition(OrderState.FILLED)
        assert order.state == OrderState.FILLED
        assert order.filled_at is not None
        assert order.is_terminal
        assert not order.is_active

    def test_invalid_transition(self):
        order = Order()
        # Can't go directly from PENDING to FILLED
        assert not order.transition(OrderState.FILLED)
        assert order.state == OrderState.PENDING  # unchanged

    def test_terminal_states_block_transitions(self):
        order = Order()
        order.transition(OrderState.FAILED)
        assert order.is_terminal
        assert not order.transition(OrderState.SUBMITTED)

    def test_partial_fill(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.PARTIAL_FILL)
        assert order.is_active
        assert order.transition(OrderState.FILLED)
        assert order.is_terminal

    def test_cancellation(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.CANCELLED)
        assert order.is_terminal
        assert order.cancelled_at is not None


class TestOrderManager:
    """Test OrderManager registry."""

    def test_create_and_get(self):
        mgr = OrderManager()
        order = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        assert order.state == OrderState.PENDING
        assert mgr.get_order(order.oms_id) is order

    def test_update_state(self):
        mgr = OrderManager()
        order = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        assert mgr.update_state(order.oms_id, OrderState.SUBMITTED, broker_order_id="abc-123")
        assert order.state == OrderState.SUBMITTED
        assert mgr.get_by_broker_id("abc-123") is order

    def test_idempotency(self):
        mgr = OrderManager()
        o1 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10, idempotency_key="key1")
        o2 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10, idempotency_key="key1")
        assert o1.oms_id == o2.oms_id
        assert mgr.stats["total"] == 1

    def test_active_orders(self):
        mgr = OrderManager()
        o1 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        o2 = mgr.create_order("MSFT", "VWAP", "sell", "bracket", 5)
        mgr.update_state(o1.oms_id, OrderState.SUBMITTED)
        mgr.update_state(o2.oms_id, OrderState.SUBMITTED)
        mgr.update_state(o1.oms_id, OrderState.FILLED, filled_qty=10)

        active = mgr.get_active_orders()
        assert len(active) == 1
        assert active[0].symbol == "MSFT"

    def test_active_orders_by_symbol(self):
        mgr = OrderManager()
        mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        mgr.create_order("MSFT", "VWAP", "sell", "bracket", 5)
        assert len(mgr.get_active_orders("AAPL")) == 1
        assert len(mgr.get_active_orders("TSLA")) == 0

    def test_cancel_all(self):
        mgr = OrderManager()
        o1 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        o2 = mgr.create_order("MSFT", "VWAP", "sell", "bracket", 5)
        cancelled = mgr.cancel_all()
        assert len(cancelled) == 2
        assert o1.is_terminal
        assert o2.is_terminal

    def test_audit_trail(self):
        mgr = OrderManager()
        order = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        mgr.update_state(order.oms_id, OrderState.SUBMITTED)
        mgr.update_state(order.oms_id, OrderState.FILLED)
        trail = mgr.get_audit_trail()
        assert len(trail) == 3  # PENDING, SUBMITTED, FILLED
        assert trail[0]["new_state"] == "pending"
        assert trail[2]["new_state"] == "filled"

    def test_stats(self):
        mgr = OrderManager()
        o1 = mgr.create_order("AAPL", "STAT_MR", "buy", "bracket", 10)
        mgr.update_state(o1.oms_id, OrderState.SUBMITTED)
        mgr.update_state(o1.oms_id, OrderState.FILLED)
        o2 = mgr.create_order("MSFT", "VWAP", "sell", "bracket", 5)
        mgr.update_state(o2.oms_id, OrderState.SUBMITTED)
        mgr.update_state(o2.oms_id, OrderState.REJECTED)

        stats = mgr.stats
        assert stats["total"] == 2
        assert stats["filled"] == 1
        assert stats["rejected"] == 1
        assert stats["active"] == 0

    def test_unknown_order(self):
        mgr = OrderManager()
        assert not mgr.update_state("nonexistent", OrderState.FILLED)
        assert mgr.get_order("nonexistent") is None


class TestKillSwitch:
    """Test emergency kill switch."""

    def test_default_state(self):
        ks = KillSwitch()
        assert ks.is_trading_allowed()
        assert not ks.active

    def test_activate_deactivate(self):
        ks = KillSwitch()
        ks.activate("test_reason")
        assert not ks.is_trading_allowed()
        assert ks.active
        assert ks.reason == "test_reason"
        assert ks.activated_at is not None

        ks.deactivate()
        assert ks.is_trading_allowed()
        assert not ks.active

    def test_double_activate(self):
        ks = KillSwitch()
        ks.activate("first")
        ks.activate("second")  # Should be no-op
        assert ks.reason == "first"

    def test_status_dict(self):
        ks = KillSwitch()
        status = ks.status
        assert status["active"] is False
        assert status["activated_at"] is None


class TestTransactionCost:
    """Test pre-trade transaction cost model."""

    def test_basic_cost_estimate(self):
        costs = estimate_round_trip_cost(100.0, 10)
        assert costs["total_cost"] > 0
        assert costs["cost_bps"] > 0
        assert costs["spread_cost"] > 0
        assert costs["slippage_cost"] > 0

    def test_zero_commission_alpaca(self):
        costs = estimate_round_trip_cost(100.0, 10, commission_per_share=0.0)
        assert costs["commission_cost"] == 0.0

    def test_cost_scales_with_size(self):
        small = estimate_round_trip_cost(100.0, 1)
        large = estimate_round_trip_cost(100.0, 100)
        assert large["total_cost"] > small["total_cost"]
        # But cost_bps should be the same (proportional)
        assert abs(small["cost_bps"] - large["cost_bps"]) < 0.01

    def test_profitable_trade(self):
        # Wide TP/SL = should be profitable
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0, take_profit=110.0, stop_loss=95.0,
            qty=10, side="buy", win_rate=0.55,
        )
        assert profitable
        assert details["expected_value"] > 0

    def test_unprofitable_trade(self):
        # Tiny TP, big SL = should be unprofitable
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0, take_profit=100.05, stop_loss=99.0,
            qty=10, side="buy", win_rate=0.50,
        )
        assert not profitable
        assert details["expected_value"] < 0

    def test_short_side(self):
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0, take_profit=90.0, stop_loss=105.0,
            qty=10, side="sell", win_rate=0.55,
        )
        assert details["gross_profit"] > 0
        assert details["gross_loss"] > 0
