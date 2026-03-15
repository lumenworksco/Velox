"""Tests for V8 Broker Abstraction Layer."""

import pytest
from unittest.mock import patch


class TestOrderResult:

    def test_successful_order(self):
        from broker.base import OrderResult
        r = OrderResult(success=True, order_id="123", filled_price=100.0, filled_qty=10)
        assert r.success
        assert r.order_id == "123"

    def test_failed_order(self):
        from broker.base import OrderResult
        r = OrderResult(success=False, message="insufficient funds")
        assert not r.success


class TestBrokerInterface:

    def test_broker_is_abstract(self):
        from broker.base import Broker
        with pytest.raises(TypeError):
            Broker()


class TestPaperBroker:

    def _make_broker(self, **kwargs):
        from broker.paper_broker import PaperBroker
        return PaperBroker(**kwargs)

    def test_submit_buy_order(self):
        broker = self._make_broker(initial_equity=100000)
        result = broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        assert result.success
        assert result.filled_qty == 10
        assert result.order_id.startswith("paper-")

    def test_slippage_applied(self):
        broker = self._make_broker()
        result = broker.submit_order("AAPL", 10, "buy", limit_price=100.0)
        # Buy slippage should increase price
        assert result.filled_price > 100.0

    def test_sell_slippage(self):
        broker = self._make_broker()
        # First buy
        broker.submit_order("AAPL", 10, "buy", limit_price=100.0)
        # Then sell
        result = broker.submit_order("AAPL", 10, "sell", limit_price=100.0)
        # Sell slippage should decrease price
        assert result.filled_price < 100.0

    def test_get_account(self):
        broker = self._make_broker(initial_equity=100000)
        account = broker.get_account()
        assert account.equity == 100000
        assert account.cash == 100000

    def test_get_positions_empty(self):
        broker = self._make_broker()
        assert broker.get_positions() == []

    def test_get_positions_after_buy(self):
        broker = self._make_broker()
        broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10

    def test_close_position(self):
        broker = self._make_broker()
        broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        result = broker.close_position("AAPL")
        assert result.success
        assert len(broker.get_positions()) == 0

    def test_close_nonexistent_position(self):
        broker = self._make_broker()
        result = broker.close_position("AAPL")
        assert not result.success

    def test_close_all_positions(self):
        broker = self._make_broker()
        broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        broker.submit_order("MSFT", 5, "buy", limit_price=400.0)
        results = broker.close_all_positions()
        assert all(r.success for r in results)
        assert len(broker.get_positions()) == 0

    def test_can_short_disabled(self, override_config):
        broker = self._make_broker()
        with override_config(ALLOW_SHORT=False):
            allowed, reason = broker.can_short("AAPL", 10, 150.0)
            assert not allowed
            assert reason == "shorting_disabled"

    def test_can_short_no_short_symbol(self, override_config):
        broker = self._make_broker()
        with override_config(ALLOW_SHORT=True, NO_SHORT_SYMBOLS={"SPY"}):
            allowed, reason = broker.can_short("SPY", 10, 500.0)
            assert not allowed

    def test_cancel_order(self):
        broker = self._make_broker()
        result = broker.submit_order("AAPL", 10, "buy", limit_price=150.0)
        assert broker.cancel_order(result.order_id)
        assert not broker.cancel_order("nonexistent")
