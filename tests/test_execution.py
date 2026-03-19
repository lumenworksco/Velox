"""Tests for execution.py — order submission, position closes, max hold checks, TWAP."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

import pytest

from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from strategies.base import Signal
from conftest import _make_trade, MockTradingClient, ET


class TestSubmitBracketOrder:
    def test_orb_order_succeeds(self, mock_trading_client):
        """ORB submit returns order ID."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="AAPL", strategy="ORB", side="buy",
                entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                reason="test",
            )
            order_id = submit_bracket_order(signal, qty=10)
            assert order_id is not None
            assert order_id.startswith("mock-")

    def test_vwap_order_succeeds(self, mock_trading_client):
        """VWAP submit returns order ID."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="MSFT", strategy="VWAP", side="sell",
                entry_price=400.0, take_profit=395.0, stop_loss=404.0,
                reason="test",
            )
            order_id = submit_bracket_order(signal, qty=5)
            assert order_id is not None

    def test_momentum_order_succeeds(self, mock_trading_client):
        """MOMENTUM submit returns order ID (GTC limit)."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="NVDA", strategy="MOMENTUM", side="buy",
                entry_price=800.0, take_profit=850.0, stop_loss=780.0,
                reason="test", hold_type="swing",
            )
            order_id = submit_bracket_order(signal, qty=3)
            assert order_id is not None

    def test_order_failure_returns_none(self):
        """Order failure (both attempts) returns None."""
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("API down")

        with patch("execution.get_trading_client", return_value=mock_client), \
             patch("execution.time.sleep"):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="AAPL", strategy="ORB", side="buy",
                entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                reason="test",
            )
            order_id = submit_bracket_order(signal, qty=10)
            assert order_id is None


class TestClosePosition:
    def test_close_position_success(self, mock_trading_client):
        """close_position returns True on success."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import close_position
            result = close_position("AAPL", reason="eod")
            assert result is True

    def test_close_position_failure(self):
        """close_position returns False on error."""
        mock_client = MagicMock()
        mock_client.close_position.side_effect = Exception("not found")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import close_position
            result = close_position("AAPL")
            assert result is False


class TestClosePartialPosition:
    def test_close_partial_position(self):
        """close_partial_position calls client with qty string."""
        mock_client = MagicMock()
        mock_client.close_position.return_value = True

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import close_partial_position
            result = close_partial_position("AAPL", qty=5)
            assert result is True
            mock_client.close_position.assert_called_with("AAPL", qty="5")


class TestMaxHoldChecks:
    # V10: Removed tests for deleted legacy functions:
    # - test_sector_max_hold (SECTOR_ROTATION strategy removed)
    # - test_sector_max_hold_not_expired
    # - test_pairs_max_hold (legacy PAIRS strategy replaced by KALMAN_PAIRS)

    def test_orb_close(self, mock_trading_client):
        """close_orb_positions closes ORB trades only."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import close_orb_positions

            now = datetime(2026, 3, 13, 15, 45, tzinfo=ET)
            orb_trade = _make_trade(symbol="AAPL", strategy="ORB")
            vwap_trade = _make_trade(symbol="MSFT", strategy="VWAP")

            closed = close_orb_positions(
                {"AAPL": orb_trade, "MSFT": vwap_trade}, now
            )
            assert "AAPL" in closed
            assert "MSFT" not in closed


class TestCanShort:
    def test_can_short_disabled(self, override_config):
        """Returns False when ALLOW_SHORT=False."""
        with override_config(ALLOW_SHORT=False):
            from execution import can_short
            allowed, reason = can_short("AAPL", 10, 150.0)
            assert allowed is False
            assert reason == "shorting_disabled"

    def test_can_short_no_short_symbol(self, override_config, mock_trading_client):
        """Returns False for symbols in NO_SHORT_SYMBOLS."""
        with override_config(ALLOW_SHORT=True, NO_SHORT_SYMBOLS={"AAPL"}):
            from execution import can_short
            allowed, reason = can_short("AAPL", 10, 150.0)
            assert allowed is False
            assert reason == "no_short_symbol"

    def test_can_short_allowed(self, override_config, mock_trading_client):
        """Returns True when shortable and has buying power."""
        with override_config(ALLOW_SHORT=True, NO_SHORT_SYMBOLS=set()):
            with patch("execution.get_trading_client", return_value=mock_trading_client):
                from execution import can_short
                allowed, reason = can_short("AAPL", 10, 150.0)
                assert allowed is True


class TestV6StrategyRouting:
    """V6: strategy-specific order types and TWAP splitting."""

    def test_stat_mr_uses_limit_order(self):
        """STAT_MR signals produce a LimitOrderRequest."""
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="limit-001")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import _submit_order

            signal = Signal(
                symbol="AAPL", strategy="STAT_MR", side="buy",
                entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                reason="stat_mr test",
            )
            _submit_order(signal, qty=10, client=mock_client)

            args, _kwargs = mock_client.submit_order.call_args
            assert isinstance(args[0], LimitOrderRequest)

    def test_kalman_pairs_uses_limit_order(self):
        """KALMAN_PAIRS signals produce a LimitOrderRequest."""
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="limit-002")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import _submit_order

            signal = Signal(
                symbol="MSFT", strategy="KALMAN_PAIRS", side="sell",
                entry_price=400.0, take_profit=390.0, stop_loss=410.0,
                reason="kalman test", pair_id="MSFT-GOOG-001",
            )
            _submit_order(signal, qty=5, client=mock_client)

            args, _kwargs = mock_client.submit_order.call_args
            assert isinstance(args[0], LimitOrderRequest)

    def test_micro_mom_uses_market_order(self):
        """MICRO_MOM signals produce a MarketOrderRequest."""
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="mkt-001")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import _submit_order

            signal = Signal(
                symbol="NVDA", strategy="MICRO_MOM", side="buy",
                entry_price=800.0, take_profit=820.0, stop_loss=790.0,
                reason="micro_mom test",
            )
            _submit_order(signal, qty=3, client=mock_client)

            args, _kwargs = mock_client.submit_order.call_args
            assert isinstance(args[0], MarketOrderRequest)

    def test_beta_hedge_uses_market_order(self):
        """BETA_HEDGE signals produce a MarketOrderRequest."""
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="mkt-002")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import _submit_order

            signal = Signal(
                symbol="SPY", strategy="BETA_HEDGE", side="sell",
                entry_price=500.0, take_profit=490.0, stop_loss=505.0,
                reason="beta_hedge test",
            )
            _submit_order(signal, qty=20, client=mock_client)

            args, _kwargs = mock_client.submit_order.call_args
            assert isinstance(args[0], MarketOrderRequest)

    def test_twap_splits_order(self):
        """Large STAT_MR order auto-routes to TWAP and produces multiple slices."""
        mock_client = MagicMock()
        order_counter = {"n": 0}

        def _fake_submit(req):
            order_counter["n"] += 1
            return MagicMock(id=f"twap-{order_counter['n']:03d}")

        mock_client.submit_order.side_effect = _fake_submit

        with patch("execution.get_trading_client", return_value=mock_client), \
             patch("execution.time.sleep"):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="AAPL", strategy="STAT_MR", side="buy",
                entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                reason="large order test",
            )
            # qty=50 * $150 = $7500 > $2000 threshold
            result = submit_bracket_order(signal, qty=50)

            assert isinstance(result, list)
            assert len(result) == 5  # default 5 slices
            assert mock_client.submit_order.call_count == 5

    def test_small_order_not_twap(self):
        """Small STAT_MR order submits as a single bracket order (no TWAP)."""
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="single-001")

        with patch("execution.get_trading_client", return_value=mock_client):
            from execution import submit_bracket_order

            signal = Signal(
                symbol="AAPL", strategy="STAT_MR", side="buy",
                entry_price=10.0, take_profit=11.0, stop_loss=9.5,
                reason="small order test",
            )
            # qty=5 * $10 = $50 < $2000 threshold
            result = submit_bracket_order(signal, qty=5)

            assert isinstance(result, str)
            assert result == "single-001"
            assert mock_client.submit_order.call_count == 1
