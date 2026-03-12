"""Tests for execution.py — order submission, position closes, max hold checks."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

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
    def test_sector_max_hold(self, mock_trading_client):
        """check_sector_max_hold returns expired sector trades."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import check_sector_max_hold

            now = datetime(2026, 3, 23, 10, 0, tzinfo=ET)
            trade = _make_trade(
                symbol="XLK", strategy="SECTOR_ROTATION",
                entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
            )
            trade.max_hold_date = datetime(2026, 3, 20, 10, 0, tzinfo=ET)

            expired = check_sector_max_hold({"XLK": trade}, now)
            assert "XLK" in expired

    def test_sector_max_hold_not_expired(self, mock_trading_client):
        """check_sector_max_hold skips non-expired trades."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import check_sector_max_hold

            now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)
            trade = _make_trade(symbol="XLK", strategy="SECTOR_ROTATION")
            trade.max_hold_date = datetime(2026, 3, 20, 10, 0, tzinfo=ET)

            expired = check_sector_max_hold({"XLK": trade}, now)
            assert expired == []

    def test_pairs_max_hold(self, mock_trading_client):
        """check_pairs_max_hold returns expired pairs trades."""
        with patch("execution.get_trading_client", return_value=mock_trading_client):
            from execution import check_pairs_max_hold

            now = datetime(2026, 3, 30, 10, 0, tzinfo=ET)
            trade = _make_trade(
                symbol="AAPL", strategy="PAIRS",
                pair_id="pair-001",
                entry_time=datetime(2026, 3, 10, 10, 0, tzinfo=ET),
            )
            trade.max_hold_date = datetime(2026, 3, 25, 10, 0, tzinfo=ET)

            expired = check_pairs_max_hold({"AAPL": trade}, now)
            assert "AAPL" in expired

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
