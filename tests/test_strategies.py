"""Tests for strategy signal generation and Signal dataclass."""

import pytest

from strategies.base import Signal


class TestSignalDataclass:
    def test_signal_fields(self):
        """Signal has correct fields."""
        sig = Signal(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, take_profit=155.0, stop_loss=148.0,
            reason="test", hold_type="day",
        )
        assert sig.symbol == "AAPL"
        assert sig.strategy == "ORB"
        assert sig.side == "buy"
        assert sig.entry_price == 150.0
        assert sig.take_profit == 155.0
        assert sig.stop_loss == 148.0

    def test_signal_pair_id_default(self):
        """pair_id defaults to empty string."""
        sig = Signal(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, take_profit=155.0, stop_loss=148.0,
            reason="test",
        )
        assert sig.pair_id == ""

    def test_signal_hold_type_default(self):
        """hold_type defaults to 'day'."""
        sig = Signal(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, take_profit=155.0, stop_loss=148.0,
            reason="test",
        )
        assert sig.hold_type == "day"

    def test_signal_with_pair_id(self):
        """Signal with pair_id set correctly."""
        sig = Signal(
            symbol="AAPL", strategy="PAIRS", side="buy",
            entry_price=150.0, take_profit=155.0, stop_loss=145.0,
            reason="pairs long", hold_type="swing", pair_id="pair-001",
        )
        assert sig.pair_id == "pair-001"
        assert sig.hold_type == "swing"



# TestSectorRotation removed — strategy archived in V10
