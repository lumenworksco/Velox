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


class TestSectorRotation:
    def test_sector_rotation_disabled(self, override_config):
        """Returns empty when SECTOR_ROTATION_ENABLED=False."""
        with override_config(SECTOR_ROTATION_ENABLED=False):
            try:
                from strategies.sector_rotation import SectorRotationStrategy
                sr = SectorRotationStrategy()
                from datetime import datetime
                from conftest import ET
                signals = sr.scan([], datetime(2026, 3, 13, 10, 30, tzinfo=ET))
                assert signals == []
            except ImportError:
                pytest.skip("sector_rotation not available")

    def test_sector_rotation_scanned_today_skips(self, override_config):
        """Returns empty if already scanned today."""
        with override_config(SECTOR_ROTATION_ENABLED=True):
            try:
                from strategies.sector_rotation import SectorRotationStrategy
                sr = SectorRotationStrategy()
                sr.scanned_today = True
                from datetime import datetime
                from conftest import ET
                signals = sr.scan([], datetime(2026, 3, 13, 10, 30, tzinfo=ET))
                assert signals == []
            except ImportError:
                pytest.skip("sector_rotation not available")
