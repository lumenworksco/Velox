"""Tests for V8 execution analytics."""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch

ET = ZoneInfo("America/New_York")


class TestExecutionAnalytics:

    def _make_analytics(self):
        from analytics.execution_analytics import ExecutionAnalytics
        return ExecutionAnalytics()

    def test_record_submission(self):
        ea = self._make_analytics()
        now = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        ea.record_submission("ord-1", "AAPL", "STAT_MR", "buy", 150.0, 10, now)
        assert "ord-1" in ea._pending

    def test_record_fill(self):
        ea = self._make_analytics()
        now = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        ea.record_submission("ord-1", "AAPL", "STAT_MR", "buy", 150.0, 10, now)

        with patch("database.save_execution_analytics"):
            record = ea.record_fill("ord-1", 150.10, 10, now + timedelta(milliseconds=200))

        assert record is not None
        assert record.filled_price == 150.10
        assert record.latency_ms == 200
        assert record.slippage_pct > 0  # Buy slippage positive

    def test_sell_slippage_calculation(self):
        ea = self._make_analytics()
        now = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        ea.record_submission("ord-1", "AAPL", "STAT_MR", "sell", 150.0, 10, now)

        with patch("database.save_execution_analytics"):
            record = ea.record_fill("ord-1", 149.90, 10, now + timedelta(milliseconds=100))

        # Sell: slippage = (expected - filled) / expected = 0.1/150 > 0
        assert record.slippage_pct > 0

    def test_fill_rate(self):
        ea = self._make_analytics()
        now = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        ea.record_submission("ord-1", "AAPL", "STAT_MR", "buy", 150.0, 100, now)

        with patch("database.save_execution_analytics"):
            record = ea.record_fill("ord-1", 150.0, 50, now)

        assert record.fill_rate == 0.5

    def test_unknown_order_fill(self):
        ea = self._make_analytics()
        with patch("database.save_execution_analytics"):
            record = ea.record_fill("nonexistent", 150.0, 10)
        assert record is None

    def test_execution_db_save(self, in_memory_db):
        import database
        now = datetime(2026, 3, 13, 10, 0, tzinfo=ET)
        database.save_execution_analytics(
            order_id="ord-1", symbol="AAPL", strategy="STAT_MR",
            side="buy", expected_price=150.0, filled_price=150.10,
            slippage_pct=0.00067, submitted_at=now,
            filled_at=now + timedelta(milliseconds=200),
            latency_ms=200, qty_requested=10, qty_filled=10, fill_rate=1.0,
        )
        rows = in_memory_db.execute("SELECT * FROM execution_analytics").fetchall()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
