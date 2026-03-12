"""Tests for database.py — CRUD operations, schema, V4 fields."""

from datetime import datetime

import pytest

import database
from conftest import _make_trade, ET


class TestInitDb:
    def test_init_db_creates_tables(self):
        """All expected tables exist after init_db."""
        conn = database._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}

        expected = {
            "trades", "signals", "open_positions",
            "daily_snapshots", "backtest_results",
            "model_performance", "optimization_history",
            "allocation_history",
        }
        assert expected.issubset(table_names)

    def test_open_positions_has_v4_columns(self):
        """open_positions table has V4 columns."""
        conn = database._get_conn()
        cursor = conn.execute("PRAGMA table_info(open_positions)")
        cols = {row["name"] for row in cursor.fetchall()}

        assert "pair_id" in cols
        assert "partial_exits" in cols
        assert "highest_price_seen" in cols
        assert "entry_atr" in cols


class TestTradeLogging:
    def test_log_and_get_trade(self):
        """Round-trip: log a trade and retrieve it."""
        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        database.log_trade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, exit_price=155.0, qty=10,
            entry_time=now, exit_time=now,
            exit_reason="take_profit", pnl=50.0, pnl_pct=0.033,
        )

        trades = database.get_all_trades()
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "AAPL"
        assert t["strategy"] == "ORB"
        assert t["pnl"] == 50.0


class TestOpenPositions:
    def test_save_load_open_positions(self):
        """Save and restore open positions with V4 fields."""
        trade = _make_trade(
            symbol="AAPL", strategy="PAIRS", pair_id="pair-001",
            partial_exits=1, highest_price_seen=155.0, entry_atr=1.5,
        )
        database.save_open_positions({"AAPL": trade})

        rows = database.load_open_positions()
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "AAPL"
        assert row["strategy"] == "PAIRS"
        assert row["pair_id"] == "pair-001"
        assert row["partial_exits"] == 1
        assert row["highest_price_seen"] == 155.0
        assert row["entry_atr"] == 1.5

    def test_save_clears_previous(self):
        """save_open_positions replaces all rows."""
        trade1 = _make_trade(symbol="AAPL")
        trade2 = _make_trade(symbol="MSFT")

        database.save_open_positions({"AAPL": trade1, "MSFT": trade2})
        assert len(database.load_open_positions()) == 2

        # Save only one
        database.save_open_positions({"NVDA": _make_trade(symbol="NVDA")})
        rows = database.load_open_positions()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "NVDA"


class TestDailySnapshot:
    def test_daily_snapshot(self):
        """Insert and query daily snapshot."""
        database.save_daily_snapshot(
            date="2026-03-13", portfolio_value=100_500.0,
            cash=50_000.0, day_pnl=500.0, day_pnl_pct=0.005,
            total_trades=5, win_rate=0.60, sharpe_rolling=1.5,
        )

        snapshots = database.get_daily_snapshots(days=7)
        assert len(snapshots) == 1
        s = snapshots[0]
        assert s["date"] == "2026-03-13"
        assert s["portfolio_value"] == 100_500.0
        assert s["win_rate"] == 0.60


class TestSignalLogging:
    def test_signal_logging(self):
        """Log and query signals."""
        # Use actual "today" in ET so get_signal_stats_today matches
        import config
        now = datetime.now(config.ET).replace(hour=10, minute=30, second=0, microsecond=0)
        database.log_signal(now, "AAPL", "ORB", "buy", True, "")
        database.log_signal(now, "MSFT", "VWAP", "sell", False, "ml_filter_0.35")

        stats = database.get_signal_stats_today()
        assert stats["total"] == 2
        assert stats["acted"] == 1
        assert stats["skipped"] == 1
