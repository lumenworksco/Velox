"""Tests for database.py — CRUD for all V6/V7/V8 tables.

Complements test_database.py which already covers trades, open_positions,
daily_snapshots, and signals.  This file covers the newer tables:
kelly_params, monte_carlo_results, and execution_analytics, and also
provides explicit round-trip tests for signals and snapshots aligned with
the test_database_tables spec.
"""

from datetime import datetime, timezone

import pytest

import database
from conftest import ET, _make_trade


# ---------------------------------------------------------------------------
# Schema completeness
# ---------------------------------------------------------------------------

class TestInitDb:
    """Verify that init_db creates all expected tables."""

    def test_all_expected_tables_exist(self):
        """Every required table must be present after init_db."""
        conn = database._get_conn()
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "trades",
            "signals",
            "open_positions",
            "daily_snapshots",
            "backtest_results",
            "model_performance",
            "optimization_history",
            "allocation_history",
            "shadow_trades",
            "ou_parameters",
            "kalman_pairs",
            "consistency_log",
            "kelly_params",
            "monte_carlo_results",
            "execution_analytics",
        }
        missing = expected - tables
        assert not missing, f"Missing tables: {missing}"

    def test_v8_tables_exist(self):
        """V8 tables (kelly, monte_carlo, execution_analytics) must exist."""
        conn = database._get_conn()
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in ("kelly_params", "monte_carlo_results", "execution_analytics"):
            assert t in tables, f"V8 table '{t}' not found"


# ---------------------------------------------------------------------------
# Trade round-trip
# ---------------------------------------------------------------------------

class TestLogTradeRoundTrip:
    """log_trade / get_recent_trades round-trip."""

    def test_log_and_retrieve_trade(self):
        """A logged trade is retrievable via get_recent_trades."""
        now = datetime(2026, 3, 15, 14, 0, tzinfo=ET)
        database.log_trade(
            symbol="NVDA",
            strategy="ORB",
            side="buy",
            entry_price=800.0,
            exit_price=850.0,
            qty=5,
            entry_time=now,
            exit_time=now,
            exit_reason="take_profit",
            pnl=250.0,
            pnl_pct=0.0625,
        )

        trades = database.get_recent_trades(days=1)
        assert len(trades) == 1
        t = trades[0]
        assert t["symbol"] == "NVDA"
        assert t["strategy"] == "ORB"
        assert t["side"] == "buy"
        assert t["pnl"] == pytest.approx(250.0)
        assert t["pnl_pct"] == pytest.approx(0.0625)
        assert t["exit_reason"] == "take_profit"

    def test_multiple_trades_ordered_desc(self):
        """get_recent_trades returns rows ordered by exit_time DESC."""
        t1 = datetime(2026, 3, 15, 10, 0, tzinfo=ET)
        t2 = datetime(2026, 3, 15, 14, 0, tzinfo=ET)
        for ts, sym in [(t1, "AAPL"), (t2, "MSFT")]:
            database.log_trade(
                symbol=sym, strategy="VWAP", side="sell",
                entry_price=100.0, exit_price=95.0, qty=10,
                entry_time=ts, exit_time=ts,
                exit_reason="stop_loss", pnl=-50.0, pnl_pct=-0.05,
            )

        trades = database.get_recent_trades(days=1)
        # Most recent first
        assert trades[0]["symbol"] == "MSFT"
        assert trades[1]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Signal round-trip
# ---------------------------------------------------------------------------

class TestLogSignalRoundTrip:
    """log_signal / get_signals_by_date round-trip."""

    def test_log_and_retrieve_signals_by_date(self):
        """Signals logged for a date are returned by get_signals_by_date."""
        ts = datetime(2026, 3, 15, 10, 30, tzinfo=ET)
        database.log_signal(ts, "AAPL", "ORB", "buy", acted_on=True, skip_reason="")
        database.log_signal(ts, "MSFT", "VWAP", "sell", acted_on=False, skip_reason="vol_low")

        rows = database.get_signals_by_date("2026-03-15")
        assert len(rows) == 2
        symbols = {r["symbol"] for r in rows}
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_acted_on_flag_preserved(self):
        """acted_on flag is stored and retrieved correctly."""
        ts = datetime(2026, 3, 15, 11, 0, tzinfo=ET)
        database.log_signal(ts, "AMD", "ORB", "buy", acted_on=True, skip_reason="")
        database.log_signal(ts, "AMD", "ORB", "buy", acted_on=False, skip_reason="no_volume")

        rows = database.get_signals_by_date("2026-03-15")
        acted = [r for r in rows if r["acted_on"] == 1]
        skipped = [r for r in rows if r["acted_on"] == 0]
        assert len(acted) == 1
        assert len(skipped) == 1
        assert skipped[0]["skip_reason"] == "no_volume"

    def test_signals_for_different_date_not_returned(self):
        """get_signals_by_date excludes signals from other dates."""
        ts_today = datetime(2026, 3, 15, 9, 35, tzinfo=ET)
        ts_yesterday = datetime(2026, 3, 14, 9, 35, tzinfo=ET)
        database.log_signal(ts_today, "SPY", "ORB", "buy", acted_on=True)
        database.log_signal(ts_yesterday, "QQQ", "ORB", "sell", acted_on=True)

        rows = database.get_signals_by_date("2026-03-15")
        assert all(r["symbol"] == "SPY" for r in rows)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Open positions round-trip
# ---------------------------------------------------------------------------

class TestOpenPositionsRoundTrip:
    """save_open_positions / load_open_positions round-trip."""

    def test_save_and_load_single_position(self):
        """A saved position is returned by load_open_positions."""
        trade = _make_trade(
            symbol="TSLA", strategy="VWAP", side="sell",
            entry_price=250.0, qty=8,
        )
        database.save_open_positions({"TSLA": trade})

        rows = database.load_open_positions()
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "TSLA"
        assert r["strategy"] == "VWAP"
        assert r["side"] == "sell"
        assert r["entry_price"] == pytest.approx(250.0)
        assert r["qty"] == 8

    def test_save_clears_old_positions(self):
        """Calling save_open_positions replaces all existing rows."""
        t1 = _make_trade(symbol="AAPL")
        t2 = _make_trade(symbol="MSFT")
        database.save_open_positions({"AAPL": t1, "MSFT": t2})
        assert len(database.load_open_positions()) == 2

        t3 = _make_trade(symbol="NVDA")
        database.save_open_positions({"NVDA": t3})
        rows = database.load_open_positions()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "NVDA"

    def test_save_empty_clears_all(self):
        """Saving an empty dict removes all open positions."""
        database.save_open_positions({"AAPL": _make_trade()})
        database.save_open_positions({})
        assert database.load_open_positions() == []

    def test_v4_fields_preserved(self):
        """V4 fields (pair_id, partial_exits, etc.) survive the round-trip."""
        trade = _make_trade(
            symbol="AAPL", strategy="KALMAN_PAIRS",
            pair_id="AAPL-MSFT-001", partial_exits=2,
            highest_price_seen=160.0, entry_atr=2.5,
        )
        database.save_open_positions({"AAPL": trade})

        rows = database.load_open_positions()
        r = rows[0]
        assert r["pair_id"] == "AAPL-MSFT-001"
        assert r["partial_exits"] == 2
        assert r["highest_price_seen"] == pytest.approx(160.0)
        assert r["entry_atr"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Daily snapshots round-trip
# ---------------------------------------------------------------------------

class TestDailySnapshotRoundTrip:
    """save_daily_snapshot / get_daily_snapshots round-trip."""

    def test_save_and_retrieve_snapshot(self):
        """A saved snapshot is returned by get_daily_snapshots."""
        database.save_daily_snapshot(
            date="2026-03-15",
            portfolio_value=102_000.0,
            cash=55_000.0,
            day_pnl=2_000.0,
            day_pnl_pct=0.02,
            total_trades=8,
            win_rate=0.75,
            sharpe_rolling=1.8,
        )

        rows = database.get_daily_snapshots(days=7)
        assert len(rows) == 1
        s = rows[0]
        assert s["date"] == "2026-03-15"
        assert s["portfolio_value"] == pytest.approx(102_000.0)
        assert s["cash"] == pytest.approx(55_000.0)
        assert s["day_pnl"] == pytest.approx(2_000.0)
        assert s["day_pnl_pct"] == pytest.approx(0.02)
        assert s["total_trades"] == 8
        assert s["win_rate"] == pytest.approx(0.75)
        assert s["sharpe_rolling"] == pytest.approx(1.8)

    def test_upsert_updates_existing_date(self):
        """INSERT OR REPLACE means saving the same date twice updates the row."""
        database.save_daily_snapshot(
            date="2026-03-15", portfolio_value=100_000.0,
            cash=50_000.0, day_pnl=0.0, day_pnl_pct=0.0,
            total_trades=0, win_rate=0.0, sharpe_rolling=0.0,
        )
        database.save_daily_snapshot(
            date="2026-03-15", portfolio_value=105_000.0,
            cash=52_000.0, day_pnl=5_000.0, day_pnl_pct=0.05,
            total_trades=3, win_rate=0.67, sharpe_rolling=1.2,
        )

        rows = database.get_daily_snapshots(days=7)
        assert len(rows) == 1
        assert rows[0]["portfolio_value"] == pytest.approx(105_000.0)

    def test_multiple_snapshots_ordered_desc(self):
        """get_daily_snapshots returns most-recent date first."""
        for date in ("2026-03-13", "2026-03-14", "2026-03-15"):
            database.save_daily_snapshot(
                date=date, portfolio_value=100_000.0, cash=50_000.0,
                day_pnl=0.0, day_pnl_pct=0.0, total_trades=0,
                win_rate=0.0, sharpe_rolling=0.0,
            )
        rows = database.get_daily_snapshots(days=7)
        assert rows[0]["date"] == "2026-03-15"
        assert rows[-1]["date"] == "2026-03-13"


# ---------------------------------------------------------------------------
# Kelly params
# ---------------------------------------------------------------------------

class TestSaveKellyParams:
    """save_kelly_params inserts a record into kelly_params."""

    def test_save_kelly_params_basic(self, in_memory_db):
        """Saved Kelly params are queryable from the DB."""
        database.save_kelly_params(
            strategy="STAT_MR",
            win_rate=0.58,
            avg_win_loss=1.8,
            kelly_f=0.36,
            half_kelly_f=0.18,
            sample_size=80,
        )

        rows = in_memory_db.execute(
            "SELECT * FROM kelly_params WHERE strategy = 'STAT_MR'"
        ).fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["win_rate"] == pytest.approx(0.58)
        assert r["avg_win_loss"] == pytest.approx(1.8)
        assert r["kelly_f"] == pytest.approx(0.36)
        assert r["half_kelly_f"] == pytest.approx(0.18)
        assert r["sample_size"] == 80

    def test_save_kelly_params_multiple_strategies(self, in_memory_db):
        """Multiple strategies can each have their own Kelly params."""
        for strategy in ("STAT_MR", "ORB", "VWAP"):
            database.save_kelly_params(
                strategy=strategy,
                win_rate=0.5,
                avg_win_loss=1.5,
                kelly_f=0.17,
                half_kelly_f=0.085,
                sample_size=50,
            )

        rows = in_memory_db.execute("SELECT * FROM kelly_params").fetchall()
        assert len(rows) == 3
        strategies = {r["strategy"] for r in rows}
        assert strategies == {"STAT_MR", "ORB", "VWAP"}

    def test_save_kelly_params_computed_at_set(self, in_memory_db):
        """computed_at timestamp is populated automatically."""
        database.save_kelly_params(
            strategy="ORB",
            win_rate=0.55,
            avg_win_loss=2.0,
            kelly_f=0.275,
            half_kelly_f=0.138,
            sample_size=40,
        )
        row = in_memory_db.execute(
            "SELECT computed_at FROM kelly_params WHERE strategy = 'ORB'"
        ).fetchone()
        assert row is not None
        assert row["computed_at"] is not None
        assert len(row["computed_at"]) > 0


# ---------------------------------------------------------------------------
# Monte Carlo results
# ---------------------------------------------------------------------------

class TestSaveMonteCarloResult:
    """save_monte_carlo_result inserts a record into monte_carlo_results."""

    def test_save_monte_carlo_basic(self, in_memory_db):
        """Saved Monte Carlo result is queryable from the DB."""
        database.save_monte_carlo_result(
            date="2026-03-15",
            var_95=-0.035,
            var_99=-0.055,
            cvar_95=-0.042,
            cvar_99=-0.065,
            horizon_days=21,
            simulations=10_000,
        )

        rows = in_memory_db.execute(
            "SELECT * FROM monte_carlo_results WHERE date = '2026-03-15'"
        ).fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["var_95"] == pytest.approx(-0.035)
        assert r["var_99"] == pytest.approx(-0.055)
        assert r["cvar_95"] == pytest.approx(-0.042)
        assert r["cvar_99"] == pytest.approx(-0.065)
        assert r["horizon_days"] == 21
        assert r["simulations"] == 10_000

    def test_save_multiple_monte_carlo_results(self, in_memory_db):
        """Multiple Monte Carlo results can exist for different dates."""
        for date in ("2026-03-13", "2026-03-14", "2026-03-15"):
            database.save_monte_carlo_result(
                date=date, var_95=-0.03, var_99=-0.05,
                cvar_95=-0.04, cvar_99=-0.06,
                horizon_days=21, simulations=1000,
            )

        rows = in_memory_db.execute("SELECT * FROM monte_carlo_results").fetchall()
        assert len(rows) == 3

    def test_monte_carlo_computed_at_populated(self, in_memory_db):
        """computed_at field is set when saving Monte Carlo results."""
        database.save_monte_carlo_result(
            date="2026-03-15", var_95=-0.03, var_99=-0.05,
            cvar_95=-0.04, cvar_99=-0.06, horizon_days=21, simulations=5000,
        )
        row = in_memory_db.execute(
            "SELECT computed_at FROM monte_carlo_results"
        ).fetchone()
        assert row["computed_at"] is not None


# ---------------------------------------------------------------------------
# Execution analytics
# ---------------------------------------------------------------------------

class TestSaveExecutionAnalytics:
    """save_execution_analytics / get_execution_analytics round-trip."""

    def _save_sample(self, order_id="ord-001", symbol="AAPL", strategy="ORB",
                     side="buy", expected=150.0, filled=150.05):
        """Helper to insert one execution analytics record."""
        now = datetime(2026, 3, 15, 10, 0, tzinfo=ET)
        database.save_execution_analytics(
            order_id=order_id,
            symbol=symbol,
            strategy=strategy,
            side=side,
            expected_price=expected,
            filled_price=filled,
            slippage_pct=(filled - expected) / expected,
            submitted_at=now,
            filled_at=now,
            latency_ms=42,
            qty_requested=10,
            qty_filled=10,
            fill_rate=1.0,
        )

    def test_save_and_retrieve(self):
        """A saved execution analytics record is returned by get_execution_analytics."""
        self._save_sample(order_id="ord-001", symbol="AAPL", strategy="ORB")

        rows = database.get_execution_analytics(days=1)
        assert len(rows) == 1
        r = rows[0]
        assert r["order_id"] == "ord-001"
        assert r["symbol"] == "AAPL"
        assert r["strategy"] == "ORB"
        assert r["latency_ms"] == 42
        assert r["qty_filled"] == 10
        assert r["fill_rate"] == pytest.approx(1.0)

    def test_filter_by_strategy(self):
        """get_execution_analytics returns only records for requested strategy."""
        self._save_sample(order_id="ord-001", strategy="ORB")
        self._save_sample(order_id="ord-002", strategy="VWAP")
        self._save_sample(order_id="ord-003", strategy="ORB")

        orb_rows = database.get_execution_analytics(strategy="ORB", days=1)
        assert len(orb_rows) == 2
        assert all(r["strategy"] == "ORB" for r in orb_rows)

        vwap_rows = database.get_execution_analytics(strategy="VWAP", days=1)
        assert len(vwap_rows) == 1

    def test_slippage_pct_stored_correctly(self):
        """slippage_pct is stored and retrieved with correct sign."""
        self._save_sample(expected=150.0, filled=150.15)
        rows = database.get_execution_analytics(days=1)
        assert rows[0]["slippage_pct"] == pytest.approx(0.15 / 150.0, rel=1e-4)

    def test_all_records_returned_without_strategy_filter(self):
        """get_execution_analytics with no strategy returns all records."""
        for sym in ("AAPL", "MSFT", "NVDA"):
            self._save_sample(order_id=f"ord-{sym}", symbol=sym, strategy="ORB")

        rows = database.get_execution_analytics(days=1)
        assert len(rows) == 3
