"""Tests for watchdog.py — Watchdog, PositionReconciler, AuditTrail."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pytest

import config
from watchdog import (
    AuditTrail,
    HealthStatus,
    PositionReconciler,
    ReconciliationResult,
    Watchdog,
)


# =========================================================================
# HealthStatus dataclass
# =========================================================================

class TestHealthStatus:

    def test_creation_healthy(self):
        now = datetime.now(ZoneInfo("America/New_York"))
        hs = HealthStatus(
            timestamp=now,
            overall_healthy=True,
            checks={"scan_loop": {"healthy": True, "detail": "ok", "last_ok": now}},
        )
        assert hs.overall_healthy is True
        assert hs.timestamp == now
        assert hs.issues == []
        assert hs.recoveries_attempted == []

    def test_creation_unhealthy(self):
        now = datetime.now(ZoneInfo("America/New_York"))
        hs = HealthStatus(
            timestamp=now,
            overall_healthy=False,
            checks={},
            issues=["scan_loop stale"],
            recoveries_attempted=["logged_critical_stale_scan"],
        )
        assert hs.overall_healthy is False
        assert len(hs.issues) == 1
        assert len(hs.recoveries_attempted) == 1

    def test_default_lists(self):
        hs = HealthStatus(
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            overall_healthy=True,
            checks={},
        )
        assert hs.issues == []
        assert hs.recoveries_attempted == []
        # Verify independent default instances
        hs2 = HealthStatus(
            timestamp=datetime.now(ZoneInfo("America/New_York")), overall_healthy=True, checks={}
        )
        hs.issues.append("test")
        assert hs2.issues == []


# =========================================================================
# ReconciliationResult dataclass
# =========================================================================

class TestReconciliationResult:

    def test_creation(self):
        rr = ReconciliationResult(
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            positions_checked=5,
            phantoms_removed=["AAPL"],
            unknowns_found=["TSLA"],
            size_mismatches_fixed=["MSFT"],
            all_reconciled=False,
        )
        assert rr.positions_checked == 5
        assert rr.all_reconciled is False

    def test_defaults(self):
        rr = ReconciliationResult(
            timestamp=datetime.now(ZoneInfo("America/New_York")),
            positions_checked=0,
        )
        assert rr.phantoms_removed == []
        assert rr.unknowns_found == []
        assert rr.size_mismatches_fixed == []
        assert rr.all_reconciled is True


# =========================================================================
# Watchdog
# =========================================================================

class TestWatchdog:

    def test_check_health_all_healthy(self, in_memory_db):
        """All checks pass when everything is nominal."""
        now = datetime.now(ZoneInfo("America/New_York"))
        wd = Watchdog(
            last_scan_time_fn=lambda: now - timedelta(seconds=30),
            trading_client_fn=None,  # skip API/orphan checks
        )
        status = wd.check_health()
        assert isinstance(status, HealthStatus)
        assert status.checks["scan_loop"]["healthy"] is True
        assert status.checks["disk_space"]["healthy"] is True
        assert status.checks["database"]["healthy"] is True

    def test_scan_loop_stale(self, in_memory_db):
        """Detect a stale scan loop."""
        stale_time = datetime.now(ZoneInfo("America/New_York")) - timedelta(minutes=10)
        wd = Watchdog(
            last_scan_time_fn=lambda: stale_time,
            trading_client_fn=None,
        )
        status = wd.check_health()
        assert status.checks["scan_loop"]["healthy"] is False
        assert "stale" in status.checks["scan_loop"]["detail"]

    def test_scan_loop_none(self, in_memory_db):
        """Scan loop returns None (never run)."""
        wd = Watchdog(
            last_scan_time_fn=lambda: None,
            trading_client_fn=None,
        )
        status = wd.check_health()
        assert status.checks["scan_loop"]["healthy"] is False

    def test_scan_loop_no_fn(self, in_memory_db):
        """No scan function configured — should be healthy."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        status = wd.check_health()
        assert status.checks["scan_loop"]["healthy"] is True

    def test_api_check_success(self, in_memory_db, mock_trading_client):
        """API check succeeds with mock client."""
        wd = Watchdog(
            last_scan_time_fn=None,
            trading_client_fn=lambda: mock_trading_client,
        )
        status = wd.check_health()
        assert status.checks["api_responsive"]["healthy"] is True

    def test_api_check_failure(self, in_memory_db):
        """API check fails when client raises."""
        def bad_client():
            client = MagicMock()
            client.get_account.side_effect = Exception("connection refused")
            return client

        wd = Watchdog(
            last_scan_time_fn=None,
            trading_client_fn=bad_client,
        )
        status = wd.check_health()
        assert status.checks["api_responsive"]["healthy"] is False
        assert "API error" in status.checks["api_responsive"]["detail"]

    def test_database_writable(self, in_memory_db):
        """DB check succeeds with in-memory DB."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        status = wd.check_health()
        assert status.checks["database"]["healthy"] is True

    def test_disk_space_check(self, in_memory_db):
        """Disk space check runs without error."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        status = wd.check_health()
        # Should be healthy on any reasonable dev machine
        assert "disk_space" in status.checks

    def test_memory_check(self, in_memory_db):
        """Memory check runs without error."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        status = wd.check_health()
        assert "memory" in status.checks

    def test_orphaned_orders_none(self, in_memory_db):
        """No orphaned orders when client has no open orders."""
        client = MagicMock()
        client.get_account.return_value = MagicMock(equity="100000")
        client.get_orders.return_value = []
        client.get_all_positions.return_value = []

        wd = Watchdog(
            last_scan_time_fn=None,
            trading_client_fn=lambda: client,
        )
        status = wd.check_health()
        assert status.checks["orphaned_orders"]["healthy"] is True

    def test_health_history_tracked(self, in_memory_db):
        """Health history is accumulated."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd.check_health()
        wd.check_health()
        assert len(wd.health_history) == 2

    def test_overall_healthy_flag(self, in_memory_db):
        """Overall healthy is True when all checks pass."""
        wd = Watchdog(
            last_scan_time_fn=lambda: datetime.now(ZoneInfo("America/New_York")),
            trading_client_fn=None,
        )
        status = wd.check_health()
        assert status.overall_healthy is True
        assert status.issues == []


# =========================================================================
# Watchdog — Recovery
# =========================================================================

class TestWatchdogRecovery:

    def test_recover_scan_loop(self, in_memory_db):
        """Stale scan triggers critical log."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd.recover("scan_loop")
        # Should not raise

    def test_recover_api(self, in_memory_db):
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd.recover("api_responsive")

    def test_recover_orphaned_orders(self, in_memory_db):
        client = MagicMock()
        client.cancel_orders.return_value = None

        wd = Watchdog(
            last_scan_time_fn=None,
            trading_client_fn=lambda: client,
        )
        wd.recover("orphaned_orders")
        client.cancel_orders.assert_called_once()

    def test_recover_memory(self, in_memory_db):
        """Memory recovery clears data cache if available."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd.recover("memory")
        # Should not raise

    def test_recover_unknown_issue(self, in_memory_db):
        """Unknown issue type does nothing (no crash)."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd.recover("unknown_issue")


# =========================================================================
# Watchdog — Fail-open behavior
# =========================================================================

class TestWatchdogFailOpen:

    def test_check_health_internal_error(self, in_memory_db):
        """Watchdog.check_health never propagates exceptions."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        # Sabotage an internal method
        wd._check_health_inner = MagicMock(side_effect=RuntimeError("boom"))
        status = wd.check_health()
        assert isinstance(status, HealthStatus)
        assert status.overall_healthy is False
        assert any("watchdog_internal_error" in i for i in status.issues)

    def test_recover_internal_error(self, in_memory_db):
        """Watchdog.recover never propagates exceptions."""
        wd = Watchdog(last_scan_time_fn=None, trading_client_fn=None)
        wd._recover_inner = MagicMock(side_effect=RuntimeError("boom"))
        # Must not raise
        wd.recover("anything")

    def test_scan_fn_raises(self, in_memory_db):
        """If the scan time function raises, check still completes."""
        def bad_fn():
            raise ValueError("oops")

        wd = Watchdog(last_scan_time_fn=bad_fn, trading_client_fn=None)
        status = wd.check_health()
        assert status.checks["scan_loop"]["healthy"] is False
        assert "error" in status.checks["scan_loop"]["detail"]


# =========================================================================
# PositionReconciler
# =========================================================================

class TestPositionReconciler:

    def test_reconcile_all_match(self, in_memory_db, mock_trading_client):
        """No discrepancies when DB and broker match."""
        import database

        # Add a position to DB
        conn = database._get_conn()
        conn.execute(
            """INSERT INTO open_positions
               (symbol, strategy, side, entry_price, qty, entry_time,
                take_profit, stop_loss, alpaca_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AAPL", "ORB", "buy", 150.0, 10, "2026-03-13T10:00:00",
             155.0, 148.0, "order-1"),
        )
        conn.commit()

        # Set up broker positions to match
        from conftest import MockPosition
        mock_trading_client._positions = [
            MockPosition("AAPL", "10", "1500.0")
        ]

        rec = PositionReconciler(
            trading_client_fn=lambda: mock_trading_client
        )
        result = rec.reconcile()
        assert isinstance(result, ReconciliationResult)
        assert result.all_reconciled is True
        assert result.positions_checked == 1

    def test_phantom_position(self, in_memory_db, mock_trading_client):
        """Position in DB but not at broker gets removed."""
        import database

        conn = database._get_conn()
        conn.execute(
            """INSERT INTO open_positions
               (symbol, strategy, side, entry_price, qty, entry_time,
                take_profit, stop_loss, alpaca_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("GHOST", "ORB", "buy", 50.0, 5, "2026-03-13T10:00:00",
             55.0, 48.0, "order-2"),
        )
        conn.commit()

        mock_trading_client._positions = []  # nothing at broker

        rec = PositionReconciler(
            trading_client_fn=lambda: mock_trading_client
        )
        result = rec.reconcile()
        assert "GHOST" in result.phantoms_removed
        assert result.all_reconciled is False

        # Verify removed from DB
        rows = conn.execute(
            "SELECT * FROM open_positions WHERE symbol = 'GHOST'"
        ).fetchall()
        assert len(rows) == 0

    def test_unknown_position(self, in_memory_db, mock_trading_client):
        """Position at broker but not in DB gets flagged."""
        from conftest import MockPosition
        mock_trading_client._positions = [
            MockPosition("SURPRISE", "20", "2000.0")
        ]

        rec = PositionReconciler(
            trading_client_fn=lambda: mock_trading_client
        )
        result = rec.reconcile()
        assert "SURPRISE" in result.unknowns_found
        assert result.all_reconciled is False

    def test_size_mismatch(self, in_memory_db, mock_trading_client):
        """Qty mismatch between DB and broker gets fixed."""
        import database

        conn = database._get_conn()
        conn.execute(
            """INSERT INTO open_positions
               (symbol, strategy, side, entry_price, qty, entry_time,
                take_profit, stop_loss, alpaca_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("MSFT", "VWAP", "buy", 400.0, 10, "2026-03-13T10:00:00",
             410.0, 395.0, "order-3"),
        )
        conn.commit()

        from conftest import MockPosition
        mock_trading_client._positions = [
            MockPosition("MSFT", "15", "6000.0")  # broker has 15, DB has 10
        ]

        rec = PositionReconciler(
            trading_client_fn=lambda: mock_trading_client
        )
        result = rec.reconcile()
        assert "MSFT" in result.size_mismatches_fixed
        assert result.all_reconciled is False

        # Verify DB was updated
        row = conn.execute(
            "SELECT qty FROM open_positions WHERE symbol = 'MSFT'"
        ).fetchone()
        assert float(row["qty"]) == 15.0

    def test_reconcile_no_client(self, in_memory_db):
        """Reconciler works with no trading client (no broker positions)."""
        rec = PositionReconciler(trading_client_fn=None)
        result = rec.reconcile()
        assert result.all_reconciled is True

    def test_reconcile_fail_open(self, in_memory_db):
        """Reconciler never raises even if broker client explodes."""
        def exploding_client():
            raise ConnectionError("broker down")

        rec = PositionReconciler(trading_client_fn=exploding_client)
        result = rec.reconcile()
        assert isinstance(result, ReconciliationResult)
        assert result.all_reconciled is False


# =========================================================================
# AuditTrail
# =========================================================================

class TestAuditTrail:

    def test_generate_trace_id(self):
        tid = AuditTrail.generate_trace_id()
        assert tid.startswith("T-")
        assert len(tid) == 14  # "T-" + 12 hex chars

    def test_generate_trace_id_unique(self):
        ids = {AuditTrail.generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_log_event_and_get_trace(self, override_config):
        """Log events and retrieve them by trace_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "audit.log")
            with override_config(
                STRUCTURED_LOGGING_ENABLED=True,
                AUDIT_LOG_FILE=log_file,
            ):
                trail = AuditTrail(log_dir=tmpdir)
                tid = trail.generate_trace_id()

                trail.log_event(tid, "signal", {"symbol": "AAPL", "strategy": "ORB"})
                trail.log_event(tid, "risk_decision", {"approved": True})
                trail.log_event("other-trace", "signal", {"symbol": "MSFT"})

                # Flush handler
                if trail._logger:
                    for h in trail._logger.handlers:
                        h.flush()

                events = trail.get_trace(tid)
                assert len(events) == 2
                assert events[0]["event_type"] == "signal"
                assert events[1]["event_type"] == "risk_decision"

    def test_log_event_disabled(self, override_config):
        """When disabled, log_event is a no-op."""
        with override_config(STRUCTURED_LOGGING_ENABLED=False):
            trail = AuditTrail()
            trail.log_event("T-123", "signal", {"test": True})
            # No crash, no file written

    def test_get_trace_disabled(self, override_config):
        """When disabled, get_trace returns empty list."""
        with override_config(STRUCTURED_LOGGING_ENABLED=False):
            trail = AuditTrail()
            assert trail.get_trace("T-123") == []

    def test_log_event_includes_timestamp(self, override_config):
        """Each logged event has a timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "audit.log")
            with override_config(
                STRUCTURED_LOGGING_ENABLED=True,
                AUDIT_LOG_FILE=log_file,
            ):
                trail = AuditTrail(log_dir=tmpdir)
                tid = trail.generate_trace_id()
                trail.log_event(tid, "order", {"qty": 10})

                if trail._logger:
                    for h in trail._logger.handlers:
                        h.flush()

                events = trail.get_trace(tid)
                assert len(events) == 1
                assert "timestamp" in events[0]

    def test_get_trace_no_file(self, override_config):
        """get_trace returns empty when log file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "nonexistent.log")
            with override_config(
                STRUCTURED_LOGGING_ENABLED=True,
                AUDIT_LOG_FILE=log_file,
            ):
                trail = AuditTrail(log_dir=tmpdir)
                assert trail.get_trace("T-missing") == []

    def test_audit_trail_fail_open(self, override_config):
        """AuditTrail never raises from public methods."""
        with override_config(STRUCTURED_LOGGING_ENABLED=True):
            trail = AuditTrail()
            # Force logger to None to simulate setup failure
            trail._logger = None
            trail._enabled = True
            # Should not raise
            trail.log_event("T-123", "signal", {"test": True})
            assert trail.get_trace("T-123") == []


# =========================================================================
# Config flag gating
# =========================================================================

class TestConfigFlags:

    def test_watchdog_enabled_flag(self):
        assert hasattr(config, "WATCHDOG_ENABLED")
        assert isinstance(config.WATCHDOG_ENABLED, bool)

    def test_watchdog_check_interval(self):
        assert hasattr(config, "WATCHDOG_CHECK_INTERVAL")
        assert config.WATCHDOG_CHECK_INTERVAL == 300

    def test_reconciliation_interval(self):
        assert hasattr(config, "RECONCILIATION_INTERVAL")
        assert config.RECONCILIATION_INTERVAL == 1800

    def test_structured_logging_flag(self):
        assert hasattr(config, "STRUCTURED_LOGGING_ENABLED")
        assert isinstance(config.STRUCTURED_LOGGING_ENABLED, bool)

    def test_audit_trail_retention(self):
        assert hasattr(config, "AUDIT_TRAIL_RETENTION_DAYS")
        assert config.AUDIT_TRAIL_RETENTION_DAYS == 365

    def test_reconciliation_enabled_flag(self):
        assert hasattr(config, "RECONCILIATION_ENABLED")
        assert isinstance(config.RECONCILIATION_ENABLED, bool)
