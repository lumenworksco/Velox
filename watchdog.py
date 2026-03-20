"""Resilience & Operational Hardening — Watchdog, Position Reconciler, Audit Trail."""

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HealthStatus:
    """Result of a full health check cycle."""

    timestamp: datetime
    overall_healthy: bool
    checks: dict  # {check_name: {"healthy": bool, "detail": str, "last_ok": datetime}}
    issues: list[str] = field(default_factory=list)
    recoveries_attempted: list[str] = field(default_factory=list)


@dataclass
class ReconciliationResult:
    """Result of a position reconciliation run."""

    timestamp: datetime
    positions_checked: int
    phantoms_removed: list[str] = field(default_factory=list)
    unknowns_found: list[str] = field(default_factory=list)
    size_mismatches_fixed: list[str] = field(default_factory=list)
    all_reconciled: bool = True


# =============================================================================
# Watchdog
# =============================================================================

class Watchdog:
    """
    Monitors bot health and automatically recovers from failures.

    Completely fail-open: the watchdog itself must never crash the bot.
    All public methods catch exceptions and log them rather than propagating.
    """

    def __init__(self, last_scan_time_fn=None, trading_client_fn=None):
        """
        Args:
            last_scan_time_fn: callable returning datetime of last scan loop tick.
            trading_client_fn: callable returning the Alpaca TradingClient instance.
        """
        self._last_scan_time_fn = last_scan_time_fn
        self._trading_client_fn = trading_client_fn
        self._health_history: list[HealthStatus] = []
        self._last_ok: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_health(self) -> HealthStatus:
        """Run all health checks and return a consolidated HealthStatus.

        Never raises — returns a degraded HealthStatus on internal error.
        """
        try:
            return self._check_health_inner()
        except Exception as e:
            logger.error(f"Watchdog.check_health itself failed: {e}")
            return HealthStatus(
                timestamp=datetime.now(config.ET),
                overall_healthy=False,
                checks={},
                issues=[f"watchdog_internal_error: {e}"],
            )

    def recover(self, issue: str):
        """Attempt automatic recovery for a known issue type.

        Never raises.
        """
        try:
            self._recover_inner(issue)
        except Exception as e:
            logger.error(f"Watchdog.recover failed for '{issue}': {e}")

    @property
    def health_history(self) -> list[HealthStatus]:
        return list(self._health_history)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _check_health_inner(self) -> HealthStatus:
        checks: dict = {}
        issues: list[str] = []
        recoveries: list[str] = []
        now = datetime.now(config.ET)

        # 1. Main scan loop freshness
        checks["scan_loop"] = self._check_scan_loop(now)
        # 2. Alpaca API responsive
        checks["api_responsive"] = self._check_api(now)
        # 3. Database writable
        checks["database"] = self._check_database(now)
        # 4. Disk space
        checks["disk_space"] = self._check_disk_space(now)
        # 5. Memory usage
        checks["memory"] = self._check_memory(now)
        # 6. Orphaned orders
        checks["orphaned_orders"] = self._check_orphaned_orders(now)

        for name, result in checks.items():
            if result["healthy"]:
                self._last_ok[name] = now
            else:
                issues.append(f"{name}: {result['detail']}")
                # Attempt recovery
                action = self._recover_inner(name)
                if action:
                    recoveries.append(action)

        overall = len(issues) == 0

        status = HealthStatus(
            timestamp=now,
            overall_healthy=overall,
            checks=checks,
            issues=issues,
            recoveries_attempted=recoveries,
        )
        self._health_history.append(status)
        # Keep last 100 entries
        if len(self._health_history) > 100:
            self._health_history = self._health_history[-100:]
        return status

    def _check_scan_loop(self, now: datetime) -> dict:
        """Check that the main scan loop ran within the last 5 minutes."""
        try:
            if self._last_scan_time_fn is None:
                return {"healthy": True, "detail": "no scan tracker configured",
                        "last_ok": now}
            last_scan = self._last_scan_time_fn()
            if last_scan is None:
                return {"healthy": False, "detail": "scan loop has never run",
                        "last_ok": self._last_ok.get("scan_loop")}
            # Ensure both are timezone-aware for comparison
            if last_scan.tzinfo is None:
                last_scan = last_scan.replace(tzinfo=config.ET)
            stale = (now - last_scan) > timedelta(
                seconds=config.WATCHDOG_CHECK_INTERVAL
            )
            if stale:
                return {"healthy": False,
                        "detail": f"last scan {last_scan.isoformat()} is stale",
                        "last_ok": self._last_ok.get("scan_loop")}
            return {"healthy": True, "detail": "ok", "last_ok": now}
        except Exception as e:
            return {"healthy": False, "detail": f"error: {e}",
                    "last_ok": self._last_ok.get("scan_loop")}

    def _check_api(self, now: datetime) -> dict:
        """Test Alpaca API with account endpoint."""
        try:
            if self._trading_client_fn is None:
                return {"healthy": True, "detail": "no client configured",
                        "last_ok": now}
            client = self._trading_client_fn()
            acct = client.get_account()
            return {"healthy": True,
                    "detail": f"equity={acct.equity}", "last_ok": now}
        except Exception as e:
            return {"healthy": False, "detail": f"API error: {e}",
                    "last_ok": self._last_ok.get("api_responsive")}

    def _check_database(self, now: datetime) -> dict:
        """Verify the database is writable."""
        try:
            import database
            conn = database._get_conn()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _watchdog_ping (ts TEXT)"
            )
            conn.execute(
                "INSERT INTO _watchdog_ping (ts) VALUES (?)",
                (now.isoformat(),),
            )
            conn.execute("DELETE FROM _watchdog_ping")
            conn.commit()
            return {"healthy": True, "detail": "ok", "last_ok": now}
        except Exception as e:
            return {"healthy": False, "detail": f"DB error: {e}",
                    "last_ok": self._last_ok.get("database")}

    def _check_disk_space(self, now: datetime) -> dict:
        """Check free disk space > 500 MB."""
        try:
            usage = shutil.disk_usage(os.getcwd())
            free_mb = usage.free / (1024 * 1024)
            if free_mb < 500:
                return {"healthy": False,
                        "detail": f"low disk: {free_mb:.0f}MB free",
                        "last_ok": self._last_ok.get("disk_space")}
            return {"healthy": True,
                    "detail": f"{free_mb:.0f}MB free", "last_ok": now}
        except Exception as e:
            return {"healthy": False, "detail": f"disk check error: {e}",
                    "last_ok": self._last_ok.get("disk_space")}

    def _check_memory(self, now: datetime) -> dict:
        """Check memory usage < 80%. Falls back to os-based if psutil unavailable."""
        try:
            try:
                import psutil
                mem = psutil.virtual_memory()
                pct = mem.percent
            except ImportError:
                # Fallback: read from /proc/meminfo on Linux, or skip on macOS
                import platform
                if platform.system() == "Linux":
                    with open("/proc/meminfo") as f:
                        lines = f.readlines()
                    info = {}
                    for line in lines:
                        parts = line.split()
                        info[parts[0].rstrip(":")] = int(parts[1])
                    total = info.get("MemTotal", 1)
                    avail = info.get("MemAvailable", total)
                    pct = (1 - avail / total) * 100
                else:
                    # macOS / other: report healthy, can't check without psutil
                    return {"healthy": True,
                            "detail": "psutil unavailable, skipping",
                            "last_ok": now}

            if pct > 80:
                return {"healthy": False,
                        "detail": f"memory at {pct:.1f}%",
                        "last_ok": self._last_ok.get("memory")}
            return {"healthy": True,
                    "detail": f"memory at {pct:.1f}%", "last_ok": now}
        except Exception as e:
            return {"healthy": False, "detail": f"memory check error: {e}",
                    "last_ok": self._last_ok.get("memory")}

    def _check_orphaned_orders(self, now: datetime) -> dict:
        """Check for orders submitted at broker but not tracked internally."""
        try:
            if self._trading_client_fn is None:
                return {"healthy": True, "detail": "no client configured",
                        "last_ok": now}
            client = self._trading_client_fn()
            # Get broker's open orders
            open_orders = []
            if hasattr(client, "get_orders"):
                open_orders = client.get_orders() or []
            if len(open_orders) == 0:
                return {"healthy": True, "detail": "no open orders",
                        "last_ok": now}

            # Get tracked positions from DB
            import database
            tracked = database.load_open_positions()
            tracked_order_ids = {
                pos.get("alpaca_order_id", "") for pos in tracked
            }
            orphaned = [
                o for o in open_orders
                if str(getattr(o, "id", "")) not in tracked_order_ids
            ]
            if orphaned:
                symbols = [getattr(o, "symbol", "?") for o in orphaned]
                return {"healthy": False,
                        "detail": f"orphaned orders: {symbols}",
                        "last_ok": self._last_ok.get("orphaned_orders")}
            return {"healthy": True, "detail": "ok", "last_ok": now}
        except Exception as e:
            return {"healthy": False,
                    "detail": f"orphan check error: {e}",
                    "last_ok": self._last_ok.get("orphaned_orders")}

    def _recover_inner(self, issue: str) -> str | None:
        """Attempt auto-recovery for a known issue. Returns action taken or None."""
        if issue == "scan_loop":
            logger.critical(
                "WATCHDOG: Scan loop appears stale — possible hang detected"
            )
            return "logged_critical_stale_scan"

        if issue == "api_responsive":
            logger.warning("WATCHDOG: Alpaca API unresponsive — will retry next check")
            return "logged_api_retry"

        if issue == "orphaned_orders":
            try:
                if self._trading_client_fn is None:
                    return None
                client = self._trading_client_fn()
                if hasattr(client, "cancel_orders"):
                    client.cancel_orders()
                    logger.warning("WATCHDOG: Cancelled all orphaned orders")
                    return "cancelled_orphaned_orders"
            except Exception as e:
                logger.error(f"WATCHDOG: Failed to cancel orphaned orders: {e}")
            return None

        if issue == "memory":
            # Clear caches in the data module if available
            try:
                import data
                if hasattr(data, "_cache") and isinstance(data._cache, dict):
                    data._cache.clear()
                    logger.info("WATCHDOG: Cleared data cache to free memory")
                    return "cleared_data_cache"
            except Exception:
                pass
            return None

        return None


# =============================================================================
# Position Reconciler
# =============================================================================

class PositionReconciler:
    """
    Ensures bot's internal state matches broker's actual state.
    Run every 30 minutes and at market open.

    Completely fail-open: never raises from public methods.
    """

    def __init__(self, trading_client_fn=None):
        """
        Args:
            trading_client_fn: callable returning the Alpaca TradingClient.
        """
        self._trading_client_fn = trading_client_fn

    def reconcile(self) -> ReconciliationResult:
        """Compare open_positions with broker's actual positions.

        Never raises — returns a partial result on error.
        """
        try:
            return self._reconcile_inner()
        except Exception as e:
            logger.error(f"PositionReconciler.reconcile failed: {e}")
            return ReconciliationResult(
                timestamp=datetime.now(config.ET),
                positions_checked=0,
                all_reconciled=False,
            )

    def _reconcile_inner(self) -> ReconciliationResult:
        import database

        now = datetime.now(config.ET)
        phantoms: list[str] = []
        unknowns: list[str] = []
        mismatches: list[str] = []

        # Load DB positions
        db_positions = database.load_open_positions()
        db_by_symbol: dict[str, dict] = {p["symbol"]: p for p in db_positions}

        # Load broker positions
        broker_positions = []
        if self._trading_client_fn is not None:
            client = self._trading_client_fn()
            broker_positions = client.get_all_positions() or []
        broker_by_symbol: dict = {
            pos.symbol: pos for pos in broker_positions
        }

        all_symbols = set(db_by_symbol.keys()) | set(broker_by_symbol.keys())

        for symbol in all_symbols:
            in_db = symbol in db_by_symbol
            at_broker = symbol in broker_by_symbol

            if in_db and not at_broker:
                # Phantom position — in DB but not at broker
                phantoms.append(symbol)
                logger.warning(
                    f"RECONCILER: Phantom position {symbol} — removing from DB"
                )
                self._remove_from_db(symbol)

            elif at_broker and not in_db:
                # Unknown position — at broker but not in DB
                unknowns.append(symbol)
                logger.warning(
                    f"RECONCILER: Unknown position {symbol} at broker — not in DB"
                )

            else:
                # Both exist — check size
                db_qty = float(db_by_symbol[symbol].get("qty", 0))
                broker_qty = float(getattr(broker_by_symbol[symbol], "qty", 0))
                if abs(db_qty - broker_qty) > 0.001:
                    mismatches.append(symbol)
                    logger.warning(
                        f"RECONCILER: Size mismatch {symbol}: "
                        f"DB={db_qty}, broker={broker_qty} — updating DB"
                    )
                    self._update_qty_in_db(symbol, broker_qty)

        all_reconciled = len(phantoms) == 0 and len(unknowns) == 0 and len(mismatches) == 0

        return ReconciliationResult(
            timestamp=now,
            positions_checked=len(all_symbols),
            phantoms_removed=phantoms,
            unknowns_found=unknowns,
            size_mismatches_fixed=mismatches,
            all_reconciled=all_reconciled,
        )

    @staticmethod
    def _remove_from_db(symbol: str):
        """Remove a phantom position from the database."""
        try:
            import database
            conn = database._get_conn()
            conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
            conn.commit()
        except Exception as e:
            logger.error(f"RECONCILER: Failed to remove {symbol} from DB: {e}")

    @staticmethod
    def _update_qty_in_db(symbol: str, new_qty: float):
        """Update qty in database to match broker."""
        try:
            import database
            conn = database._get_conn()
            conn.execute(
                "UPDATE open_positions SET qty = ? WHERE symbol = ?",
                (new_qty, symbol),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"RECONCILER: Failed to update qty for {symbol}: {e}")


# =============================================================================
# Structured Audit Trail
# =============================================================================

class AuditTrail:
    """
    Structured JSON audit trail for trade lifecycle tracing.

    Every trade decision gets a trace_id that connects:
    signal -> risk_decision -> order -> fill -> exit

    Gated by STRUCTURED_LOGGING_ENABLED config flag.
    All public methods are fail-open.
    """

    _instance_counter = 0

    def __init__(self, log_dir: str | None = None):
        self._enabled = getattr(config, "STRUCTURED_LOGGING_ENABLED", False)
        self._log_dir = log_dir or os.path.dirname(
            os.path.abspath(config.AUDIT_LOG_FILE)
        ) or "."
        self._log_file = os.path.join(
            self._log_dir,
            os.path.basename(config.AUDIT_LOG_FILE),
        )
        self._logger: logging.Logger | None = None
        AuditTrail._instance_counter += 1
        self._logger_name = f"audit_trail_{AuditTrail._instance_counter}"
        if self._enabled:
            self._setup_logger()

    def _setup_logger(self):
        """Configure a dedicated JSON logger with daily rotation."""
        try:
            self._logger = logging.getLogger(self._logger_name)
            self._logger.setLevel(logging.INFO)
            # Avoid duplicate handlers
            if not self._logger.handlers:
                handler = TimedRotatingFileHandler(
                    self._log_file,
                    when="midnight",
                    backupCount=getattr(
                        config, "AUDIT_TRAIL_RETENTION_DAYS", 365
                    ),
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
                self._logger.propagate = False
        except Exception as e:
            logger.error(f"AuditTrail: Failed to set up logger: {e}")
            self._logger = None

    def log_event(self, trace_id: str, event_type: str, data: dict):
        """Log a structured event to the audit trail.

        Never raises.
        """
        if not self._enabled or self._logger is None:
            return
        try:
            entry = {
                "timestamp": datetime.now(config.ET).isoformat(),
                "trace_id": trace_id,
                "event_type": event_type,
                "data": data,
            }
            self._logger.info(json.dumps(entry, default=str))
        except Exception as e:
            logger.error(f"AuditTrail.log_event failed: {e}")

    def get_trace(self, trace_id: str) -> list[dict]:
        """Get all events for a given trace_id by scanning the audit log.

        Never raises — returns empty list on error.
        """
        if not self._enabled:
            return []
        try:
            events = []
            log_path = Path(self._log_file)
            if not log_path.exists():
                return []
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("trace_id") == trace_id:
                            events.append(entry)
                    except json.JSONDecodeError:
                        continue
            return events
        except Exception as e:
            logger.error(f"AuditTrail.get_trace failed: {e}")
            return []

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a unique trace ID for a new signal.

        Never raises.
        """
        try:
            return f"T-{uuid.uuid4().hex[:12]}"
        except Exception:
            return f"T-{datetime.now(config.ET).strftime('%Y%m%d%H%M%S')}"
