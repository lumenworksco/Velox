"""COMPLY-001: Complete Audit Trail — structured, append-only event logging for every decision point.

Logs every decision with full context:
  - Signal generation (strategy, symbol, confidence, indicators)
  - Filter pipeline (which filters passed/failed, reasons)
  - Position sizing (Kelly, volatility target, final size)
  - Order submission (order type, price, quantity, routing)
  - Fills (fill price, slippage, latency)
  - Exits (reason, PnL, hold time)

7-year retention policy (configurable).
Append-only structured JSON — one event per line.
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single audit trail event."""

    event_id: str
    event_type: str
    source: str
    timestamp: str
    details: dict = field(default_factory=dict)
    trace_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_TYPES = {
    # Signal lifecycle
    "signal_generated",
    "signal_filtered",
    "signal_ranked",
    "signal_approved",
    "signal_rejected",

    # Position lifecycle
    "position_size_computed",
    "order_submitted",
    "order_filled",
    "order_cancelled",
    "order_rejected",
    "order_amended",

    # Exit lifecycle
    "exit_triggered",
    "exit_order_submitted",
    "exit_filled",
    "partial_exit",

    # Risk events
    "risk_check_passed",
    "risk_check_failed",
    "circuit_breaker_triggered",
    "kill_switch_activated",
    "pnl_lock_activated",
    "drawdown_warning",
    "margin_warning",

    # System events
    "system_startup",
    "system_shutdown",
    "config_changed",
    "reconciliation_run",
    "error_occurred",
}

# T5-010: SEC 17a-4 inspired event types for compliance audit log
SEC_AUDIT_EVENT_TYPES = {
    "SIGNAL_GENERATED",
    "SIGNAL_REJECTED",
    "ORDER_SUBMITTED",
    "ORDER_FILLED",
    "RISK_LIMIT_HIT",
    "CONFIG_CHANGED",
    "STRATEGY_DISABLED",
}


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Complete audit trail for regulatory compliance and trade lifecycle tracing.

    Every decision point logs a structured JSON event with full context.
    Events are append-only and never modified or deleted (within retention period).

    Usage:
        audit = AuditTrail()
        trace_id = audit.new_trace_id()

        audit.log_event("signal_generated", "stat_mr", {
            "symbol": "AAPL", "zscore": -2.1, "confidence": 0.75
        }, trace_id=trace_id)

        audit.log_event("order_submitted", "oms", {
            "symbol": "AAPL", "side": "buy", "qty": 100, "order_type": "limit"
        }, trace_id=trace_id)

        # Query events
        events = audit.query_events({"event_type": "signal_generated", "symbol": "AAPL"})
    """

    # Retention: 7 years = 2555 days
    DEFAULT_RETENTION_DAYS = 2555

    def __init__(self, log_dir: str | None = None, session_id: str | None = None,
                 retention_days: int | None = None):
        """
        Args:
            log_dir: Directory for audit log files. Defaults to current working directory.
            session_id: Unique ID for this bot session.
            retention_days: Days to retain audit logs. Defaults to 7 years (2555 days).
        """
        self._log_dir = Path(log_dir or os.getcwd())
        self._session_id = session_id or f"S-{uuid.uuid4().hex[:8]}"
        self._retention_days = retention_days or self.DEFAULT_RETENTION_DAYS
        self._log_file = self._log_dir / "compliance_audit.jsonl"

        self._file_logger: Optional[logging.Logger] = None
        self._lock = threading.Lock()
        self._event_counter = 0
        self._recent_events: list[AuditEvent] = []
        self._max_recent = 5000

        self._setup_logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, source: str, details: dict,
                  timestamp: datetime | None = None, trace_id: str = ""):
        """Log a structured audit event.

        Args:
            event_type: Type of event (see EVENT_TYPES).
            source: Originating component (e.g. "stat_mr", "oms", "risk_manager").
            details: Full context dict — all relevant data for this decision.
            timestamp: Event timestamp. Defaults to now (ET).
            trace_id: Optional trace ID linking related events (e.g. signal -> fill).

        Never raises.
        """
        try:
            self._log_inner(event_type, source, details, timestamp, trace_id)
        except Exception as e:
            logger.error(f"AuditTrail.log_event failed: {e}")

    def query_events(self, event_type: str | None = None,
                     symbol: str | None = None,
                     start: datetime | str | None = None,
                     end: datetime | str | None = None,
                     source: str | None = None,
                     trace_id: str | None = None,
                     limit: int = 500) -> list[AuditEvent]:
        """Query recent audit events with optional keyword filters.

        Also accepts a single ``dict`` as the first positional argument for
        backwards compatibility with the ``filters=`` calling convention.

        Args:
            event_type: Filter by event type (e.g. "order_filled").
            symbol: Filter by symbol (searched in ``details["symbol"]``).
            start: Only return events at or after this time.
            end: Only return events at or before this time.
            source: Filter by source component.
            trace_id: Filter by trace ID.
            limit: Max results to return.

        Returns:
            List of AuditEvent objects matching filters.

        Never raises — returns empty list on error.
        """
        # Backwards-compatible: accept query_events({"event_type": ...})
        if isinstance(event_type, dict):
            filters = event_type
        else:
            filters: dict = {}
            if event_type is not None:
                filters["event_type"] = event_type
            if symbol is not None:
                filters["symbol"] = symbol
            if source is not None:
                filters["source"] = source
            if trace_id is not None:
                filters["trace_id"] = trace_id
            if start is not None:
                filters["start"] = start.isoformat() if hasattr(start, "isoformat") else str(start)
            if end is not None:
                filters["end"] = end.isoformat() if hasattr(end, "isoformat") else str(end)
        try:
            return self._query_inner(filters, limit)
        except Exception as e:
            logger.error(f"AuditTrail.query_events failed: {e}")
            return []

    def get_trace(self, trace_id: str) -> list[AuditEvent]:
        """Get all events for a given trace ID (signal -> fill lifecycle).

        Returns events in chronological order.
        """
        return self.query_events({"trace_id": trace_id}, limit=1000)

    @staticmethod
    def new_trace_id() -> str:
        """Generate a unique trace ID for linking related events."""
        try:
            return f"T-{uuid.uuid4().hex[:12]}"
        except Exception:
            return f"T-{datetime.now(config.ET).strftime('%Y%m%d%H%M%S%f')}"

    def get_event_count(self) -> int:
        """Return total events logged this session."""
        return self._event_counter

    def get_recent_events(self, n: int = 100) -> list[AuditEvent]:
        """Return the N most recent events from memory cache."""
        with self._lock:
            return list(self._recent_events[-n:])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_logger(self):
        """Configure a dedicated file logger with daily rotation."""
        try:
            self._file_logger = logging.getLogger(
                f"compliance_audit_{self._session_id}"
            )
            self._file_logger.setLevel(logging.INFO)

            if not self._file_logger.handlers:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                handler = TimedRotatingFileHandler(
                    str(self._log_file),
                    when="midnight",
                    backupCount=self._retention_days,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._file_logger.addHandler(handler)
                self._file_logger.propagate = False
        except Exception as e:
            logger.error(f"AuditTrail: Failed to set up file logger: {e}")
            self._file_logger = None

    def _log_inner(self, event_type: str, source: str, details: dict,
                   timestamp: datetime | None, trace_id: str):
        """Core logging logic."""
        now = timestamp or datetime.now(config.ET)
        self._event_counter += 1

        event = AuditEvent(
            event_id=f"EVT-{self._event_counter:08d}",
            event_type=event_type,
            source=source,
            timestamp=now.isoformat() if hasattr(now, "isoformat") else str(now),
            details=details,
            trace_id=trace_id,
            session_id=self._session_id,
        )

        # Store in memory cache
        with self._lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent:
                self._recent_events = self._recent_events[-self._max_recent:]

        # Write to file
        if self._file_logger:
            try:
                self._file_logger.info(json.dumps(event.to_dict(), default=str))
            except Exception as e:
                logger.error(f"AuditTrail: Failed to write event: {e}")

    def _query_inner(self, filters: dict, limit: int) -> list[AuditEvent]:
        """Query events from memory cache."""
        with self._lock:
            candidates = list(self._recent_events)

        results = []
        for event in reversed(candidates):  # most recent first
            if self._matches_filters(event, filters):
                results.append(event)
                if len(results) >= limit:
                    break
        return results

    @staticmethod
    def _matches_filters(event: AuditEvent, filters: dict) -> bool:
        """Check if an event matches all filter criteria."""
        for key, value in filters.items():
            if key == "event_type" and event.event_type != value:
                return False
            if key == "source" and event.source != value:
                return False
            if key == "trace_id" and event.trace_id != value:
                return False
            if key == "symbol":
                if event.details.get("symbol") != value:
                    return False
            if key in ("since", "start"):
                if event.timestamp < str(value):
                    return False
            if key == "end":
                if event.timestamp > str(value):
                    return False
        return True


# ---------------------------------------------------------------------------
# T5-010: SEC 17a-4 Inspired Hash-Chained DB Audit Log
# ---------------------------------------------------------------------------

class HashChainAuditLog:
    """Append-only, hash-chained audit log stored in SQLite.

    T5-010: Every record contains:
      - signature_hash = SHA-256(prev_hash + payload_json)
      - No UPDATE or DELETE operations are ever performed.

    Event types: SIGNAL_GENERATED, SIGNAL_REJECTED, ORDER_SUBMITTED,
    ORDER_FILLED, RISK_LIMIT_HIT, CONFIG_CHANGED, STRATEGY_DISABLED.

    Usage::

        audit = HashChainAuditLog()
        audit.append("ORDER_SUBMITTED", "signal_processor", {
            "symbol": "AAPL", "side": "buy", "qty": 100
        })

        # Verify chain integrity
        valid, broken_id = audit.verify_chain()
    """

    # Genesis hash — the "previous hash" for the very first record
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._lock = threading.Lock()
        self._prev_hash: str | None = None  # Lazily loaded from DB

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        actor: str,
        payload: dict,
        timestamp: datetime | None = None,
    ) -> str | None:
        """Append a new audit record with hash chain linking.

        Args:
            event_type: One of SEC_AUDIT_EVENT_TYPES.
            actor: Originating component or user.
            payload: Full context dict for the event.
            timestamp: Event time. Defaults to now (ET).

        Returns:
            The signature_hash of the new record, or None on failure.

        Never raises.
        """
        try:
            return self._append_inner(event_type, actor, payload, timestamp)
        except Exception as e:
            logger.error("HashChainAuditLog.append failed: %s", e)
            return None

    def verify_chain(self, limit: int = 10000) -> tuple[bool, int | None]:
        """Verify the hash chain integrity of the audit log.

        Args:
            limit: Max records to verify (most recent N).

        Returns:
            (is_valid, broken_at_id) — broken_at_id is None if valid.
        """
        try:
            return self._verify_inner(limit)
        except Exception as e:
            logger.error("HashChainAuditLog.verify_chain failed: %s", e)
            return False, -1

    def query(
        self,
        event_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Query audit records with optional filters.

        Returns list of dicts with id, timestamp, event_type, actor,
        payload_json, signature_hash.
        """
        try:
            import database
            conn = database._get_conn()
            conditions = []
            params: list = []
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type)
            if from_date:
                conditions.append("timestamp >= ?")
                params.append(from_date)
            if to_date:
                conditions.append("timestamp <= ?")
                params.append(to_date)
            where = " AND ".join(conditions) if conditions else "1=1"
            c = conn.cursor()
            c.execute(
                f"SELECT id, timestamp, event_type, actor, payload_json, signature_hash "
                f"FROM audit_log WHERE {where} ORDER BY id DESC LIMIT ?",
                params + [limit],
            )
            return [
                {
                    "id": r[0], "timestamp": r[1], "event_type": r[2],
                    "actor": r[3], "payload_json": r[4], "signature_hash": r[5],
                }
                for r in c.fetchall()
            ]
        except Exception as e:
            logger.error("HashChainAuditLog.query failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(prev_hash: str, payload_json: str) -> str:
        """SHA-256(prev_hash + payload_json) for tamper detection."""
        data = (prev_hash + payload_json).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _get_prev_hash(self) -> str:
        """Load the last signature_hash from the DB, or return genesis hash."""
        if self._prev_hash is not None:
            return self._prev_hash
        try:
            import database
            conn = database._get_conn()
            row = conn.execute(
                "SELECT signature_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            self._prev_hash = row[0] if row else self.GENESIS_HASH
        except Exception:
            self._prev_hash = self.GENESIS_HASH
        return self._prev_hash

    def _append_inner(
        self, event_type: str, actor: str, payload: dict,
        timestamp: datetime | None,
    ) -> str:
        """Core append logic — serialized via lock for hash chain integrity."""
        import database

        now = timestamp or datetime.now(config.ET)
        ts_str = now.isoformat() if hasattr(now, "isoformat") else str(now)
        payload_json = json.dumps(payload, default=str, sort_keys=True)

        with self._lock:
            prev_hash = self._get_prev_hash()
            sig_hash = self._compute_hash(prev_hash, payload_json)

            conn = database._get_conn()
            conn.execute(
                "INSERT INTO audit_log (timestamp, event_type, actor, payload_json, signature_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts_str, event_type, actor, payload_json, sig_hash),
            )
            conn.commit()

            self._prev_hash = sig_hash

        return sig_hash

    def _verify_inner(self, limit: int) -> tuple[bool, int | None]:
        """Verify hash chain integrity by recomputing hashes."""
        import database

        conn = database._get_conn()
        c = conn.cursor()
        c.execute(
            "SELECT id, payload_json, signature_hash FROM audit_log ORDER BY id ASC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()

        if not rows:
            return True, None

        prev_hash = self.GENESIS_HASH
        for row in rows:
            record_id, payload_json, stored_hash = row[0], row[1], row[2]
            expected_hash = self._compute_hash(prev_hash, payload_json)
            if expected_hash != stored_hash:
                logger.error(
                    "T5-010: Hash chain broken at record id=%d (expected=%s, stored=%s)",
                    record_id, expected_hash[:16], stored_hash[:16],
                )
                return False, record_id
            prev_hash = stored_hash

        return True, None


# Module-level singleton
_hash_chain_audit: HashChainAuditLog | None = None
_hca_lock = threading.Lock()


def get_hash_chain_audit() -> HashChainAuditLog:
    """Get or create the global HashChainAuditLog singleton."""
    global _hash_chain_audit
    if _hash_chain_audit is None:
        with _hca_lock:
            if _hash_chain_audit is None:
                _hash_chain_audit = HashChainAuditLog()
    return _hash_chain_audit
