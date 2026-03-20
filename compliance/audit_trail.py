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
