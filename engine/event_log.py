"""Structured event logging for audit trail.

V10: Provides a centralized event log for all trading actions,
enabling post-hoc analysis and regulatory audit compliance.
"""

import logging
from enum import Enum

import database

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Standardized event types for the trading system."""
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_FILTERED = "signal.filtered"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_CANCELLED = "position.cancelled"  # BUG-FIX (2026-04-14): rolled-back entry
    EXIT_TRIGGERED = "exit.triggered"
    CIRCUIT_BREAKER = "circuit_breaker.triggered"
    KILL_SWITCH = "kill_switch.activated"
    PDT_BLOCKED = "pdt.blocked"
    REGIME_CHANGE = "regime.changed"
    RISK_LIMIT = "risk.limit_hit"
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    ERROR = "system.error"


def log_event(event_type: EventType | str, source: str,
              symbol: str | None = None, strategy: str | None = None,
              details: str | None = None, severity: str = "INFO"):
    """Log a structured event to both the database and Python logger.

    Args:
        event_type: The type of event (use EventType enum).
        source: Module or component that generated the event.
        symbol: Trading symbol involved (if applicable).
        strategy: Strategy name (if applicable).
        details: Free-form details string (JSON recommended).
        severity: Log level — INFO, WARNING, ERROR, CRITICAL.
    """
    event_str = event_type.value if isinstance(event_type, EventType) else str(event_type)

    # Log to Python logger
    log_msg = f"[EVENT] {event_str} | src={source}"
    if symbol:
        log_msg += f" | sym={symbol}"
    if strategy:
        log_msg += f" | strat={strategy}"
    if details:
        log_msg += f" | {details}"

    log_level = getattr(logging, severity.upper(), logging.INFO)
    logger.log(log_level, log_msg)

    # Persist to database
    try:
        database.insert_event_log(
            event_type=event_str,
            source=source,
            symbol=symbol,
            strategy=strategy,
            details=details,
            severity=severity,
        )
    except Exception as e:
        # Never let event logging crash the trading system
        logger.error(f"Failed to persist event log: {e}")
