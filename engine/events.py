"""V10 Engine — Internal event bus for decoupled component communication.

Events flow through the bus instead of direct function calls, enabling:
- Loose coupling between components
- Audit trail (all events logged)
- Easy addition of new subscribers (monitoring, notifications, analytics)
- Testing via event inspection

Usage:
    bus = EventBus()
    bus.subscribe("order.filled", my_handler)
    bus.publish(Event("order.filled", {"symbol": "AAPL", "qty": 10}))
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any

import config

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single event on the bus."""
    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(config.ET))
    source: str = ""

    def __repr__(self):
        return f"Event({self.event_type}, {self.data})"


# Standard event types
class EventTypes:
    # Signals
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_FILTERED = "signal.filtered"
    SIGNAL_ACCEPTED = "signal.accepted"

    # Orders
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_FAILED = "order.failed"

    # Positions
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_PARTIAL_CLOSE = "position.partial_close"

    # Risk
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker.triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch.activated"
    DRAWDOWN_WARNING = "drawdown.warning"

    # Regime
    REGIME_CHANGED = "regime.changed"

    # System
    DAY_RESET = "system.day_reset"
    EOD_CLOSE = "system.eod_close"
    SCAN_COMPLETE = "system.scan_complete"
    BROKER_SYNC = "system.broker_sync"
    RECONCILIATION_MISMATCH = "system.reconciliation_mismatch"


class EventBus:
    """Publish-subscribe event bus for intra-process communication.

    Thread-safe. Handlers are called synchronously in subscription order.
    If a handler raises, it's logged and other handlers continue.
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe a handler to an event type.

        Handler signature: handler(event: Event) -> None
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)
        logger.debug(f"EventBus: {handler.__name__} subscribed to {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable):
        """Remove a handler from an event type."""
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h is not handler
                ]

    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))
            # Also notify wildcard subscribers
            handlers.extend(self._handlers.get("*", []))

            # Store in history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"EventBus: handler {handler.__name__} failed on {event.event_type}: {e}")

    def get_history(self, event_type: str = None, limit: int = 50) -> list[Event]:
        """Get recent events, optionally filtered by type."""
        with self._lock:
            if event_type:
                filtered = [e for e in self._history if e.event_type == event_type]
                return filtered[-limit:]
            return self._history[-limit:]

    @property
    def stats(self) -> dict:
        """Get event bus statistics."""
        with self._lock:
            return {
                "total_events": len(self._history),
                "subscribers": {k: len(v) for k, v in self._handlers.items()},
            }


# Global event bus instance (singleton)
_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get or create the global event bus (thread-safe)."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:  # Double-checked locking
                _bus = EventBus()
    return _bus
