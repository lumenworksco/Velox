"""ARCH-003: Event-Driven Architecture — Enhanced event bus.

Builds on the existing ``engine.events`` bus with:
- Typed event dataclasses for each domain event
- Async publish/subscribe via asyncio
- Priority-based handler ordering
- Type-safe handler registration

This module is additive: the original ``engine.events.EventBus`` remains
the canonical synchronous bus.  Import from here when you need async
support or typed events.

Usage::

    from engine.event_bus import AsyncEventBus, BarUpdate, QuoteUpdate

    bus = AsyncEventBus()
    bus.subscribe(BarUpdate, my_handler, priority=10)
    await bus.publish(BarUpdate(symbol="AAPL", close=185.0, volume=100_000))
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, TypeVar

from utils.timezone import now_et

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority levels — lower number = higher priority (dispatched first)
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    CRITICAL = 0   # Risk / kill-switch handlers
    HIGH = 10      # Order fills, circuit breakers
    NORMAL = 50    # Strategy signals, regime updates
    LOW = 90       # Analytics, logging, dashboard
    BACKGROUND = 100  # Replay recording, telemetry


# ---------------------------------------------------------------------------
# Typed event hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseEvent:
    """Root class for all typed events."""
    timestamp: datetime = field(default_factory=now_et)
    source: str = ""


@dataclass(frozen=True)
class BarUpdate(BaseEvent):
    """A new OHLCV bar is available."""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    timeframe: str = "1min"


@dataclass(frozen=True)
class QuoteUpdate(BaseEvent):
    """A new top-of-book quote."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0


@dataclass(frozen=True)
class TradeUpdate(BaseEvent):
    """A trade execution reported by the exchange / broker."""
    symbol: str = ""
    price: float = 0.0
    size: int = 0
    side: str = ""


@dataclass(frozen=True)
class OrderFill(BaseEvent):
    """Our order was filled (partially or fully)."""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    fill_price: float = 0.0
    strategy: str = ""
    remaining_qty: float = 0.0


@dataclass(frozen=True)
class RiskAlert(BaseEvent):
    """A risk threshold was breached."""
    alert_type: str = ""       # e.g. "drawdown", "circuit_breaker", "heat"
    severity: str = "warning"  # "warning" | "critical"
    message: str = ""
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeChange(BaseEvent):
    """Market regime has changed."""
    previous_regime: str = ""
    new_regime: str = ""
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SignalGenerated(BaseEvent):
    """A strategy produced a trading signal."""
    symbol: str = ""
    strategy: str = ""
    side: str = ""             # "buy" | "sell"
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    metadata: dict = field(default_factory=dict)


# Mapping for convenience / iteration
EVENT_TYPES: dict[str, type[BaseEvent]] = {
    "bar_update": BarUpdate,
    "quote_update": QuoteUpdate,
    "trade_update": TradeUpdate,
    "order_fill": OrderFill,
    "risk_alert": RiskAlert,
    "regime_change": RegimeChange,
    "signal_generated": SignalGenerated,
}

# Type variable for generic handler registration
E = TypeVar("E", bound=BaseEvent)


# ---------------------------------------------------------------------------
# Handler wrapper that carries priority
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PrioritizedHandler:
    priority: int
    handler: Any = field(compare=False)
    is_async: bool = field(compare=False, default=False)


# ---------------------------------------------------------------------------
# Async Event Bus
# ---------------------------------------------------------------------------

class AsyncEventBus:
    """Publish-subscribe bus with async support and priority ordering.

    Handlers are dispatched in priority order (lowest number first).
    Both sync and async handlers are supported:
    - sync handlers are called directly
    - async handlers are awaited when using ``publish()`` (async)
    - ``publish_sync()`` skips async handlers (useful from sync code)

    Thread-safe for subscription; publish should ideally happen from
    a single asyncio event loop or via ``publish_sync()`` from threads.
    """

    def __init__(self, max_history: int = 1000, max_dead_letters: int = 500) -> None:
        self._handlers: dict[type[BaseEvent], list[_PrioritizedHandler]] = {}
        self._wildcard_handlers: list[_PrioritizedHandler] = []
        self._lock = threading.Lock()
        self._history: list[BaseEvent] = []
        self._max_history = max_history
        # HIGH-011: Dead letter queue for failed handler dispatches
        self._dead_letters: list[dict] = []
        self._max_dead_letters = max_dead_letters

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: type[E],
        handler: Callable[[E], Any],
        priority: int = Priority.NORMAL,
    ) -> None:
        """Register *handler* for *event_type*.

        Lower *priority* values are dispatched first.
        Handler may be a regular function or an async coroutine function.
        """
        is_async = asyncio.iscoroutinefunction(handler)
        entry = _PrioritizedHandler(priority, handler, is_async)
        with self._lock:
            self._handlers.setdefault(event_type, []).append(entry)
            self._handlers[event_type].sort()
        name = getattr(handler, "__name__", repr(handler))
        logger.debug("AsyncEventBus: %s subscribed to %s (pri=%d, async=%s)",
                      name, event_type.__name__, priority, is_async)

    def subscribe_all(
        self,
        handler: Callable[[BaseEvent], Any],
        priority: int = Priority.LOW,
    ) -> None:
        """Register *handler* to receive ALL event types (wildcard)."""
        is_async = asyncio.iscoroutinefunction(handler)
        entry = _PrioritizedHandler(priority, handler, is_async)
        with self._lock:
            self._wildcard_handlers.append(entry)
            self._wildcard_handlers.sort()

    def unsubscribe(
        self,
        event_type: type[BaseEvent],
        handler: Callable,
    ) -> None:
        """Remove *handler* from *event_type*."""
        with self._lock:
            entries = self._handlers.get(event_type, [])
            self._handlers[event_type] = [
                e for e in entries if e.handler is not handler
            ]

    # ------------------------------------------------------------------
    # Publish (async)
    # ------------------------------------------------------------------

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event, dispatching to all matching handlers.

        Async handlers are awaited; sync handlers are called directly.
        Errors in individual handlers are logged but do not stop dispatch.
        """
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
            handlers.extend(self._wildcard_handlers)
            handlers.sort()

            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        for entry in handlers:
            try:
                if entry.is_async:
                    await entry.handler(event)
                else:
                    entry.handler(event)
            except Exception as exc:
                name = getattr(entry.handler, "__name__", repr(entry.handler))
                logger.error("AsyncEventBus: handler %s failed on %s: %s",
                             name, type(event).__name__, exc, exc_info=True)
                self._record_dead_letter(event, name, exc)

    # ------------------------------------------------------------------
    # Publish (sync — for use from threads / non-async code)
    # ------------------------------------------------------------------

    def publish_sync(self, event: BaseEvent) -> None:
        """Publish synchronously. Only sync handlers are invoked.

        Async handlers are silently skipped. Use this from synchronous
        code paths (scanner loop, broker callbacks, etc.).
        """
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
            handlers.extend(self._wildcard_handlers)
            handlers.sort()

            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        for entry in handlers:
            if entry.is_async:
                continue
            try:
                entry.handler(event)
            except Exception as exc:
                name = getattr(entry.handler, "__name__", repr(entry.handler))
                logger.error("AsyncEventBus: handler %s failed on %s: %s",
                             name, type(event).__name__, exc, exc_info=True)
                self._record_dead_letter(event, name, exc)

    # ------------------------------------------------------------------
    # Dead letter queue (HIGH-011)
    # ------------------------------------------------------------------

    def _record_dead_letter(self, event: BaseEvent, handler_name: str, exc: Exception) -> None:
        """Record a failed event dispatch for later inspection."""
        entry = {
            "event_type": type(event).__name__,
            "event": event,
            "handler": handler_name,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "timestamp": now_et(),
        }
        with self._lock:
            self._dead_letters.append(entry)
            if len(self._dead_letters) > self._max_dead_letters:
                self._dead_letters = self._dead_letters[-self._max_dead_letters:]

    def get_dead_letters(self, limit: int = 50) -> list[dict]:
        """Return recent dead letter entries."""
        with self._lock:
            return list(self._dead_letters[-limit:])

    # ------------------------------------------------------------------
    # History / introspection
    # ------------------------------------------------------------------

    def get_history(
        self,
        event_type: type[BaseEvent] | None = None,
        limit: int = 50,
    ) -> list[BaseEvent]:
        """Return recent events, optionally filtered by type."""
        with self._lock:
            if event_type is not None:
                filtered = [e for e in self._history if isinstance(e, event_type)]
                return filtered[-limit:]
            return list(self._history[-limit:])

    @property
    def stats(self) -> dict:
        """Return subscription and history statistics."""
        with self._lock:
            return {
                "total_events": len(self._history),
                "subscribers": {
                    et.__name__: len(handlers)
                    for et, handlers in self._handlers.items()
                },
                "wildcard_subscribers": len(self._wildcard_handlers),
                "dead_letters": len(self._dead_letters),
            }


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors engine.events pattern)
# ---------------------------------------------------------------------------

_async_bus: AsyncEventBus | None = None
_async_bus_lock = threading.Lock()


def get_async_event_bus() -> AsyncEventBus:
    """Get or create the global async event bus (thread-safe)."""
    global _async_bus
    if _async_bus is None:
        with _async_bus_lock:
            if _async_bus is None:
                _async_bus = AsyncEventBus()
    return _async_bus
