"""Tests for V10 Event Bus and Tiered Circuit Breaker."""

import pytest
from datetime import datetime

from engine.events import EventBus, Event, EventTypes, get_event_bus
from risk.circuit_breaker import TieredCircuitBreaker, CircuitTier, TierConfig


class TestEventBus:
    """Test the publish-subscribe event bus."""

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda e: received.append(e))
        bus.publish(Event("test.event", {"key": "value"}))
        assert len(received) == 1
        assert received[0].data["key"] == "value"

    def test_multiple_subscribers(self):
        bus = EventBus()
        counts = {"a": 0, "b": 0}
        bus.subscribe("test.event", lambda e: counts.__setitem__("a", counts["a"] + 1))
        bus.subscribe("test.event", lambda e: counts.__setitem__("b", counts["b"] + 1))
        bus.publish(Event("test.event"))
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e.event_type))
        bus.publish(Event("signal.generated"))
        bus.publish(Event("order.filled"))
        assert received == ["signal.generated", "order.filled"]

    def test_no_cross_talk(self):
        bus = EventBus()
        received = []
        bus.subscribe("signal.generated", lambda e: received.append(e))
        bus.publish(Event("order.filled"))
        assert len(received) == 0

    def test_handler_exception_doesnt_break_others(self):
        bus = EventBus()
        results = []

        def bad_handler(e):
            raise ValueError("broken")

        def good_handler(e):
            results.append("ok")

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)
        bus.publish(Event("test"))
        assert results == ["ok"]

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("test", handler)
        bus.publish(Event("test"))
        assert len(received) == 1

        bus.unsubscribe("test", handler)
        bus.publish(Event("test"))
        assert len(received) == 1  # No new events

    def test_event_history(self):
        bus = EventBus()
        bus.publish(Event("a"))
        bus.publish(Event("b"))
        bus.publish(Event("c"))
        history = bus.get_history()
        assert len(history) == 3
        assert history[0].event_type == "a"

    def test_history_filtered(self):
        bus = EventBus()
        bus.publish(Event("a"))
        bus.publish(Event("b"))
        bus.publish(Event("a"))
        assert len(bus.get_history("a")) == 2
        assert len(bus.get_history("b")) == 1

    def test_history_limit(self):
        bus = EventBus()
        bus._max_history = 5
        for i in range(10):
            bus.publish(Event(f"event_{i}"))
        assert len(bus.get_history()) == 5

    def test_event_types_defined(self):
        """Verify standard event types are defined."""
        assert EventTypes.SIGNAL_GENERATED == "signal.generated"
        assert EventTypes.ORDER_FILLED == "order.filled"
        assert EventTypes.POSITION_CLOSED == "position.closed"
        assert EventTypes.KILL_SWITCH_ACTIVATED == "kill_switch.activated"

    def test_stats(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.subscribe("b", lambda e: None)
        stats = bus.stats
        assert stats["subscribers"]["a"] == 1
        assert stats["subscribers"]["b"] == 2


class TestTieredCircuitBreaker:
    """Test the 4-tier progressive circuit breaker."""

    def test_normal_state(self):
        cb = TieredCircuitBreaker()
        assert cb.current_tier == CircuitTier.NORMAL
        assert cb.allow_new_entries
        assert cb.size_multiplier == 1.0
        assert not cb.should_close_day_trades
        assert not cb.should_close_all

    def test_yellow_tier(self):
        cb = TieredCircuitBreaker()
        tier = cb.update(-0.015)  # -1.5% loss
        assert tier == CircuitTier.YELLOW
        assert cb.allow_new_entries
        assert cb.size_multiplier == 0.5  # Reduced sizing

    def test_orange_tier(self):
        cb = TieredCircuitBreaker()
        tier = cb.update(-0.025)  # -2.5% loss
        assert tier == CircuitTier.ORANGE
        assert not cb.allow_new_entries
        assert cb.size_multiplier == 0.0

    def test_red_tier(self):
        cb = TieredCircuitBreaker()
        tier = cb.update(-0.035)  # -3.5% loss
        assert tier == CircuitTier.RED
        assert cb.should_close_day_trades
        assert not cb.should_close_all

    def test_black_tier(self):
        cb = TieredCircuitBreaker()
        tier = cb.update(-0.05)  # -5% loss
        assert tier == CircuitTier.BLACK
        assert cb.should_close_all
        assert not cb.allow_new_entries

    def test_de_escalation(self):
        cb = TieredCircuitBreaker()
        cb.update(-0.03)
        assert cb.current_tier == CircuitTier.RED
        cb.update(-0.005)  # Recovery
        assert cb.current_tier == CircuitTier.NORMAL

    def test_tier_history(self):
        cb = TieredCircuitBreaker()
        cb.update(-0.015)  # Yellow
        cb.update(-0.025)  # Orange
        assert len(cb.tier_history) == 2
        assert cb.tier_history[0][1] == CircuitTier.YELLOW
        assert cb.tier_history[1][1] == CircuitTier.ORANGE

    def test_daily_reset(self):
        cb = TieredCircuitBreaker()
        cb.update(-0.03)
        assert cb.current_tier == CircuitTier.RED
        cb.reset_daily()
        assert cb.current_tier == CircuitTier.NORMAL
        assert len(cb.tier_history) == 0

    def test_custom_tiers(self):
        custom = {
            CircuitTier.NORMAL: TierConfig(0.0, 1.0, True, False, False),
            CircuitTier.YELLOW: TierConfig(-0.005, 0.5, True, False, False),  # Tighter
        }
        cb = TieredCircuitBreaker(tiers=custom)
        cb.update(-0.006)
        assert cb.current_tier == CircuitTier.YELLOW

    def test_no_change_no_log(self):
        cb = TieredCircuitBreaker()
        cb.update(-0.001)  # Still normal
        assert cb.current_tier == CircuitTier.NORMAL
        assert len(cb.tier_history) == 0

    def test_status_dict(self):
        cb = TieredCircuitBreaker()
        status = cb.status
        assert status["tier"] == "NORMAL"
        assert status["size_multiplier"] == 1.0
        assert status["allow_new_entries"] is True
