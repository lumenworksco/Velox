"""ARCH-002: Dependency Injection Container.

Lightweight DI system that holds all shared dependencies and manages their
lifecycle.  Components are lazily created on first access and cached for reuse.

Usage::

    container = Container.instance()
    rm = container.risk_manager
    bus = container.event_bus

Thread safety: the singleton is created under a lock (double-checked locking).
Individual component factories are also guarded so each dependency is created
exactly once even under concurrent access.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class Container:
    """Central dependency container for the trading bot.

    Holds references to all major subsystems.  Dependencies are created lazily
    via factory methods and cached for the lifetime of the container.
    """

    _instance: Container | None = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> Container:
        """Return the global container singleton (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Container: singleton created")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton.  Intended for tests only."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: dict[str, Any] = {}
        self._factories: dict[str, Any] = {}

        # Register default factories
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default factory functions for known components."""
        self.register_factory("config", self._create_config)
        self.register_factory("database", self._create_database)
        self.register_factory("event_bus", self._create_event_bus)
        self.register_factory("broker_client", self._create_broker_client)
        self.register_factory("data_client", self._create_data_client)
        self.register_factory("risk_manager", self._create_risk_manager)
        self.register_factory("oms", self._create_oms)
        self.register_factory("circuit_breaker", self._create_circuit_breaker)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_factory(self, name: str, factory) -> None:
        """Register a factory callable for a named component.

        The factory is called with no arguments and should return the
        component instance.  Overwrites any existing factory.
        """
        with self._lock:
            self._factories[name] = factory
            # Clear cached instance so the new factory takes effect
            self._components.pop(name, None)

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-built instance (useful for testing)."""
        with self._lock:
            self._components[name] = instance

    def get(self, name: str) -> Any:
        """Retrieve a component by name, creating it if necessary."""
        # Fast path — no lock
        inst = self._components.get(name)
        if inst is not None:
            return inst

        with self._lock:
            # Double-check under lock
            inst = self._components.get(name)
            if inst is not None:
                return inst

            factory = self._factories.get(name)
            if factory is None:
                raise KeyError(f"Container: no factory registered for '{name}'")

            logger.debug("Container: creating '%s'", name)
            inst = factory()
            self._components[name] = inst
            return inst

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def config(self):
        """The global config module."""
        return self.get("config")

    @property
    def database(self):
        """Database connection / helper."""
        return self.get("database")

    @property
    def event_bus(self):
        """The global EventBus instance."""
        return self.get("event_bus")

    @property
    def broker_client(self):
        """Broker abstraction client."""
        return self.get("broker_client")

    @property
    def data_client(self):
        """Market data client."""
        return self.get("data_client")

    @property
    def risk_manager(self):
        """RiskManager instance."""
        return self.get("risk_manager")

    @property
    def oms(self):
        """Order Management System (OrderManager)."""
        return self.get("oms")

    @property
    def circuit_breaker(self):
        """TieredCircuitBreaker instance."""
        return self.get("circuit_breaker")

    # ------------------------------------------------------------------
    # Default factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def _create_config():
        import config as _cfg
        return _cfg

    @staticmethod
    def _create_database():
        import database as _db
        _db.init_db()
        return _db

    @staticmethod
    def _create_event_bus():
        from engine.events import get_event_bus
        return get_event_bus()

    @staticmethod
    def _create_broker_client():
        from broker.base import Broker
        import config as _cfg

        if _cfg.PAPER_MODE:
            from broker.paper_broker import PaperBroker
            return PaperBroker()
        # Default: return the base Broker (Alpaca REST wrapper)
        return Broker()

    @staticmethod
    def _create_data_client():
        import data as _data
        return _data

    @staticmethod
    def _create_risk_manager():
        from risk import RiskManager
        return RiskManager()

    @staticmethod
    def _create_oms():
        from oms import OrderManager
        return OrderManager()

    @staticmethod
    def _create_circuit_breaker():
        from risk.circuit_breaker import TieredCircuitBreaker
        return TieredCircuitBreaker()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Clean up components that need explicit teardown."""
        logger.info("Container: shutting down")
        self._components.clear()

    def __repr__(self) -> str:
        with self._lock:
            created = list(self._components.keys())
            registered = list(self._factories.keys())
        return (
            f"Container(created={created}, registered={registered})"
        )
