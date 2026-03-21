"""PROD-011: DegradedModuleTracker — singleton that tracks fail-open/degraded modules.

When a module enters degraded mode (e.g., ML inference unavailable, data feed
stale, feature store errors), it registers itself here. The dashboard endpoint
can then expose which modules are currently degraded for operational visibility.

Usage:
    from engine.degraded_tracker import degraded_tracker

    # In any module that degrades gracefully:
    try:
        result = some_risky_operation()
        degraded_tracker.mark_healthy("ml_inference")
    except Exception as e:
        degraded_tracker.mark_degraded("ml_inference", reason=str(e))
        result = fallback_value

    # In the web dashboard:
    @app.get("/api/degraded")
    def get_degraded():
        return degraded_tracker.status()
"""

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DegradedModule:
    """State record for a degraded module."""
    module: str
    reason: str
    degraded_since: float  # monotonic timestamp
    last_updated: float
    error_count: int = 0


class DegradedModuleTracker:
    """PROD-011: Singleton tracker for modules in fail-open/degraded state.

    Thread-safe. Modules self-report their degraded status via
    `mark_degraded()` / `mark_healthy()`. The dashboard can query
    `status()` to see which modules are currently degraded.
    """

    _instance: Optional["DegradedModuleTracker"] = None
    _lock_class = threading.Lock()

    def __new__(cls):
        with cls._lock_class:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._modules: Dict[str, DegradedModule] = {}
        self._initialized = True
        logger.info("PROD-011: DegradedModuleTracker initialized")

    def mark_degraded(self, module: str, reason: str = "unknown"):
        """Mark a module as degraded/fail-open.

        Args:
            module: Module identifier (e.g., "ml_inference", "feature_store").
            reason: Human-readable reason for degradation.
        """
        now = _time.monotonic()
        with self._lock:
            existing = self._modules.get(module)
            if existing:
                existing.reason = reason
                existing.last_updated = now
                existing.error_count += 1
            else:
                self._modules[module] = DegradedModule(
                    module=module,
                    reason=reason,
                    degraded_since=now,
                    last_updated=now,
                    error_count=1,
                )
                logger.warning(
                    "PROD-011: Module '%s' entered degraded state: %s",
                    module, reason,
                )

    def mark_healthy(self, module: str):
        """Mark a module as healthy (remove from degraded list).

        Args:
            module: Module identifier.
        """
        with self._lock:
            removed = self._modules.pop(module, None)
            if removed:
                logger.info(
                    "PROD-011: Module '%s' recovered from degraded state "
                    "(was degraded for %.1fs, %d errors)",
                    module,
                    _time.monotonic() - removed.degraded_since,
                    removed.error_count,
                )

    def is_degraded(self, module: str) -> bool:
        """Check if a specific module is currently degraded."""
        with self._lock:
            return module in self._modules

    def get_degraded_modules(self) -> list[str]:
        """Get list of currently degraded module names."""
        with self._lock:
            return list(self._modules.keys())

    def status(self) -> dict:
        """Return full degradation status for dashboard/API.

        Returns:
            Dict with summary and per-module details.
        """
        now = _time.monotonic()
        with self._lock:
            modules = {}
            for name, mod in self._modules.items():
                modules[name] = {
                    "reason": mod.reason,
                    "degraded_for_sec": round(now - mod.degraded_since, 1),
                    "error_count": mod.error_count,
                    "last_updated_sec_ago": round(now - mod.last_updated, 1),
                }
            return {
                "degraded_count": len(self._modules),
                "healthy": len(self._modules) == 0,
                "modules": modules,
            }

    def clear(self):
        """Clear all degraded state (e.g., on restart)."""
        with self._lock:
            self._modules.clear()


# Module-level singleton instance
degraded_tracker = DegradedModuleTracker()
