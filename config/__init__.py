"""ARCH-006: Configuration management with typed access, validation, and hot-reload.

Backward compatibility: everything previously available via ``import config``
(the old ``config.py``, now ``config/settings.py``) is re-exported here so
that ``config.ET``, ``config.PAPER_MODE``, ``config.validate()`` etc. all
continue to work unchanged.

Additionally, this package provides:
- ``ConfigManager`` — typed access to YAML-based strategy/risk params with
  hot-reload support.
- ``config.base`` — immutable market-structure constants.

Usage (new code)::

    from config import ConfigManager

    cfg = ConfigManager.instance()
    entry_z = cfg.strategy("stat_mr", "zscore_entry", float)  # 1.5
    max_pos = cfg.risk("position_sizing", "max_positions", int)  # 12
    cfg.reload()

Usage (legacy, still works)::

    import config
    print(config.PAPER_MODE)
    print(config.ET)
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# Re-export EVERYTHING from the original config.py (now config/settings.py)
# so that ``import config; config.PAPER_MODE`` keeps working.
# -----------------------------------------------------------------------
from config.settings import *  # noqa: F401, F403
from config.settings import (  # explicit re-exports for type checkers
    ET, PAPER_MODE, API_KEY, API_SECRET,
    MARKET_OPEN, MARKET_CLOSE, TRADING_START,
    SYMBOLS, CORE_SYMBOLS, LEVERAGED_ETFS, STANDARD_SYMBOLS,
    STRATEGY_ALLOCATIONS, SECTOR_MAP, SECTOR_GROUPS,
    validate, get_param, set_param,
)

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Directory containing the YAML files (same directory as this __init__.py)
_CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning an empty dict on any error."""
    try:
        import yaml  # type: ignore
    except ImportError:
        # Fall back to a minimal parser if PyYAML is not installed
        logger.warning("PyYAML not installed — YAML config files will not be loaded")
        return {}

    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return {}


class ConfigManager:
    """Centralized, typed configuration with hot-reload support.

    All access is thread-safe. The singleton is created lazily.
    """

    _instance: ConfigManager | None = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> ConfigManager:
        """Return the global ConfigManager singleton (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._load_all()
                    logger.info("ConfigManager: initialized from %s", _CONFIG_DIR)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for tests)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strategies: dict = {}
        self._risk: dict = {}
        self._last_loaded: datetime | None = None
        self._strategies_path = _CONFIG_DIR / "strategies.yaml"
        self._risk_path = _CONFIG_DIR / "risk.yaml"
        self._watchers: list = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        with self._lock:
            self._strategies = _load_yaml(self._strategies_path)
            self._risk = _load_yaml(self._risk_path)
            self._last_loaded = datetime.now()

    def reload(self) -> bool:
        """Reload YAML files from disk.  Returns True if anything changed."""
        old_s = self._strategies.copy() if self._strategies else {}
        old_r = self._risk.copy() if self._risk else {}
        self._load_all()
        changed = (self._strategies != old_s) or (self._risk != old_r)
        if changed:
            logger.info("ConfigManager: configuration reloaded (changed=True)")
            self._notify_watchers()
        return changed

    # ------------------------------------------------------------------
    # Typed access — strategies
    # ------------------------------------------------------------------

    def strategy(self, section: str, key: str, type_: type[T] = float, default: T | None = None) -> T:
        """Get a strategy parameter with type coercion.

        Args:
            section: Top-level key in strategies.yaml (e.g. "stat_mr").
            key: Parameter name within the section.
            type_: Expected return type (float, int, bool, str).
            default: Value returned if key is missing.

        Returns:
            The parameter value cast to *type_*, or *default*.
        """
        with self._lock:
            section_data = self._strategies.get(section, {})
            if not isinstance(section_data, dict):
                return default  # type: ignore[return-value]
            raw = section_data.get(key)

        if raw is None:
            return default  # type: ignore[return-value]
        try:
            return type_(raw)
        except (TypeError, ValueError):
            logger.warning("ConfigManager: cannot cast %s.%s=%r to %s", section, key, raw, type_)
            return default  # type: ignore[return-value]

    def strategy_section(self, section: str) -> dict:
        """Return an entire strategy section as a dict (shallow copy)."""
        with self._lock:
            data = self._strategies.get(section, {})
            return dict(data) if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Typed access — risk
    # ------------------------------------------------------------------

    def risk(self, section: str, key: str, type_: type[T] = float, default: T | None = None) -> T:
        """Get a risk parameter with type coercion.

        Same semantics as :meth:`strategy`.
        """
        with self._lock:
            section_data = self._risk.get(section, {})
            if not isinstance(section_data, dict):
                return default  # type: ignore[return-value]
            raw = section_data.get(key)

        if raw is None:
            return default  # type: ignore[return-value]
        try:
            return type_(raw)
        except (TypeError, ValueError):
            logger.warning("ConfigManager: cannot cast %s.%s=%r to %s", section, key, raw, type_)
            return default  # type: ignore[return-value]

    def risk_section(self, section: str) -> dict:
        """Return an entire risk section as a dict (shallow copy)."""
        with self._lock:
            data = self._risk.get(section, {})
            return dict(data) if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Allocations convenience
    # ------------------------------------------------------------------

    @property
    def allocations(self) -> dict[str, float]:
        """Strategy allocation weights from strategies.yaml."""
        with self._lock:
            raw = self._strategies.get("allocations", {})
            if not isinstance(raw, dict):
                return {}
            return {k: float(v) for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run validation checks, returning a list of error messages.

        Returns an empty list if everything is valid.
        """
        errors: list[str] = []

        # Allocation sum
        alloc = self.allocations
        if alloc:
            total = sum(alloc.values())
            if total > 1.01:
                errors.append(f"Strategy allocations sum to {total:.2%}, must be <= 100%")

        # Kelly bounds
        kelly_min = self.risk("kelly", "min_risk", float, 0.003)
        kelly_max = self.risk("kelly", "max_risk", float, 0.02)
        if kelly_min >= kelly_max:
            errors.append(f"kelly.min_risk ({kelly_min}) must be < kelly.max_risk ({kelly_max})")

        # Risk per trade
        rpt = self.risk("position_sizing", "risk_per_trade_pct", float, 0.008)
        if rpt <= 0 or rpt > 0.10:
            errors.append(f"position_sizing.risk_per_trade_pct ({rpt}) should be 0-10%")

        # Max position
        mpp = self.risk("position_sizing", "max_position_pct", float, 0.08)
        if mpp <= 0 or mpp > 0.50:
            errors.append(f"position_sizing.max_position_pct ({mpp}) should be 0-50%")

        if errors:
            for e in errors:
                logger.error("ConfigManager validation: %s", e)
        else:
            logger.info("ConfigManager: validation passed")

        return errors

    # ------------------------------------------------------------------
    # Hot-reload watchers
    # ------------------------------------------------------------------

    def on_reload(self, callback) -> None:
        """Register a callback to be invoked after a successful reload.

        Callback signature: ``callback(config_manager: ConfigManager) -> None``
        """
        self._watchers.append(callback)

    def _notify_watchers(self) -> None:
        for cb in self._watchers:
            try:
                cb(self)
            except Exception as exc:
                logger.error("ConfigManager: watcher %s failed: %s", cb, exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def last_loaded(self) -> datetime | None:
        return self._last_loaded

    def __repr__(self) -> str:
        strat_keys = list(self._strategies.keys()) if self._strategies else []
        risk_keys = list(self._risk.keys()) if self._risk else []
        return f"ConfigManager(strategies={strat_keys}, risk={risk_keys}, loaded={self._last_loaded})"
