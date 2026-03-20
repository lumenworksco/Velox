"""MON-003: Alerting System — tiered alerts with rate limiting, suppression, and audit.

Alert levels:
  INFO     — Trade executed, daily summary        (rate limit: 1/min)
  WARNING  — Strategy underperforming, high VaR   (rate limit: 1/5min)
  CRITICAL — Circuit breaker, position sync issue  (rate limit: 1/5min per issue)
  EMERGENCY — Kill switch, margin call             (no rate limit)

Integrates with existing Telegram notifications (notifications.py).
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------

class AlertLevel(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


LEVEL_NAMES = {
    AlertLevel.INFO: "INFO",
    AlertLevel.WARNING: "WARNING",
    AlertLevel.CRITICAL: "CRITICAL",
    AlertLevel.EMERGENCY: "EMERGENCY",
}

LEVEL_EMOJI = {
    AlertLevel.INFO: "\u2139\ufe0f",
    AlertLevel.WARNING: "\u26a0\ufe0f",
    AlertLevel.CRITICAL: "\U0001f6a8",
    AlertLevel.EMERGENCY: "\U0001f198",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single alert record."""

    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(config.ET))
    alert_id: str = ""
    suppressed: bool = False
    delivered: bool = False


@dataclass
class MaintenanceWindow:
    """Time window during which alerts are suppressed."""

    start: datetime
    end: datetime
    reason: str = ""
    suppress_levels: tuple = (AlertLevel.INFO, AlertLevel.WARNING)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket-style rate limiter per alert key."""

    def __init__(self):
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, min_interval_sec: float) -> bool:
        """Return True if the alert is allowed (not rate-limited)."""
        now = time.monotonic()
        with self._lock:
            last = self._last_sent.get(key, 0.0)
            if now - last >= min_interval_sec:
                self._last_sent[key] = now
                return True
            return False

    def reset(self, key: Optional[str] = None):
        """Reset rate limit state."""
        with self._lock:
            if key:
                self._last_sent.pop(key, None)
            else:
                self._last_sent.clear()


# Rate limit intervals by level (seconds)
_RATE_LIMITS = {
    AlertLevel.INFO: 60,        # 1 per minute
    AlertLevel.WARNING: 300,    # 1 per 5 minutes
    AlertLevel.CRITICAL: 300,   # 1 per 5 minutes (keyed by source)
    AlertLevel.EMERGENCY: 0,    # no rate limit
}


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Centralized alerting with tiered delivery, rate limiting, and audit logging.

    Usage:
        manager = AlertManager()
        manager.send_alert("WARNING", "Sharpe ratio below threshold", "metrics_pipeline")
        manager.add_maintenance_window(start, end, reason="deploy")
    """

    def __init__(self, alert_log_path: str | None = None, max_history: int = 5000):
        """
        Args:
            alert_log_path: Path for append-only alert log. Defaults to alert_history.jsonl
                            in the working directory.
            max_history: Max number of alerts to keep in memory.
        """
        self._rate_limiter = _RateLimiter()
        self._maintenance_windows: list[MaintenanceWindow] = []
        self._history: list[Alert] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._alert_counter = 0

        self._log_path = Path(alert_log_path or "alert_history.jsonl")
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, level: str | AlertLevel, message: str, source: str) -> Alert:
        """Send an alert through the pipeline.

        Steps:
          1. Create Alert record
          2. Check maintenance window suppression
          3. Check rate limiting
          4. Deliver via Telegram (if enabled)
          5. Log to audit file

        Args:
            level: "INFO", "WARNING", "CRITICAL", "EMERGENCY" or AlertLevel enum.
            message: Human-readable alert message.
            source: Originating component (e.g. "metrics_pipeline", "reconciler").

        Returns:
            The Alert object (with suppressed/delivered flags set).
        """
        try:
            return self._send_inner(level, message, source)
        except Exception as e:
            logger.error(f"AlertManager.send_alert failed: {e}")
            return Alert(
                level=AlertLevel.INFO,
                message=message,
                source=source,
                suppressed=True,
            )

    def add_maintenance_window(self, start: datetime, end: datetime,
                               reason: str = "",
                               suppress_levels: tuple | None = None):
        """Add a maintenance window during which alerts are suppressed.

        Args:
            start: Window start (timezone-aware, ET).
            end: Window end.
            reason: Human-readable reason for maintenance.
            suppress_levels: Alert levels to suppress. Defaults to INFO + WARNING.
        """
        window = MaintenanceWindow(
            start=start,
            end=end,
            reason=reason,
            suppress_levels=suppress_levels or (AlertLevel.INFO, AlertLevel.WARNING),
        )
        with self._lock:
            self._maintenance_windows.append(window)
        logger.info(f"Maintenance window added: {start} to {end} — {reason}")

    def remove_expired_windows(self):
        """Remove maintenance windows that have ended."""
        now = datetime.now(config.ET)
        with self._lock:
            self._maintenance_windows = [
                w for w in self._maintenance_windows if w.end > now
            ]

    def get_alert_history(self, level: AlertLevel | None = None,
                          source: str | None = None,
                          since: datetime | None = None,
                          limit: int = 100) -> list[Alert]:
        """Query alert history with optional filters.

        Args:
            level: Filter by alert level.
            source: Filter by source component.
            since: Only return alerts after this time.
            limit: Max number of results.

        Returns:
            List of Alert objects (most recent first).
        """
        with self._lock:
            results = list(reversed(self._history))

        if level is not None:
            results = [a for a in results if a.level == level]
        if source is not None:
            results = [a for a in results if a.source == source]
        if since is not None:
            results = [a for a in results if a.timestamp >= since]

        return results[:limit]

    def get_stats(self) -> dict:
        """Return summary statistics about alerts."""
        with self._lock:
            total = len(self._history)
            by_level = {}
            suppressed = 0
            for a in self._history:
                name = LEVEL_NAMES.get(a.level, "UNKNOWN")
                by_level[name] = by_level.get(name, 0) + 1
                if a.suppressed:
                    suppressed += 1
        return {
            "total_alerts": total,
            "by_level": by_level,
            "suppressed": suppressed,
            "active_maintenance_windows": len(self._maintenance_windows),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_inner(self, level: str | AlertLevel, message: str, source: str) -> Alert:
        """Core alert processing pipeline."""
        # Normalize level
        if isinstance(level, str):
            level_enum = {
                "INFO": AlertLevel.INFO,
                "WARNING": AlertLevel.WARNING,
                "CRITICAL": AlertLevel.CRITICAL,
                "EMERGENCY": AlertLevel.EMERGENCY,
            }.get(level.upper(), AlertLevel.INFO)
        else:
            level_enum = level

        # Create alert
        self._alert_counter += 1
        alert = Alert(
            level=level_enum,
            message=message,
            source=source,
            alert_id=f"ALT-{self._alert_counter:06d}",
        )

        # Check maintenance window
        if self._is_in_maintenance(level_enum):
            alert.suppressed = True
            logger.debug(
                f"Alert suppressed (maintenance): [{LEVEL_NAMES[level_enum]}] {message}"
            )
            self._store_alert(alert)
            return alert

        # Check rate limiting (EMERGENCY bypasses)
        if level_enum != AlertLevel.EMERGENCY:
            rate_key = f"{LEVEL_NAMES[level_enum]}:{source}"
            interval = _RATE_LIMITS.get(level_enum, 60)
            if not self._rate_limiter.allow(rate_key, interval):
                alert.suppressed = True
                logger.debug(
                    f"Alert rate-limited: [{LEVEL_NAMES[level_enum]}] {source}: {message}"
                )
                self._store_alert(alert)
                return alert

        # Deliver
        self._deliver(alert)
        alert.delivered = True

        # Store
        self._store_alert(alert)

        return alert

    def _is_in_maintenance(self, level: AlertLevel) -> bool:
        """Check if current time falls within a maintenance window for this level."""
        now = datetime.now(config.ET)
        with self._lock:
            for w in self._maintenance_windows:
                if w.start <= now <= w.end and level in w.suppress_levels:
                    return True
        return False

    def _deliver(self, alert: Alert):
        """Deliver alert via Telegram and local logging."""
        level_name = LEVEL_NAMES.get(alert.level, "INFO")
        emoji = LEVEL_EMOJI.get(alert.level, "")

        # Always log locally
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
            AlertLevel.EMERGENCY: logger.critical,
        }.get(alert.level, logger.info)
        log_fn(f"ALERT [{level_name}] ({alert.source}): {alert.message}")

        # Send via Telegram
        try:
            import notifications
            telegram_msg = (
                f"{emoji} *{level_name}*\n"
                f"Source: {alert.source}\n"
                f"{alert.message}"
            )
            notifications._send_telegram(telegram_msg)
        except Exception as e:
            logger.error(f"AlertManager: Telegram delivery failed: {e}")

    def _store_alert(self, alert: Alert):
        """Store alert in memory and append to log file."""
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Append to log file (audit)
        try:
            entry = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "level": LEVEL_NAMES.get(alert.level, "INFO"),
                "source": alert.source,
                "message": alert.message,
                "suppressed": alert.suppressed,
                "delivered": alert.delivered,
            }
            with self._log_lock:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"AlertManager: Failed to write alert log: {e}")
