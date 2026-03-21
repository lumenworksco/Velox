"""PROD-004: Heartbeat watchdog thread — monitors main loop liveness.

Runs as a daemon thread checking that the main trading loop is still alive.
The main loop calls `heartbeat()` on each cycle. If no heartbeat is received
within the timeout window (default 120s), the watchdog logs a CRITICAL message
and optionally triggers an alert callback.

Usage:
    from monitoring.watchdog import HeartbeatWatchdog

    watchdog = HeartbeatWatchdog(timeout_sec=120, check_interval_sec=60)
    watchdog.start()

    # In main loop:
    while running:
        watchdog.heartbeat()
        # ... scan, trade, etc.

    watchdog.stop()
"""

import logging
import threading
import time as _time
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HeartbeatWatchdog:
    """Daemon thread that monitors main loop heartbeats.

    If no heartbeat is received within `timeout_sec`, logs CRITICAL and
    invokes the optional `alert_callback` for external notification
    (e.g., Telegram, PagerDuty, email).
    """

    def __init__(
        self,
        timeout_sec: float = 120.0,
        check_interval_sec: float = 60.0,
        alert_callback: Optional[Callable[[str], None]] = None,
        name: str = "MainLoop",
    ):
        """Initialize the heartbeat watchdog.

        Args:
            timeout_sec: Max seconds between heartbeats before alerting.
            check_interval_sec: How often the watchdog checks for staleness.
            alert_callback: Optional callable(message: str) for external alerts.
            name: Human-readable name of the monitored component.
        """
        self._timeout_sec = timeout_sec
        self._check_interval_sec = check_interval_sec
        self._alert_callback = alert_callback
        self._name = name

        # Heartbeat signaling
        self._heartbeat_event = threading.Event()
        self._last_heartbeat: float = _time.monotonic()
        self._heartbeat_count: int = 0

        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._alert_fired = False

        logger.info(
            "PROD-004: HeartbeatWatchdog initialized (timeout=%ds, check_interval=%ds, name=%s)",
            timeout_sec, check_interval_sec, name,
        )

    def heartbeat(self):
        """Signal that the main loop is still alive. Call every scan cycle."""
        self._last_heartbeat = _time.monotonic()
        self._heartbeat_count += 1
        self._heartbeat_event.set()
        # Reset alert state on successful heartbeat
        if self._alert_fired:
            logger.info("PROD-004: %s heartbeat restored after timeout alert", self._name)
            self._alert_fired = False

    def start(self):
        """Start the watchdog daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("PROD-004: Watchdog already running")
            return

        self._stop_event.clear()
        self._last_heartbeat = _time.monotonic()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name=f"Watchdog-{self._name}",
            daemon=True,
        )
        self._thread.start()
        logger.info("PROD-004: Watchdog started for %s", self._name)

    def stop(self):
        """Stop the watchdog thread gracefully."""
        self._stop_event.set()
        self._heartbeat_event.set()  # Unblock any wait
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("PROD-004: Watchdog stopped for %s", self._name)

    def _monitor_loop(self):
        """Internal loop: periodically check heartbeat freshness."""
        while not self._stop_event.is_set():
            # Wait for check interval or stop signal
            self._stop_event.wait(timeout=self._check_interval_sec)
            if self._stop_event.is_set():
                break

            elapsed = _time.monotonic() - self._last_heartbeat

            if elapsed > self._timeout_sec:
                if not self._alert_fired:
                    self._alert_fired = True
                    msg = (
                        f"PROD-004 WATCHDOG ALERT: {self._name} has not sent a heartbeat "
                        f"for {elapsed:.0f}s (timeout={self._timeout_sec}s). "
                        f"Last heartbeat count: {self._heartbeat_count}."
                    )
                    logger.critical(msg)

                    # Fire external alert if configured
                    if self._alert_callback:
                        try:
                            self._alert_callback(msg)
                        except Exception as e:
                            logger.error(
                                "PROD-004: Alert callback failed: %s", e
                            )
            else:
                logger.debug(
                    "PROD-004: %s heartbeat OK (last=%.0fs ago, count=%d)",
                    self._name, elapsed, self._heartbeat_count,
                )

    @property
    def is_alive(self) -> bool:
        """Whether the watchdog thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds elapsed since the last heartbeat."""
        return _time.monotonic() - self._last_heartbeat

    def stats(self) -> dict:
        """Return watchdog status for monitoring/dashboard."""
        return {
            "name": self._name,
            "running": self.is_alive,
            "timeout_sec": self._timeout_sec,
            "heartbeat_count": self._heartbeat_count,
            "seconds_since_heartbeat": round(self.seconds_since_heartbeat, 1),
            "alert_fired": self._alert_fired,
        }
