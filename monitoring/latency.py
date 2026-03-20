"""MON-004: Latency Monitoring — instrument critical paths with timing and alert on SLA breaches.

Tracks latency for:
  - Bar fetch per symbol
  - Signal computation per strategy
  - Filter pipeline per filter
  - Order submission
  - Fill latency
  - Full cycle time (signal -> fill)

Alert thresholds:
  - Bar fetch > 5s
  - Scan cycle > 90s
  - Order submission > 2s
"""

import logging
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Aggregated latency statistics for an operation."""

    operation: str
    count: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    total_ms: float = 0.0
    breaches: int = 0  # number of times SLA was exceeded


# ---------------------------------------------------------------------------
# SLA thresholds (milliseconds)
# ---------------------------------------------------------------------------

DEFAULT_SLA_THRESHOLDS: dict[str, float] = {
    "bar_fetch": 5000.0,
    "scan_cycle": 90000.0,
    "order_submission": 2000.0,
    "signal_computation": 10000.0,
    "filter_pipeline": 5000.0,
    "fill_latency": 30000.0,
    "full_cycle": 120000.0,
}


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Instruments critical paths with timing, computes statistics, and alerts on breaches.

    Usage:
        tracker = LatencyTracker()

        # Context-manager style (preferred)
        with tracker.time("bar_fetch", symbol="AAPL"):
            bars = fetch_bars("AAPL")

        # Manual style
        tid = tracker.start_timer("order_submission")
        submit_order(...)
        elapsed = tracker.stop_timer(tid)

        # Get stats
        stats = tracker.get_stats("bar_fetch")
    """

    def __init__(self, sla_thresholds: dict[str, float] | None = None,
                 alert_callback=None,
                 max_samples: int = 10000):
        """
        Args:
            sla_thresholds: Operation -> max allowed ms. Uses defaults if None.
            alert_callback: Optional callable(level, message, source) for alerts.
            max_samples: Max latency samples to retain per operation.
        """
        self._sla = sla_thresholds or dict(DEFAULT_SLA_THRESHOLDS)
        self._alert_callback = alert_callback
        self._max_samples = max_samples

        # Active timers: timer_id -> (operation, start_monotonic, metadata)
        self._active: dict[str, tuple[str, float, dict]] = {}
        self._active_lock = threading.Lock()

        # Completed samples: operation -> list of elapsed_ms
        self._samples: dict[str, list[float]] = {}
        self._samples_lock = threading.Lock()

        # Breach counters
        self._breaches: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API — manual timer
    # ------------------------------------------------------------------

    def start_timer(self, operation: str, **metadata) -> str:
        """Start a latency timer for an operation.

        Args:
            operation: Name of the operation (e.g. "bar_fetch", "order_submission").
            **metadata: Optional key-value pairs (e.g. symbol="AAPL").

        Returns:
            timer_id (str) — pass to stop_timer() when done.
        """
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        with self._active_lock:
            self._active[timer_id] = (operation, time.monotonic(), metadata)
        return timer_id

    def stop_timer(self, timer_id: str) -> float:
        """Stop a timer and record the elapsed time.

        Args:
            timer_id: The ID returned by start_timer().

        Returns:
            Elapsed time in milliseconds.
        """
        end = time.monotonic()
        with self._active_lock:
            entry = self._active.pop(timer_id, None)

        if entry is None:
            logger.warning(f"LatencyTracker: Unknown timer_id {timer_id}")
            return 0.0

        operation, start, metadata = entry
        elapsed_ms = (end - start) * 1000.0

        self._record(operation, elapsed_ms, metadata)
        return elapsed_ms

    # ------------------------------------------------------------------
    # Public API — context manager
    # ------------------------------------------------------------------

    class _TimerContext:
        """Context manager for timing an operation."""

        def __init__(self, tracker: "LatencyTracker", operation: str, metadata: dict):
            self._tracker = tracker
            self._operation = operation
            self._metadata = metadata
            self._timer_id: str | None = None
            self.elapsed_ms: float = 0.0

        def __enter__(self):
            self._timer_id = self._tracker.start_timer(self._operation, **self._metadata)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._timer_id:
                self.elapsed_ms = self._tracker.stop_timer(self._timer_id)
            return False  # don't suppress exceptions

    def time(self, operation: str, **metadata) -> _TimerContext:
        """Return a context manager that times the enclosed block.

        Usage:
            with tracker.time("bar_fetch", symbol="AAPL") as t:
                do_work()
            print(f"Took {t.elapsed_ms:.1f}ms")
        """
        return self._TimerContext(self, operation, metadata)

    # ------------------------------------------------------------------
    # Public API — statistics
    # ------------------------------------------------------------------

    def get_stats(self, operation: str) -> LatencyStats:
        """Get aggregated latency statistics for an operation.

        Returns a LatencyStats dataclass. Returns zeroed stats if no data.
        """
        with self._samples_lock:
            samples = list(self._samples.get(operation, []))
            breaches = self._breaches.get(operation, 0)

        if not samples:
            return LatencyStats(operation=operation)

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return LatencyStats(
            operation=operation,
            count=n,
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            p95_ms=sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
            p99_ms=sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
            total_ms=sum(samples),
            breaches=breaches,
        )

    def get_all_stats(self) -> dict[str, LatencyStats]:
        """Get stats for all tracked operations."""
        with self._samples_lock:
            operations = list(self._samples.keys())
        return {op: self.get_stats(op) for op in operations}

    def get_active_timers(self) -> list[dict]:
        """Return info about currently running timers (for debugging stalls)."""
        now = time.monotonic()
        result = []
        with self._active_lock:
            for tid, (op, start, meta) in self._active.items():
                result.append({
                    "timer_id": tid,
                    "operation": op,
                    "elapsed_ms": (now - start) * 1000.0,
                    "metadata": meta,
                })
        return result

    # ------------------------------------------------------------------
    # Public API — decorator
    # ------------------------------------------------------------------

    def track_latency(self, operation: str):
        """Decorator for tracking a function's latency.

        Usage:
            @tracker.track_latency("scan_cycle")
            def run_scan():
                ...
        """
        import functools

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.monotonic()
                try:
                    return fn(*args, **kwargs)
                finally:
                    elapsed_ms = (time.monotonic() - start) * 1000.0
                    self._record(operation, elapsed_ms, {})
            return wrapper
        return decorator

    def reset(self, operation: str | None = None):
        """Reset latency data for an operation or all operations."""
        with self._samples_lock:
            if operation:
                self._samples.pop(operation, None)
                self._breaches.pop(operation, None)
            else:
                self._samples.clear()
                self._breaches.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, operation: str, elapsed_ms: float, metadata: dict):
        """Record a latency sample and check SLA."""
        with self._samples_lock:
            if operation not in self._samples:
                self._samples[operation] = []
            self._samples[operation].append(elapsed_ms)

            # Trim to max samples
            if len(self._samples[operation]) > self._max_samples:
                self._samples[operation] = self._samples[operation][-self._max_samples:]

        # Check SLA breach
        sla = self._sla.get(operation)
        if sla is not None and elapsed_ms > sla:
            with self._samples_lock:
                self._breaches[operation] = self._breaches.get(operation, 0) + 1

            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items()) if metadata else ""
            msg = (
                f"Latency SLA breach: {operation} took {elapsed_ms:.0f}ms "
                f"(limit: {sla:.0f}ms)"
                + (f" [{meta_str}]" if meta_str else "")
            )
            logger.warning(f"LATENCY: {msg}")

            if self._alert_callback:
                try:
                    level = "CRITICAL" if elapsed_ms > sla * 3 else "WARNING"
                    self._alert_callback(level, msg, "latency_tracker")
                except Exception as e:
                    logger.error(f"LatencyTracker alert callback failed: {e}")
