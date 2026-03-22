"""T2-003: Structured failure modes — replace silent `except Exception: pass`.

Provides a FailureMode enum and a handle_failure() dispatcher so that every
catch site explicitly declares its intended degradation behaviour.

Usage::

    from engine.failure_modes import FailureMode, handle_failure

    try:
        do_something()
    except Exception as exc:
        handle_failure(FailureMode.SKIP_SIGNAL, "signal_processor.cost_filter", exc,
                       symbol="AAPL", strategy="STAT_MR")
"""

import enum
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class FailureMode(enum.Enum):
    """Declared failure behaviour for a catch site."""

    SKIP_SIGNAL = "skip_signal"
    """Drop the current signal / trade but continue processing others."""

    DEGRADE_GRACEFULLY = "degrade_gracefully"
    """Use a safe default and continue (e.g., multiplier=1.0)."""

    ABORT_WITH_ALERT = "abort_with_alert"
    """Log at CRITICAL and raise — caller should catch at the top level."""


# ---------------------------------------------------------------------------
# Failure counters — exposed for monitoring / dashboards
# ---------------------------------------------------------------------------

_failure_counts: dict[str, int] = {}  # context_key -> count


def get_failure_counts() -> dict[str, int]:
    """Return a snapshot of per-context failure counts since startup."""
    return dict(_failure_counts)


def reset_failure_counts():
    """Clear failure counters (useful for tests or daily reset)."""
    _failure_counts.clear()


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------

def handle_failure(
    mode: FailureMode,
    context: str,
    exc: Exception,
    *,
    symbol: str = "",
    strategy: str = "",
    details: str = "",
) -> None:
    """Handle a caught exception according to the declared failure mode.

    Args:
        mode: How the failure should be handled.
        context: Human-readable location identifier (e.g., "signal_processor.cost_filter").
        exc: The caught exception.
        symbol: Optional symbol context for log enrichment.
        strategy: Optional strategy context for log enrichment.
        details: Optional extra detail string.

    Raises:
        RuntimeError: When mode is ABORT_WITH_ALERT, the original exception
                      is re-raised wrapped in a RuntimeError.
    """
    _failure_counts[context] = _failure_counts.get(context, 0) + 1

    sym_str = f" [{symbol}]" if symbol else ""
    strat_str = f" ({strategy})" if strategy else ""
    detail_str = f" — {details}" if details else ""
    msg = (
        f"T2-003 [{mode.value}] {context}{sym_str}{strat_str}: "
        f"{exc.__class__.__name__}: {exc}{detail_str}"
    )

    if mode is FailureMode.SKIP_SIGNAL:
        logger.warning(msg)

    elif mode is FailureMode.DEGRADE_GRACEFULLY:
        logger.info(msg)

    elif mode is FailureMode.ABORT_WITH_ALERT:
        logger.critical(msg)
        logger.critical("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        # Attempt to emit an alert via the event bus
        try:
            from engine.event_log import log_event, EventType
            log_event(
                EventType.SYSTEM_ERROR,
                f"failure_mode.{context}",
                symbol=symbol,
                strategy=strategy,
                details=msg,
                severity="CRITICAL",
            )
        except Exception:
            pass  # Alert emission itself must not throw
        raise RuntimeError(f"ABORT_WITH_ALERT in {context}: {exc}") from exc
