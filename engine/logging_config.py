"""V10 Engine — Structured logging with structlog.

Provides structured, JSON-capable logging that works alongside the existing
stdlib logging. In development mode, renders human-readable colored output.
In production (STRUCTURED_LOGGING=true), outputs JSON lines for log aggregation.

PROD-010: Added CorrelationIdFilter for request/scan tracing across modules.
Use `set_correlation_id()` at the start of each scan cycle or request handler.

Usage:
    from engine.logging_config import get_logger, set_correlation_id
    logger = get_logger(__name__)

    set_correlation_id()  # Auto-generates UUID for this scan cycle
    logger.info("trade_opened", symbol="AAPL", qty=10, strategy="STAT_MR")

    # Produces (production mode):
    # {"event":"trade_opened","symbol":"AAPL","qty":10,"strategy":"STAT_MR",
    #  "timestamp":"2026-03-19T14:30:00","level":"info","logger":"engine.signal_processor",
    #  "correlation_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import uuid
import threading

import structlog

import config

# Determine mode from environment/config
PRODUCTION_MODE = os.getenv("STRUCTURED_LOGGING", "").lower() in ("true", "1", "yes")
LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# PROD-010: Correlation ID support
# ---------------------------------------------------------------------------

# Thread-local storage for correlation IDs
_correlation_local = threading.local()


def set_correlation_id(cid: str | None = None) -> str:
    """Set a correlation ID for the current thread/scan cycle.

    Args:
        cid: Explicit correlation ID. If None, auto-generates a UUID.

    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = str(uuid.uuid4())
    _correlation_local.correlation_id = cid
    # Also bind to structlog contextvars for structured logging
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def get_correlation_id() -> str:
    """Get the current thread's correlation ID (or 'none' if not set)."""
    return getattr(_correlation_local, "correlation_id", "none")


def clear_correlation_id():
    """Clear the correlation ID for the current thread."""
    _correlation_local.correlation_id = "none"
    try:
        structlog.contextvars.unbind_contextvars("correlation_id")
    except Exception:
        pass


class CorrelationIdFilter(logging.Filter):
    """PROD-010: Stdlib logging filter that adds correlation_id to every LogRecord.

    Attach to handlers so that formatters can include %(correlation_id)s.
    Works with both stdlib and structlog-formatted output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True


def _add_correlation_id(logger, method_name, event_dict):
    """Structlog processor that adds correlation_id to event dict."""
    cid = get_correlation_id()
    if cid != "none":
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging():
    """Configure structlog + stdlib logging integration.

    Call once at startup (before any logging). Safe to call multiple times.
    """
    # Shared processors for both structlog and stdlib
    # PROD-010: Added _add_correlation_id for request tracing
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if PRODUCTION_MODE:
        # JSON output for log aggregation (ELK, Datadog, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable colored output for development
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Root handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    # PROD-010: Add correlation ID filter to all handlers
    handler.addFilter(CorrelationIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # File handler (always JSON for machine parsing)
    if hasattr(config, "LOG_FILE") and config.LOG_FILE:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=10_000_000,  # 10 MB per file
            backupCount=5,        # Keep 5 rotated files
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        # PROD-010: correlation ID filter on file handler too
        file_handler.addFilter(CorrelationIdFilter())
        root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str = None):
    """Get a structlog logger.

    Wraps stdlib logging so all existing `logging.getLogger()` calls
    continue to work, while new code gets structured logging.
    """
    return structlog.get_logger(name)


# Trade-specific context binding helpers
def bind_trade_context(symbol: str, strategy: str, side: str = "", qty: int = 0):
    """Bind trade context to all subsequent log calls in this context."""
    structlog.contextvars.bind_contextvars(
        symbol=symbol, strategy=strategy, side=side, qty=qty
    )


def clear_trade_context():
    """Clear trade-specific context."""
    structlog.contextvars.unbind_contextvars("symbol", "strategy", "side", "qty")
