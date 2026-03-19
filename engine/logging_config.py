"""V10 Engine — Structured logging with structlog.

Provides structured, JSON-capable logging that works alongside the existing
stdlib logging. In development mode, renders human-readable colored output.
In production (STRUCTURED_LOGGING=true), outputs JSON lines for log aggregation.

Usage:
    from engine.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("trade_opened", symbol="AAPL", qty=10, strategy="STAT_MR")

    # Produces (dev mode):
    # 2026-03-19 14:30:00 [info] trade_opened  symbol=AAPL qty=10 strategy=STAT_MR

    # Produces (production mode):
    # {"event":"trade_opened","symbol":"AAPL","qty":10,"strategy":"STAT_MR",
    #  "timestamp":"2026-03-19T14:30:00","level":"info","logger":"engine.signal_processor"}
"""

import os
import logging
import sys

import structlog

import config

# Determine mode from environment/config
PRODUCTION_MODE = os.getenv("STRUCTURED_LOGGING", "").lower() in ("true", "1", "yes")
LOG_LEVEL = getattr(config, "LOG_LEVEL", "INFO")


def configure_logging():
    """Configure structlog + stdlib logging integration.

    Call once at startup (before any logging). Safe to call multiple times.
    """
    # Shared processors for both structlog and stdlib
    shared_processors = [
        structlog.contextvars.merge_contextvars,
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
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
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
