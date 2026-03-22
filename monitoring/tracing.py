"""T6-007: Distributed tracing with OpenTelemetry.

Optional tracing setup that instruments key trading operations with trace spans.
Fails gracefully if opentelemetry is not installed — all trace functions become
no-ops.

Instrumented operations:
- scan_cycle: Full scan loop iteration
- fetch_bars: Bar data fetching (per symbol or batch)
- compute_indicators: Indicator computation
- evaluate_signal: Signal evaluation and scoring
- submit_order: Order submission to broker

Usage:
    from monitoring.tracing import tracer, trace_span

    with trace_span("scan_cycle", attributes={"regime": "BULL"}):
        # ... scan logic ...

    # Or as decorator:
    @traced("fetch_bars")
    def my_fetch(symbol):
        ...

Setup:
    Call `setup_tracing()` once at startup. If opentelemetry is not installed,
    this is a no-op and all trace functions return dummy contexts.
"""

import logging
import functools
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- Conditional OpenTelemetry import ---
_OTEL_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    Resource = None  # type: ignore


# Module-level tracer (real or no-op)
_tracer = None
_initialized = False


class _NoOpSpan:
    """Dummy span that does nothing when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """Dummy tracer that returns no-op spans."""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()


def setup_tracing(
    service_name: str = "velox-trading-bot",
    exporter: str = "console",
    endpoint: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for the trace resource.
        exporter: Exporter type: "console", "otlp", or "jaeger".
        endpoint: OTLP/Jaeger endpoint URL (e.g., "http://localhost:4317").

    Returns:
        True if tracing was initialized, False if OpenTelemetry is not available.
    """
    global _tracer, _initialized

    if not _OTEL_AVAILABLE:
        logger.info("T6-007: OpenTelemetry not installed — tracing disabled (pip install opentelemetry-sdk)")
        _tracer = _NoOpTracer()
        _initialized = True
        return False

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Select exporter
        if exporter == "otlp" and endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                span_exporter = OTLPSpanExporter(endpoint=endpoint)
            except ImportError:
                logger.warning("T6-007: OTLP exporter not available, falling back to console")
                span_exporter = ConsoleSpanExporter()
        elif exporter == "jaeger" and endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                span_exporter = JaegerExporter(agent_host_name=endpoint.split(":")[0])
            except ImportError:
                logger.warning("T6-007: Jaeger exporter not available, falling back to console")
                span_exporter = ConsoleSpanExporter()
        else:
            span_exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer("velox.trading", "11.2")
        _initialized = True

        logger.info("T6-007: OpenTelemetry tracing initialized (exporter=%s)", exporter)
        return True

    except Exception as e:
        logger.warning("T6-007: Tracing setup failed (degrading gracefully): %s", e)
        _tracer = _NoOpTracer()
        _initialized = True
        return False


def get_tracer():
    """Get the configured tracer (real or no-op)."""
    global _tracer, _initialized
    if not _initialized:
        # Auto-initialize with defaults on first use
        setup_tracing()
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for creating a trace span.

    Usage:
        with trace_span("scan_cycle", {"regime": "BULL", "symbols": 120}):
            # ... code ...

    If OpenTelemetry is not installed, this is a no-op.
    """
    tracer = get_tracer()

    if isinstance(tracer, _NoOpTracer):
        yield _NoOpSpan()
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            if _OTEL_AVAILABLE:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def traced(span_name: str | None = None, attributes: dict[str, Any] | None = None):
    """Decorator for adding a trace span to a function.

    Usage:
        @traced("fetch_bars")
        def fetch_bars(symbol):
            ...

        @traced()  # Uses function name as span name
        def compute_indicators(bars):
            ...
    """
    def decorator(fn):
        name = span_name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with trace_span(name, attributes):
                return fn(*args, **kwargs)

        return wrapper
    return decorator


# Pre-defined span names for consistency across the codebase
SPAN_SCAN_CYCLE = "velox.scan_cycle"
SPAN_FETCH_BARS = "velox.fetch_bars"
SPAN_COMPUTE_INDICATORS = "velox.compute_indicators"
SPAN_EVALUATE_SIGNAL = "velox.evaluate_signal"
SPAN_SUBMIT_ORDER = "velox.submit_order"
SPAN_RISK_CHECK = "velox.risk_check"
SPAN_BROKER_SYNC = "velox.broker_sync"
