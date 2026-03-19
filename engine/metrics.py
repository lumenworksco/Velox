"""V10 Engine — Prometheus metrics for monitoring and alerting.

Exposes metrics at /metrics endpoint (added to web dashboard).
Use with Prometheus + Grafana for production monitoring.

Metrics exposed:
- velox_trades_total (counter): Total trades by strategy, side, outcome
- velox_signals_total (counter): Total signals by strategy, action (accepted/filtered)
- velox_open_positions (gauge): Current open position count
- velox_portfolio_value (gauge): Current portfolio value
- velox_day_pnl_pct (gauge): Today's P&L percentage
- velox_circuit_breaker_tier (gauge): Current circuit breaker tier (0-4)
- velox_order_latency_seconds (histogram): Order submission latency
- velox_scan_duration_seconds (histogram): Strategy scan duration
"""

import logging
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY,
)

logger = logging.getLogger(__name__)

# ── Counters ────────────────────────────────────────────────────────────────

trades_total = Counter(
    'velox_trades_total',
    'Total trades executed',
    ['strategy', 'side', 'outcome'],  # outcome: win, loss, breakeven
)

signals_total = Counter(
    'velox_signals_total',
    'Total signals generated and processed',
    ['strategy', 'action'],  # action: accepted, filtered, pnl_halt
)

orders_total = Counter(
    'velox_orders_total',
    'Total orders by state',
    ['strategy', 'state'],  # state: submitted, filled, failed, cancelled, rejected
)

circuit_breaker_activations = Counter(
    'velox_circuit_breaker_activations_total',
    'Circuit breaker tier changes',
    ['tier'],
)

# ── Gauges ──────────────────────────────────────────────────────────────────

open_positions = Gauge(
    'velox_open_positions',
    'Current number of open positions',
)

portfolio_value = Gauge(
    'velox_portfolio_value_dollars',
    'Current portfolio value in dollars',
)

day_pnl_pct = Gauge(
    'velox_day_pnl_pct',
    'Today P&L as percentage',
)

circuit_breaker_tier = Gauge(
    'velox_circuit_breaker_tier',
    'Current circuit breaker tier (0=normal, 1=yellow, 2=orange, 3=red, 4=black)',
)

vol_scalar = Gauge(
    'velox_vol_scalar',
    'Current volatility scalar',
)

consistency_score = Gauge(
    'velox_consistency_score',
    'Latest consistency score',
)

# ── Histograms ──────────────────────────────────────────────────────────────

order_latency = Histogram(
    'velox_order_latency_seconds',
    'Time from signal to order submission',
    ['strategy'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

scan_duration = Histogram(
    'velox_scan_duration_seconds',
    'Strategy scan cycle duration',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ── Info ────────────────────────────────────────────────────────────────────

system_info = Info(
    'velox',
    'Velox trading system information',
)
system_info.info({
    'version': 'V10',
    'mode': 'paper',  # Updated at runtime
})


# ── Helper functions ────────────────────────────────────────────────────────

def record_trade(strategy: str, side: str, pnl: float):
    """Record a completed trade."""
    if pnl > 0:
        outcome = "win"
    elif pnl < 0:
        outcome = "loss"
    else:
        outcome = "breakeven"
    trades_total.labels(strategy=strategy, side=side, outcome=outcome).inc()


def record_signal(strategy: str, accepted: bool, reason: str = ""):
    """Record a signal generation."""
    action = "accepted" if accepted else "filtered"
    signals_total.labels(strategy=strategy, action=action).inc()


def update_portfolio_state(value: float, pnl_pct: float, open_count: int,
                           cb_tier: int = 0, vol: float = 1.0, consistency: float = 0.0):
    """Update all portfolio gauges at once (called from main loop)."""
    portfolio_value.set(value)
    day_pnl_pct.set(pnl_pct)
    open_positions.set(open_count)
    circuit_breaker_tier.set(cb_tier)
    vol_scalar.set(vol)
    consistency_score.set(consistency)


def get_metrics_text() -> bytes:
    """Generate Prometheus text format metrics."""
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Return the Prometheus content type header."""
    return CONTENT_TYPE_LATEST


def add_metrics_endpoint(app):
    """Add /metrics endpoint to a FastAPI app.

    Usage:
        from engine.metrics import add_metrics_endpoint
        add_metrics_endpoint(app)
    """
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        return Response(
            content=get_metrics_text(),
            media_type=get_content_type(),
        )
    logger.info("Prometheus metrics endpoint registered at /metrics")
