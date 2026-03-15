"""V8: Execution quality analytics.

Tracks actual vs. expected execution quality to measure real slippage,
fill rates, and latency.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import config
import database

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of an order execution."""
    order_id: str
    symbol: str
    strategy: str
    side: str
    expected_price: float
    filled_price: float = 0.0
    slippage_pct: float = 0.0
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    latency_ms: int = 0
    qty_requested: int = 0
    qty_filled: int = 0
    fill_rate: float = 0.0


class ExecutionAnalytics:
    """Track and analyze execution quality."""

    def __init__(self):
        self._pending: dict[str, ExecutionRecord] = {}

    def record_submission(self, order_id: str, symbol: str, strategy: str,
                         side: str, expected_price: float, qty: int,
                         submitted_at: datetime | None = None):
        """Record an order submission."""
        self._pending[order_id] = ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            strategy=strategy,
            side=side,
            expected_price=expected_price,
            qty_requested=qty,
            submitted_at=submitted_at or datetime.now(config.ET),
        )

    def record_fill(self, order_id: str, filled_price: float, filled_qty: int,
                    filled_at: datetime | None = None):
        """Record an order fill and compute metrics."""
        record = self._pending.pop(order_id, None)
        if not record:
            return None

        record.filled_price = filled_price
        record.qty_filled = filled_qty
        record.filled_at = filled_at or datetime.now(config.ET)
        record.fill_rate = filled_qty / record.qty_requested if record.qty_requested > 0 else 0

        # Compute slippage (adjusted for side)
        if record.expected_price > 0:
            if record.side == "buy":
                record.slippage_pct = (filled_price - record.expected_price) / record.expected_price
            else:
                record.slippage_pct = (record.expected_price - filled_price) / record.expected_price

        # Compute latency
        if record.submitted_at and record.filled_at:
            record.latency_ms = int(
                (record.filled_at - record.submitted_at).total_seconds() * 1000
            )

        # Save to database
        try:
            database.save_execution_analytics(
                order_id=record.order_id,
                symbol=record.symbol,
                strategy=record.strategy,
                side=record.side,
                expected_price=record.expected_price,
                filled_price=record.filled_price,
                slippage_pct=record.slippage_pct,
                submitted_at=record.submitted_at,
                filled_at=record.filled_at,
                latency_ms=record.latency_ms,
                qty_requested=record.qty_requested,
                qty_filled=record.qty_filled,
                fill_rate=record.fill_rate,
            )
        except Exception as e:
            logger.debug(f"Failed to save execution analytics: {e}")

        return record

    def get_strategy_stats(self, strategy: str | None = None) -> dict:
        """Get execution quality stats, optionally filtered by strategy."""
        try:
            records = database.get_execution_analytics(strategy=strategy)
        except Exception:
            return {}

        if not records:
            return {}

        slippages = [r["slippage_pct"] for r in records if r["slippage_pct"] is not None]
        latencies = [r["latency_ms"] for r in records if r["latency_ms"]]
        fill_rates = [r["fill_rate"] for r in records if r["fill_rate"]]

        import numpy as np
        return {
            "avg_slippage_pct": float(np.mean(slippages)) if slippages else 0.0,
            "median_slippage_pct": float(np.median(slippages)) if slippages else 0.0,
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "avg_fill_rate": float(np.mean(fill_rates)) if fill_rates else 0.0,
            "total_executions": len(records),
        }
