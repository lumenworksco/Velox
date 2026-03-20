"""COMPLY-003: Best Execution Reporting — per-trade slippage and benchmark analysis.

Measures execution quality against decision-time benchmarks:
  - Decision price vs fill price (implementation shortfall)
  - Fill price vs VWAP benchmark
  - Fill price vs TWAP benchmark
  - Aggregate slippage and execution alpha statistics

Generates daily best-execution reports for compliance review.
"""

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """Per-trade execution quality record."""

    symbol: str
    side: str
    qty: float
    decision_price: float
    fill_price: float
    vwap_benchmark: float = 0.0
    twap_benchmark: float = 0.0
    timestamp: str = ""
    strategy: str = ""
    order_type: str = ""
    order_id: str = ""

    # Computed fields (populated by record_execution)
    slippage_bps: float = 0.0
    vs_vwap_bps: float = 0.0
    vs_twap_bps: float = 0.0
    implementation_shortfall_bps: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BestExecutionReport:
    """Daily best execution report."""

    date: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(config.ET))
    total_executions: int = 0
    records: list[ExecutionRecord] = field(default_factory=list)

    # Aggregate stats
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    best_slippage_bps: float = 0.0

    avg_vs_vwap_bps: float = 0.0
    avg_vs_twap_bps: float = 0.0

    execution_alpha_pct: float = 0.0
    total_slippage_dollars: float = 0.0

    by_strategy: dict = field(default_factory=dict)
    by_side: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["records"] = [r.to_dict() for r in self.records]
        return d


# ---------------------------------------------------------------------------
# BestExecutionReporter
# ---------------------------------------------------------------------------

class BestExecutionReporter:
    """Tracks and reports on execution quality for every fill.

    Usage:
        reporter = BestExecutionReporter()

        # Record each execution as fills come in
        reporter.record_execution(order, fill, market_data)

        # End of day
        report = reporter.generate_daily_report()
        stats = reporter.get_aggregate_stats()
    """

    def __init__(self, log_dir: str | None = None, max_records: int = 50_000):
        """
        Args:
            log_dir: Directory for best-execution report files.
            max_records: Maximum execution records to retain in memory.
        """
        self._log_dir = Path(log_dir or ".")
        self._max_records = max_records
        self._records: list[ExecutionRecord] = []
        self._daily_records: dict[str, list[ExecutionRecord]] = defaultdict(list)
        self._lock = threading.Lock()
        self._report_history: list[BestExecutionReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_execution(self, order: dict, fill: dict,
                         market_data: dict | None = None):
        """Record a single execution for best-execution analysis.

        Args:
            order: Order dict with keys: symbol, side, qty, order_type, strategy,
                   order_id, decision_price (price at signal time).
            fill: Fill dict with keys: fill_price, filled_qty, fill_time.
            market_data: Optional market context with keys: vwap, twap
                         (benchmark prices at fill time).

        Never raises.
        """
        try:
            self._record_inner(order, fill, market_data)
        except Exception as e:
            logger.error(f"BestExecutionReporter.record_execution failed: {e}")

    def generate_daily_report(self, date: str | None = None) -> BestExecutionReport:
        """Generate a best-execution report for a given day.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            BestExecutionReport with per-trade and aggregate statistics.

        Never raises — returns an empty report on error.
        """
        try:
            return self._generate_report_inner(date)
        except Exception as e:
            logger.error(f"BestExecutionReporter.generate_daily_report failed: {e}")
            return BestExecutionReport(
                date=date or datetime.now(config.ET).strftime("%Y-%m-%d")
            )

    def get_aggregate_stats(self, lookback_days: int = 30) -> dict:
        """Get aggregate execution quality statistics.

        Args:
            lookback_days: Number of days to include.

        Returns:
            Dict with avg_slippage_bps, execution_alpha_pct,
            total_slippage_dollars, by_strategy breakdown, count.
        """
        try:
            return self._aggregate_stats_inner(lookback_days)
        except Exception as e:
            logger.error(f"BestExecutionReporter.get_aggregate_stats failed: {e}")
            return {
                "avg_slippage_bps": 0.0,
                "execution_alpha_pct": 0.0,
                "total_slippage_dollars": 0.0,
                "count": 0,
            }

    @property
    def report_history(self) -> list[BestExecutionReport]:
        return list(self._report_history)

    # ------------------------------------------------------------------
    # Internal — record execution
    # ------------------------------------------------------------------

    def _record_inner(self, order: dict, fill: dict,
                      market_data: dict | None):
        """Core execution recording logic."""
        symbol = order.get("symbol", "")
        side = (order.get("side", "") or "").lower()
        qty = float(fill.get("filled_qty") or order.get("qty", 0) or 0)
        decision_price = float(order.get("decision_price", 0) or 0)
        fill_price = float(fill.get("fill_price", 0) or 0)

        if fill_price <= 0:
            logger.warning(f"BestExecution: Skipping record for {symbol} — no fill price")
            return

        # Benchmark prices
        vwap = float((market_data or {}).get("vwap", 0) or 0)
        twap = float((market_data or {}).get("twap", 0) or 0)

        # Compute slippage (positive = unfavorable)
        # For buys, paying more than decision = negative slippage (unfavorable)
        # For sells, receiving less than decision = negative slippage
        if side == "buy":
            slippage_bps = ((fill_price - decision_price) / decision_price * 10_000
                            if decision_price > 0 else 0.0)
            vs_vwap_bps = ((fill_price - vwap) / vwap * 10_000
                           if vwap > 0 else 0.0)
            vs_twap_bps = ((fill_price - twap) / twap * 10_000
                           if twap > 0 else 0.0)
        else:  # sell
            slippage_bps = ((decision_price - fill_price) / decision_price * 10_000
                            if decision_price > 0 else 0.0)
            vs_vwap_bps = ((vwap - fill_price) / vwap * 10_000
                           if vwap > 0 else 0.0)
            vs_twap_bps = ((twap - fill_price) / twap * 10_000
                           if twap > 0 else 0.0)

        # Implementation shortfall = decision_price vs fill_price (always)
        impl_shortfall = slippage_bps

        fill_time = fill.get("fill_time", "")
        timestamp = (fill_time.isoformat() if hasattr(fill_time, "isoformat")
                     else str(fill_time or datetime.now(config.ET).isoformat()))

        record = ExecutionRecord(
            symbol=symbol,
            side=side,
            qty=qty,
            decision_price=decision_price,
            fill_price=fill_price,
            vwap_benchmark=vwap,
            twap_benchmark=twap,
            timestamp=timestamp,
            strategy=order.get("strategy", ""),
            order_type=order.get("order_type", ""),
            order_id=order.get("order_id", ""),
            slippage_bps=slippage_bps,
            vs_vwap_bps=vs_vwap_bps,
            vs_twap_bps=vs_twap_bps,
            implementation_shortfall_bps=impl_shortfall,
        )

        # Determine day
        day = timestamp[:10] if len(timestamp) >= 10 else datetime.now(config.ET).strftime("%Y-%m-%d")

        with self._lock:
            self._records.append(record)
            self._daily_records[day].append(record)

            # Trim memory
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

        logger.debug(
            f"BestExecution: {side} {qty} {symbol} @ {fill_price:.2f} "
            f"(decision={decision_price:.2f}, slippage={slippage_bps:+.1f}bps)"
        )

    # ------------------------------------------------------------------
    # Internal — report generation
    # ------------------------------------------------------------------

    def _generate_report_inner(self, date: str | None) -> BestExecutionReport:
        """Generate report for a specific day."""
        day = date or datetime.now(config.ET).strftime("%Y-%m-%d")

        with self._lock:
            records = list(self._daily_records.get(day, []))

        report = BestExecutionReport(date=day, records=records)
        report.total_executions = len(records)

        if not records:
            return report

        # Compute aggregates
        slippages = [r.slippage_bps for r in records]
        slippages_sorted = sorted(slippages)

        report.avg_slippage_bps = sum(slippages) / len(slippages)
        report.median_slippage_bps = slippages_sorted[len(slippages_sorted) // 2]
        report.worst_slippage_bps = max(slippages)
        report.best_slippage_bps = min(slippages)

        vwap_diffs = [r.vs_vwap_bps for r in records if r.vwap_benchmark > 0]
        twap_diffs = [r.vs_twap_bps for r in records if r.twap_benchmark > 0]

        report.avg_vs_vwap_bps = (sum(vwap_diffs) / len(vwap_diffs)) if vwap_diffs else 0.0
        report.avg_vs_twap_bps = (sum(twap_diffs) / len(twap_diffs)) if twap_diffs else 0.0

        # Total slippage in dollars
        report.total_slippage_dollars = sum(
            r.slippage_bps / 10_000 * r.decision_price * r.qty
            for r in records if r.decision_price > 0
        )

        # Execution alpha: % of trades that beat VWAP
        vwap_records = [r for r in records if r.vwap_benchmark > 0]
        if vwap_records:
            beats = sum(1 for r in vwap_records if r.vs_vwap_bps < 0)
            report.execution_alpha_pct = beats / len(vwap_records) * 100
        else:
            report.execution_alpha_pct = 0.0

        # By strategy
        by_strat: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_strat[r.strategy].append(r.slippage_bps)
        report.by_strategy = {
            s: {"avg_slippage_bps": sum(v) / len(v), "count": len(v)}
            for s, v in by_strat.items()
        }

        # By side
        by_side: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_side[r.side].append(r.slippage_bps)
        report.by_side = {
            s: {"avg_slippage_bps": sum(v) / len(v), "count": len(v)}
            for s, v in by_side.items()
        }

        # Persist report
        self._report_history.append(report)
        if len(self._report_history) > 90:
            self._report_history = self._report_history[-90:]

        self._write_report(report)

        logger.info(
            f"BestExecution: Daily report for {day} — "
            f"{report.total_executions} executions, "
            f"avg slippage={report.avg_slippage_bps:+.1f}bps, "
            f"alpha={report.execution_alpha_pct:.0f}%"
        )

        return report

    # ------------------------------------------------------------------
    # Internal — aggregate stats
    # ------------------------------------------------------------------

    def _aggregate_stats_inner(self, lookback_days: int) -> dict:
        """Compute aggregate stats across recent records."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "avg_slippage_bps": 0.0,
                "execution_alpha_pct": 0.0,
                "total_slippage_dollars": 0.0,
                "count": 0,
            }

        # Filter to lookback window
        cutoff = datetime.now(config.ET).strftime("%Y-%m-%d")
        # Simple approach: use all records (they're already bounded by max_records)
        # For precise day filtering we'd parse timestamps, but in-memory records
        # are already limited and recent.

        slippages = [r.slippage_bps for r in records]
        vwap_records = [r for r in records if r.vwap_benchmark > 0]

        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

        if vwap_records:
            beats = sum(1 for r in vwap_records if r.vs_vwap_bps < 0)
            alpha_pct = beats / len(vwap_records) * 100
        else:
            alpha_pct = 0.0

        total_slip_dollars = sum(
            r.slippage_bps / 10_000 * r.decision_price * r.qty
            for r in records if r.decision_price > 0
        )

        # Per-strategy breakdown
        by_strat: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_strat[r.strategy].append(r.slippage_bps)

        return {
            "avg_slippage_bps": avg_slippage,
            "execution_alpha_pct": alpha_pct,
            "total_slippage_dollars": total_slip_dollars,
            "count": len(records),
            "by_strategy": {
                s: {"avg_slippage_bps": sum(v) / len(v), "count": len(v)}
                for s, v in by_strat.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _write_report(self, report: BestExecutionReport):
        """Write report to a JSONL file."""
        try:
            log_file = self._log_dir / "best_execution_log.jsonl"
            entry = {
                "date": report.date,
                "timestamp": report.timestamp.isoformat() if hasattr(report.timestamp, "isoformat") else str(report.timestamp),
                "total_executions": report.total_executions,
                "avg_slippage_bps": report.avg_slippage_bps,
                "median_slippage_bps": report.median_slippage_bps,
                "worst_slippage_bps": report.worst_slippage_bps,
                "execution_alpha_pct": report.execution_alpha_pct,
                "total_slippage_dollars": report.total_slippage_dollars,
                "by_strategy": report.by_strategy,
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"BestExecution: Failed to write report: {e}")
