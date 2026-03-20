"""EXEC-004: Fill Quality Analytics — post-trade execution analysis.

Tracks per-fill execution quality metrics:
- Implementation Shortfall (IS): decision price vs fill price
- VWAP benchmark: fill vs interval VWAP
- Arrival price benchmark: fill vs price at order submission
- Spread capture: how much spread we captured (limit orders)
- Latency: time from signal to fill

Provides daily and per-strategy summaries for continuous improvement.
"""

import logging
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """A single fill with all benchmark data."""
    # Identity
    fill_id: str = ""
    oms_id: str = ""
    symbol: str = ""
    strategy: str = ""
    side: str = ""  # "buy" or "sell"

    # Prices
    fill_price: float = 0.0
    fill_qty: int = 0
    decision_price: float = 0.0     # Price when signal was generated
    arrival_price: float = 0.0      # Price when order was submitted
    interval_vwap: float = 0.0      # VWAP over the execution interval
    bid_at_fill: float = 0.0        # Bid at time of fill
    ask_at_fill: float = 0.0        # Ask at time of fill
    mid_at_fill: float = 0.0        # Mid at time of fill

    # Timing
    signal_time: datetime | None = None
    submit_time: datetime | None = None
    fill_time: datetime | None = None

    # Computed metrics (populated by record_fill)
    impl_shortfall_bps: float = 0.0
    vwap_slippage_bps: float = 0.0
    arrival_slippage_bps: float = 0.0
    spread_capture_pct: float = 0.0  # % of spread captured (1.0 = full spread)
    latency_ms: float = 0.0          # Signal to fill latency


@dataclass
class ExecutionReport:
    """Aggregated execution quality report."""
    period: str = ""  # "daily", "weekly", etc.
    start_date: str = ""
    end_date: str = ""
    total_fills: int = 0
    total_notional: float = 0.0

    # Aggregate metrics (in bps)
    avg_impl_shortfall_bps: float = 0.0
    median_impl_shortfall_bps: float = 0.0
    avg_vwap_slippage_bps: float = 0.0
    avg_arrival_slippage_bps: float = 0.0
    avg_spread_capture_pct: float = 0.0
    avg_latency_ms: float = 0.0

    # Percentiles
    p25_impl_shortfall_bps: float = 0.0
    p75_impl_shortfall_bps: float = 0.0
    p95_impl_shortfall_bps: float = 0.0

    # Cost summary
    total_execution_cost_bps: float = 0.0  # Total IS across all fills
    total_execution_cost_usd: float = 0.0

    # By strategy breakdown
    by_strategy: dict = field(default_factory=dict)

    # Trend (improvement over time)
    trend_direction: str = ""   # "improving", "stable", "degrading"
    trend_change_bps: float = 0.0


class FillAnalytics:
    """Fill quality analytics engine.

    Records fills, computes execution quality metrics, and generates
    reports for monitoring execution performance over time.

    Usage:
        analytics = FillAnalytics()
        analytics.record_fill(
            oms_id="abc123", symbol="AAPL", strategy="STAT_MR",
            side="buy", fill_price=175.50, fill_qty=100,
            decision_price=175.45, arrival_price=175.48,
            interval_vwap=175.47,
            bid_at_fill=175.49, ask_at_fill=175.51,
            signal_time=t0, submit_time=t1, fill_time=t2,
        )
        report = analytics.get_daily_summary()
    """

    def __init__(self):
        self._fills: list[FillRecord] = []
        self._max_fills = 50_000
        self._daily_fills: dict[str, list[FillRecord]] = defaultdict(list)
        self._strategy_fills: dict[str, list[FillRecord]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_fill(
        self,
        oms_id: str,
        symbol: str,
        strategy: str,
        side: str,
        fill_price: float,
        fill_qty: int,
        decision_price: float,
        arrival_price: float,
        interval_vwap: float = 0.0,
        bid_at_fill: float = 0.0,
        ask_at_fill: float = 0.0,
        signal_time: datetime | None = None,
        submit_time: datetime | None = None,
        fill_time: datetime | None = None,
    ) -> FillRecord:
        """Record a fill and compute execution quality metrics.

        Args:
            oms_id: OMS order ID.
            symbol: Ticker symbol.
            strategy: Strategy that generated the signal.
            side: "buy" or "sell".
            fill_price: Actual fill price.
            fill_qty: Number of shares filled.
            decision_price: Price when signal was generated.
            arrival_price: Price when order was submitted.
            interval_vwap: VWAP over the execution interval.
            bid_at_fill: Bid price at time of fill.
            ask_at_fill: Ask price at time of fill.
            signal_time: When the signal was generated.
            submit_time: When the order was submitted.
            fill_time: When the fill was received.

        Returns:
            FillRecord with computed metrics.
        """
        if fill_time is None:
            fill_time = datetime.now()

        mid_at_fill = (bid_at_fill + ask_at_fill) / 2 if bid_at_fill > 0 and ask_at_fill > 0 else fill_price
        spread_at_fill = ask_at_fill - bid_at_fill if ask_at_fill > bid_at_fill else 0

        # Side multiplier: buys pay more -> positive IS, sells receive less -> positive IS
        side_mult = 1.0 if side == "buy" else -1.0

        # Implementation Shortfall: difference between decision price and fill price
        impl_shortfall_bps = 0.0
        if decision_price > 0:
            impl_shortfall_bps = side_mult * (fill_price - decision_price) / decision_price * 10_000

        # VWAP slippage: fill vs interval VWAP
        vwap_slippage_bps = 0.0
        if interval_vwap > 0:
            vwap_slippage_bps = side_mult * (fill_price - interval_vwap) / interval_vwap * 10_000

        # Arrival price slippage
        arrival_slippage_bps = 0.0
        if arrival_price > 0:
            arrival_slippage_bps = side_mult * (fill_price - arrival_price) / arrival_price * 10_000

        # Spread capture: how much of the bid-ask spread did we capture?
        # 1.0 = filled at our side of BBO, 0.0 = filled at opposite side
        spread_capture_pct = 0.0
        if spread_at_fill > 0:
            if side == "buy":
                # Best case: fill at bid. Worst case: fill at ask.
                spread_capture_pct = (ask_at_fill - fill_price) / spread_at_fill
            else:
                # Best case: fill at ask. Worst case: fill at bid.
                spread_capture_pct = (fill_price - bid_at_fill) / spread_at_fill
            spread_capture_pct = max(0.0, min(1.0, spread_capture_pct))

        # Latency
        latency_ms = 0.0
        if signal_time and fill_time:
            latency_ms = (fill_time - signal_time).total_seconds() * 1000

        record = FillRecord(
            fill_id=f"{oms_id}_{fill_time.strftime('%H%M%S')}",
            oms_id=oms_id,
            symbol=symbol,
            strategy=strategy,
            side=side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            decision_price=decision_price,
            arrival_price=arrival_price,
            interval_vwap=interval_vwap,
            bid_at_fill=bid_at_fill,
            ask_at_fill=ask_at_fill,
            mid_at_fill=mid_at_fill,
            signal_time=signal_time,
            submit_time=submit_time,
            fill_time=fill_time,
            impl_shortfall_bps=round(impl_shortfall_bps, 2),
            vwap_slippage_bps=round(vwap_slippage_bps, 2),
            arrival_slippage_bps=round(arrival_slippage_bps, 2),
            spread_capture_pct=round(spread_capture_pct, 4),
            latency_ms=round(latency_ms, 1),
        )

        # Store (thread-safe)
        with self._lock:
            self._fills.append(record)
            if len(self._fills) > self._max_fills:
                self._fills = self._fills[-self._max_fills:]

            day_key = fill_time.strftime("%Y-%m-%d")
            self._daily_fills[day_key].append(record)
            self._strategy_fills[strategy].append(record)

        logger.info(
            f"FillAnalytics [{symbol}]: IS={impl_shortfall_bps:+.1f}bps "
            f"VWAP={vwap_slippage_bps:+.1f}bps "
            f"SpreadCap={spread_capture_pct:.0%} "
            f"Latency={latency_ms:.0f}ms"
        )
        return record

    def get_daily_summary(self, target_date: date | None = None) -> ExecutionReport:
        """Generate daily execution quality report.

        Args:
            target_date: Date to report on (default: today).

        Returns:
            ExecutionReport with aggregated metrics.
        """
        if target_date is None:
            target_date = date.today()

        day_key = target_date.strftime("%Y-%m-%d")
        with self._lock:
            fills = list(self._daily_fills.get(day_key, []))

        if not fills:
            return ExecutionReport(
                period="daily",
                start_date=day_key,
                end_date=day_key,
                total_fills=0,
            )

        report = self._build_report(fills, "daily", day_key, day_key)

        # Add by-strategy breakdown
        strat_groups: dict[str, list[FillRecord]] = defaultdict(list)
        for f in fills:
            strat_groups[f.strategy].append(f)

        for strat, strat_fills in strat_groups.items():
            report.by_strategy[strat] = self._compute_aggregate_metrics(strat_fills)

        # Compute trend (compare to previous 5 trading days)
        report.trend_direction, report.trend_change_bps = self._compute_trend(target_date)

        return report

    def get_strategy_summary(self, strategy: str, lookback_days: int = 30) -> dict:
        """Get execution quality summary for a specific strategy.

        Args:
            strategy: Strategy name (e.g. "STAT_MR").
            lookback_days: Number of calendar days to look back.

        Returns:
            Dict with aggregate metrics and trend data.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        with self._lock:
            all_strat_fills = list(self._strategy_fills.get(strategy, []))
        fills = [
            f for f in all_strat_fills
            if f.fill_time and f.fill_time >= cutoff
        ]

        if not fills:
            return {
                "strategy": strategy,
                "period_days": lookback_days,
                "total_fills": 0,
                "message": "No fills in lookback period",
            }

        metrics = self._compute_aggregate_metrics(fills)
        metrics["strategy"] = strategy
        metrics["period_days"] = lookback_days

        # Weekly trend within the period
        weekly_is: list[float] = []
        for week_offset in range(0, lookback_days, 7):
            week_start = datetime.now() - timedelta(days=week_offset + 7)
            week_end = datetime.now() - timedelta(days=week_offset)
            week_fills = [
                f for f in fills
                if f.fill_time and week_start <= f.fill_time < week_end
            ]
            if week_fills:
                avg_is = statistics.mean(f.impl_shortfall_bps for f in week_fills)
                weekly_is.append(round(avg_is, 2))

        metrics["weekly_impl_shortfall_trend"] = weekly_is

        return metrics

    def get_fill_history(self, symbol: str = "", limit: int = 50) -> list[FillRecord]:
        """Get recent fill records, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (empty = all).
            limit: Maximum records to return.

        Returns:
            List of FillRecords, most recent first.
        """
        with self._lock:
            fills = list(self._fills)
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        return list(reversed(fills[-limit:]))

    def _build_report(
        self,
        fills: list[FillRecord],
        period: str,
        start: str,
        end: str,
    ) -> ExecutionReport:
        """Build an ExecutionReport from a list of fills."""
        metrics = self._compute_aggregate_metrics(fills)

        total_notional = sum(f.fill_price * f.fill_qty for f in fills)
        total_cost_usd = sum(
            f.impl_shortfall_bps / 10_000 * f.fill_price * f.fill_qty
            for f in fills
        )

        return ExecutionReport(
            period=period,
            start_date=start,
            end_date=end,
            total_fills=metrics["total_fills"],
            total_notional=round(total_notional, 2),
            avg_impl_shortfall_bps=metrics["avg_impl_shortfall_bps"],
            median_impl_shortfall_bps=metrics["median_impl_shortfall_bps"],
            avg_vwap_slippage_bps=metrics["avg_vwap_slippage_bps"],
            avg_arrival_slippage_bps=metrics["avg_arrival_slippage_bps"],
            avg_spread_capture_pct=metrics["avg_spread_capture_pct"],
            avg_latency_ms=metrics["avg_latency_ms"],
            p25_impl_shortfall_bps=metrics["p25_impl_shortfall_bps"],
            p75_impl_shortfall_bps=metrics["p75_impl_shortfall_bps"],
            p95_impl_shortfall_bps=metrics["p95_impl_shortfall_bps"],
            total_execution_cost_bps=metrics["avg_impl_shortfall_bps"],
            total_execution_cost_usd=round(total_cost_usd, 2),
        )

    def _compute_aggregate_metrics(self, fills: list[FillRecord]) -> dict:
        """Compute aggregate metrics from a list of fills."""
        if not fills:
            return {
                "total_fills": 0,
                "avg_impl_shortfall_bps": 0,
                "median_impl_shortfall_bps": 0,
                "avg_vwap_slippage_bps": 0,
                "avg_arrival_slippage_bps": 0,
                "avg_spread_capture_pct": 0,
                "avg_latency_ms": 0,
                "p25_impl_shortfall_bps": 0,
                "p75_impl_shortfall_bps": 0,
                "p95_impl_shortfall_bps": 0,
            }

        is_values = [f.impl_shortfall_bps for f in fills]
        is_sorted = sorted(is_values)
        n = len(is_sorted)

        return {
            "total_fills": n,
            "avg_impl_shortfall_bps": round(statistics.mean(is_values), 2),
            "median_impl_shortfall_bps": round(statistics.median(is_values), 2),
            "avg_vwap_slippage_bps": round(
                statistics.mean(f.vwap_slippage_bps for f in fills), 2
            ),
            "avg_arrival_slippage_bps": round(
                statistics.mean(f.arrival_slippage_bps for f in fills), 2
            ),
            "avg_spread_capture_pct": round(
                statistics.mean(f.spread_capture_pct for f in fills), 4
            ),
            "avg_latency_ms": round(
                statistics.mean(f.latency_ms for f in fills), 1
            ),
            "p25_impl_shortfall_bps": round(
                is_sorted[max(0, int(n * 0.25) - 1)], 2
            ),
            "p75_impl_shortfall_bps": round(
                is_sorted[min(n - 1, int(n * 0.75))], 2
            ),
            "p95_impl_shortfall_bps": round(
                is_sorted[min(n - 1, int(n * 0.95))], 2
            ),
        }

    def _compute_trend(self, target_date: date) -> tuple[str, float]:
        """Compute execution quality trend vs prior period.

        Returns:
            (direction, change_bps) where direction is "improving", "stable", or "degrading".
        """
        # Current day IS
        day_key = target_date.strftime("%Y-%m-%d")
        with self._lock:
            current_fills = list(self._daily_fills.get(day_key, []))
        if not current_fills:
            return "stable", 0.0

        current_is = statistics.mean(f.impl_shortfall_bps for f in current_fills)

        # Previous 5 trading days
        prior_is_values: list[float] = []
        for offset in range(1, 8):
            prior_date = target_date - timedelta(days=offset)
            prior_key = prior_date.strftime("%Y-%m-%d")
            with self._lock:
                prior_fills = list(self._daily_fills.get(prior_key, []))
            if prior_fills:
                prior_is_values.append(
                    statistics.mean(f.impl_shortfall_bps for f in prior_fills)
                )

        if not prior_is_values:
            return "stable", 0.0

        prior_avg = statistics.mean(prior_is_values)
        change = current_is - prior_avg

        if change < -0.5:
            return "improving", round(change, 2)
        elif change > 0.5:
            return "degrading", round(change, 2)
        return "stable", round(change, 2)

    @property
    def stats(self) -> dict:
        """Overall analytics statistics."""
        with self._lock:
            return {
                "total_fills_recorded": len(self._fills),
                "days_tracked": len(self._daily_fills),
                "strategies_tracked": list(self._strategy_fills.keys()),
            }
