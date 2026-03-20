"""BACKTEST-004: Performance Attribution Analysis.

Decomposes P&L into strategy-level, factor-level, timing-level,
and execution-level contributions to understand where returns come from.

Usage:
    pa = PerformanceAttribution()
    report = pa.compute_attribution(trades, market_data)
    print(report.strategy_attribution)
    print(report.timing_attribution)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Data classes
# ===================================================================== #

@dataclass
class StrategyAttribution:
    """P&L attribution for a single strategy."""
    strategy: str
    total_pnl: float = 0.0
    pnl_pct: float = 0.0           # As % of total portfolio P&L
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    sharpe: float = 0.0

    # Temporal breakdown
    daily_pnl: dict[str, float] = field(default_factory=dict)
    weekly_pnl: dict[str, float] = field(default_factory=dict)
    monthly_pnl: dict[str, float] = field(default_factory=dict)


@dataclass
class FactorAttribution:
    """P&L attribution by market factor."""
    market_beta_pnl: float = 0.0       # P&L explained by market exposure
    sector_pnl: dict[str, float] = field(default_factory=dict)
    momentum_pnl: float = 0.0          # P&L from momentum factor
    mean_reversion_pnl: float = 0.0    # P&L from mean reversion factor
    residual_alpha: float = 0.0        # Unexplained alpha


@dataclass
class TimingAttribution:
    """P&L attribution from entry/exit timing and position sizing."""
    entry_timing_pnl: float = 0.0      # P&L from entry timing
    exit_timing_pnl: float = 0.0       # P&L from exit timing
    sizing_contribution: float = 0.0    # P&L from position sizing decisions
    hold_time_contribution: float = 0.0 # Effect of hold time on returns


@dataclass
class ExecutionAttribution:
    """P&L attribution from execution quality."""
    total_slippage_cost: float = 0.0
    total_spread_cost: float = 0.0
    total_market_impact: float = 0.0
    total_commission: float = 0.0
    execution_shortfall: float = 0.0     # Total execution cost


@dataclass
class BrinsonFachlerResult:
    """Brinson-Fachler sector decomposition of active return.

    Breaks down the portfolio's active return (vs equal-weight benchmark)
    into allocation effect, selection effect, and interaction effect.
    """
    allocation_effect: float = 0.0      # Over-weighting outperforming sectors
    selection_effect: float = 0.0       # Picking winners within sectors
    interaction_effect: float = 0.0     # Combined allocation + selection
    total_active_return: float = 0.0
    sector_details: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class AttributionReport:
    """Complete performance attribution report."""
    # Period
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_pnl: float = 0.0
    total_trades: int = 0

    # Breakdowns
    strategy_attribution: dict[str, StrategyAttribution] = field(default_factory=dict)
    factor_attribution: Optional[FactorAttribution] = None
    timing_attribution: Optional[TimingAttribution] = None
    execution_attribution: Optional[ExecutionAttribution] = None
    brinson_fachler: Optional[BrinsonFachlerResult] = None

    # Time-of-day analysis
    hour_of_day_pnl: dict[int, float] = field(default_factory=dict)
    day_of_week_pnl: dict[str, float] = field(default_factory=dict)


_DEFAULT_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ADBE": "Technology", "AVGO": "Technology", "CSCO": "Technology",
    "ORCL": "Technology", "QCOM": "Technology", "TXN": "Technology",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MCD": "Consumer",
    "DIS": "Consumer", "NFLX": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "TGT": "Consumer", "LOW": "Consumer",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "V": "Financials",
    "MA": "Financials", "AXP": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "BMY": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "LMT": "Industrials",
    # ETFs
    "SPY": "Broad Market", "QQQ": "Technology", "IWM": "Small Cap",
    "DIA": "Broad Market", "XLF": "Financials", "XLE": "Energy",
    "XLK": "Technology", "XLV": "Healthcare",
}


class PerformanceAttribution:
    """Multi-level performance attribution engine.

    Decomposes trading P&L into:
    1. Strategy level: which strategies contributed most
    2. Factor level: market beta, sector, momentum, mean reversion
    3. Timing level: entry timing, exit timing, position sizing
    4. Execution level: slippage, spread, market impact, commission
    5. Brinson-Fachler: sector allocation vs stock selection

    Args:
        risk_free_rate: Annual risk-free rate for Sharpe computation.
        sector_map: Symbol-to-sector mapping. Uses built-in default if None.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        sector_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._rf = risk_free_rate
        self._sector_map = sector_map or _DEFAULT_SECTOR_MAP

    def compute_attribution(
        self,
        trades: list[dict],
        market_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> AttributionReport:
        """Compute full performance attribution.

        Args:
            trades: List of trade dicts. Required keys:
                - pnl (float): Trade P&L
                - strategy (str): Strategy name
                - symbol (str): Ticker
                - entry_time (datetime): Entry timestamp
                - exit_time (datetime): Exit timestamp
                - entry_price (float): Entry price
                - exit_price (float): Exit price
                - qty (int): Position size
                Optional keys:
                - side (str): "buy" or "sell"
                - commission (float): Commission cost
                - slippage (float): Slippage cost
            market_data: Dict of symbol -> DataFrame with OHLCV.
                Required for factor attribution. Optional.

        Returns:
            AttributionReport with all attribution levels.
        """
        if not trades:
            logger.warning("No trades for attribution analysis")
            return AttributionReport()

        logger.info("Computing attribution for %d trades", len(trades))

        report = AttributionReport(
            total_pnl=sum(t.get("pnl", 0.0) for t in trades),
            total_trades=len(trades),
        )

        # Extract date range
        entry_times = [t["entry_time"] for t in trades if "entry_time" in t]
        if entry_times:
            report.start_date = min(entry_times)
            report.end_date = max(entry_times)

        # 1. Strategy-level attribution
        report.strategy_attribution = self._strategy_attribution(trades)

        # 2. Factor-level attribution
        if market_data:
            report.factor_attribution = self._factor_attribution(trades, market_data)
        else:
            report.factor_attribution = FactorAttribution()

        # 3. Timing-level attribution
        report.timing_attribution = self._timing_attribution(trades, market_data)

        # 4. Execution-level attribution
        report.execution_attribution = self._execution_attribution(trades)

        # 5. Brinson-Fachler decomposition (requires market data)
        if market_data:
            report.brinson_fachler = self._brinson_fachler(trades, market_data)

        # 6. Time-of-day and day-of-week analysis
        report.hour_of_day_pnl = self._hour_of_day_analysis(trades)
        report.day_of_week_pnl = self._day_of_week_analysis(trades)

        logger.info(
            "Attribution complete: total P&L=$%.2f across %d strategies",
            report.total_pnl, len(report.strategy_attribution),
        )

        return report

    # ------------------------------------------------------------------ #
    #  Strategy-level attribution
    # ------------------------------------------------------------------ #

    def _strategy_attribution(
        self, trades: list[dict],
    ) -> dict[str, StrategyAttribution]:
        """Break down P&L by strategy with temporal detail."""
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            strategy = t.get("strategy", "UNKNOWN")
            by_strategy[strategy].append(t)

        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        result = {}

        for strategy, strat_trades in by_strategy.items():
            pnls = [t.get("pnl", 0.0) for t in strat_trades]
            winners = [p for p in pnls if p > 0]

            attr = StrategyAttribution(
                strategy=strategy,
                total_pnl=sum(pnls),
                pnl_pct=sum(pnls) / total_pnl if total_pnl != 0 else 0.0,
                trade_count=len(strat_trades),
                win_rate=len(winners) / len(pnls) if pnls else 0.0,
                avg_pnl_per_trade=float(np.mean(pnls)) if pnls else 0.0,
            )

            # Sharpe
            if len(pnls) > 5:
                arr = np.array(pnls)
                std = float(np.std(arr, ddof=1))
                if std > 0:
                    attr.sharpe = float(np.mean(arr) / std * np.sqrt(252))

            # Temporal breakdown
            for t in strat_trades:
                entry_time = t.get("entry_time")
                if entry_time is None:
                    continue
                pnl = t.get("pnl", 0.0)

                day_key = entry_time.strftime("%Y-%m-%d")
                attr.daily_pnl[day_key] = attr.daily_pnl.get(day_key, 0.0) + pnl

                week_key = entry_time.strftime("%Y-W%W")
                attr.weekly_pnl[week_key] = attr.weekly_pnl.get(week_key, 0.0) + pnl

                month_key = entry_time.strftime("%Y-%m")
                attr.monthly_pnl[month_key] = attr.monthly_pnl.get(month_key, 0.0) + pnl

            result[strategy] = attr

        return result

    # ------------------------------------------------------------------ #
    #  Factor-level attribution
    # ------------------------------------------------------------------ #

    def _factor_attribution(
        self,
        trades: list[dict],
        market_data: dict[str, pd.DataFrame],
    ) -> FactorAttribution:
        """Decompose P&L by market factors: beta, sector, momentum, mean reversion."""
        result = FactorAttribution()

        # Get market (SPY) returns as benchmark
        spy_data = market_data.get("SPY")
        if spy_data is None:
            logger.debug("No SPY data for factor attribution; using trade-level estimates")
            return self._estimate_factor_attribution(trades)

        spy_returns = spy_data["close"].pct_change().dropna()

        total_beta_pnl = 0.0
        sector_pnl: dict[str, float] = defaultdict(float)
        momentum_pnl = 0.0
        mr_pnl = 0.0

        for t in trades:
            pnl = t.get("pnl", 0.0)
            strategy = t.get("strategy", "")
            symbol = t.get("symbol", "")
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")

            if entry_time is None or exit_time is None:
                continue

            # Market beta component: approximate by SPY return during hold
            try:
                spy_subset = spy_returns.loc[
                    (spy_returns.index >= entry_time) &
                    (spy_returns.index <= exit_time)
                ]
                market_return = float(spy_subset.sum()) if len(spy_subset) > 0 else 0.0
            except (KeyError, TypeError):
                market_return = 0.0

            entry_price = t.get("entry_price", 0.0)
            qty = t.get("qty", 0)
            notional = entry_price * qty if entry_price and qty else abs(pnl) * 10

            # Beta contribution (assume beta ~1 for simplicity)
            beta_contribution = market_return * notional
            side = t.get("side", "buy")
            if side == "sell":
                beta_contribution = -beta_contribution
            total_beta_pnl += beta_contribution

            # Sector (from config SECTOR_MAP)
            try:
                import config as cfg
                sector = cfg.SECTOR_MAP.get(symbol, "OTHER")
                sector_pnl[sector] += pnl
            except ImportError:
                sector_pnl["UNKNOWN"] += pnl

            # Momentum vs. mean reversion classification
            if strategy in ("ORB", "MICRO_MOM", "PEAD"):
                momentum_pnl += pnl
            elif strategy in ("STAT_MR", "VWAP", "KALMAN_PAIRS"):
                mr_pnl += pnl

        result.market_beta_pnl = total_beta_pnl
        result.sector_pnl = dict(sector_pnl)
        result.momentum_pnl = momentum_pnl
        result.mean_reversion_pnl = mr_pnl
        result.residual_alpha = sum(t.get("pnl", 0.0) for t in trades) - total_beta_pnl

        return result

    def _estimate_factor_attribution(self, trades: list[dict]) -> FactorAttribution:
        """Estimate factor attribution without market data."""
        result = FactorAttribution()

        momentum_pnl = 0.0
        mr_pnl = 0.0

        for t in trades:
            pnl = t.get("pnl", 0.0)
            strategy = t.get("strategy", "")

            if strategy in ("ORB", "MICRO_MOM", "PEAD"):
                momentum_pnl += pnl
            elif strategy in ("STAT_MR", "VWAP", "KALMAN_PAIRS"):
                mr_pnl += pnl

        result.momentum_pnl = momentum_pnl
        result.mean_reversion_pnl = mr_pnl
        result.residual_alpha = sum(t.get("pnl", 0.0) for t in trades)

        return result

    # ------------------------------------------------------------------ #
    #  Timing-level attribution
    # ------------------------------------------------------------------ #

    def _timing_attribution(
        self,
        trades: list[dict],
        market_data: Optional[dict[str, pd.DataFrame]],
    ) -> TimingAttribution:
        """Analyze entry timing, exit timing, and sizing contributions."""
        result = TimingAttribution()

        if not market_data:
            return result

        entry_timing_values = []
        exit_timing_values = []
        sizing_values = []

        for t in trades:
            symbol = t.get("symbol", "")
            entry_price = t.get("entry_price", 0.0)
            exit_price = t.get("exit_price", 0.0)
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")
            qty = t.get("qty", 0)
            side = t.get("side", "buy")

            if not all([symbol, entry_price, exit_price, entry_time, exit_time]):
                continue

            if symbol not in market_data:
                continue

            df = market_data[symbol]

            # Entry timing: compare entry price vs bar's OHLC range
            try:
                entry_bar = df.loc[df.index.asof(entry_time)]
                if isinstance(entry_bar, pd.DataFrame):
                    entry_bar = entry_bar.iloc[0]

                bar_range = entry_bar["high"] - entry_bar["low"]
                if bar_range > 0:
                    if side == "buy":
                        # Lower entry = better timing for longs
                        position_in_range = (entry_price - entry_bar["low"]) / bar_range
                        entry_timing = 1.0 - position_in_range  # 1.0 = best, 0.0 = worst
                    else:
                        position_in_range = (entry_bar["high"] - entry_price) / bar_range
                        entry_timing = 1.0 - position_in_range
                    entry_timing_values.append(entry_timing)
            except (KeyError, IndexError, TypeError):
                pass

            # Exit timing: compare exit price vs bar's range
            try:
                exit_bar = df.loc[df.index.asof(exit_time)]
                if isinstance(exit_bar, pd.DataFrame):
                    exit_bar = exit_bar.iloc[0]

                bar_range = exit_bar["high"] - exit_bar["low"]
                if bar_range > 0:
                    if side == "buy":
                        position_in_range = (exit_price - exit_bar["low"]) / bar_range
                        exit_timing = position_in_range
                    else:
                        position_in_range = (exit_bar["high"] - exit_price) / bar_range
                        exit_timing = position_in_range
                    exit_timing_values.append(exit_timing)
            except (KeyError, IndexError, TypeError):
                pass

            # Sizing contribution: bigger sizes on winners, smaller on losers
            pnl = t.get("pnl", 0.0)
            if qty > 0 and entry_price > 0:
                pnl_per_share = pnl / qty
                sizing_values.append(pnl_per_share * qty)

        # Entry timing score: 0-1 where 1 = always entered at the best price
        if entry_timing_values:
            result.entry_timing_pnl = float(np.mean(entry_timing_values))

        if exit_timing_values:
            result.exit_timing_pnl = float(np.mean(exit_timing_values))

        if sizing_values:
            # Compare actual PnL to equal-weight PnL
            actual_total = sum(sizing_values)
            avg_pnl_per_share = np.mean([v / max(t.get("qty", 1), 1)
                                          for v, t in zip(sizing_values, trades)
                                          if t.get("qty", 0) > 0])
            equal_weight_total = avg_pnl_per_share * sum(
                t.get("qty", 0) for t in trades if t.get("qty", 0) > 0
            ) / max(len(sizing_values), 1)
            result.sizing_contribution = actual_total - equal_weight_total

        # Hold time contribution
        hold_times = []
        pnls_by_hold = []
        for t in trades:
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")
            if entry_time and exit_time:
                hold_min = (exit_time - entry_time).total_seconds() / 60.0
                hold_times.append(hold_min)
                pnls_by_hold.append(t.get("pnl", 0.0))

        if len(hold_times) > 5:
            # Correlation between hold time and P&L
            corr = float(np.corrcoef(hold_times, pnls_by_hold)[0, 1])
            result.hold_time_contribution = corr if not np.isnan(corr) else 0.0

        return result

    # ------------------------------------------------------------------ #
    #  Execution-level attribution
    # ------------------------------------------------------------------ #

    def _execution_attribution(self, trades: list[dict]) -> ExecutionAttribution:
        """Compute execution quality metrics."""
        result = ExecutionAttribution()

        for t in trades:
            result.total_commission += t.get("commission", 0.0)
            result.total_slippage_cost += t.get("slippage", 0.0)

            # Estimate spread cost from entry price vs midpoint if available
            entry_price = t.get("entry_price", 0.0)
            bid = t.get("bid_at_entry", 0.0)
            ask = t.get("ask_at_entry", 0.0)
            if bid > 0 and ask > 0:
                midpoint = (bid + ask) / 2
                spread_cost = abs(entry_price - midpoint)
                qty = t.get("qty", 0)
                result.total_spread_cost += spread_cost * qty

        result.execution_shortfall = (
            result.total_slippage_cost +
            result.total_spread_cost +
            result.total_market_impact +
            result.total_commission
        )

        return result

    # ------------------------------------------------------------------ #
    #  Time-of-day and day-of-week analysis
    # ------------------------------------------------------------------ #

    def _hour_of_day_analysis(self, trades: list[dict]) -> dict[int, float]:
        """P&L breakdown by hour of day (market hours)."""
        hour_pnl: dict[int, float] = defaultdict(float)
        for t in trades:
            entry_time = t.get("entry_time")
            if entry_time is not None:
                hour_pnl[entry_time.hour] += t.get("pnl", 0.0)
        return dict(sorted(hour_pnl.items()))

    def _day_of_week_analysis(self, trades: list[dict]) -> dict[str, float]:
        """P&L breakdown by day of week."""
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                     "Saturday", "Sunday"]
        day_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            entry_time = t.get("entry_time")
            if entry_time is not None:
                day_name = day_names[entry_time.weekday()]
                day_pnl[day_name] += t.get("pnl", 0.0)
        return dict(day_pnl)

    # ------------------------------------------------------------------ #
    #  Brinson-Fachler decomposition
    # ------------------------------------------------------------------ #

    def _brinson_fachler(
        self,
        trades: list[dict],
        market_data: dict[str, pd.DataFrame],
    ) -> BrinsonFachlerResult:
        """Brinson-Fachler decomposition of active return.

        Decomposes the portfolio's active return (vs an equal-weight sector
        benchmark) into:
        - Allocation effect: over-weighting sectors that outperform
        - Selection effect: picking winners within each sector
        - Interaction effect: combined allocation + selection

        The benchmark is constructed as equal-weight across all sectors
        present in the portfolio, using each sector's average market return
        from market_data.
        """
        if not trades:
            return BrinsonFachlerResult()

        # --- Portfolio sector weights and returns ---
        sector_notional: dict[str, float] = defaultdict(float)
        sector_pnl: dict[str, float] = defaultdict(float)

        for t in trades:
            symbol = t.get("symbol", "")
            base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            sector = self._sector_map.get(base_symbol, "Other")
            notional = abs(t.get("entry_price", 0.0) * t.get("qty", 1))
            sector_notional[sector] += notional
            sector_pnl[sector] += t.get("pnl", 0.0)

        total_notional = sum(sector_notional.values())
        if total_notional <= 0:
            return BrinsonFachlerResult()

        all_sectors = sorted(set(sector_notional.keys()))
        n_sectors = len(all_sectors)
        if n_sectors == 0:
            return BrinsonFachlerResult()

        # Portfolio weights and returns per sector
        port_weights: dict[str, float] = {}
        port_returns: dict[str, float] = {}
        for sector in all_sectors:
            port_weights[sector] = sector_notional[sector] / total_notional
            port_returns[sector] = (
                sector_pnl[sector] / sector_notional[sector]
                if sector_notional[sector] > 0 else 0.0
            )

        # --- Benchmark: equal-weight sector returns from market data ---
        bench_weights: dict[str, float] = {}
        bench_returns: dict[str, float] = {}
        equal_weight = 1.0 / n_sectors

        for sector in all_sectors:
            bench_weights[sector] = equal_weight

            # Average return of symbols in this sector from market_data
            sector_syms = [
                sym for sym, sec in self._sector_map.items()
                if sec == sector and sym in market_data
            ]
            if sector_syms:
                sector_rets = []
                for sym in sector_syms:
                    df = market_data[sym]
                    df_lower = {c.lower(): c for c in df.columns}
                    close_col = df_lower.get("close")
                    if close_col is None:
                        continue
                    closes = df[close_col]
                    if len(closes) >= 2:
                        ret = (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0])
                        sector_rets.append(ret)
                bench_returns[sector] = float(np.mean(sector_rets)) if sector_rets else 0.0
            else:
                bench_returns[sector] = 0.0

        # --- Decomposition ---
        total_bench_return = sum(
            bench_weights[s] * bench_returns[s] for s in all_sectors
        )

        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        sector_details: dict[str, dict[str, float]] = {}

        for sector in all_sectors:
            wp = port_weights[sector]
            wb = bench_weights[sector]
            rp = port_returns[sector]
            rb = bench_returns.get(sector, 0.0)

            alloc = (wp - wb) * (rb - total_bench_return)
            select = wb * (rp - rb)
            interact = (wp - wb) * (rp - rb)

            allocation_effect += alloc
            selection_effect += select
            interaction_effect += interact

            sector_details[sector] = {
                "portfolio_weight": wp,
                "benchmark_weight": wb,
                "portfolio_return": rp,
                "benchmark_return": rb,
                "allocation": alloc,
                "selection": select,
                "interaction": interact,
            }

        total_active = allocation_effect + selection_effect + interaction_effect

        return BrinsonFachlerResult(
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_active_return=total_active,
            sector_details=sector_details,
        )

    def __repr__(self) -> str:
        return f"PerformanceAttribution(rf={self._rf:.3f})"
