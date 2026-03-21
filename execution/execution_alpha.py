"""COMP-017: Execution alpha decomposition — separate alpha P&L from execution P&L.

Tracks what portion of returns comes from signal alpha versus execution
quality.  This is critical for understanding whether a strategy's edge is
being eroded by poor execution or enhanced by smart routing.

Key concepts:
    - **Alpha P&L**: profit attributable to the signal's predictive power,
      measured as the return between signal generation and a theoretical
      benchmark execution (e.g. VWAP, arrival price).
    - **Execution P&L**: difference between the theoretical benchmark and
      the actual fill price — captures slippage, market impact, and timing.
    - **Implementation shortfall**: total cost of implementing a trading
      decision, decomposed into delay cost, market impact, and opportunity cost.

Usage:
    analyzer = ExecutionAlphaAnalyzer()
    alpha, execution = analyzer.decompose_pnl(trades)
    shortfall = analyzer.compute_implementation_shortfall(signal, fill)
    report = analyzer.generate_report(trades)

Dependencies: numpy, pandas (required).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Record of a single trade for P&L decomposition.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    side : str
        ``"buy"`` or ``"sell"``.
    quantity : float
        Number of shares.
    signal_price : float
        Price at the time the signal was generated (decision price).
    arrival_price : float
        Price when the order hit the market (arrival price).
    fill_price : float
        Actual average fill price.
    benchmark_price : float
        Benchmark execution price (e.g. VWAP, TWAP, close).
    signal_timestamp : Optional[str]
        When the signal was generated.
    fill_timestamp : Optional[str]
        When the fill was completed.
    """

    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0
    signal_price: float = 0.0
    arrival_price: float = 0.0
    fill_price: float = 0.0
    benchmark_price: float = 0.0
    signal_timestamp: Optional[str] = None
    fill_timestamp: Optional[str] = None


@dataclass
class PnLDecomposition:
    """Result of decomposing a trade's P&L into alpha and execution."""

    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0

    # P&L components (in dollars)
    total_pnl: float = 0.0
    alpha_pnl: float = 0.0
    execution_pnl: float = 0.0

    # Per-share costs
    alpha_per_share: float = 0.0
    execution_per_share: float = 0.0

    # Basis points
    alpha_bps: float = 0.0
    execution_bps: float = 0.0

    # Implementation shortfall components
    delay_cost: float = 0.0
    market_impact: float = 0.0
    opportunity_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "total_pnl": self.total_pnl,
            "alpha_pnl": self.alpha_pnl,
            "execution_pnl": self.execution_pnl,
            "alpha_per_share": self.alpha_per_share,
            "execution_per_share": self.execution_per_share,
            "alpha_bps": self.alpha_bps,
            "execution_bps": self.execution_bps,
            "delay_cost": self.delay_cost,
            "market_impact": self.market_impact,
            "opportunity_cost": self.opportunity_cost,
        }


# ---------------------------------------------------------------------------
# Core decomposition logic
# ---------------------------------------------------------------------------


def decompose_pnl(
    trades: List[TradeRecord],
) -> Tuple[float, float]:
    """Decompose total P&L into alpha and execution components.

    Alpha P&L measures how much profit came from the signal's predictive
    power (signal_price -> benchmark_price).  Execution P&L measures how
    much was gained or lost from execution quality (benchmark_price ->
    fill_price).

    Parameters
    ----------
    trades : list of TradeRecord
        List of trade records to decompose.

    Returns
    -------
    Tuple[float, float]
        (total_alpha_pnl, total_execution_pnl) in dollar terms.
    """
    total_alpha = 0.0
    total_execution = 0.0

    for trade in trades:
        try:
            direction = 1.0 if trade.side == "buy" else -1.0
            qty = abs(trade.quantity)

            # Alpha P&L: signal predicted direction correctly?
            # Measured as (benchmark - signal_price) * direction * qty
            alpha = (trade.benchmark_price - trade.signal_price) * direction * qty

            # Execution P&L: did we execute better or worse than benchmark?
            # For buys: fill < benchmark = positive (saved money)
            # For sells: fill > benchmark = positive (got more)
            exec_pnl = (trade.benchmark_price - trade.fill_price) * direction * qty

            total_alpha += alpha
            total_execution += exec_pnl

        except Exception as e:
            logger.warning("Failed to decompose trade %s: %s", trade.symbol, e)

    logger.info(
        "P&L decomposition: alpha=$%.2f, execution=$%.2f, total=$%.2f",
        total_alpha, total_execution, total_alpha + total_execution,
    )
    return total_alpha, total_execution


def compute_implementation_shortfall(
    signal_price: float,
    fill_price: float,
    arrival_price: Optional[float] = None,
    benchmark_price: Optional[float] = None,
    side: str = "buy",
    quantity: float = 1.0,
) -> Dict[str, float]:
    """Compute implementation shortfall and its components.

    Implementation shortfall = total cost of implementing a trading
    decision, measured as the difference between the paper return and
    the actual return.

    Components:
        - Delay cost: slippage from signal to order arrival
        - Market impact: cost of the order moving the market
        - Opportunity cost: unfilled portion (assumed zero here)

    Parameters
    ----------
    signal_price : float
        Price when the trading signal was generated.
    fill_price : float
        Actual fill price.
    arrival_price : float, optional
        Price when the order arrived at the market.  If None, assumed
        equal to signal_price (no delay).
    benchmark_price : float, optional
        Benchmark price for impact measurement.  If None, uses
        midpoint of arrival and fill.
    side : str
        ``"buy"`` or ``"sell"``.
    quantity : float
        Number of shares.

    Returns
    -------
    dict
        Keys: total_shortfall, delay_cost, market_impact, total_bps.
    """
    try:
        direction = 1.0 if side == "buy" else -1.0

        if arrival_price is None:
            arrival_price = signal_price

        if benchmark_price is None:
            benchmark_price = (arrival_price + fill_price) / 2.0

        # Delay cost: price moved against us between signal and arrival
        delay = (arrival_price - signal_price) * direction * quantity

        # Market impact: price moved against us between arrival and fill
        impact = (fill_price - arrival_price) * direction * quantity

        # Total shortfall
        total = (fill_price - signal_price) * direction * quantity

        # Basis points
        ref_price = max(abs(signal_price), 1e-8)
        total_bps = (abs(fill_price - signal_price) / ref_price) * 10000

        return {
            "total_shortfall": total,
            "delay_cost": delay,
            "market_impact": impact,
            "opportunity_cost": 0.0,
            "total_bps": total_bps,
            "reference_price": signal_price,
        }

    except Exception as e:
        logger.error("Implementation shortfall calculation failed: %s", e)
        return {
            "total_shortfall": 0.0,
            "delay_cost": 0.0,
            "market_impact": 0.0,
            "opportunity_cost": 0.0,
            "total_bps": 0.0,
            "reference_price": signal_price,
        }


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------


class ExecutionAlphaAnalyzer:
    """Comprehensive execution alpha analysis.

    Tracks and decomposes P&L across multiple trades, providing aggregate
    statistics and per-symbol breakdowns.

    Parameters
    ----------
    benchmark_type : str
        Default benchmark: ``"vwap"``, ``"arrival"``, ``"close"``.
    """

    def __init__(self, benchmark_type: str = "vwap") -> None:
        self.benchmark_type = benchmark_type
        self._trades: List[TradeRecord] = []
        self._decompositions: List[PnLDecomposition] = []
        logger.info("ExecutionAlphaAnalyzer initialised (benchmark=%s).", benchmark_type)

    def add_trade(self, trade: TradeRecord) -> PnLDecomposition:
        """Add a trade and compute its decomposition.

        Parameters
        ----------
        trade : TradeRecord
            Trade to analyse.

        Returns
        -------
        PnLDecomposition
            Decomposed P&L for this trade.
        """
        self._trades.append(trade)
        decomp = self._decompose_single(trade)
        self._decompositions.append(decomp)
        return decomp

    def decompose_pnl(
        self, trades: List[TradeRecord],
    ) -> Tuple[float, float]:
        """Decompose P&L for a batch of trades.

        Parameters
        ----------
        trades : list of TradeRecord
            Trades to decompose.

        Returns
        -------
        Tuple[float, float]
            (total_alpha_pnl, total_execution_pnl).
        """
        return decompose_pnl(trades)

    def compute_implementation_shortfall(
        self,
        signal_price: float,
        fill_price: float,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute implementation shortfall for a single execution.

        Parameters
        ----------
        signal_price : float
            Decision price.
        fill_price : float
            Actual fill price.
        **kwargs
            Additional arguments passed to the module-level function.

        Returns
        -------
        dict
            Shortfall components.
        """
        return compute_implementation_shortfall(signal_price, fill_price, **kwargs)

    def generate_report(
        self,
        trades: Optional[List[TradeRecord]] = None,
    ) -> Dict[str, Any]:
        """Generate an aggregate execution quality report.

        Parameters
        ----------
        trades : list of TradeRecord, optional
            Trades to report on.  If None, uses previously added trades.

        Returns
        -------
        dict
            Aggregate report with per-symbol and overall metrics.
        """
        if trades is not None:
            decomps = [self._decompose_single(t) for t in trades]
        else:
            decomps = self._decompositions

        if not decomps:
            logger.warning("No trades to report on.")
            return {"n_trades": 0}

        try:
            total_alpha = sum(d.alpha_pnl for d in decomps)
            total_exec = sum(d.execution_pnl for d in decomps)
            total_pnl = sum(d.total_pnl for d in decomps)

            alpha_bps = [d.alpha_bps for d in decomps]
            exec_bps = [d.execution_bps for d in decomps]

            # Per-symbol breakdown
            by_symbol: Dict[str, Dict[str, float]] = {}
            for d in decomps:
                if d.symbol not in by_symbol:
                    by_symbol[d.symbol] = {
                        "alpha_pnl": 0.0, "execution_pnl": 0.0, "n_trades": 0,
                    }
                by_symbol[d.symbol]["alpha_pnl"] += d.alpha_pnl
                by_symbol[d.symbol]["execution_pnl"] += d.execution_pnl
                by_symbol[d.symbol]["n_trades"] += 1

            report = {
                "n_trades": len(decomps),
                "total_pnl": total_pnl,
                "alpha_pnl": total_alpha,
                "execution_pnl": total_exec,
                "alpha_pct_of_total": (
                    total_alpha / abs(total_pnl) * 100 if total_pnl != 0 else 0.0
                ),
                "execution_pct_of_total": (
                    total_exec / abs(total_pnl) * 100 if total_pnl != 0 else 0.0
                ),
                "avg_alpha_bps": float(np.mean(alpha_bps)) if alpha_bps else 0.0,
                "avg_execution_bps": float(np.mean(exec_bps)) if exec_bps else 0.0,
                "median_alpha_bps": float(np.median(alpha_bps)) if alpha_bps else 0.0,
                "median_execution_bps": float(np.median(exec_bps)) if exec_bps else 0.0,
                "by_symbol": by_symbol,
            }

            logger.info(
                "Execution report: %d trades, alpha=$%.2f (%.1f%%), "
                "execution=$%.2f (%.1f%%)",
                report["n_trades"], total_alpha,
                report["alpha_pct_of_total"],
                total_exec, report["execution_pct_of_total"],
            )
            return report

        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return {"n_trades": len(decomps), "error": str(e)}

    def get_execution_quality_score(self) -> float:
        """Compute an aggregate execution quality score in [0, 1].

        1.0 = perfect execution (always beat benchmark).
        0.0 = worst execution (always behind benchmark).

        Returns
        -------
        float
            Quality score.
        """
        if not self._decompositions:
            return 0.5  # neutral

        positive = sum(
            1 for d in self._decompositions if d.execution_pnl >= 0
        )
        return positive / len(self._decompositions)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all decompositions to a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per trade with all decomposition fields.
        """
        if not self._decompositions:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self._decompositions])

    def reset(self) -> None:
        """Clear all stored trades and decompositions."""
        self._trades.clear()
        self._decompositions.clear()

    def _decompose_single(self, trade: TradeRecord) -> PnLDecomposition:
        """Decompose a single trade."""
        try:
            direction = 1.0 if trade.side == "buy" else -1.0
            qty = abs(trade.quantity)
            ref_price = max(abs(trade.signal_price), 1e-8)

            # Alpha P&L
            alpha = (trade.benchmark_price - trade.signal_price) * direction * qty
            alpha_per_share = (trade.benchmark_price - trade.signal_price) * direction

            # Execution P&L
            exec_pnl = (trade.benchmark_price - trade.fill_price) * direction * qty
            exec_per_share = (trade.benchmark_price - trade.fill_price) * direction

            # Total
            total = alpha + exec_pnl

            # Basis points
            alpha_bps = abs(alpha_per_share) / ref_price * 10000
            exec_bps = abs(exec_per_share) / ref_price * 10000

            # Implementation shortfall components
            delay = (trade.arrival_price - trade.signal_price) * direction * qty
            impact = (trade.fill_price - trade.arrival_price) * direction * qty

            return PnLDecomposition(
                symbol=trade.symbol,
                side=trade.side,
                quantity=qty,
                total_pnl=total,
                alpha_pnl=alpha,
                execution_pnl=exec_pnl,
                alpha_per_share=alpha_per_share,
                execution_per_share=exec_per_share,
                alpha_bps=alpha_bps,
                execution_bps=exec_bps,
                delay_cost=delay,
                market_impact=impact,
                opportunity_cost=0.0,
            )

        except Exception as e:
            logger.warning("Single trade decomposition failed: %s", e)
            return PnLDecomposition(symbol=trade.symbol, side=trade.side)
