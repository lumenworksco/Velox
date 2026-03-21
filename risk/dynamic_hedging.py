"""RISK-002: Dynamic hedging module — portfolio-aware hedge recommendations.

Analyzes current portfolio exposures and generates hedge recommendations:
1. Beta hedging: When portfolio beta > 1.5, suggest SPY puts.
2. Tail risk hedging: When tail risk is elevated (high VaR/CVaR), suggest VIX calls.
3. Sector concentration hedging: When a single sector exceeds concentration
   threshold, suggest the corresponding inverse sector ETF.

Output: list of HedgeRecommendation named tuples (symbol, side, size_pct).

Usage:
    hedger = DynamicHedgingEngine()
    recs = hedger.evaluate(positions, portfolio_equity, var_result)
    for rec in recs:
        print(f"{rec.symbol} {rec.side} {rec.size_pct:.1%}")
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# Thresholds
BETA_HEDGE_THRESHOLD = 1.5        # Portfolio beta above which to suggest SPY puts
TAIL_RISK_VAR_PCT = 0.03          # 95% VaR > 3% triggers tail risk hedge
SECTOR_CONCENTRATION_PCT = 0.40   # Single sector > 40% of portfolio

# Inverse sector ETFs for hedging
SECTOR_INVERSE_ETFS: dict[str, str] = {
    "XLK": "PSQ",    # Tech -> short Nasdaq proxy
    "XLF": "SKF",    # Financials -> ProShares UltraShort Financials
    "XLE": "ERY",    # Energy -> Direxion Daily Energy Bear
    "XLV": "RXD",    # Healthcare -> ProShares UltraShort Health Care
    "XLY": "SCC",    # Consumer Disc -> ProShares UltraShort Consumer Services
    "XLP": "SZK",    # Consumer Staples -> ProShares UltraShort Consumer Goods
    "XLI": "SIJ",    # Industrials -> ProShares UltraShort Industrials
    "XLU": "SDP",    # Utilities -> ProShares UltraShort Utilities
    "XLC": "PSQ",    # Comm Services -> short Nasdaq proxy
    "XLRE": "SRS",   # Real Estate -> ProShares UltraShort Real Estate
    "XLB": "SMN",    # Materials -> ProShares UltraShort Basic Materials
}


@dataclass
class HedgeRecommendation:
    """A single hedge recommendation."""
    symbol: str
    side: str             # "buy" for protective puts/calls, "sell" for shorts
    size_pct: float       # Suggested hedge size as fraction of portfolio (0.0–1.0)
    hedge_type: str       # "beta", "tail_risk", or "sector_concentration"
    reason: str           # Human-readable explanation
    priority: int = 1     # 1 = highest priority


@dataclass
class HedgeAnalysis:
    """Full hedge analysis result."""
    recommendations: list[HedgeRecommendation] = field(default_factory=list)
    portfolio_beta: float = 1.0
    tail_risk_elevated: bool = False
    sector_concentrations: dict[str, float] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.now)


class DynamicHedgingEngine:
    """Generate dynamic hedge recommendations based on portfolio exposures.

    Fail-open: if any analysis step fails, it is skipped and remaining
    checks continue. An empty recommendation list means no hedges needed.
    """

    def __init__(
        self,
        beta_threshold: float = BETA_HEDGE_THRESHOLD,
        tail_risk_var_pct: float = TAIL_RISK_VAR_PCT,
        sector_concentration_pct: float = SECTOR_CONCENTRATION_PCT,
    ):
        self._beta_threshold = beta_threshold
        self._tail_risk_var_pct = tail_risk_var_pct
        self._sector_concentration_pct = sector_concentration_pct
        self._last_analysis: HedgeAnalysis = HedgeAnalysis()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        positions: dict[str, Any],
        portfolio_equity: float,
        var_result: Any | None = None,
    ) -> list[HedgeRecommendation]:
        """Evaluate current portfolio and return hedge recommendations.

        Args:
            positions: Dict of symbol -> position info (TradeRecord or dict).
                       Must have: qty, entry_price, side, strategy.
            portfolio_equity: Total portfolio equity in dollars.
            var_result: Optional VaRResult from VaRMonitor for tail risk check.

        Returns:
            List of HedgeRecommendation sorted by priority.
        """
        if not positions or portfolio_equity <= 0:
            return []

        recommendations: list[HedgeRecommendation] = []

        # 1. Beta hedging
        try:
            beta_recs = self._check_beta_exposure(positions, portfolio_equity)
            recommendations.extend(beta_recs)
        except Exception as e:
            logger.warning(f"RISK-002: Beta hedge check failed (fail-open): {e}")

        # 2. Tail risk hedging
        try:
            tail_recs = self._check_tail_risk(var_result, portfolio_equity)
            recommendations.extend(tail_recs)
        except Exception as e:
            logger.warning(f"RISK-002: Tail risk check failed (fail-open): {e}")

        # 3. Sector concentration hedging
        try:
            sector_recs = self._check_sector_concentration(positions, portfolio_equity)
            recommendations.extend(sector_recs)
        except Exception as e:
            logger.warning(f"RISK-002: Sector concentration check failed (fail-open): {e}")

        # Sort by priority (lower = more urgent)
        recommendations.sort(key=lambda r: r.priority)

        # Store analysis
        with self._lock:
            self._last_analysis = HedgeAnalysis(
                recommendations=recommendations,
                portfolio_beta=self._compute_portfolio_beta(positions, portfolio_equity),
                tail_risk_elevated=any(r.hedge_type == "tail_risk" for r in recommendations),
                computed_at=datetime.now(),
            )

        if recommendations:
            logger.info(
                f"RISK-002: {len(recommendations)} hedge recommendation(s): "
                + ", ".join(f"{r.symbol} {r.side} {r.size_pct:.1%}" for r in recommendations)
            )

        return recommendations

    # ------------------------------------------------------------------
    # Beta exposure
    # ------------------------------------------------------------------

    def _check_beta_exposure(
        self,
        positions: dict[str, Any],
        portfolio_equity: float,
    ) -> list[HedgeRecommendation]:
        """Check if portfolio beta exceeds threshold; recommend SPY puts."""
        portfolio_beta = self._compute_portfolio_beta(positions, portfolio_equity)

        if portfolio_beta <= self._beta_threshold:
            return []

        # Size the hedge to bring beta back to ~1.0
        excess_beta = portfolio_beta - 1.0
        # SPY put notional needed = excess_beta * deployed_capital
        deployed = self._total_deployed(positions)
        hedge_notional = excess_beta * deployed
        hedge_pct = min(hedge_notional / portfolio_equity, 0.10)  # Cap at 10%

        return [HedgeRecommendation(
            symbol="SPY",
            side="buy",  # Buy puts
            size_pct=hedge_pct,
            hedge_type="beta",
            reason=f"Portfolio beta {portfolio_beta:.2f} > {self._beta_threshold:.1f}; "
                   f"suggest SPY puts at {hedge_pct:.1%} of portfolio",
            priority=1,
        )]

    def _compute_portfolio_beta(
        self,
        positions: dict[str, Any],
        portfolio_equity: float,
    ) -> float:
        """Compute dollar-weighted portfolio beta."""
        total_beta_dollars = 0.0
        total_notional = 0.0

        for symbol, pos in positions.items():
            qty = self._get_attr(pos, "qty", 0)
            price = self._get_attr(pos, "entry_price", 0.0)
            side = self._get_attr(pos, "side", "buy")

            if qty <= 0 or price <= 0:
                continue

            notional = qty * price
            side_mult = 1.0 if side == "buy" else -1.0
            beta = getattr(config, "MICRO_BETA_TABLE", {}).get(symbol, 1.0)

            total_beta_dollars += notional * beta * side_mult
            total_notional += notional

        if total_notional <= 0:
            return 1.0

        return total_beta_dollars / total_notional

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    def _check_tail_risk(
        self,
        var_result: Any | None,
        portfolio_equity: float,
    ) -> list[HedgeRecommendation]:
        """Check if tail risk is elevated; recommend VIX calls."""
        if var_result is None:
            return []

        var_95_pct = getattr(var_result, "var_95_pct", 0.0)
        if var_95_pct < self._tail_risk_var_pct:
            return []

        # Scale hedge size: 1–3% of portfolio based on severity
        severity = min(var_95_pct / self._tail_risk_var_pct, 3.0)
        hedge_pct = 0.01 * severity  # 1% per unit of severity, max 3%
        hedge_pct = min(hedge_pct, 0.05)

        return [HedgeRecommendation(
            symbol="VIX",
            side="buy",  # Buy VIX calls
            size_pct=hedge_pct,
            hedge_type="tail_risk",
            reason=f"Tail risk elevated: VaR95={var_95_pct:.2%} > {self._tail_risk_var_pct:.2%}; "
                   f"suggest VIX calls at {hedge_pct:.1%} of portfolio",
            priority=1,
        )]

    # ------------------------------------------------------------------
    # Sector concentration
    # ------------------------------------------------------------------

    def _check_sector_concentration(
        self,
        positions: dict[str, Any],
        portfolio_equity: float,
    ) -> list[HedgeRecommendation]:
        """Check for sector over-concentration; recommend inverse ETFs."""
        sector_notional: dict[str, float] = {}
        total_notional = 0.0

        sector_map = getattr(config, "SECTOR_MAP", {})

        for symbol, pos in positions.items():
            qty = self._get_attr(pos, "qty", 0)
            price = self._get_attr(pos, "entry_price", 0.0)
            side = self._get_attr(pos, "side", "buy")

            if qty <= 0 or price <= 0:
                continue

            notional = qty * price
            total_notional += notional

            sector = sector_map.get(symbol, "UNKNOWN")
            if side == "buy":
                sector_notional[sector] = sector_notional.get(sector, 0.0) + notional
            else:
                sector_notional[sector] = sector_notional.get(sector, 0.0) - notional

        if total_notional <= 0:
            return []

        recs: list[HedgeRecommendation] = []
        for sector, notional in sector_notional.items():
            concentration = abs(notional) / total_notional
            if concentration > self._sector_concentration_pct:
                inverse_etf = SECTOR_INVERSE_ETFS.get(sector)
                if inverse_etf is None:
                    continue

                # Hedge the excess over the threshold
                excess_pct = concentration - self._sector_concentration_pct
                hedge_pct = min(excess_pct * 0.5, 0.05)  # Hedge half the excess, cap 5%

                recs.append(HedgeRecommendation(
                    symbol=inverse_etf,
                    side="buy",
                    size_pct=hedge_pct,
                    hedge_type="sector_concentration",
                    reason=f"Sector {sector} concentration {concentration:.1%} > "
                           f"{self._sector_concentration_pct:.0%}; suggest {inverse_etf} "
                           f"at {hedge_pct:.1%} of portfolio",
                    priority=2,
                ))

        # Store for status
        with self._lock:
            self._last_analysis.sector_concentrations = {
                s: n / total_notional for s, n in sector_notional.items()
            }

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _total_deployed(positions: dict[str, Any]) -> float:
        """Sum of absolute notional exposure across all positions."""
        total = 0.0
        for pos in positions.values():
            qty = DynamicHedgingEngine._get_attr(pos, "qty", 0)
            price = DynamicHedgingEngine._get_attr(pos, "entry_price", 0.0)
            if qty > 0 and price > 0:
                total += qty * price
        return total

    @staticmethod
    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        """Get attribute from object or dict."""
        if hasattr(obj, name):
            return getattr(obj, name, default)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def last_analysis(self) -> HedgeAnalysis:
        with self._lock:
            return self._last_analysis

    @property
    def status(self) -> dict:
        with self._lock:
            a = self._last_analysis
            return {
                "recommendations_count": len(a.recommendations),
                "portfolio_beta": round(a.portfolio_beta, 3),
                "tail_risk_elevated": a.tail_risk_elevated,
                "sector_concentrations": {
                    s: round(v, 3) for s, v in a.sector_concentrations.items()
                },
                "computed_at": a.computed_at.isoformat() if a.computed_at else None,
                "recommendations": [
                    {"symbol": r.symbol, "side": r.side, "size_pct": round(r.size_pct, 4),
                     "type": r.hedge_type, "reason": r.reason}
                    for r in a.recommendations
                ],
            }
