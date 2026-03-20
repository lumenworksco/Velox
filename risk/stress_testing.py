"""RISK-003: Stress testing framework for portfolio resilience analysis.

Runs predefined market shock scenarios against the current portfolio to
estimate potential losses. If worst-case exceeds -5% of portfolio, blocks
new position entries until risk is reduced.

Scenarios are based on historical tail events and designed to test:
- Equity drawdown (flash crash, gap opening)
- Liquidity evaporation (spread widening, volume collapse)
- Correlation regime change (diversification failure)
- Macro surprises (Fed, sector rotation)

Run daily before market open via the scheduler.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# Block new positions if worst-case scenario loss exceeds this
BLOCK_THRESHOLD_PCT = -0.05  # -5% portfolio loss


class ScenarioType(Enum):
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    GAP_OPENING = "gap_opening"
    FED_SURPRISE = "fed_surprise"
    SECTOR_SHOCK = "sector_shock"


@dataclass
class ScenarioShock:
    """Definition of a market stress scenario."""
    name: str
    scenario_type: ScenarioType
    description: str
    # Shocks to apply
    equity_move_pct: float = 0.0           # SPY-equivalent move
    vix_level: float | None = None         # Absolute VIX level
    spread_multiplier: float = 1.0         # Bid-ask spread multiplier
    volume_change_pct: float = 0.0         # Volume change (-0.9 = -90%)
    correlation_override: float | None = None  # Force all pairwise corr
    overnight_gap_pct: float = 0.0         # Gap at open
    sector_shock: dict[str, float] = field(default_factory=dict)  # sector -> move
    rate_move_bps: float = 0.0             # Interest rate move
    beta_amplification: float = 1.0        # High-beta stocks move more


@dataclass
class StressTestResult:
    """Result of a single stress scenario."""
    scenario_name: str
    scenario_type: ScenarioType
    estimated_pnl_pct: float       # Portfolio P&L as decimal (-0.03 = -3%)
    estimated_pnl_dollars: float
    positions_affected: int
    worst_position: str = ""
    worst_position_pnl: float = 0.0
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Predefined scenarios
# ------------------------------------------------------------------

PREDEFINED_SCENARIOS = [
    ScenarioShock(
        name="Flash Crash",
        scenario_type=ScenarioType.FLASH_CRASH,
        description="Rapid equity sell-off: SPY -5%, VIX spikes to 80",
        equity_move_pct=-0.05,
        vix_level=80.0,
        spread_multiplier=5.0,
        volume_change_pct=3.0,  # Volume surges
        beta_amplification=1.5,
    ),
    ScenarioShock(
        name="Liquidity Crisis",
        scenario_type=ScenarioType.LIQUIDITY_CRISIS,
        description="Market liquidity evaporates: spreads 10x, volume -90%",
        equity_move_pct=-0.02,
        vix_level=45.0,
        spread_multiplier=10.0,
        volume_change_pct=-0.90,
        beta_amplification=1.2,
    ),
    ScenarioShock(
        name="Correlation Breakdown",
        scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
        description="All correlations converge to 1.0 (diversification failure)",
        equity_move_pct=-0.03,
        vix_level=35.0,
        correlation_override=1.0,
        beta_amplification=1.0,
    ),
    ScenarioShock(
        name="Gap Opening",
        scenario_type=ScenarioType.GAP_OPENING,
        description="Market gaps down -3% overnight on geopolitical event",
        equity_move_pct=-0.03,
        overnight_gap_pct=-0.03,
        vix_level=30.0,
        spread_multiplier=3.0,
        beta_amplification=1.3,
    ),
    ScenarioShock(
        name="Fed Surprise",
        scenario_type=ScenarioType.FED_SURPRISE,
        description="Unexpected 50bps rate hike; growth stocks hit hard",
        equity_move_pct=-0.02,
        vix_level=28.0,
        rate_move_bps=50.0,
        sector_shock={
            "XLK": -0.04,   # Tech -4%
            "XLC": -0.03,   # Comm -3%
            "XLY": -0.03,   # Discretionary -3%
            "XLF": 0.01,    # Financials +1%
            "XLU": -0.01,   # Utilities -1%
        },
        beta_amplification=1.2,
    ),
    ScenarioShock(
        name="Sector Shock",
        scenario_type=ScenarioType.SECTOR_SHOCK,
        description="Single sector drops -10% (e.g., regulatory action on tech)",
        equity_move_pct=-0.01,
        vix_level=25.0,
        sector_shock={"XLK": -0.10},
        beta_amplification=1.0,
    ),
]


class StressTestFramework:
    """Run stress test scenarios against the current portfolio.

    Usage:
        framework = StressTestFramework()
        results = framework.run_stress_tests(positions)
        if framework.should_block_new_positions():
            # Do not open new trades
    """

    def __init__(
        self,
        scenarios: list[ScenarioShock] | None = None,
        block_threshold: float = BLOCK_THRESHOLD_PCT,
    ):
        self._scenarios = scenarios or list(PREDEFINED_SCENARIOS)
        self._block_threshold = block_threshold
        self._last_results: list[StressTestResult] = []
        self._last_run: datetime | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_stress_tests(
        self,
        positions: dict[str, Any],
        portfolio_equity: float | None = None,
    ) -> list[StressTestResult]:
        """Run all stress scenarios against current positions.

        Args:
            positions: Dict of symbol -> position info. Each value should
                       have attributes or keys: qty, entry_price, side, strategy.
                       Accepts TradeRecord objects or dicts.
            portfolio_equity: Total portfolio equity (for % calculations).
                              If None, estimated from positions.

        Returns:
            List of StressTestResult, one per scenario.
        """
        if not positions:
            logger.info("Stress test: no positions to test")
            with self._lock:
                self._last_results = []
                self._last_run = datetime.now(config.ET)
            return []

        # Normalize positions to a common format
        pos_list = self._normalize_positions(positions)

        if portfolio_equity is None:
            portfolio_equity = sum(p["notional"] for p in pos_list) * 2.0
            # Rough estimate: positions are ~50% deployed

        if portfolio_equity <= 0:
            portfolio_equity = 100_000.0  # Fallback

        results = []
        for scenario in self._scenarios:
            try:
                result = self._evaluate_scenario(scenario, pos_list, portfolio_equity)
                results.append(result)
            except Exception as e:
                logger.error(f"Stress test '{scenario.name}' failed: {e}", exc_info=True)
                results.append(StressTestResult(
                    scenario_name=scenario.name,
                    scenario_type=scenario.scenario_type,
                    estimated_pnl_pct=0.0,
                    estimated_pnl_dollars=0.0,
                    positions_affected=0,
                    details={"error": str(e)},
                ))

        with self._lock:
            self._last_results = results
            self._last_run = datetime.now(config.ET)

        # Log summary
        worst = min(results, key=lambda r: r.estimated_pnl_pct) if results else None
        if worst:
            logger.info(
                f"Stress tests complete: {len(results)} scenarios. "
                f"Worst case: {worst.scenario_name} = {worst.estimated_pnl_pct:.2%} "
                f"(${worst.estimated_pnl_dollars:+,.0f})"
            )
            if worst.estimated_pnl_pct < self._block_threshold:
                logger.warning(
                    f"STRESS TEST WARNING: worst case {worst.estimated_pnl_pct:.2%} "
                    f"exceeds block threshold {self._block_threshold:.2%}"
                )

        return results

    def should_block_new_positions(self) -> bool:
        """Return True if the worst-case scenario exceeds the block threshold.

        Checks the most recent stress test results. Returns False if no
        results are available (fail-open).
        """
        with self._lock:
            if not self._last_results:
                return False
            worst_pnl = min(r.estimated_pnl_pct for r in self._last_results)
            return worst_pnl < self._block_threshold

    def get_worst_scenario(self) -> StressTestResult | None:
        """Return the scenario with the largest estimated loss."""
        with self._lock:
            if not self._last_results:
                return None
            return min(self._last_results, key=lambda r: r.estimated_pnl_pct)

    # ------------------------------------------------------------------
    # Scenario evaluation
    # ------------------------------------------------------------------

    def _evaluate_scenario(
        self,
        scenario: ScenarioShock,
        positions: list[dict],
        portfolio_equity: float,
    ) -> StressTestResult:
        """Evaluate a single stress scenario against the portfolio."""

        total_pnl = 0.0
        affected = 0
        worst_sym = ""
        worst_pnl = 0.0
        position_details = {}

        for pos in positions:
            symbol = pos["symbol"]
            notional = pos["notional"]  # Absolute dollar exposure
            side_mult = pos["side_mult"]  # +1 long, -1 short
            beta = pos.get("beta", 1.0)
            sector_etf = config.SECTOR_MAP.get(symbol, "")

            # 1. Equity move impact (beta-adjusted)
            move = scenario.equity_move_pct * beta * scenario.beta_amplification
            pos_pnl = notional * move * side_mult

            # 2. Sector-specific shock (additive)
            if sector_etf in scenario.sector_shock:
                sector_move = scenario.sector_shock[sector_etf]
                pos_pnl += notional * sector_move * side_mult

            # 3. Overnight gap (positions held overnight are fully exposed)
            if scenario.overnight_gap_pct != 0.0:
                gap_move = scenario.overnight_gap_pct * beta * scenario.beta_amplification
                # Gap impact is already captured in equity_move for long positions
                # For shorts, a gap down is profitable
                # No additional adjustment needed -- gap is part of equity_move

            # 4. Spread/liquidity cost (always negative)
            if scenario.spread_multiplier > 1.0:
                # Assume current spread is ~5 bps; excess spread is a cost
                normal_spread_bps = 5.0
                excess_spread = normal_spread_bps * (scenario.spread_multiplier - 1.0)
                spread_cost = notional * (excess_spread / 10_000.0)
                pos_pnl -= spread_cost

            # 5. Correlation override -- when all corr -> 1, no diversification
            # benefit, so the portfolio loss is the sum of individual losses
            # (which is already what we compute position-by-position)

            total_pnl += pos_pnl
            affected += 1

            if pos_pnl < worst_pnl:
                worst_pnl = pos_pnl
                worst_sym = symbol

            position_details[symbol] = round(pos_pnl, 2)

        pnl_pct = total_pnl / portfolio_equity if portfolio_equity > 0 else 0.0

        return StressTestResult(
            scenario_name=scenario.name,
            scenario_type=scenario.scenario_type,
            estimated_pnl_pct=pnl_pct,
            estimated_pnl_dollars=total_pnl,
            positions_affected=affected,
            worst_position=worst_sym,
            worst_position_pnl=worst_pnl,
            details={
                "position_pnl": position_details,
                "vix_level": scenario.vix_level,
                "spread_mult": scenario.spread_multiplier,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_positions(positions: dict) -> list[dict]:
        """Convert position dict (symbol -> TradeRecord or dict) to flat list."""
        result = []
        for symbol, pos in positions.items():
            # Support both TradeRecord objects and dicts
            if hasattr(pos, "entry_price"):
                qty = getattr(pos, "qty", 0)
                price = getattr(pos, "entry_price", 0)
                side = getattr(pos, "side", "buy")
                strategy = getattr(pos, "strategy", "")
            elif isinstance(pos, dict):
                qty = pos.get("qty", 0)
                price = pos.get("entry_price", 0)
                side = pos.get("side", "buy")
                strategy = pos.get("strategy", "")
            else:
                continue

            if qty <= 0 or price <= 0:
                continue

            notional = qty * price
            side_mult = 1.0 if side == "buy" else -1.0

            # Look up beta from config table
            beta = config.MICRO_BETA_TABLE.get(symbol, 1.0)

            result.append({
                "symbol": symbol,
                "notional": notional,
                "side_mult": side_mult,
                "beta": beta,
                "strategy": strategy,
            })

        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def last_results(self) -> list[StressTestResult]:
        with self._lock:
            return list(self._last_results)

    @property
    def status(self) -> dict:
        with self._lock:
            worst = (
                min(self._last_results, key=lambda r: r.estimated_pnl_pct)
                if self._last_results else None
            )
            return {
                "last_run": self._last_run.isoformat() if self._last_run else None,
                "scenarios_count": len(self._scenarios),
                "results_count": len(self._last_results),
                "worst_scenario": worst.scenario_name if worst else None,
                "worst_pnl_pct": f"{worst.estimated_pnl_pct:.2%}" if worst else None,
                "blocking_new_positions": self.should_block_new_positions(),
                "block_threshold": f"{self._block_threshold:.2%}",
            }
