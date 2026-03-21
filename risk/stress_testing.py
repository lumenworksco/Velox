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
    # RISK-008: Adversarial scenario generation
    # ------------------------------------------------------------------

    def generate_adversarial_scenarios(
        self,
        positions: dict[str, Any],
        portfolio_equity: float | None = None,
    ) -> list[ScenarioShock]:
        """RISK-008: Generate adversarial scenarios targeting portfolio weaknesses.

        Analyzes the current portfolio to create worst-case scenarios that
        specifically exploit the portfolio's vulnerabilities:
        1. Concentrated position attack: max drawdown on largest positions.
        2. Correlated selloff: simultaneous drop in top holdings.
        3. Sector collapse: shock to the most concentrated sector.
        4. Beta amplification: high-beta positions get outsized moves.
        5. Liquidity crisis on illiquid holdings.

        Args:
            positions: Dict of symbol -> position info.
            portfolio_equity: Total portfolio equity.

        Returns:
            List of adversarial ScenarioShock objects.
        """
        if not positions:
            return []

        pos_list = self._normalize_positions(positions)
        if not pos_list:
            return []

        if portfolio_equity is None:
            portfolio_equity = sum(p["notional"] for p in pos_list) * 2.0

        adversarial: list[ScenarioShock] = []

        try:
            # 1. Concentrated position attack
            conc_scenario = self._adversarial_concentration(pos_list, portfolio_equity)
            if conc_scenario:
                adversarial.append(conc_scenario)
        except Exception as e:
            logger.debug(f"RISK-008: Concentration scenario generation failed: {e}")

        try:
            # 2. Correlated selloff of top holdings
            corr_scenario = self._adversarial_correlated_selloff(pos_list, portfolio_equity)
            if corr_scenario:
                adversarial.append(corr_scenario)
        except Exception as e:
            logger.debug(f"RISK-008: Correlated selloff scenario generation failed: {e}")

        try:
            # 3. Sector collapse
            sector_scenario = self._adversarial_sector_collapse(pos_list, portfolio_equity)
            if sector_scenario:
                adversarial.append(sector_scenario)
        except Exception as e:
            logger.debug(f"RISK-008: Sector collapse scenario generation failed: {e}")

        try:
            # 4. Beta-amplified crash
            beta_scenario = self._adversarial_beta_crash(pos_list, portfolio_equity)
            if beta_scenario:
                adversarial.append(beta_scenario)
        except Exception as e:
            logger.debug(f"RISK-008: Beta crash scenario generation failed: {e}")

        try:
            # 5. Anti-portfolio (everything moves against us)
            anti_scenario = self._adversarial_anti_portfolio(pos_list, portfolio_equity)
            if anti_scenario:
                adversarial.append(anti_scenario)
        except Exception as e:
            logger.debug(f"RISK-008: Anti-portfolio scenario generation failed: {e}")

        if adversarial:
            logger.info(
                f"RISK-008: Generated {len(adversarial)} adversarial scenarios "
                f"targeting portfolio weaknesses"
            )

        return adversarial

    def run_adversarial_stress_tests(
        self,
        positions: dict[str, Any],
        portfolio_equity: float | None = None,
    ) -> list[StressTestResult]:
        """RISK-008: Generate and run adversarial scenarios in one call.

        Combines generate_adversarial_scenarios + run_stress_tests using
        the adversarial scenarios instead of predefined ones.
        """
        scenarios = self.generate_adversarial_scenarios(positions, portfolio_equity)
        if not scenarios:
            return []

        # Temporarily swap scenarios
        old_scenarios = self._scenarios
        self._scenarios = scenarios

        try:
            results = self.run_stress_tests(positions, portfolio_equity)
        finally:
            self._scenarios = old_scenarios

        return results

    def _adversarial_concentration(
        self,
        pos_list: list[dict],
        portfolio_equity: float,
    ) -> ScenarioShock | None:
        """Attack concentrated positions with outsized moves."""
        if not pos_list:
            return None

        # Find largest position by notional
        largest = max(pos_list, key=lambda p: p["notional"])
        concentration = largest["notional"] / portfolio_equity

        if concentration < 0.05:  # Less than 5% — not concentrated enough
            return None

        # The larger the concentration, the bigger the shock
        shock_pct = -0.03 * (1 + concentration * 5)  # -3% to -18% depending on concentration
        shock_pct = max(shock_pct, -0.20)  # Cap at -20%

        return ScenarioShock(
            name=f"Adversarial: {largest['symbol']} Concentration Attack",
            scenario_type=ScenarioType.FLASH_CRASH,
            description=(
                f"Targeted attack on largest position {largest['symbol']} "
                f"({concentration:.0%} of portfolio) with {shock_pct:.0%} move"
            ),
            equity_move_pct=shock_pct * 0.3,  # General market sympathy
            vix_level=40.0,
            spread_multiplier=4.0,
            beta_amplification=2.0,  # Concentrated stock moves more
        )

    def _adversarial_correlated_selloff(
        self,
        pos_list: list[dict],
        portfolio_equity: float,
    ) -> ScenarioShock | None:
        """All top holdings drop simultaneously (correlation -> 1)."""
        if len(pos_list) < 2:
            return None

        # Sort by notional, take top 5
        sorted_pos = sorted(pos_list, key=lambda p: -p["notional"])[:5]
        top_notional = sum(p["notional"] for p in sorted_pos)
        top_pct = top_notional / portfolio_equity

        return ScenarioShock(
            name="Adversarial: Correlated Top-Holdings Selloff",
            scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
            description=(
                f"Top {len(sorted_pos)} holdings ({top_pct:.0%} of portfolio) "
                f"drop simultaneously as correlations spike to 1.0"
            ),
            equity_move_pct=-0.05,
            vix_level=50.0,
            correlation_override=1.0,
            spread_multiplier=3.0,
            beta_amplification=1.5,
        )

    def _adversarial_sector_collapse(
        self,
        pos_list: list[dict],
        portfolio_equity: float,
    ) -> ScenarioShock | None:
        """Collapse the most concentrated sector."""
        sector_map = getattr(config, "SECTOR_MAP", {})
        sector_notional: dict[str, float] = {}

        for pos in pos_list:
            sector = sector_map.get(pos["symbol"], "UNKNOWN")
            sector_notional[sector] = sector_notional.get(sector, 0.0) + pos["notional"]

        if not sector_notional:
            return None

        # Find largest sector
        top_sector = max(sector_notional, key=sector_notional.get)
        top_sector_pct = sector_notional[top_sector] / portfolio_equity

        if top_sector_pct < 0.15:  # Not concentrated enough
            return None

        # Shock proportional to concentration
        sector_shock_pct = -0.08 * (1 + top_sector_pct)

        return ScenarioShock(
            name=f"Adversarial: {top_sector} Sector Collapse",
            scenario_type=ScenarioType.SECTOR_SHOCK,
            description=(
                f"Most concentrated sector {top_sector} ({top_sector_pct:.0%}) "
                f"drops {sector_shock_pct:.0%}"
            ),
            equity_move_pct=-0.015,
            vix_level=35.0,
            sector_shock={top_sector: sector_shock_pct},
            spread_multiplier=3.0,
            beta_amplification=1.2,
        )

    def _adversarial_beta_crash(
        self,
        pos_list: list[dict],
        portfolio_equity: float,
    ) -> ScenarioShock | None:
        """Market crash amplified by portfolio's beta exposure."""
        # Compute weighted average beta
        total_notional = sum(p["notional"] for p in pos_list)
        if total_notional <= 0:
            return None

        weighted_beta = sum(
            p["beta"] * p["notional"] / total_notional for p in pos_list
        )

        if weighted_beta <= 1.2:
            return None  # Beta not high enough to exploit

        # Scale the crash by the portfolio's beta
        market_drop = -0.04
        effective_drop = market_drop * weighted_beta

        return ScenarioShock(
            name=f"Adversarial: Beta-Amplified Crash (beta={weighted_beta:.1f})",
            scenario_type=ScenarioType.FLASH_CRASH,
            description=(
                f"Market drops {market_drop:.0%} amplified by portfolio "
                f"beta {weighted_beta:.1f} for effective {effective_drop:.1%} loss"
            ),
            equity_move_pct=market_drop,
            vix_level=60.0,
            spread_multiplier=5.0,
            beta_amplification=weighted_beta,
        )

    def _adversarial_anti_portfolio(
        self,
        pos_list: list[dict],
        portfolio_equity: float,
    ) -> ScenarioShock | None:
        """Every position moves against its direction (longs drop, shorts rally)."""
        # Compute the portfolio's directional bias
        long_notional = sum(p["notional"] for p in pos_list if p["side_mult"] > 0)
        short_notional = sum(p["notional"] for p in pos_list if p["side_mult"] < 0)
        net_exposure = (long_notional - short_notional) / portfolio_equity if portfolio_equity > 0 else 0

        # If net long, scenario is a market crash; if net short, a rally
        if net_exposure > 0:
            move = -0.04  # Market drops
            scenario_desc = f"market drops against net-long exposure ({net_exposure:.0%})"
        elif net_exposure < 0:
            move = 0.04   # Market rallies
            scenario_desc = f"market rallies against net-short exposure ({net_exposure:.0%})"
        else:
            return None  # Market-neutral, not much to attack

        return ScenarioShock(
            name="Adversarial: Anti-Portfolio (Everything Moves Against You)",
            scenario_type=ScenarioType.GAP_OPENING,
            description=(
                f"Anti-portfolio scenario: {scenario_desc}. "
                f"All positions move against their direction."
            ),
            equity_move_pct=move,
            overnight_gap_pct=move,
            vix_level=45.0,
            spread_multiplier=4.0,
            beta_amplification=1.3,
        )

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
