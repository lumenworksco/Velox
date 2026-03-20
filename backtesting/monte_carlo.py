"""BACKTEST-003: Monte Carlo Robustness Testing.

Tests strategy robustness through randomization:
- Trade shuffling: random reordering of trade sequence
- Trade skipping: randomly remove 10-30% of trades
- Parameter perturbation: vary parameters by +/-10-20%
- Data bootstrapping: resample trades with replacement

Usage:
    mc = MonteCarloTester(n_simulations=1000)
    result = mc.run_simulations(trades, initial_capital=100_000)
    print(result.median_sharpe, result.prob_of_ruin)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int = 0

    # Sharpe distribution
    median_sharpe: float = 0.0
    mean_sharpe: float = 0.0
    sharpe_5th_pct: float = 0.0
    sharpe_25th_pct: float = 0.0
    sharpe_75th_pct: float = 0.0
    sharpe_95th_pct: float = 0.0

    # Return distribution
    median_return: float = 0.0
    mean_return: float = 0.0
    return_5th_pct: float = 0.0
    return_95th_pct: float = 0.0

    # Drawdown distribution
    median_max_drawdown: float = 0.0
    max_drawdown_95th_pct: float = 0.0  # 95th percentile of drawdowns (worst case)
    mean_max_drawdown: float = 0.0

    # Risk metrics
    probability_of_ruin: float = 0.0      # P(portfolio < 50% initial)
    probability_of_loss: float = 0.0      # P(total return < 0)
    probability_of_target: float = 0.0    # P(return > 10%)

    # Raw arrays for further analysis
    sharpe_distribution: list[float] = field(default_factory=list)
    return_distribution: list[float] = field(default_factory=list)
    drawdown_distribution: list[float] = field(default_factory=list)


class MonteCarloTester:
    """Monte Carlo simulation engine for strategy robustness testing.

    Runs multiple randomized variations of a trade sequence to assess
    how robust a strategy's performance is to luck and path dependency.

    Args:
        n_simulations: Number of Monte Carlo trials (default 1000).
        risk_free_rate: Annual risk-free rate for Sharpe computation.
        ruin_threshold: Portfolio fraction below which "ruin" is declared.
        target_return: Return threshold for probability of target metric.
        random_seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        risk_free_rate: float = 0.045,
        ruin_threshold: float = 0.50,
        target_return: float = 0.10,
        random_seed: Optional[int] = None,
    ) -> None:
        if n_simulations < 10:
            raise ValueError(f"n_simulations must be >= 10, got {n_simulations}")

        self._n_sims = n_simulations
        self._rf = risk_free_rate
        self._ruin_threshold = ruin_threshold
        self._target_return = target_return
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def run_simulations(
        self,
        trades: list[dict],
        initial_capital: float = 100_000.0,
        n_simulations: Optional[int] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulations on a set of historical trades.

        Each trade dict must have at minimum: ``pnl`` (float).
        Optional keys: ``pnl_pct``, ``hold_minutes``, ``strategy``.

        Runs four types of simulations:
        1. Trade shuffling (25% of sims)
        2. Trade skipping (25%)
        3. Parameter perturbation (25%)
        4. Data bootstrapping (25%)

        Args:
            trades: List of trade dicts, each containing at least ``pnl``.
            initial_capital: Starting capital for each simulation.
            n_simulations: Override default number of simulations.

        Returns:
            MonteCarloResult with distribution statistics.
        """
        n_sims = n_simulations or self._n_sims

        if not trades:
            logger.warning("No trades provided for Monte Carlo simulation")
            return MonteCarloResult(n_simulations=0)

        pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)

        if len(pnls) < 5:
            logger.warning("Too few trades (%d) for meaningful Monte Carlo", len(pnls))
            return MonteCarloResult(n_simulations=0)

        logger.info(
            "Running %d Monte Carlo simulations on %d trades (capital=$%.0f)",
            n_sims, len(pnls), initial_capital,
        )

        # Allocate simulation types
        n_shuffle = n_sims // 4
        n_skip = n_sims // 4
        n_perturb = n_sims // 4
        n_bootstrap = n_sims - n_shuffle - n_skip - n_perturb

        all_sharpes = []
        all_returns = []
        all_drawdowns = []

        # 1. Trade shuffling — random order
        for _ in range(n_shuffle):
            shuffled = self._rng.permutation(pnls)
            metrics = self._simulate_equity_curve(shuffled, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 2. Trade skipping — remove 10-30% of trades
        for _ in range(n_skip):
            skip_pct = self._rng.uniform(0.10, 0.30)
            mask = self._rng.random(len(pnls)) > skip_pct
            if mask.sum() < 3:
                mask[:3] = True  # Keep at least 3 trades
            skipped = pnls[mask]
            metrics = self._simulate_equity_curve(skipped, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 3. Parameter perturbation — scale PnLs by +/-10-20%
        for _ in range(n_perturb):
            perturbation = self._rng.uniform(0.80, 1.20, size=len(pnls))
            perturbed = pnls * perturbation
            metrics = self._simulate_equity_curve(perturbed, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 4. Data bootstrapping — resample with replacement
        for _ in range(n_bootstrap):
            indices = self._rng.integers(0, len(pnls), size=len(pnls))
            bootstrapped = pnls[indices]
            metrics = self._simulate_equity_curve(bootstrapped, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # Compile results
        sharpes = np.array(all_sharpes)
        returns = np.array(all_returns)
        drawdowns = np.array(all_drawdowns)

        result = MonteCarloResult(
            n_simulations=n_sims,
            # Sharpe
            median_sharpe=float(np.median(sharpes)),
            mean_sharpe=float(np.mean(sharpes)),
            sharpe_5th_pct=float(np.percentile(sharpes, 5)),
            sharpe_25th_pct=float(np.percentile(sharpes, 25)),
            sharpe_75th_pct=float(np.percentile(sharpes, 75)),
            sharpe_95th_pct=float(np.percentile(sharpes, 95)),
            # Returns
            median_return=float(np.median(returns)),
            mean_return=float(np.mean(returns)),
            return_5th_pct=float(np.percentile(returns, 5)),
            return_95th_pct=float(np.percentile(returns, 95)),
            # Drawdowns
            median_max_drawdown=float(np.median(drawdowns)),
            max_drawdown_95th_pct=float(np.percentile(drawdowns, 95)),
            mean_max_drawdown=float(np.mean(drawdowns)),
            # Probabilities
            probability_of_ruin=float(np.mean(returns < -(1.0 - self._ruin_threshold))),
            probability_of_loss=float(np.mean(returns < 0)),
            probability_of_target=float(np.mean(returns > self._target_return)),
            # Raw data
            sharpe_distribution=sharpes.tolist(),
            return_distribution=returns.tolist(),
            drawdown_distribution=drawdowns.tolist(),
        )

        logger.info(
            "Monte Carlo complete: median Sharpe=%.2f, 5th pct Sharpe=%.2f, "
            "P(ruin)=%.1f%%, P(loss)=%.1f%%",
            result.median_sharpe, result.sharpe_5th_pct,
            result.probability_of_ruin * 100, result.probability_of_loss * 100,
        )

        return result

    # ------------------------------------------------------------------ #
    #  Internal simulation
    # ------------------------------------------------------------------ #

    def _simulate_equity_curve(
        self,
        pnls: np.ndarray,
        initial_capital: float,
    ) -> dict[str, float]:
        """Simulate an equity curve from a PnL sequence and compute metrics.

        Returns dict with: sharpe, total_return, max_drawdown.
        """
        equity = np.empty(len(pnls) + 1)
        equity[0] = initial_capital
        for i, pnl in enumerate(pnls):
            equity[i + 1] = equity[i] + pnl

        # Total return
        total_return = (equity[-1] - initial_capital) / initial_capital

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
        max_drawdown = float(np.max(drawdown))

        # Sharpe (treat each PnL as a "daily" return for simplicity)
        if len(pnls) > 1 and initial_capital > 0:
            returns = pnls / initial_capital
            daily_rf = self._rf / 252
            excess = returns - daily_rf
            std = float(np.std(excess, ddof=1))
            if std > 0:
                sharpe = float(np.mean(excess) / std * np.sqrt(252))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
        }

    def __repr__(self) -> str:
        return f"MonteCarloTester(n_simulations={self._n_sims})"
