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
        strategy_params: Optional[dict[str, dict[str, float]]] = None,
        regime_labels: Optional[list[str]] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulations on a set of historical trades.

        Each trade dict must have at minimum: ``pnl`` (float).
        Optional keys: ``pnl_pct``, ``hold_minutes``, ``strategy``, ``regime``.

        Runs six types of simulations:
        1. Trade shuffling (16% of sims) — random reordering
        2. Trade skipping (16%) — randomly remove 10-30% of trades
        3. Parameter perturbation (17%) — vary strategy parameters by +/-20%
        4. Block bootstrap (17%) — resample contiguous blocks of trades
        5. Regime randomization (17%) — shuffle regime labels to test robustness
        6. Combined perturbation (17%) — parameter noise + block bootstrap

        IMPL-002 enhancements:
        - Parameter perturbation uses per-strategy param dictionaries with +/-20% noise
        - Block bootstrap preserves temporal autocorrelation in trade sequences
        - Regime randomization shuffles regime labels to test regime-dependent performance

        Args:
            trades: List of trade dicts, each containing at least ``pnl``.
            initial_capital: Starting capital for each simulation.
            n_simulations: Override default number of simulations.
            strategy_params: Optional dict of {strategy_name: {param_name: value}}.
                If provided, parameter perturbation varies these params +/-20%
                and scales PnLs by the ratio of perturbed/original param values.
            regime_labels: Optional list of regime labels (one per trade).
                If provided, regime randomization shuffles these labels and
                adjusts PnLs based on regime-strategy performance patterns.

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

        # Allocate simulation types across 6 categories
        n_shuffle = n_sims // 6
        n_skip = n_sims // 6
        n_perturb = n_sims // 6
        n_block_boot = n_sims // 6
        n_regime = n_sims // 6
        n_combined = n_sims - n_shuffle - n_skip - n_perturb - n_block_boot - n_regime

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

        # 3. Parameter perturbation — vary strategy params by +/-20%
        for _ in range(n_perturb):
            perturbed = self._perturb_with_params(pnls, trades, strategy_params)
            metrics = self._simulate_equity_curve(perturbed, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 4. Block bootstrap — resample contiguous blocks preserving autocorrelation
        for _ in range(n_block_boot):
            bootstrapped = self._block_bootstrap(pnls)
            metrics = self._simulate_equity_curve(bootstrapped, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 5. Regime randomization — shuffle regime labels
        for _ in range(n_regime):
            randomized = self._regime_randomize(pnls, trades, regime_labels)
            metrics = self._simulate_equity_curve(randomized, initial_capital)
            all_sharpes.append(metrics["sharpe"])
            all_returns.append(metrics["total_return"])
            all_drawdowns.append(metrics["max_drawdown"])

        # 6. Combined: parameter perturbation + block bootstrap
        for _ in range(n_combined):
            perturbed = self._perturb_with_params(pnls, trades, strategy_params)
            combined = self._block_bootstrap(perturbed)
            metrics = self._simulate_equity_curve(combined, initial_capital)
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
    #  IMPL-002: Parameter perturbation, block bootstrap, regime randomization
    # ------------------------------------------------------------------ #

    def _perturb_with_params(
        self,
        pnls: np.ndarray,
        trades: list[dict],
        strategy_params: Optional[dict[str, dict[str, float]]] = None,
    ) -> np.ndarray:
        """Perturb PnLs based on strategy parameter sensitivity.

        If strategy_params is provided, applies per-strategy multiplicative
        noise that simulates parameter variation of +/-20%. Different strategies
        get different perturbation factors to model the effect of changing
        each strategy's configuration independently.

        If strategy_params is not provided, falls back to uniform +/-20% noise.
        """
        if strategy_params is None or not strategy_params:
            # Fallback: uniform per-trade perturbation +/-20%
            perturbation = self._rng.uniform(0.80, 1.20, size=len(pnls))
            return pnls * perturbation

        # Per-strategy perturbation: each strategy gets a single random
        # scale factor derived from perturbing its parameters
        strategy_scales: dict[str, float] = {}
        for strat_name, params in strategy_params.items():
            if not params:
                strategy_scales[strat_name] = 1.0
                continue
            # Perturb each parameter by +/-20%, then average the scaling effect
            param_scales = []
            for param_name, original_val in params.items():
                if original_val == 0:
                    param_scales.append(1.0)
                    continue
                noise = self._rng.uniform(0.80, 1.20)
                param_scales.append(noise)
            # Geometric mean of parameter perturbations
            strategy_scales[strat_name] = float(np.prod(param_scales) ** (1.0 / len(param_scales)))

        # Apply per-trade scaling based on trade's strategy
        result = pnls.copy()
        for i, trade in enumerate(trades):
            strat = trade.get("strategy", "")
            scale = strategy_scales.get(strat, 1.0)
            # Add small per-trade noise on top of strategy-level perturbation
            trade_noise = self._rng.uniform(0.95, 1.05)
            result[i] = pnls[i] * scale * trade_noise

        return result

    def _block_bootstrap(
        self,
        pnls: np.ndarray,
        block_size: Optional[int] = None,
    ) -> np.ndarray:
        """Block bootstrap: resample contiguous blocks of PnLs.

        Preserves temporal autocorrelation structure (streaks of wins/losses)
        that trade-level shuffling would destroy. Block size defaults to
        sqrt(n_trades), a standard heuristic.

        Args:
            pnls: Array of trade PnLs.
            block_size: Size of each block. Defaults to sqrt(len(pnls)).

        Returns:
            Bootstrapped PnL array of the same length as input.
        """
        n = len(pnls)
        if n <= 1:
            return pnls.copy()

        if block_size is None:
            block_size = max(2, int(np.sqrt(n)))

        # Number of blocks needed to cover the full length
        n_blocks = int(np.ceil(n / block_size))

        # Sample random block start indices
        max_start = n - block_size
        if max_start <= 0:
            # Data too short for blocking; fall back to standard bootstrap
            indices = self._rng.integers(0, n, size=n)
            return pnls[indices]

        starts = self._rng.integers(0, max_start + 1, size=n_blocks)

        # Concatenate blocks and trim to original length
        blocks = []
        for start in starts:
            blocks.append(pnls[start:start + block_size])

        result = np.concatenate(blocks)[:n]
        return result

    def _regime_randomize(
        self,
        pnls: np.ndarray,
        trades: list[dict],
        regime_labels: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Randomize regime labels and adjust PnLs to test robustness.

        If regime_labels are provided, shuffles them and scales each trade's
        PnL based on how well the strategy historically performs in different
        regimes. This tests whether the strategy's alpha is regime-dependent.

        If no regime_labels, falls back to random scaling that simulates
        regime-like clustering: groups of consecutive trades get the same
        random scale factor.
        """
        n = len(pnls)

        if regime_labels is not None and len(regime_labels) == n:
            # Shuffle the regime labels
            shuffled_labels = list(regime_labels)
            self._rng.shuffle(shuffled_labels)

            # Regime multipliers: simulate different performance per regime
            unique_regimes = list(set(regime_labels))
            regime_multipliers = {}
            for regime in unique_regimes:
                # Random multiplier centered at 1.0 with moderate variance
                regime_multipliers[regime] = float(self._rng.uniform(0.6, 1.4))

            result = pnls.copy()
            for i in range(n):
                original_regime = regime_labels[i]
                shuffled_regime = shuffled_labels[i]
                # Scale by ratio of shuffled/original regime multiplier
                orig_mult = regime_multipliers.get(original_regime, 1.0)
                shuf_mult = regime_multipliers.get(shuffled_regime, 1.0)
                if orig_mult != 0:
                    result[i] = pnls[i] * (shuf_mult / orig_mult)
            return result

        # Fallback: simulate regime-like clustering
        # Divide trades into random "regime blocks" of 5-15 trades
        result = pnls.copy()
        i = 0
        while i < n:
            block_len = min(int(self._rng.integers(5, 16)), n - i)
            regime_scale = float(self._rng.uniform(0.6, 1.4))
            result[i:i + block_len] = pnls[i:i + block_len] * regime_scale
            i += block_len

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
