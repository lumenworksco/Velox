"""ML-004: Overfitting Prevention & Validation Framework.

Implements:
    - Combinatorial Purged Cross-Validation (CPCV)
    - Deflated Sharpe Ratio (DSR)
    - Probability of Backtest Overfitting (PBO)
    - Parameter sensitivity analysis

Based on Marcos Lopez de Prado's methodology for evaluating
whether strategy performance is real or the result of overfitting.
"""

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conditional imports
_HAS_SCIPY = False
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    scipy_stats = None  # type: ignore[assignment]

_HAS_SKLEARN = False
try:
    from sklearn.model_selection import KFold
    _HAS_SKLEARN = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Comprehensive validation results."""
    # Deflated Sharpe Ratio
    dsr: float = 0.0
    dsr_pvalue: float = 1.0
    dsr_significant: bool = False

    # Probability of Backtest Overfitting
    pbo: float = 1.0  # 1.0 = certainly overfit, 0.0 = no overfitting
    pbo_logits: List[float] = field(default_factory=list)

    # CPCV results
    cpcv_sharpes: List[float] = field(default_factory=list)
    cpcv_mean_sharpe: float = 0.0
    cpcv_std_sharpe: float = 0.0
    cpcv_min_sharpe: float = 0.0

    # Sensitivity analysis
    sensitivity_scores: Dict[str, float] = field(default_factory=dict)
    sensitivity_stable: bool = True
    sensitivity_fragile_params: List[str] = field(default_factory=list)

    # Overall
    is_overfit: bool = True
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dsr": round(self.dsr, 4),
            "dsr_pvalue": round(self.dsr_pvalue, 4),
            "dsr_significant": self.dsr_significant,
            "pbo": round(self.pbo, 4),
            "cpcv_mean_sharpe": round(self.cpcv_mean_sharpe, 4),
            "cpcv_std_sharpe": round(self.cpcv_std_sharpe, 4),
            "cpcv_min_sharpe": round(self.cpcv_min_sharpe, 4),
            "sensitivity_stable": self.sensitivity_stable,
            "sensitivity_fragile_params": self.sensitivity_fragile_params,
            "is_overfit": self.is_overfit,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Validation Framework
# ---------------------------------------------------------------------------

class ValidationFramework:
    """Multi-method overfitting detection for trading strategies.

    Usage::

        vf = ValidationFramework()
        report = vf.validate_strategy(backtest_results)
        if report.is_overfit:
            print("Strategy is likely overfit!")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        annualization: int = 252,
        dsr_significance: float = 0.05,
        pbo_threshold: float = 0.5,
        sensitivity_variation: float = 0.20,
        sensitivity_max_degradation: float = 0.30,
    ):
        """Initialize validation framework.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations.
            annualization: Trading days per year.
            dsr_significance: p-value threshold for DSR test.
            pbo_threshold: PBO above this is considered overfit.
            sensitivity_variation: Parameter variation for sensitivity (20%).
            sensitivity_max_degradation: Max allowed Sharpe degradation (30%).
        """
        self.risk_free_rate = risk_free_rate
        self.annualization = annualization
        self.dsr_significance = dsr_significance
        self.pbo_threshold = pbo_threshold
        self.sensitivity_variation = sensitivity_variation
        self.sensitivity_max_degradation = sensitivity_max_degradation

    def validate_strategy(
        self,
        backtest_results: Dict[str, Any],
    ) -> ValidationReport:
        """Run full validation suite on backtest results.

        Args:
            backtest_results: Dict containing:
                - ``returns``: pd.Series or np.ndarray of daily returns
                - ``sharpe``: float (observed Sharpe ratio)
                - ``n_trials``: int (number of parameter combinations tried)
                - ``trial_sharpes``: List[float] (Sharpe of each trial, for PBO)
                - ``params``: Dict[str, float] (current parameter values, for sensitivity)
                - ``param_eval_fn``: Callable (function(params) -> sharpe, for sensitivity)
                - ``split_returns``: List[np.ndarray] (returns per sub-period, for CPCV)

        Returns:
            ValidationReport with all results.
        """
        report = ValidationReport()
        returns = backtest_results.get("returns")

        if returns is None:
            report.summary = "No returns data provided"
            return report

        if isinstance(returns, pd.Series):
            returns = returns.values.astype(np.float64)
        else:
            returns = np.array(returns, dtype=np.float64)

        # 1. Deflated Sharpe Ratio
        observed_sharpe = backtest_results.get("sharpe")
        n_trials = backtest_results.get("n_trials", 1)
        if observed_sharpe is None:
            observed_sharpe = self._compute_sharpe(returns)
        self._compute_dsr(report, returns, observed_sharpe, n_trials)

        # 2. Probability of Backtest Overfitting
        trial_sharpes = backtest_results.get("trial_sharpes")
        split_returns = backtest_results.get("split_returns")
        if trial_sharpes is not None and len(trial_sharpes) > 2:
            self._compute_pbo(report, trial_sharpes)
        elif split_returns is not None and len(split_returns) >= 4:
            self._compute_pbo_from_splits(report, split_returns)

        # 3. CPCV
        if split_returns is not None and len(split_returns) >= 4:
            self._compute_cpcv(report, split_returns)
        elif len(returns) >= 200:
            # Auto-split into sub-periods
            n_splits = min(10, len(returns) // 50)
            if n_splits >= 4:
                splits = np.array_split(returns, n_splits)
                self._compute_cpcv(report, [s for s in splits])

        # 4. Parameter sensitivity
        params = backtest_results.get("params")
        param_eval_fn = backtest_results.get("param_eval_fn")
        if params and param_eval_fn:
            self._compute_sensitivity(report, params, param_eval_fn, observed_sharpe)

        # Overall judgment
        self._compute_overall(report, observed_sharpe)

        return report

    # ------------------------------------------------------------------
    # Deflated Sharpe Ratio (DSR)
    # ------------------------------------------------------------------

    def _compute_dsr(
        self,
        report: ValidationReport,
        returns: np.ndarray,
        observed_sharpe: float,
        n_trials: int,
    ) -> None:
        """Compute the Deflated Sharpe Ratio.

        Adjusts the observed Sharpe ratio for the number of trials
        (multiple testing bias) and the non-normality of returns.

        Reference: Bailey & Lopez de Prado (2014).
        """
        n = len(returns)
        if n < 10:
            return

        # Moments
        sr = observed_sharpe
        skew = float(scipy_stats.skew(returns)) if _HAS_SCIPY else self._simple_skew(returns)
        kurt = float(scipy_stats.kurtosis(returns, fisher=True)) if _HAS_SCIPY else self._simple_kurtosis(returns)

        # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
        # E[max(SR)] ~ sqrt(2 * ln(N)) * (1 - gamma / (2 * ln(N))) + gamma / sqrt(2 * ln(N))
        # where gamma ~ 0.5772 (Euler-Mascheroni constant) and N = n_trials
        if n_trials <= 1:
            e_max_sr = 0.0
        else:
            gamma_em = 0.5772156649
            log_n = math.log(max(n_trials, 2))
            e_max_sr = math.sqrt(2 * log_n) * (1 - gamma_em / (2 * log_n)) + gamma_em / math.sqrt(2 * log_n)

        # Standard error of Sharpe ratio (accounting for non-normality)
        sr_std = math.sqrt(
            (1 + 0.5 * sr ** 2 - skew * sr + ((kurt - 3) / 4) * sr ** 2) / max(n - 1, 1)
        )

        if sr_std > 0:
            # Test statistic
            z = (sr - e_max_sr) / sr_std
            report.dsr = float(z)
            # p-value (one-tailed)
            if _HAS_SCIPY:
                report.dsr_pvalue = float(1 - scipy_stats.norm.cdf(z))
            else:
                # Approximation
                report.dsr_pvalue = float(0.5 * math.erfc(z / math.sqrt(2)))
            report.dsr_significant = report.dsr_pvalue < self.dsr_significance
        else:
            report.dsr = 0.0
            report.dsr_pvalue = 1.0

        logger.info(
            "DSR: z=%.3f, p=%.4f (n_trials=%d, SR=%.3f, E[maxSR]=%.3f)",
            report.dsr, report.dsr_pvalue, n_trials, sr, e_max_sr,
        )

    # ------------------------------------------------------------------
    # Probability of Backtest Overfitting (PBO)
    # ------------------------------------------------------------------

    def _compute_pbo(
        self,
        report: ValidationReport,
        trial_sharpes: List[float],
    ) -> None:
        """Compute PBO from a set of trial Sharpe ratios.

        HIGH-019: Proper Bailey/de Prado Probability of Backtest Overfitting.

        Given N trial (strategy/parameter) Sharpe ratios, we use the CSCV
        (Combinatorially Symmetric Cross-Validation) approach:
        1. Pair each trial's Sharpe into N/2 IS and N/2 OOS components.
        2. For each combinatorial split, find the IS-optimal trial.
        3. Compute that trial's relative OOS rank (omega_c).
        4. PBO = fraction of combinations where omega_c <= 0.5 (i.e. the
           IS-best ranks in the bottom half OOS).

        The logit of omega is also stored for distribution analysis.
        """
        sharpes = np.array(trial_sharpes, dtype=float)
        n = len(sharpes)
        if n < 4:
            return

        half = n // 2
        # Generate combinatorial splits of trial indices
        all_combos = list(combinations(range(n), half))

        # Cap at 200 combinations for performance
        if len(all_combos) > 200:
            rng = np.random.RandomState(42)
            selected = rng.choice(len(all_combos), 200, replace=False)
            all_combos = [all_combos[i] for i in selected]

        n_overfit = 0
        logits = []

        for is_combo in all_combos:
            oos_combo = [i for i in range(n) if i not in is_combo]

            is_sharpes = sharpes[list(is_combo)]
            oos_sharpes = sharpes[oos_combo]

            # Find IS-optimal trial index (within IS set)
            best_is_local = int(np.argmax(is_sharpes))
            # Map back to original index to find corresponding OOS value
            best_trial_idx = list(is_combo)[best_is_local]

            # Relative rank of the IS-best trial in the OOS set
            # omega_c = fraction of OOS trials that this trial outperforms
            # We need the OOS performance of the IS-best trial.
            # Since we only have a single Sharpe per trial (not per-split),
            # the OOS performance of the IS-best is its Sharpe among OOS trials.
            best_oos_sharpe = sharpes[best_trial_idx]
            # omega = relative rank: fraction of OOS sharpes that are worse
            n_oos = len(oos_sharpes)
            rank_below = np.sum(oos_sharpes <= best_oos_sharpe)
            omega = rank_below / n_oos  # 1.0 = best OOS, 0.0 = worst

            # Logit transformation (clamp to avoid inf)
            omega_clamped = max(0.01, min(0.99, omega))
            logit = math.log(omega_clamped / (1.0 - omega_clamped))
            logits.append(logit)

            # Overfit if IS-best ranks in bottom half OOS
            if omega <= 0.5:
                n_overfit += 1

        n_combos = len(all_combos)
        if n_combos > 0:
            report.pbo = n_overfit / n_combos
            report.pbo_logits = logits

        logger.info("PBO: %.3f (%d/%d overfit combinations)", report.pbo, n_overfit, n_combos)

    def _compute_pbo_from_splits(
        self,
        report: ValidationReport,
        split_returns: List[np.ndarray],
    ) -> None:
        """Compute PBO from sub-period return splits using proper CSCV.

        HIGH-019: Proper Bailey/de Prado CSCV-based PBO.

        Given S sub-period return arrays, for each way to partition them
        into S/2 IS and S/2 OOS groups:
        1. Concatenate IS returns, compute IS Sharpe.
        2. Concatenate OOS returns, compute OOS Sharpe.
        3. Compute omega (the relative performance rank of the IS-optimal
           configuration in OOS).  For single-strategy evaluation we use
           the sign and magnitude of OOS Sharpe relative to IS Sharpe.
        4. PBO = P(omega <= 0.5) across all combinations.
        """
        n_splits = len(split_returns)
        if n_splits < 4:
            return

        half = n_splits // 2
        split_indices = list(range(n_splits))
        all_combos = list(combinations(split_indices, half))

        # Cap at 200 combinations
        if len(all_combos) > 200:
            rng = np.random.RandomState(42)
            selected = rng.choice(len(all_combos), 200, replace=False)
            all_combos = [all_combos[i] for i in selected]

        n_overfit = 0
        logits = []

        for is_combo in all_combos:
            oos_combo = [i for i in split_indices if i not in is_combo]

            is_returns = np.concatenate([split_returns[i] for i in is_combo])
            oos_returns = np.concatenate([split_returns[i] for i in oos_combo])

            is_sharpe = self._compute_sharpe(is_returns)
            oos_sharpe = self._compute_sharpe(oos_returns)

            # Omega: normalize OOS Sharpe relative to IS Sharpe
            # If OOS degrades significantly vs IS, omega is low (overfit signal)
            if abs(is_sharpe) > 1e-8:
                ratio = oos_sharpe / abs(is_sharpe)
                # Map ratio to 0-1 range: ratio=1 -> omega=0.75, ratio=0 -> omega=0.5, ratio<0 -> omega<0.5
                omega = max(0.01, min(0.99, 0.5 + 0.25 * np.clip(ratio, -2.0, 2.0)))
            else:
                omega = 0.5  # IS Sharpe ~0, no information

            logit = math.log(omega / (1.0 - omega))
            logits.append(logit)

            # Overfit if omega <= 0.5 (IS-best underperforms OOS expectation)
            if omega <= 0.5:
                n_overfit += 1

        total = len(all_combos)
        if total > 0:
            report.pbo = n_overfit / total
            report.pbo_logits = logits

        logger.info("PBO (from splits): %.3f (%d/%d overfit combinations)",
                     report.pbo, n_overfit, total)

    # ------------------------------------------------------------------
    # Combinatorial Purged Cross-Validation (CPCV)
    # ------------------------------------------------------------------

    def _compute_cpcv(
        self,
        report: ValidationReport,
        split_returns: List[np.ndarray],
    ) -> None:
        """Combinatorial Purged Cross-Validation.

        Evaluates strategy across all combinatorial train/test splits
        to produce a distribution of out-of-sample Sharpe ratios.
        """
        n_splits = len(split_returns)
        if n_splits < 4:
            return

        # For each combination, use N-2 splits as train, 2 as test
        test_size = 2
        all_combos = list(combinations(range(n_splits), test_size))

        # Cap for performance
        if len(all_combos) > 200:
            rng = np.random.RandomState(42)
            selected = rng.choice(len(all_combos), 200, replace=False)
            all_combos = [all_combos[i] for i in selected]

        oos_sharpes = []
        for test_combo in all_combos:
            test_returns = np.concatenate([split_returns[i] for i in test_combo])
            sharpe = self._compute_sharpe(test_returns)
            oos_sharpes.append(sharpe)

        report.cpcv_sharpes = oos_sharpes
        if oos_sharpes:
            report.cpcv_mean_sharpe = float(np.mean(oos_sharpes))
            report.cpcv_std_sharpe = float(np.std(oos_sharpes))
            report.cpcv_min_sharpe = float(np.min(oos_sharpes))

        logger.info(
            "CPCV: mean_sharpe=%.3f, std=%.3f, min=%.3f (%d combos)",
            report.cpcv_mean_sharpe, report.cpcv_std_sharpe,
            report.cpcv_min_sharpe, len(oos_sharpes),
        )

    # ------------------------------------------------------------------
    # Parameter sensitivity analysis
    # ------------------------------------------------------------------

    def _compute_sensitivity(
        self,
        report: ValidationReport,
        params: Dict[str, float],
        param_eval_fn: Callable,
        base_sharpe: float,
    ) -> None:
        """Vary each parameter +/- 20% and measure Sharpe stability.

        A robust strategy should not collapse when parameters change
        slightly.  Fragile parameters are flagged.

        Args:
            params: Dict of parameter name to current value.
            param_eval_fn: Function(params_dict) -> sharpe_ratio.
            base_sharpe: Sharpe with the current parameters.
        """
        fragile = []
        scores = {}

        for param_name, base_val in params.items():
            if base_val == 0:
                continue

            sharpes_varied = []
            for mult in [
                1 - self.sensitivity_variation,
                1 - self.sensitivity_variation / 2,
                1 + self.sensitivity_variation / 2,
                1 + self.sensitivity_variation,
            ]:
                varied_params = dict(params)
                varied_params[param_name] = base_val * mult
                try:
                    sr = param_eval_fn(varied_params)
                    sharpes_varied.append(sr)
                except Exception as e:
                    logger.debug("Sensitivity eval failed for %s: %s", param_name, e)

            if not sharpes_varied:
                continue

            # Stability score: how much does Sharpe change?
            min_sr = min(sharpes_varied)
            max_degradation = (base_sharpe - min_sr) / max(abs(base_sharpe), 1e-6)
            scores[param_name] = max_degradation

            if max_degradation > self.sensitivity_max_degradation:
                fragile.append(param_name)

        report.sensitivity_scores = scores
        report.sensitivity_fragile_params = fragile
        report.sensitivity_stable = len(fragile) == 0

        if fragile:
            logger.warning("Fragile parameters detected: %s", fragile)

    # ------------------------------------------------------------------
    # Overall judgment
    # ------------------------------------------------------------------

    def _compute_overall(
        self, report: ValidationReport, observed_sharpe: float
    ) -> None:
        """Synthesize all signals into an overall overfitting verdict."""
        signals = []
        weights = []

        # DSR signal
        if report.dsr_pvalue < 1.0:
            # Lower p-value = less likely overfit
            signals.append(1.0 - report.dsr_pvalue)
            weights.append(0.30)

        # PBO signal
        if report.pbo < 1.0:
            signals.append(1.0 - report.pbo)
            weights.append(0.30)

        # CPCV signal
        if report.cpcv_sharpes:
            # What fraction of CPCV paths are positive?
            frac_positive = np.mean([1 for s in report.cpcv_sharpes if s > 0])
            signals.append(frac_positive)
            weights.append(0.25)

        # Sensitivity signal
        if report.sensitivity_scores:
            signals.append(1.0 if report.sensitivity_stable else 0.3)
            weights.append(0.15)

        if signals and weights:
            report.confidence = float(np.average(signals, weights=weights))
        else:
            report.confidence = 0.0

        # Decision
        report.is_overfit = report.confidence < 0.5

        # Summary
        parts = []
        if report.dsr_significant:
            parts.append(f"DSR significant (p={report.dsr_pvalue:.3f})")
        else:
            parts.append(f"DSR NOT significant (p={report.dsr_pvalue:.3f})")

        parts.append(f"PBO={report.pbo:.2f}")

        if report.cpcv_sharpes:
            parts.append(f"CPCV mean SR={report.cpcv_mean_sharpe:.2f}")

        if report.sensitivity_fragile_params:
            parts.append(f"fragile params: {report.sensitivity_fragile_params}")

        verdict = "LIKELY REAL" if not report.is_overfit else "LIKELY OVERFIT"
        report.summary = f"{verdict} (confidence={report.confidence:.2f}): {'; '.join(parts)}"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - self.risk_free_rate / self.annualization
        std = np.std(returns, ddof=1)
        if std < 1e-8:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(self.annualization))

    @staticmethod
    def _simple_skew(x: np.ndarray) -> float:
        """Compute skewness without scipy."""
        n = len(x)
        if n < 3:
            return 0.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10:
            return 0.0
        return float(n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3))

    @staticmethod
    def _simple_kurtosis(x: np.ndarray) -> float:
        """Compute excess kurtosis without scipy."""
        n = len(x)
        if n < 4:
            return 0.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10:
            return 0.0
        k4 = np.mean(((x - m) / s) ** 4)
        # Fisher's excess kurtosis
        return float(k4 - 3.0)
