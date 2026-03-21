"""COMP-015: Synthetic crash, squeeze, and flash crash scenario generation.

Generates realistic stress-test scenarios for backtesting portfolio
resilience.  Leverages diffusion models (EDGE-008) if available, otherwise
falls back to parametric methods (GARCH shocks, jump-diffusion, and
regime-conditioned simulation).

Scenario types:
    - **Crash**: Sustained drawdown over multiple days (e.g. 2008, 2020).
    - **Short squeeze**: Rapid upward price move with momentum exhaustion.
    - **Flash crash**: Intraday liquidity vacuum — sharp drop and recovery.

Usage:
    gen = SyntheticScenarioGenerator()
    crash = gen.generate_crash(severity=0.30, duration=20)
    squeeze = gen.generate_squeeze(magnitude=0.50)
    flash = gen.generate_flash_crash()
    combined = gen.generate_stress_battery()

Dependencies: numpy, pandas (required). torch, diffusion model (optional).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: diffusion model backend (EDGE-008)
# ---------------------------------------------------------------------------

_HAS_DIFFUSION = False
try:
    from ml.synthetic_data import SyntheticDataGenerator

    _HAS_DIFFUSION = True  # We can at least use GARCH-based generation
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Parametric scenario generators
# ---------------------------------------------------------------------------


def _generate_garch_path(
    n_steps: int,
    mu: float = 0.0,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a GARCH(1,1) return path.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    mu : float
        Drift.
    omega, alpha, beta : float
        GARCH parameters.

    Returns
    -------
    np.ndarray
        (n_steps,) array of returns.
    """
    rng = np.random.RandomState(seed)
    returns = np.zeros(n_steps)
    var = omega / max(1.0 - alpha - beta, 0.01)

    for t in range(n_steps):
        returns[t] = mu + math.sqrt(max(var, 1e-10)) * rng.standard_normal()
        var = omega + alpha * returns[t] ** 2 + beta * var

    return returns


def _generate_jump_diffusion(
    n_steps: int,
    mu: float = 0.0,
    sigma: float = 0.01,
    jump_intensity: float = 0.05,
    jump_mean: float = -0.05,
    jump_std: float = 0.03,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a Merton jump-diffusion return path.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    mu : float
        Drift.
    sigma : float
        Diffusion volatility.
    jump_intensity : float
        Probability of jump per step (Poisson intensity).
    jump_mean : float
        Mean jump size.
    jump_std : float
        Jump size standard deviation.

    Returns
    -------
    np.ndarray
        (n_steps,) array of returns.
    """
    rng = np.random.RandomState(seed)
    diffusion = mu + sigma * rng.standard_normal(n_steps)

    # Poisson jumps
    n_jumps = rng.poisson(jump_intensity, n_steps)
    jumps = np.zeros(n_steps)
    for t in range(n_steps):
        if n_jumps[t] > 0:
            jumps[t] = sum(
                rng.normal(jump_mean, jump_std)
                for _ in range(n_jumps[t])
            )

    return diffusion + jumps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


class SyntheticScenarioGenerator:
    """Generate synthetic stress-test scenarios for backtesting.

    Parameters
    ----------
    base_volatility : float
        Annualised volatility of the baseline asset.  Default 0.20 (20%).
    trading_days : int
        Number of trading days per year.  Default 252.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_volatility: float = 0.20,
        trading_days: int = 252,
        seed: int = 42,
    ) -> None:
        self.base_vol = base_volatility
        self.daily_vol = base_volatility / math.sqrt(trading_days)
        self.trading_days = trading_days
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        logger.info(
            "SyntheticScenarioGenerator: base_vol=%.2f, daily_vol=%.4f",
            base_volatility, self.daily_vol,
        )

    def generate_crash(
        self,
        severity: float = 0.30,
        duration: int = 20,
        recovery_days: int = 10,
        n_assets: int = 1,
        correlation_spike: float = 0.9,
    ) -> pd.DataFrame:
        """Generate a crash scenario with sustained drawdown.

        Parameters
        ----------
        severity : float
            Total drawdown magnitude (e.g. 0.30 = -30%).
        duration : int
            Number of days for the crash phase.
        recovery_days : int
            Days of partial recovery after crash.
        n_assets : int
            Number of correlated assets.
        correlation_spike : float
            Correlation between assets during crash (correlation spikes).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns per asset, indexed by day number.
            Values are daily returns.
        """
        try:
            total_days = duration + recovery_days

            # Crash phase: increasingly negative returns with vol spike
            crash_daily_return = -severity / duration
            crash_vol = self.daily_vol * 3.0  # vol triples during crash

            crash_returns = np.zeros((duration, n_assets))
            for d in range(duration):
                # Accelerating crash pattern
                progress = d / max(duration - 1, 1)
                accel = 1.0 + progress  # accelerates into crash
                base = crash_daily_return * accel
                noise = self._rng.multivariate_normal(
                    mean=np.zeros(n_assets),
                    cov=self._correlated_cov(n_assets, crash_vol, correlation_spike),
                )
                crash_returns[d] = base + noise

            # Scale to hit target severity
            actual_dd = np.exp(crash_returns.sum(axis=0)).min() - 1.0
            if actual_dd != 0:
                scale = severity / abs(actual_dd)
                crash_returns[:, 0] *= min(scale, 2.0)

            # Recovery phase: mean-reverting positive returns
            recovery_returns = np.zeros((recovery_days, n_assets))
            recovery_vol = self.daily_vol * 2.0
            for d in range(recovery_days):
                progress = d / max(recovery_days - 1, 1)
                base = severity * 0.3 / recovery_days * (1.0 - progress)
                noise = self._rng.normal(0, recovery_vol, n_assets)
                recovery_returns[d] = base + noise

            all_returns = np.vstack([crash_returns, recovery_returns])
            columns = [f"asset_{i}" for i in range(n_assets)]
            df = pd.DataFrame(all_returns, columns=columns)
            df.index.name = "day"

            logger.info(
                "Generated crash: severity=%.1f%%, duration=%d, "
                "total_dd=%.1f%%, recovery_days=%d",
                severity * 100, duration,
                (np.exp(crash_returns.sum(axis=0)[0]) - 1) * 100,
                recovery_days,
            )
            return df

        except Exception as e:
            logger.error("Crash generation failed: %s — returning flat", e)
            return pd.DataFrame(
                np.zeros((duration + recovery_days, n_assets)),
                columns=[f"asset_{i}" for i in range(n_assets)],
            )

    def generate_squeeze(
        self,
        magnitude: float = 0.50,
        ramp_days: int = 5,
        peak_days: int = 3,
        collapse_days: int = 7,
        n_assets: int = 1,
    ) -> pd.DataFrame:
        """Generate a short squeeze scenario.

        Parameters
        ----------
        magnitude : float
            Total upward move during squeeze (e.g. 0.50 = +50%).
        ramp_days : int
            Days of accelerating upward pressure.
        peak_days : int
            Days at peak volatility.
        collapse_days : int
            Days of reversal / momentum exhaustion.

        Returns
        -------
        pd.DataFrame
            Daily returns.
        """
        try:
            total_days = ramp_days + peak_days + collapse_days
            all_returns = np.zeros((total_days, n_assets))

            # Phase 1: Ramp — accelerating positive returns
            ramp_vol = self.daily_vol * 2.0
            daily_gain = magnitude * 0.6 / max(ramp_days, 1)
            for d in range(ramp_days):
                accel = (d + 1) / ramp_days
                base = daily_gain * accel
                all_returns[d] = base + self._rng.normal(0, ramp_vol, n_assets)

            # Phase 2: Peak — extreme vol, continued gains
            peak_vol = self.daily_vol * 5.0
            daily_gain_peak = magnitude * 0.3 / max(peak_days, 1)
            for d in range(peak_days):
                offset = ramp_days + d
                all_returns[offset] = daily_gain_peak + self._rng.normal(0, peak_vol, n_assets)

            # Phase 3: Collapse — reversal
            collapse_vol = self.daily_vol * 3.0
            daily_loss = -magnitude * 0.4 / max(collapse_days, 1)
            for d in range(collapse_days):
                offset = ramp_days + peak_days + d
                decay = 1.0 - d / max(collapse_days - 1, 1)
                all_returns[offset] = daily_loss * decay + self._rng.normal(0, collapse_vol, n_assets)

            columns = [f"asset_{i}" for i in range(n_assets)]
            df = pd.DataFrame(all_returns, columns=columns)
            df.index.name = "day"

            logger.info(
                "Generated squeeze: magnitude=%.1f%%, total_days=%d",
                magnitude * 100, total_days,
            )
            return df

        except Exception as e:
            logger.error("Squeeze generation failed: %s — returning flat", e)
            total = ramp_days + peak_days + collapse_days
            return pd.DataFrame(
                np.zeros((total, n_assets)),
                columns=[f"asset_{i}" for i in range(n_assets)],
            )

    def generate_flash_crash(
        self,
        drop_magnitude: float = 0.08,
        drop_minutes: int = 15,
        recovery_minutes: int = 45,
        tick_interval_seconds: int = 60,
        n_assets: int = 1,
    ) -> pd.DataFrame:
        """Generate a flash crash scenario (intraday resolution).

        Models a sudden liquidity vacuum: sharp drop followed by recovery.

        Parameters
        ----------
        drop_magnitude : float
            Intraday drop magnitude (e.g. 0.08 = -8%).
        drop_minutes : int
            Duration of the drop phase in minutes.
        recovery_minutes : int
            Duration of the recovery in minutes.
        tick_interval_seconds : int
            Seconds between ticks.

        Returns
        -------
        pd.DataFrame
            Per-tick returns (not daily).  Index is tick number.
        """
        try:
            drop_ticks = max(1, drop_minutes * 60 // tick_interval_seconds)
            recovery_ticks = max(1, recovery_minutes * 60 // tick_interval_seconds)
            total_ticks = drop_ticks + recovery_ticks

            returns = np.zeros((total_ticks, n_assets))
            intraday_vol = self.daily_vol / math.sqrt(390 / (tick_interval_seconds / 60))

            # Drop phase: exponential decay pattern
            for t in range(drop_ticks):
                progress = (t + 1) / drop_ticks
                # Most of the drop happens in the last third
                intensity = progress ** 2
                base = -drop_magnitude * intensity / drop_ticks * 3.0
                returns[t] = base + self._rng.normal(0, intraday_vol * 5.0, n_assets)

            # Scale drop to hit target
            drop_total = returns[:drop_ticks, 0].sum()
            if abs(drop_total) > 0:
                scale = drop_magnitude / abs(drop_total)
                returns[:drop_ticks] *= min(scale, 3.0)

            # Recovery phase: exponential recovery (fast then slow)
            actual_drop = abs(returns[:drop_ticks, 0].sum())
            for t in range(recovery_ticks):
                progress = (t + 1) / recovery_ticks
                # Recovery: 80% of drop, fast at first
                recovery_rate = 0.8 * actual_drop / recovery_ticks * (1.0 - progress * 0.5)
                returns[drop_ticks + t] = recovery_rate + self._rng.normal(
                    0, intraday_vol * 3.0, n_assets,
                )

            columns = [f"asset_{i}" for i in range(n_assets)]
            df = pd.DataFrame(returns, columns=columns)
            df.index.name = "tick"

            logger.info(
                "Generated flash crash: drop=%.1f%%, drop_ticks=%d, recovery_ticks=%d",
                drop_magnitude * 100, drop_ticks, recovery_ticks,
            )
            return df

        except Exception as e:
            logger.error("Flash crash generation failed: %s — returning flat", e)
            total = max(1, drop_minutes + recovery_minutes)
            return pd.DataFrame(
                np.zeros((total, n_assets)),
                columns=[f"asset_{i}" for i in range(n_assets)],
            )

    def generate_stress_battery(
        self,
        n_assets: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        """Generate a full battery of stress scenarios.

        Returns
        -------
        dict
            Mapping of scenario name to DataFrame of returns.
        """
        scenarios = {}

        try:
            scenarios["mild_crash"] = self.generate_crash(
                severity=0.10, duration=10, recovery_days=5, n_assets=n_assets,
            )
            scenarios["moderate_crash"] = self.generate_crash(
                severity=0.25, duration=20, recovery_days=15, n_assets=n_assets,
            )
            scenarios["severe_crash"] = self.generate_crash(
                severity=0.45, duration=40, recovery_days=30, n_assets=n_assets,
            )
            scenarios["short_squeeze"] = self.generate_squeeze(
                magnitude=0.30, n_assets=n_assets,
            )
            scenarios["extreme_squeeze"] = self.generate_squeeze(
                magnitude=0.80, n_assets=n_assets,
            )
            scenarios["flash_crash"] = self.generate_flash_crash(
                drop_magnitude=0.06, n_assets=n_assets,
            )
            scenarios["severe_flash_crash"] = self.generate_flash_crash(
                drop_magnitude=0.12, n_assets=n_assets,
            )
        except Exception as e:
            logger.error("Stress battery generation failed: %s", e)

        logger.info("Generated %d stress scenarios.", len(scenarios))
        return scenarios

    def _correlated_cov(
        self,
        n: int,
        vol: float,
        correlation: float,
    ) -> np.ndarray:
        """Build a covariance matrix with uniform correlation."""
        cov = np.full((n, n), correlation * vol ** 2)
        np.fill_diagonal(cov, vol ** 2)
        return cov

    def inject_scenario(
        self,
        base_returns: pd.DataFrame,
        scenario: pd.DataFrame,
        injection_point: int,
    ) -> pd.DataFrame:
        """Inject a scenario into a base return series.

        Parameters
        ----------
        base_returns : pd.DataFrame
            Original return series.
        scenario : pd.DataFrame
            Scenario returns to inject.
        injection_point : int
            Index at which to inject the scenario.

        Returns
        -------
        pd.DataFrame
            Combined return series with scenario injected.
        """
        try:
            before = base_returns.iloc[:injection_point]
            after = base_returns.iloc[injection_point + len(scenario):]

            # Align columns
            common_cols = list(set(base_returns.columns) & set(scenario.columns))
            if not common_cols:
                common_cols = base_returns.columns[:scenario.shape[1]].tolist()
                scenario.columns = common_cols

            result = pd.concat(
                [before[common_cols], scenario[common_cols], after[common_cols]],
                ignore_index=True,
            )
            logger.info(
                "Injected scenario at day %d: %d scenario days into %d base days",
                injection_point, len(scenario), len(base_returns),
            )
            return result

        except Exception as e:
            logger.error("Scenario injection failed: %s — returning base", e)
            return base_returns
