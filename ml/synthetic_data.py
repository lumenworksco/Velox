"""ADVML-004: Synthetic Data Generation for Trading Strategy Validation.

Generates realistic synthetic financial time series that preserve the
stylized facts of asset returns:

    1. **Fat tails** — excess kurtosis beyond the Gaussian
    2. **Volatility clustering** — GARCH(1,1) conditional heteroskedasticity
    3. **Leverage effect** — negative correlation between returns and vol
    4. **Autocorrelation of squared returns** — persistent vol memory
    5. **Near-zero linear autocorrelation** — weak predictability

Generation methods:
    - Block bootstrap (preserves temporal structure)
    - GARCH(1,1) parametric simulation (GJR-GARCH with Student-t innovations)
    - Regime-specific generation (bull / bear / sideways / crisis / low_vol,
      plus HMM-compatible regimes: low_vol_bull, high_vol_bull, etc.)
    - Crash scenario injection with configurable severity and duration

Usage:
    gen = SyntheticDataGenerator()
    normal_df = gen.generate_normal(n_samples=1000, n_features=5)
    crash_df = gen.generate_crash_scenario(severity=0.3, duration=20)
    regime_df = gen.generate_regime_specific("high_vol_bear", n_samples=500)

No external GAN dependencies — uses numpy + pandas only.

Dependencies: numpy, pandas (always available in this project).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GARCH(1,1) simulator
# ---------------------------------------------------------------------------


class GARCHSimulator:
    """GARCH(1,1) process simulator.

    The conditional variance evolves as:

        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    with ``r_t = sigma_t * z_t``, ``z_t ~ N(0,1)`` (or Student-t).

    Parameters
    ----------
    omega : float
        Long-run variance weight.  Default 1e-6.
    alpha : float
        ARCH coefficient (shock sensitivity).  Default 0.08.
    beta : float
        GARCH coefficient (persistence).  Default 0.90.
    mu : float
        Mean return.  Default 0.0.
    df : float or None
        Student-t degrees of freedom for the innovation distribution.
        *None* uses Gaussian innovations.  Default 5.0 (fat tails).
    leverage : float
        Asymmetric (GJR-GARCH) leverage term.  If > 0, negative returns
        increase vol more than positive ones.  Default 0.04.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.08,
        beta: float = 0.90,
        mu: float = 0.0,
        df: Optional[float] = 5.0,
        leverage: float = 0.04,
        seed: Optional[int] = None,
    ) -> None:
        if alpha + beta >= 1.0:
            logger.warning(
                "alpha + beta = %.4f >= 1 — GARCH process is non-stationary.",
                alpha + beta,
            )
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.df = df
        self.leverage = leverage
        self._rng = np.random.RandomState(seed)

    def simulate(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate *n_steps* returns and conditional variances.

        Returns
        -------
        returns : np.ndarray
            Simulated return series of length *n_steps*.
        variances : np.ndarray
            Conditional variance series of length *n_steps*.
        """
        returns = np.zeros(n_steps)
        variances = np.zeros(n_steps)

        # Unconditional variance as initial condition
        persistence = self.alpha + self.beta
        if persistence < 1.0:
            sigma2_0 = self.omega / (1.0 - persistence)
        else:
            sigma2_0 = self.omega / 0.01  # fallback for non-stationary

        variances[0] = sigma2_0

        for t in range(n_steps):
            # Innovation
            if self.df is not None and self.df > 2:
                z = self._rng.standard_t(self.df)
            else:
                z = self._rng.standard_normal()

            sigma_t = math.sqrt(max(variances[t], 1e-15))
            returns[t] = self.mu + sigma_t * z

            if t < n_steps - 1:
                r_prev = returns[t] - self.mu
                # GJR-GARCH: leverage term activates on negative returns
                leverage_term = (
                    self.leverage * r_prev ** 2 if r_prev < 0 else 0.0
                )
                variances[t + 1] = (
                    self.omega
                    + self.alpha * r_prev ** 2
                    + self.beta * variances[t]
                    + leverage_term
                )
                variances[t + 1] = max(variances[t + 1], 1e-15)

        return returns, variances

    @classmethod
    def fit_from_returns(
        cls,
        returns: np.ndarray,
        seed: Optional[int] = None,
    ) -> "GARCHSimulator":
        """Fit GARCH(1,1) parameters from empirical returns via
        method-of-moments.

        This is a rough estimate — for production parameter estimation
        use maximum likelihood (e.g. arch library).  Useful for quick
        bootstrap calibration.

        Parameters
        ----------
        returns : array-like
            Historical return series.
        seed : int, optional
            Random seed for the fitted simulator.

        Returns
        -------
        GARCHSimulator
        """
        returns = np.asarray(returns, dtype=np.float64)
        mu = float(np.mean(returns))
        residuals = returns - mu
        var_r = float(np.var(residuals))

        # Estimate persistence from autocorrelation of squared residuals
        sq = residuals ** 2
        if len(sq) > 1:
            autocorr = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
            autocorr = max(0.0, min(autocorr, 0.99))
        else:
            autocorr = 0.9

        beta = min(autocorr * 0.95, 0.95)
        alpha = min(0.10, 1.0 - beta - 0.02)
        omega = var_r * (1.0 - alpha - beta)
        omega = max(omega, 1e-8)

        # Estimate degrees of freedom from kurtosis
        kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis
        if kurt > 0:
            df = max(4.0, 6.0 / kurt + 4.0)
        else:
            df = None  # Gaussian

        # Leverage effect: correlation between returns and future vol
        if len(returns) > 2:
            lev_corr = float(np.corrcoef(residuals[:-1], sq[1:])[0, 1])
            leverage = max(0.0, -lev_corr * alpha)
        else:
            leverage = 0.0

        logger.info(
            "Fitted GARCH(1,1): omega=%.2e, alpha=%.4f, beta=%.4f, "
            "df=%.1f, leverage=%.4f",
            omega,
            alpha,
            beta,
            df if df else float("inf"),
            leverage,
        )

        return cls(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            df=df,
            leverage=leverage,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------


# Regime presets: (mu_annual, vol_annual, garch_alpha, garch_beta, df)
_REGIME_PRESETS: Dict[str, Dict[str, float]] = {
    "bull": {
        "mu_daily": 0.0005,
        "vol_daily": 0.010,
        "alpha": 0.05,
        "beta": 0.92,
        "df": 6.0,
        "leverage": 0.02,
    },
    "bear": {
        "mu_daily": -0.0008,
        "vol_daily": 0.020,
        "alpha": 0.10,
        "beta": 0.85,
        "df": 4.0,
        "leverage": 0.08,
    },
    "sideways": {
        "mu_daily": 0.0001,
        "vol_daily": 0.008,
        "alpha": 0.04,
        "beta": 0.94,
        "df": 8.0,
        "leverage": 0.01,
    },
    "crisis": {
        "mu_daily": -0.0025,
        "vol_daily": 0.040,
        "alpha": 0.15,
        "beta": 0.80,
        "df": 3.0,
        "leverage": 0.12,
    },
    "low_vol": {
        "mu_daily": 0.0003,
        "vol_daily": 0.005,
        "alpha": 0.03,
        "beta": 0.95,
        "df": 10.0,
        "leverage": 0.01,
    },
    # HMM-compatible regime aliases (matching MarketRegimeState names)
    "low_vol_bull": {
        "mu_daily": 0.0006,
        "vol_daily": 0.008,
        "alpha": 0.05,
        "beta": 0.92,
        "df": 7.0,
        "leverage": 0.02,
    },
    "high_vol_bull": {
        "mu_daily": 0.0004,
        "vol_daily": 0.018,
        "alpha": 0.12,
        "beta": 0.83,
        "df": 4.5,
        "leverage": 0.06,
    },
    "low_vol_bear": {
        "mu_daily": -0.0003,
        "vol_daily": 0.010,
        "alpha": 0.06,
        "beta": 0.90,
        "df": 6.0,
        "leverage": 0.04,
    },
    "high_vol_bear": {
        "mu_daily": -0.0020,
        "vol_daily": 0.035,
        "alpha": 0.15,
        "beta": 0.78,
        "df": 3.0,
        "leverage": 0.12,
    },
    "mean_reverting": {
        "mu_daily": 0.0001,
        "vol_daily": 0.012,
        "alpha": 0.06,
        "beta": 0.90,
        "df": 6.0,
        "leverage": 0.03,
    },
}


class SyntheticDataGenerator:
    """Generate synthetic financial time series for backtesting and
    stress testing.

    Parameters
    ----------
    seed : int or None
        Master random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    # ----- public API -------------------------------------------------------

    def generate_normal(
        self,
        n_samples: int,
        n_features: int = 1,
        mu: float = 0.0002,
        vol: float = 0.015,
        correlation: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate normal-market synthetic returns with GARCH dynamics.

        Parameters
        ----------
        n_samples : int
            Number of time steps (e.g. trading days).
        n_features : int
            Number of assets / return series.
        mu : float
            Daily mean return.  Default ~5% annualised.
        vol : float
            Daily volatility.  Default ~24% annualised.
        correlation : np.ndarray, optional
            Target correlation matrix (n_features x n_features).  If
            *None*, assets are independent.
        feature_names : list of str, optional
            Column names.  Defaults to ``["asset_0", "asset_1", ...]``.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (n_samples, n_features) with a
            DatetimeIndex starting 2020-01-01.
        """
        if feature_names is None:
            feature_names = [f"asset_{i}" for i in range(n_features)]

        omega = vol ** 2 * 0.02
        alpha = 0.08
        beta = 0.90

        # Generate independent GARCH series
        series = np.zeros((n_samples, n_features))
        for j in range(n_features):
            sim = GARCHSimulator(
                omega=omega,
                alpha=alpha,
                beta=beta,
                mu=mu,
                df=5.0,
                leverage=0.04,
                seed=self._next_seed(),
            )
            returns, _ = sim.simulate(n_samples)
            series[:, j] = returns

        # Apply target correlation via Cholesky
        if correlation is not None and n_features > 1:
            series = self._apply_correlation(series, correlation)

        dates = pd.bdate_range(start="2020-01-01", periods=n_samples)
        return pd.DataFrame(series, index=dates, columns=feature_names)

    def generate_crash_scenario(
        self,
        severity: float = 0.05,
        duration: int = 10,
        n_features: int = 1,
        pre_crash_days: int = 100,
        post_crash_days: int = 50,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate a scenario with a market crash embedded.

        The crash period has:
        - Large negative mean return
        - Elevated volatility (3-5x normal)
        - Higher correlation among assets (contagion)
        - Heavier tails

        Parameters
        ----------
        severity : float
            Total crash magnitude (cumulative return).  E.g. 0.05 = 5%
            drawdown, 0.30 = 30% drawdown.  Default 0.05.
        duration : int
            Number of days the crash spans.  Default 10.
        n_features : int
            Number of assets.
        pre_crash_days : int
            Calm period before the crash.
        post_crash_days : int
            Recovery period after the crash.
        feature_names : list of str, optional

        Returns
        -------
        pd.DataFrame
        """
        if feature_names is None:
            feature_names = [f"asset_{i}" for i in range(n_features)]

        total_days = pre_crash_days + duration + post_crash_days

        # --- pre-crash: normal regime ---------------------------------------
        pre = self.generate_regime_specific(
            "bull", pre_crash_days, n_features, feature_names
        )

        # --- crash period ---------------------------------------------------
        daily_crash_return = math.log(1.0 - severity) / duration
        crash_vol = 0.04  # ~60%+ annualised

        crash_sim = GARCHSimulator(
            omega=crash_vol ** 2 * 0.05,
            alpha=0.15,
            beta=0.80,
            mu=daily_crash_return,
            df=3.0,
            leverage=0.15,
            seed=self._next_seed(),
        )

        crash_series = np.zeros((duration, n_features))
        for j in range(n_features):
            returns, _ = crash_sim.simulate(duration)
            crash_series[:, j] = returns
            crash_sim._rng = np.random.RandomState(self._next_seed())

        # Apply high correlation during crash (contagion)
        if n_features > 1:
            crash_corr = np.full((n_features, n_features), 0.85)
            np.fill_diagonal(crash_corr, 1.0)
            crash_series = self._apply_correlation(crash_series, crash_corr)

        crash_dates = pd.bdate_range(
            start=pre.index[-1] + pd.Timedelta(days=1), periods=duration
        )
        crash_df = pd.DataFrame(
            crash_series, index=crash_dates, columns=feature_names
        )

        # --- post-crash: recovery -------------------------------------------
        post = self.generate_regime_specific(
            "sideways", post_crash_days, n_features, feature_names
        )
        post.index = pd.bdate_range(
            start=crash_dates[-1] + pd.Timedelta(days=1), periods=post_crash_days
        )

        result = pd.concat([pre, crash_df, post])
        logger.info(
            "Generated crash scenario: severity=%.1f%%, duration=%d, "
            "total_days=%d, n_features=%d",
            severity * 100,
            duration,
            len(result),
            n_features,
        )
        return result

    def generate_regime_specific(
        self,
        regime: str,
        n_samples: int,
        n_features: int = 1,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate returns from a specific market regime.

        Parameters
        ----------
        regime : str
            One of ``"bull"``, ``"bear"``, ``"sideways"``, ``"crisis"``,
            ``"low_vol"``.
        n_samples : int
            Number of time steps.
        n_features : int
            Number of assets.
        feature_names : list of str, optional

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If *regime* is not recognised.
        """
        regime = regime.lower()
        if regime not in _REGIME_PRESETS:
            raise ValueError(
                f"Unknown regime {regime!r}. Choose from: "
                f"{list(_REGIME_PRESETS.keys())}"
            )

        params = _REGIME_PRESETS[regime]
        if feature_names is None:
            feature_names = [f"asset_{i}" for i in range(n_features)]

        vol = params["vol_daily"]
        omega = vol ** 2 * (1.0 - params["alpha"] - params["beta"])
        omega = max(omega, 1e-10)

        series = np.zeros((n_samples, n_features))
        for j in range(n_features):
            sim = GARCHSimulator(
                omega=omega,
                alpha=params["alpha"],
                beta=params["beta"],
                mu=params["mu_daily"],
                df=params["df"],
                leverage=params["leverage"],
                seed=self._next_seed(),
            )
            returns, _ = sim.simulate(n_samples)
            series[:, j] = returns

        dates = pd.bdate_range(start="2020-01-01", periods=n_samples)
        return pd.DataFrame(series, index=dates, columns=feature_names)

    def generate_multi_regime(
        self,
        regime_sequence: List[Tuple[str, int]],
        n_features: int = 1,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate a series that transitions through multiple regimes.

        Parameters
        ----------
        regime_sequence : list of (regime, duration) tuples
            E.g. ``[("bull", 200), ("crisis", 30), ("bear", 100)]``.
        n_features : int
        feature_names : list of str, optional

        Returns
        -------
        pd.DataFrame
        """
        if feature_names is None:
            feature_names = [f"asset_{i}" for i in range(n_features)]

        segments: List[pd.DataFrame] = []
        for regime, duration in regime_sequence:
            seg = self.generate_regime_specific(
                regime, duration, n_features, feature_names
            )
            segments.append(seg)

        # Re-index with continuous business dates
        combined = pd.concat(segments, ignore_index=True)
        total = len(combined)
        combined.index = pd.bdate_range(start="2020-01-01", periods=total)
        combined.columns = feature_names
        return combined

    def bootstrap_from_data(
        self,
        historical_returns: pd.DataFrame,
        n_samples: int,
        block_size: int = 20,
    ) -> pd.DataFrame:
        """Block bootstrap resampling of historical returns.

        Preserves short-term temporal dependencies (vol clustering,
        autocorrelation) by resampling contiguous blocks.

        Parameters
        ----------
        historical_returns : pd.DataFrame
            Historical return data (rows = time, columns = assets).
        n_samples : int
            Length of the synthetic series to generate.
        block_size : int
            Block length for the stationary bootstrap.  Default 20
            (approximately one trading month).

        Returns
        -------
        pd.DataFrame
        """
        data = historical_returns.values
        n_hist = len(data)
        if n_hist < block_size:
            logger.warning(
                "Historical data (%d) shorter than block_size (%d) — "
                "reducing block_size.",
                n_hist,
                block_size,
            )
            block_size = max(1, n_hist // 2)

        blocks: List[np.ndarray] = []
        remaining = n_samples

        while remaining > 0:
            # Random start point
            start = self._rng.randint(0, n_hist - block_size + 1)
            block_len = min(block_size, remaining)
            blocks.append(data[start : start + block_len])
            remaining -= block_len

        synthetic = np.concatenate(blocks, axis=0)[:n_samples]
        dates = pd.bdate_range(start="2020-01-01", periods=n_samples)
        return pd.DataFrame(
            synthetic, index=dates, columns=historical_returns.columns
        )

    def validate_stylized_facts(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        significance: float = 0.1,
    ) -> Dict[str, bool]:
        """Validate that synthetic data preserves stylized facts of
        real financial returns.

        Tests:
            1. **fat_tails** — excess kurtosis > 0
            2. **vol_clustering** — significant autocorrelation of
               squared returns at lag 1
            3. **leverage_effect** — negative correlation between returns
               and future squared returns
            4. **low_linear_autocorr** — |autocorrelation at lag 1| < 0.1
            5. **mean_close** — means within 2 standard errors
            6. **vol_close** — standard deviations within 50% of each other

        Parameters
        ----------
        real : pd.DataFrame
            Real return data (single column or first column used).
        synthetic : pd.DataFrame
            Synthetic return data.
        significance : float
            Tolerance level.  Default 0.1.

        Returns
        -------
        dict
            ``{test_name: passed}`` — True means the synthetic data
            matches the stylized fact.
        """
        r = real.iloc[:, 0].dropna().values if real.ndim > 1 else real.values
        s = synthetic.iloc[:, 0].dropna().values if synthetic.ndim > 1 else synthetic.values

        results: Dict[str, bool] = {}

        # 1. Fat tails: excess kurtosis > 0
        real_kurt = float(_excess_kurtosis(r))
        synth_kurt = float(_excess_kurtosis(s))
        results["fat_tails"] = synth_kurt > 0
        logger.debug(
            "Kurtosis — real: %.2f, synthetic: %.2f", real_kurt, synth_kurt
        )

        # 2. Vol clustering: autocorrelation of squared returns
        real_sq_ac = _lag1_autocorr(r ** 2)
        synth_sq_ac = _lag1_autocorr(s ** 2)
        results["vol_clustering"] = synth_sq_ac > significance
        logger.debug(
            "Sq-return autocorr — real: %.4f, synthetic: %.4f",
            real_sq_ac,
            synth_sq_ac,
        )

        # 3. Leverage effect: corr(r_t, r_{t+1}^2) < 0
        if len(s) > 2:
            lev = float(np.corrcoef(s[:-1], s[1:] ** 2)[0, 1])
        else:
            lev = 0.0
        results["leverage_effect"] = lev < 0
        logger.debug("Leverage effect (synthetic): %.4f", lev)

        # 4. Low linear autocorrelation
        synth_ac = abs(_lag1_autocorr(s))
        results["low_linear_autocorr"] = synth_ac < 0.10
        logger.debug("Linear autocorr (synthetic): %.4f", synth_ac)

        # 5. Mean close
        real_mean = float(np.mean(r))
        synth_mean = float(np.mean(s))
        se = float(np.std(r) / max(math.sqrt(len(r)), 1))
        results["mean_close"] = abs(synth_mean - real_mean) < 2 * se + 1e-8
        logger.debug(
            "Mean — real: %.6f, synthetic: %.6f, SE: %.6f",
            real_mean,
            synth_mean,
            se,
        )

        # 6. Vol close: within 50%
        real_vol = float(np.std(r))
        synth_vol = float(np.std(s))
        if real_vol > 1e-10:
            vol_ratio = synth_vol / real_vol
            results["vol_close"] = 0.5 < vol_ratio < 1.5
        else:
            results["vol_close"] = synth_vol < 1e-6
        logger.debug(
            "Vol — real: %.6f, synthetic: %.6f", real_vol, synth_vol
        )

        n_passed = sum(results.values())
        n_total = len(results)
        logger.info(
            "Stylized facts validation: %d/%d passed  %s",
            n_passed,
            n_total,
            results,
        )
        return results

    # ----- helpers ----------------------------------------------------------

    def _next_seed(self) -> int:
        """Draw a reproducible child seed from the master RNG."""
        return int(self._rng.randint(0, 2 ** 31))

    @staticmethod
    def _apply_correlation(
        independent: np.ndarray, target_corr: np.ndarray
    ) -> np.ndarray:
        """Apply a target correlation structure to independent series
        via Cholesky decomposition.

        Parameters
        ----------
        independent : np.ndarray
            (n_samples, n_features) — independent series.
        target_corr : np.ndarray
            (n_features, n_features) — target correlation matrix.

        Returns
        -------
        np.ndarray
            Correlated series with approximately the target structure.
        """
        # Ensure PSD
        eigvals = np.linalg.eigvalsh(target_corr)
        if np.min(eigvals) < 0:
            # Nearest PSD fix
            vals, vecs = np.linalg.eigh(target_corr)
            vals = np.maximum(vals, 1e-8)
            target_corr = vecs @ np.diag(vals) @ vecs.T
            # Re-normalize to correlation
            d = np.sqrt(np.diag(target_corr))
            target_corr = target_corr / np.outer(d, d)

        try:
            L = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError:
            logger.warning(
                "Cholesky failed on target correlation — returning "
                "independent series."
            )
            return independent

        # Standardise each column
        means = independent.mean(axis=0)
        stds = independent.std(axis=0)
        stds[stds == 0] = 1.0
        standardised = (independent - means) / stds

        # Apply correlation, then restore scale
        correlated = standardised @ L.T
        return correlated * stds + means


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition, = kurtosis - 3)."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-15:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation coefficient."""
    if len(x) < 3:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])
