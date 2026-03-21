"""COMP-007: Cross-asset lead-lag signal detection.

Computes lagged correlations between assets to find leading indicators.
Uses Granger causality tests and cross-correlation analysis to identify
which assets lead or lag others.

Key methods:
- find_leaders(target, candidates, max_lag): find assets that lead target
- compute_lead_lag_matrix(returns): full pairwise lead-lag analysis
- granger_test(x, y, max_lag): Granger causality F-test

Fail-open: returns empty results if insufficient data.

Usage::

    from analytics.lead_lag import LeadLagAnalyzer

    analyzer = LeadLagAnalyzer()
    leaders = analyzer.find_leaders(
        target=spy_returns,
        candidates={"XLF": xlf_returns, "TLT": tlt_returns, "VIX": vix_returns},
        max_lag=5,
    )
    # [LeadLagResult(leader="VIX", lag=1, correlation=-0.35, granger_pvalue=0.02)]

    matrix = analyzer.compute_lead_lag_matrix(returns_dict)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum observations needed for valid analysis
MIN_OBSERVATIONS = 60

# Default maximum lag to test (trading days)
DEFAULT_MAX_LAG = 10

# Granger causality significance level
GRANGER_SIGNIFICANCE = 0.05

# Minimum absolute correlation to consider meaningful
MIN_CORRELATION = 0.10

# Maximum number of leaders to return per target
MAX_LEADERS = 20


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeadLagResult:
    """Result of lead-lag analysis for one pair."""
    leader: str                        # Symbol of the leading asset
    follower: str                      # Symbol of the lagging asset
    optimal_lag: int                   # Lag in periods at max correlation
    correlation: float                 # Cross-correlation at optimal lag
    granger_pvalue: float              # Granger causality p-value
    granger_significant: bool          # Whether Granger test is significant
    all_lag_correlations: Dict[int, float] = field(default_factory=dict)
    information_coefficient: float = 0.0  # IC at optimal lag

    @property
    def summary(self) -> str:
        direction = "positively" if self.correlation > 0 else "negatively"
        sig = " (Granger significant)" if self.granger_significant else ""
        return (
            f"{self.leader} leads {self.follower} by {self.optimal_lag} periods "
            f"({direction} correlated: {self.correlation:.3f}){sig}"
        )


@dataclass
class LeadLagMatrix:
    """Full pairwise lead-lag matrix for a set of assets."""
    symbols: List[str]
    optimal_lags: Dict[Tuple[str, str], int] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    granger_pvalues: Dict[Tuple[str, str], float] = field(default_factory=dict)
    net_leadership_scores: Dict[str, float] = field(default_factory=dict)

    def get_top_leaders(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N net leaders (assets that lead most others)."""
        sorted_scores = sorted(
            self.net_leadership_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_scores[:n]

    def get_top_followers(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top N net followers (assets that lag most others)."""
        sorted_scores = sorted(
            self.net_leadership_scores.items(),
            key=lambda x: x[1],
        )
        return sorted_scores[:n]


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def _cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> Dict[int, float]:
    """Compute cross-correlation between x and y for lags -max_lag to +max_lag.

    Positive lag means x leads y (x at time t predicts y at time t+lag).

    Args:
        x: First time series (standardized).
        y: Second time series (standardized).
        max_lag: Maximum lag to compute.

    Returns:
        Dict mapping lag -> correlation.
    """
    n = len(x)
    if n < max_lag + 10:
        return {}

    correlations: Dict[int, float] = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # x leads y: correlate x[:-lag] with y[lag:]
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        elif lag < 0:
            # y leads x: correlate x[-lag:] with y[:lag]
            corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]

        if np.isfinite(corr):
            correlations[lag] = float(corr)

    return correlations


def _granger_causality_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> Tuple[float, float, int]:
    """Granger causality F-test: does x Granger-cause y?

    Tests whether past values of x help predict y beyond what past
    values of y alone can predict.

    Args:
        x: Potential causal (leading) series.
        y: Potential effect (lagging) series.
        max_lag: Maximum number of lags to include.

    Returns:
        Tuple of (best_f_stat, best_p_value, best_lag).
    """
    n = len(y)
    if n < max_lag * 2 + 10:
        return 0.0, 1.0, 0

    best_f = 0.0
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        try:
            # Restricted model: y_t = a0 + a1*y_{t-1} + ... + a_k*y_{t-k}
            # Unrestricted: y_t = a0 + a1*y_{t-1} + ... + a_k*y_{t-k} + b1*x_{t-1} + ... + b_k*x_{t-k}

            Y = y[lag:]
            n_obs = len(Y)

            # Build lagged matrices
            Y_lags = np.column_stack([y[lag - i - 1: n - i - 1] for i in range(lag)])
            X_lags = np.column_stack([x[lag - i - 1: n - i - 1] for i in range(lag)])

            # Restricted model (y lags only)
            Z_restricted = np.column_stack([np.ones(n_obs), Y_lags])
            try:
                beta_r = np.linalg.lstsq(Z_restricted, Y, rcond=None)[0]
                resid_r = Y - Z_restricted @ beta_r
                rss_r = np.sum(resid_r ** 2)
            except np.linalg.LinAlgError:
                continue

            # Unrestricted model (y lags + x lags)
            Z_unrestricted = np.column_stack([np.ones(n_obs), Y_lags, X_lags])
            try:
                beta_u = np.linalg.lstsq(Z_unrestricted, Y, rcond=None)[0]
                resid_u = Y - Z_unrestricted @ beta_u
                rss_u = np.sum(resid_u ** 2)
            except np.linalg.LinAlgError:
                continue

            # F-statistic
            df1 = lag  # Additional parameters in unrestricted
            df2 = n_obs - 2 * lag - 1

            if df2 <= 0 or rss_u < 1e-15:
                continue

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

            if f_stat < 0:
                continue

            # Compute p-value using F-distribution survival function
            p_value = _f_survival(f_stat, df1, df2)

            if p_value < best_p:
                best_f = f_stat
                best_p = p_value
                best_lag = lag

        except Exception as exc:
            logger.debug("Granger test failed at lag %d: %s", lag, exc)
            continue

    return best_f, best_p, best_lag


def _f_survival(f_stat: float, df1: int, df2: int) -> float:
    """Approximate survival function for F-distribution.

    Uses the regularized incomplete beta function approximation.
    For production, scipy.stats.f.sf() would be more accurate.
    """
    try:
        from scipy.stats import f as f_dist
        return float(f_dist.sf(f_stat, df1, df2))
    except ImportError:
        pass

    # Fallback: crude approximation using normal approximation to F
    # Valid for large df2
    if df2 < 5:
        return 0.5  # Can't approximate well, return neutral

    # Approximation via z-transform
    z = (
        (f_stat ** (1 / 3) * (1 - 2 / (9 * df2)) - (1 - 2 / (9 * df1)))
        / ((2 / (9 * df1) + f_stat ** (2 / 3) * 2 / (9 * df2)) ** 0.5)
    )

    # Standard normal survival
    return float(0.5 * (1 - _erf(z / 2 ** 0.5))) if np.isfinite(z) else 0.5


def _erf(x: float) -> float:
    """Approximate error function."""
    # Abramowitz and Stegun approximation
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y


def _information_coefficient(
    signal: np.ndarray,
    forward_returns: np.ndarray,
) -> float:
    """Rank IC (Spearman correlation) between signal and forward returns."""
    if len(signal) < 10 or len(signal) != len(forward_returns):
        return 0.0

    try:
        from scipy.stats import spearmanr
        ic, _ = spearmanr(signal, forward_returns)
        return float(ic) if np.isfinite(ic) else 0.0
    except ImportError:
        # Fallback: Pearson on ranks
        ranks_s = np.argsort(np.argsort(signal)).astype(float)
        ranks_r = np.argsort(np.argsort(forward_returns)).astype(float)
        corr = np.corrcoef(ranks_s, ranks_r)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LeadLagAnalyzer:
    """Cross-asset lead-lag analysis engine.

    Identifies which assets lead or lag others using cross-correlation
    and Granger causality tests.
    """

    def __init__(
        self,
        max_lag: int = DEFAULT_MAX_LAG,
        min_correlation: float = MIN_CORRELATION,
        significance_level: float = GRANGER_SIGNIFICANCE,
    ):
        self._max_lag = max_lag
        self._min_correlation = min_correlation
        self._significance_level = significance_level

    # ------------------------------------------------------------------
    # Find leaders for a target
    # ------------------------------------------------------------------

    def find_leaders(
        self,
        target: pd.Series,
        candidates: Dict[str, pd.Series],
        max_lag: Optional[int] = None,
        target_name: str = "target",
    ) -> List[LeadLagResult]:
        """Find assets that lead the target.

        Args:
            target: Target asset returns (pd.Series with datetime index).
            candidates: Dict of candidate name -> return series.
            max_lag: Max lag to test (default: self._max_lag).
            target_name: Name for the target in results.

        Returns:
            List of LeadLagResult sorted by |correlation| descending.
        """
        if max_lag is None:
            max_lag = self._max_lag

        target_arr = np.asarray(target.dropna(), dtype=float)
        if len(target_arr) < MIN_OBSERVATIONS:
            logger.info("Insufficient target data (%d < %d)", len(target_arr), MIN_OBSERVATIONS)
            return []

        results: List[LeadLagResult] = []

        for name, series in candidates.items():
            try:
                result = self._analyze_pair(
                    leader_name=name,
                    follower_name=target_name,
                    leader_returns=series,
                    follower_returns=target,
                    max_lag=max_lag,
                )
                if result and abs(result.correlation) >= self._min_correlation:
                    results.append(result)
            except Exception as exc:
                logger.debug("Lead-lag analysis failed for %s: %s", name, exc)

        results.sort(key=lambda r: abs(r.correlation), reverse=True)
        return results[:MAX_LEADERS]

    # ------------------------------------------------------------------
    # Full pairwise matrix
    # ------------------------------------------------------------------

    def compute_lead_lag_matrix(
        self,
        returns: Dict[str, pd.Series],
        max_lag: Optional[int] = None,
    ) -> LeadLagMatrix:
        """Compute pairwise lead-lag relationships for all assets.

        Args:
            returns: Dict of symbol -> return series.
            max_lag: Maximum lag to test.

        Returns:
            LeadLagMatrix with full pairwise results.
        """
        if max_lag is None:
            max_lag = self._max_lag

        symbols = list(returns.keys())
        matrix = LeadLagMatrix(symbols=symbols)
        leadership_tally: Dict[str, float] = {s: 0.0 for s in symbols}

        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if i == j:
                    continue

                try:
                    result = self._analyze_pair(
                        leader_name=sym_a,
                        follower_name=sym_b,
                        leader_returns=returns[sym_a],
                        follower_returns=returns[sym_b],
                        max_lag=max_lag,
                    )
                    if result is None:
                        continue

                    pair = (sym_a, sym_b)
                    matrix.optimal_lags[pair] = result.optimal_lag
                    matrix.correlations[pair] = result.correlation
                    matrix.granger_pvalues[pair] = result.granger_pvalue

                    # Update leadership tally
                    if result.optimal_lag > 0 and abs(result.correlation) >= self._min_correlation:
                        leadership_tally[sym_a] += abs(result.correlation)
                        leadership_tally[sym_b] -= abs(result.correlation)

                except Exception as exc:
                    logger.debug("Matrix computation failed for %s->%s: %s", sym_a, sym_b, exc)

        matrix.net_leadership_scores = leadership_tally
        return matrix

    # ------------------------------------------------------------------
    # Pair analysis
    # ------------------------------------------------------------------

    def _analyze_pair(
        self,
        leader_name: str,
        follower_name: str,
        leader_returns: pd.Series,
        follower_returns: pd.Series,
        max_lag: int,
    ) -> Optional[LeadLagResult]:
        """Analyze lead-lag relationship between two return series."""
        # Align series
        aligned = pd.concat([leader_returns, follower_returns], axis=1).dropna()
        if len(aligned) < MIN_OBSERVATIONS:
            return None

        x = aligned.iloc[:, 0].values.astype(float)
        y = aligned.iloc[:, 1].values.astype(float)

        # Cross-correlation
        xcorr = _cross_correlation(x, y, max_lag)
        if not xcorr:
            return None

        # Find optimal lag (max absolute correlation at positive lags = x leads y)
        positive_lags = {lag: corr for lag, corr in xcorr.items() if lag > 0}
        if not positive_lags:
            return None

        optimal_lag = max(positive_lags, key=lambda k: abs(positive_lags[k]))
        optimal_corr = positive_lags[optimal_lag]

        # Granger causality
        f_stat, p_value, granger_lag = _granger_causality_test(x, y, max_lag)

        # Information coefficient at optimal lag
        if optimal_lag > 0 and optimal_lag < len(x):
            ic = _information_coefficient(x[:-optimal_lag], y[optimal_lag:])
        else:
            ic = 0.0

        return LeadLagResult(
            leader=leader_name,
            follower=follower_name,
            optimal_lag=optimal_lag,
            correlation=optimal_corr,
            granger_pvalue=p_value,
            granger_significant=p_value < self._significance_level,
            all_lag_correlations=xcorr,
            information_coefficient=ic,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compute_predictive_signal(
        self,
        leader_returns: pd.Series,
        optimal_lag: int,
        correlation_sign: float,
    ) -> pd.Series:
        """Generate a predictive signal from a leader's returns.

        Shifts leader returns forward by optimal_lag and applies the
        correlation direction to produce a forecast signal.

        Args:
            leader_returns: Leader asset returns.
            optimal_lag: Number of periods the leader leads by.
            correlation_sign: +1.0 or -1.0 for direction.

        Returns:
            pd.Series with predictive signal values.
        """
        signal = leader_returns.shift(optimal_lag) * np.sign(correlation_sign)
        return signal.dropna()

    @staticmethod
    def rolling_lead_lag(
        x: pd.Series,
        y: pd.Series,
        window: int = 60,
        lag: int = 1,
    ) -> pd.Series:
        """Compute rolling cross-correlation at a fixed lag.

        Useful for monitoring if a lead-lag relationship is stable
        or time-varying.

        Args:
            x: Leader series.
            y: Follower series.
            window: Rolling window size.
            lag: Fixed lag to compute correlation at.

        Returns:
            pd.Series of rolling correlations.
        """
        aligned = pd.concat([x, y], axis=1).dropna()
        if len(aligned) < window + lag:
            return pd.Series(dtype=float)

        x_vals = aligned.iloc[:, 0]
        y_vals = aligned.iloc[:, 1]

        correlations: List[float] = []
        dates: List = []

        for i in range(window + lag, len(aligned)):
            x_window = x_vals.iloc[i - window - lag: i - lag].values
            y_window = y_vals.iloc[i - window: i].values
            if len(x_window) != len(y_window):
                continue
            corr = np.corrcoef(x_window, y_window)[0, 1]
            correlations.append(float(corr) if np.isfinite(corr) else 0.0)
            dates.append(aligned.index[i])

        return pd.Series(correlations, index=dates)
