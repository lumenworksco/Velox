"""LPRADO-001: Fractional Differentiation — memory-preserving stationarity.

Implements the fixed-width window fractional differentiation method from
Lopez de Prado's *Advances in Financial Machine Learning* (Chapter 5).

Standard differencing (d=1) achieves stationarity but destroys memory.
Fractional differencing with d in [0.3, 0.5] achieves stationarity while
retaining long-range dependence — critical for ML features derived from
price, volume, VWAP, OBV, and cumulative returns.

Key method: find the minimum d that makes a series stationary (ADF test
p-value < threshold), then apply that d to produce a feature that is
both stationary and memory-preserving.

Usage:
    fd = FractionalDifferentiator()
    d_star = fd.find_min_d(price_series)             # typically 0.3-0.5
    stationary = fd.frac_diff(price_series, d_star)   # apply
    features_df = fd.transform_features(ohlcv_df)     # batch transform
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.debug("statsmodels not available — ADF-based find_min_d disabled")


class FractionalDifferentiator:
    """Fixed-width window fractional differentiation (FFD).

    The FFD method truncates the infinite weight series at a threshold,
    producing a finite-length filter that can be applied as a convolution.
    This avoids the expanding-window problem of the standard fracdiff.
    """

    def __init__(self, threshold: float = 1e-5):
        """
        Args:
            threshold: Minimum absolute weight to include in the filter.
                       Smaller values give longer filters with more memory
                       but slower computation. Default 1e-5 works well for
                       daily price data up to ~10 years.
        """
        self.threshold = threshold
        # Cache computed optimal d values per column name
        self._d_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core: compute FFD weights
    # ------------------------------------------------------------------

    @staticmethod
    def _get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
        """Compute fixed-width window fracdiff weights.

        The weights w_k for lag k are:
            w_0 = 1
            w_k = -w_{k-1} * (d - k + 1) / k

        We keep generating weights until |w_k| < threshold.

        Args:
            d: Fractional differencing order (0 < d < 1 typically).
            threshold: Truncation threshold for weights.

        Returns:
            1-D array of weights, w[0] = 1, w[1], w[2], ...
        """
        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1
            # Safety: cap at 10000 lags to prevent runaway
            if k > 10000:
                break
        return np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public: fractional differentiation
    # ------------------------------------------------------------------

    def frac_diff(
        self,
        series: pd.Series,
        d: float,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """Apply fractional differentiation of order d to a series.

        Uses the fixed-width window (FFD) method. The output series is
        shorter than the input by (filter_length - 1) samples, which are
        set to NaN to preserve index alignment.

        Args:
            series: Input time series (e.g., log prices).
            d: Fractional differencing order. Typical range [0.0, 1.0].
                d=0 returns the original series; d=1 is standard differencing.
            threshold: Override the instance threshold for weight truncation.

        Returns:
            Fractionally differenced series with the same index.
            Leading values where the filter has insufficient history are NaN.
        """
        if d < 1e-8:
            return series.copy()

        thresh = threshold if threshold is not None else self.threshold
        weights = self._get_weights_ffd(d, thresh)
        width = len(weights)

        values = series.values.astype(np.float64)
        n = len(values)

        if width > n:
            logger.warning(
                f"FFD filter width ({width}) exceeds series length ({n}); "
                f"returning NaN series"
            )
            return pd.Series(np.nan, index=series.index, name=series.name)

        # Apply convolution (dot product of reversed weights with window)
        result = np.full(n, np.nan, dtype=np.float64)
        w_rev = weights[::-1]  # Reverse for dot product with [x_{t-width+1}, ..., x_t]

        for t in range(width - 1, n):
            result[t] = np.dot(w_rev, values[t - width + 1: t + 1])

        return pd.Series(result, index=series.index, name=series.name)

    # ------------------------------------------------------------------
    # Public: find minimum d for stationarity
    # ------------------------------------------------------------------

    def find_min_d(
        self,
        series: pd.Series,
        max_d: float = 1.0,
        p_threshold: float = 0.05,
        step: float = 0.05,
    ) -> float:
        """Find the minimum d that makes the series stationary via ADF test.

        Performs a grid search over d in [0, max_d] with the given step size.
        For each d, applies FFD and runs the Augmented Dickey-Fuller test.
        Returns the smallest d whose ADF p-value is below p_threshold.

        Args:
            series: Input time series (raw prices or log prices).
            max_d: Maximum d to search. Default 1.0.
            p_threshold: ADF p-value threshold for stationarity. Default 0.05.
            step: Grid search step size. Default 0.05.

        Returns:
            Minimum d achieving stationarity. Returns max_d if stationarity
            is never achieved (rare for financial price series).

        Raises:
            RuntimeError: If statsmodels is not installed.
        """
        if not _HAS_STATSMODELS:
            raise RuntimeError(
                "statsmodels is required for find_min_d (ADF test). "
                "Install with: pip install statsmodels"
            )

        # Use log prices for numerical stability
        log_series = np.log(series.replace(0, np.nan).dropna())
        if len(log_series) < 50:
            logger.warning(
                f"Series too short ({len(log_series)}) for reliable ADF test; "
                f"returning default d=0.5"
            )
            return 0.5

        log_series = pd.Series(log_series.values, index=log_series.index)

        d_values = np.arange(0.0, max_d + step / 2, step)
        best_d = max_d

        for d in d_values:
            if d < 1e-8:
                continue  # Skip d=0 (original series, likely non-stationary)

            diffed = self.frac_diff(log_series, d)
            clean = diffed.dropna()

            if len(clean) < 30:
                continue

            try:
                adf_stat, p_value, *_ = _adfuller(clean.values, maxlag=10)
                if p_value < p_threshold:
                    best_d = d
                    logger.debug(
                        f"find_min_d: d={d:.2f} achieves stationarity "
                        f"(ADF p={p_value:.4f})"
                    )
                    break
            except Exception as e:
                logger.debug(f"ADF test failed at d={d:.2f}: {e}")
                continue

        logger.info(f"Optimal fractional differencing order: d*={best_d:.2f}")
        return round(best_d, 2)

    # ------------------------------------------------------------------
    # Public: batch transform a DataFrame of features
    # ------------------------------------------------------------------

    def transform_features(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Apply optimal fractional differencing to each column of a DataFrame.

        For each column, finds the minimum d for stationarity (or uses cached
        value), then applies FFD with that d.

        Default columns if not specified: close, volume, vwap, obv, cum_return.
        Non-existent columns are silently skipped.

        Args:
            df: DataFrame with time-series columns.
            columns: Specific columns to transform. If None, uses defaults.
            use_cache: Whether to reuse previously computed d values.

        Returns:
            New DataFrame with fractionally differenced columns.
            Column names are suffixed with '_fd'.
        """
        default_columns = ["close", "volume", "vwap", "obv", "cum_return"]
        target_cols = columns or default_columns

        # Filter to columns that actually exist
        available = [c for c in target_cols if c in df.columns]
        if not available:
            logger.warning(
                f"No target columns found in DataFrame. "
                f"Available: {list(df.columns)}, requested: {target_cols}"
            )
            return pd.DataFrame(index=df.index)

        result = pd.DataFrame(index=df.index)

        for col in available:
            series = df[col].dropna()
            if len(series) < 50:
                logger.debug(f"Column '{col}' too short ({len(series)}), skipping")
                continue

            # Find or retrieve optimal d
            if use_cache and col in self._d_cache:
                d = self._d_cache[col]
            else:
                try:
                    d = self.find_min_d(series)
                    if use_cache:
                        self._d_cache[col] = d
                except Exception as e:
                    logger.warning(f"find_min_d failed for '{col}': {e}, using d=0.5")
                    d = 0.5

            # Apply fractional differencing
            diffed = self.frac_diff(pd.Series(series.values, index=series.index), d)
            result[f"{col}_fd"] = diffed.reindex(df.index)

            logger.debug(f"Transformed '{col}' with d={d:.2f}")

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_cached_d_values(self) -> dict[str, float]:
        """Return cached optimal d values from previous transform_features calls."""
        return dict(self._d_cache)

    def clear_cache(self) -> None:
        """Clear the cached d values."""
        self._d_cache.clear()
