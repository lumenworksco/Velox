"""RISK-007: Dynamic Correlation Matrix with Exponential Weighting and Shrinkage.

Computes and maintains a dynamic correlation matrix for portfolio risk:
- Exponentially-weighted moving correlation (configurable half-life)
- Ledoit-Wolf shrinkage for numerical stability with small samples
- Periodic refresh during market hours (every 30 minutes)

Used by:
1. Portfolio-level risk aggregation (correlated VaR)
2. Concentration limiter (effective number of bets)
3. Pairs/stat-arb strategy signal generation
"""

import logging
import threading
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Market hours (ET)
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_REFRESH_INTERVAL = timedelta(minutes=30)


class DynamicCorrelation:
    """Exponentially-weighted correlation with Ledoit-Wolf shrinkage.

    Maintains a dynamic correlation matrix that adapts to recent market
    conditions via exponential weighting, with shrinkage applied for
    numerical stability when the number of observations is limited
    relative to the number of assets.

    Thread-safe: all state mutations are protected by a lock.

    Usage:
        dc = DynamicCorrelation(half_life=10)
        corr = dc.get_correlation_matrix(returns_df.values, shrink=True)
        # corr is an (N x N) ndarray

        # Periodic refresh during market hours
        dc.maybe_refresh(returns_df.values)

    Args:
        half_life: Half-life for exponential weighting in periods (days).
            Default 10 (roughly 2 trading weeks).
        refresh_interval_minutes: Minutes between automatic refreshes
            during market hours. Default 30.
        min_observations: Minimum number of observations required to
            compute a correlation matrix. Default 20.
    """

    def __init__(
        self,
        half_life: int = 10,
        refresh_interval_minutes: int = 30,
        min_observations: int = 20,
    ) -> None:
        self._half_life = half_life
        self._refresh_interval = timedelta(minutes=refresh_interval_minutes)
        self._min_obs = min_observations

        self._cached_corr: Optional[np.ndarray] = None
        self._cached_symbols: list[str] = []
        self._last_refresh: Optional[datetime] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_ewm_correlation(
        self,
        returns: np.ndarray,
        half_life: Optional[int] = None,
    ) -> np.ndarray:
        """Compute exponentially-weighted correlation matrix.

        Args:
            returns: (T x N) array of asset returns, rows are time periods,
                columns are assets. Most recent observation is the last row.
            half_life: Override half-life (in periods). Uses instance default
                if not provided.

        Returns:
            (N x N) correlation matrix.

        Raises:
            ValueError: If returns has fewer than min_observations rows or
                fewer than 2 columns.
        """
        if returns.ndim != 2:
            raise ValueError(f"returns must be 2-D, got shape {returns.shape}")

        T, N = returns.shape
        if T < self._min_obs:
            raise ValueError(
                f"Need at least {self._min_obs} observations, got {T}"
            )
        if N < 2:
            raise ValueError(f"Need at least 2 assets, got {N}")

        hl = half_life if half_life is not None else self._half_life
        decay = np.log(2) / max(hl, 1)

        # Exponential weights: most recent = highest weight
        # weights[t] = exp(-decay * (T - 1 - t))  for t in [0, T-1]
        ages = np.arange(T - 1, -1, -1, dtype=float)  # T-1, T-2, ..., 0
        weights = np.exp(-decay * ages)
        weights /= weights.sum()

        # Weighted mean
        w_col = weights.reshape(-1, 1)  # (T, 1) for broadcasting
        mean = (w_col * returns).sum(axis=0)  # (N,)

        # Weighted centered returns
        centered = returns - mean  # (T, N)

        # Weighted covariance: C = sum_t w_t * centered_t * centered_t^T
        weighted_centered = centered * np.sqrt(w_col)  # (T, N)
        cov = weighted_centered.T @ weighted_centered  # (N, N)

        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        std = np.where(std > 1e-12, std, 1e-12)  # avoid division by zero
        corr = cov / np.outer(std, std)

        # Clamp to [-1, 1] for numerical safety
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 1.0)

        return corr

    def shrink_correlation(
        self,
        corr: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage to a correlation matrix.

        Shrinks toward the identity matrix (equal correlation = 0 prior).
        The optimal shrinkage intensity is estimated analytically following
        Ledoit & Wolf (2004).

        Args:
            corr: (N x N) sample correlation matrix.
            returns: (T x N) returns used to compute the correlation (needed
                for shrinkage intensity estimation).

        Returns:
            (N x N) shrunk correlation matrix.
        """
        T, N = returns.shape
        if N < 2 or T < 2:
            return corr.copy()

        target = np.eye(N)

        # Standardize returns for shrinkage estimation
        std = returns.std(axis=0)
        std = np.where(std > 1e-12, std, 1e-12)
        X = (returns - returns.mean(axis=0)) / std  # (T, N) standardized

        # Estimate optimal shrinkage intensity (Ledoit-Wolf)
        # delta = ||corr - I||^2 (Frobenius norm of off-diagonal)
        delta = np.sum((corr - target) ** 2)

        # Estimate variance of sample correlations
        X2 = X ** 2  # (T, N)
        # pi_hat: sum of asymptotic variances of entries of sqrt(T) * corr
        pi_mat = (X2.T @ X2) / T - corr ** 2  # (N, N)
        pi_hat = np.sum(pi_mat)

        # Optimal shrinkage intensity
        kappa = (pi_hat / delta) / T if delta > 1e-12 else 1.0
        shrinkage = max(0.0, min(1.0, kappa))

        shrunk = (1.0 - shrinkage) * corr + shrinkage * target

        # Ensure valid correlation matrix
        np.clip(shrunk, -1.0, 1.0, out=shrunk)
        np.fill_diagonal(shrunk, 1.0)

        logger.debug(
            "Ledoit-Wolf shrinkage: intensity=%.4f (T=%d, N=%d)",
            shrinkage, T, N,
        )

        return shrunk

    def get_correlation_matrix(
        self,
        returns: np.ndarray,
        shrink: bool = True,
        half_life: Optional[int] = None,
        symbols: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Compute and cache the full correlation matrix.

        This is the primary entry point. Computes EWM correlation,
        optionally applies Ledoit-Wolf shrinkage, and caches the result.

        Args:
            returns: (T x N) array of asset returns.
            shrink: Whether to apply Ledoit-Wolf shrinkage. Default True.
            half_life: Override half-life for EWM weighting.
            symbols: Optional list of symbol names (length N) for cache
                association. If provided, enables symbol-based lookups.

        Returns:
            (N x N) correlation matrix.
        """
        try:
            corr = self.compute_ewm_correlation(returns, half_life=half_life)
        except ValueError as exc:
            logger.warning("Correlation computation failed: %s", exc)
            N = returns.shape[1] if returns.ndim == 2 else 1
            return np.eye(N)

        if shrink:
            corr = self.shrink_correlation(corr, returns)

        with self._lock:
            self._cached_corr = corr.copy()
            self._cached_symbols = list(symbols) if symbols else []
            self._last_refresh = datetime.now()

        logger.info(
            "Correlation matrix updated: %dx%d, shrink=%s, half_life=%d",
            corr.shape[0], corr.shape[1], shrink,
            half_life if half_life is not None else self._half_life,
        )

        return corr

    # ------------------------------------------------------------------
    # Cached lookups
    # ------------------------------------------------------------------

    def get_pairwise(self, sym1: str, sym2: str) -> float:
        """Look up a pairwise correlation from the cached matrix.

        Args:
            sym1: First symbol.
            sym2: Second symbol.

        Returns:
            Correlation coefficient, or 0.0 if symbols are not in the cache.
        """
        with self._lock:
            if self._cached_corr is None or not self._cached_symbols:
                return 0.0
            try:
                i = self._cached_symbols.index(sym1)
                j = self._cached_symbols.index(sym2)
                return float(self._cached_corr[i, j])
            except ValueError:
                return 0.0

    def get_cached_matrix(self) -> Optional[np.ndarray]:
        """Return the most recently computed correlation matrix, or None."""
        with self._lock:
            return self._cached_corr.copy() if self._cached_corr is not None else None

    # ------------------------------------------------------------------
    # Periodic refresh
    # ------------------------------------------------------------------

    def maybe_refresh(
        self,
        returns: np.ndarray,
        shrink: bool = True,
        symbols: Optional[list[str]] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Refresh the correlation matrix if enough time has elapsed.

        Intended to be called on every scan cycle. Only recomputes if:
        1. At least refresh_interval has passed since the last refresh.
        2. Current time is during market hours (9:30 AM - 4:00 PM).

        Args:
            returns: (T x N) returns array.
            shrink: Apply shrinkage.
            symbols: Symbol names.
            now: Current time override (for testing).

        Returns:
            True if a refresh was performed, False otherwise.
        """
        if now is None:
            now = datetime.now()

        current_time = now.time()

        # Only refresh during market hours
        if current_time < _MARKET_OPEN or current_time > _MARKET_CLOSE:
            return False

        with self._lock:
            if (
                self._last_refresh is not None
                and (now - self._last_refresh) < self._refresh_interval
            ):
                return False

        # Perform refresh (get_correlation_matrix acquires the lock internally)
        self.get_correlation_matrix(
            returns, shrink=shrink, symbols=symbols,
        )
        return True

    def needs_refresh(self, now: Optional[datetime] = None) -> bool:
        """Check whether a refresh is due without actually refreshing.

        Args:
            now: Current time override.

        Returns:
            True if refresh_interval has elapsed since last refresh.
        """
        if now is None:
            now = datetime.now()
        with self._lock:
            if self._last_refresh is None:
                return True
            return (now - self._last_refresh) >= self._refresh_interval

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        """Diagnostic status snapshot."""
        with self._lock:
            return {
                "half_life": self._half_life,
                "refresh_interval_minutes": int(
                    self._refresh_interval.total_seconds() / 60
                ),
                "min_observations": self._min_obs,
                "cached_matrix_size": (
                    list(self._cached_corr.shape)
                    if self._cached_corr is not None
                    else None
                ),
                "cached_symbols": len(self._cached_symbols),
                "last_refresh": (
                    self._last_refresh.isoformat()
                    if self._last_refresh
                    else None
                ),
            }

    def __repr__(self) -> str:
        with self._lock:
            n = self._cached_corr.shape[0] if self._cached_corr is not None else 0
        return (
            f"DynamicCorrelation(half_life={self._half_life}, "
            f"cached_assets={n})"
        )
