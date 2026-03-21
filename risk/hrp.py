"""RISK-002: Hierarchical Risk Parity allocation.

Implementation of Lopez de Prado (2016) "Building Diversified Portfolios
that Outperform Out-of-Sample". Allocates capital across strategies using:

1. Compute correlation/covariance matrix from strategy returns
2. Hierarchical clustering (Ward linkage) on correlation distance
3. Quasi-diagonalize the covariance matrix along the dendrogram
4. Recursive bisection with inverse-variance allocation within clusters

Provides more robust diversification than mean-variance optimization,
particularly when the correlation matrix is noisy or non-stationary.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import config

logger = logging.getLogger(__name__)

# Allocation bounds per strategy
DEFAULT_MIN_ALLOC = 0.05   # 5% floor
DEFAULT_MAX_ALLOC = 0.50   # 50% ceiling
REBALANCE_INTERVAL_DAYS = 7  # Weekly rebalancing


@dataclass
class HRPResult:
    """Result of an HRP allocation computation."""
    allocations: dict[str, float]         # strategy -> weight (sum = 1.0)
    cluster_order: list[str]              # strategies ordered by dendrogram
    avg_correlation: float = 0.0
    timestamp: datetime | None = None


class HierarchicalRiskParity:
    """Hierarchical Risk Parity allocator for strategy capital distribution.

    Call compute_allocations() weekly (or on-demand) with a DataFrame of
    strategy returns. Returns a weight dict that sums to 1.0, respecting
    per-strategy floor/ceiling bounds.
    """

    def __init__(
        self,
        min_alloc: float = DEFAULT_MIN_ALLOC,
        max_alloc: float = DEFAULT_MAX_ALLOC,
        rebalance_days: int = REBALANCE_INTERVAL_DAYS,
    ):
        self._min_alloc = min_alloc
        self._max_alloc = max_alloc
        self._rebalance_days = rebalance_days

        self._last_result: HRPResult | None = None
        self._last_rebalance: datetime | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_allocations(
        self, strategy_returns: pd.DataFrame
    ) -> dict[str, float]:
        """Compute HRP allocations from a DataFrame of strategy returns.

        Args:
            strategy_returns: DataFrame with columns = strategy names,
                              rows = dates, values = daily returns.
                              Requires at least 20 observations.

        Returns:
            Dict of strategy -> allocation weight (sums to 1.0)
        """
        strategies = list(strategy_returns.columns)
        n = len(strategies)

        if n == 0:
            return {}
        if n == 1:
            return {strategies[0]: 1.0}

        # Need minimum observations for meaningful covariance
        min_obs = max(20, n + 1)
        if len(strategy_returns) < min_obs:
            logger.warning(
                f"HRP: only {len(strategy_returns)} observations, need {min_obs}. "
                f"Falling back to equal weight."
            )
            return self._equal_weight(strategies)

        # Drop rows with any NaN
        clean = strategy_returns.dropna()
        if len(clean) < min_obs:
            logger.warning("HRP: too many NaN rows after cleaning, using equal weight")
            return self._equal_weight(strategies)

        try:
            # Step 1: Correlation and covariance
            corr = clean.corr()
            cov = clean.cov()

            # MED-021: Clean correlation/covariance matrices
            # Replace NaN with 0 (uncorrelated assumption) and ensure PSD
            if corr.isna().any().any():
                logger.warning("HRP: NaN in correlation matrix — replacing with 0.0")
                corr = corr.fillna(0.0)
                np.fill_diagonal(corr.values, 1.0)
            if cov.isna().any().any():
                logger.warning("HRP: NaN in covariance matrix — replacing with 0.0")
                cov = cov.fillna(0.0)
            # Ensure covariance is positive semi-definite
            cov_vals = cov.values
            eigvals = np.linalg.eigvalsh(cov_vals)
            if np.any(eigvals < -1e-10):
                logger.warning("HRP: Covariance matrix not PSD — clamping negative eigenvalues")
                evals, evecs = np.linalg.eigh(cov_vals)
                evals = np.maximum(evals, 0.0)
                cov_clean = evecs @ np.diag(evals) @ evecs.T
                cov = pd.DataFrame(cov_clean, index=cov.index, columns=cov.columns)

            # Step 2: Hierarchical clustering on correlation distance
            dist = self._correlation_distance(corr)
            link = linkage(squareform(dist), method="ward")
            sort_idx = list(leaves_list(link))
            sorted_strategies = [strategies[i] for i in sort_idx]

            # Step 3: Quasi-diagonalize covariance matrix
            sorted_cov = cov.iloc[sort_idx, sort_idx]

            # Step 4: Recursive bisection
            raw_weights = self._recursive_bisection(sorted_cov, sorted_strategies)

            # Step 5: Apply bounds and normalize
            bounded = self._apply_bounds(raw_weights)

            avg_corr = float(
                corr.values[np.triu_indices_from(corr.values, k=1)].mean()
            ) if n > 1 else 0.0

            result = HRPResult(
                allocations=bounded,
                cluster_order=sorted_strategies,
                avg_correlation=avg_corr,
                timestamp=datetime.now(config.ET),
            )

            with self._lock:
                self._last_result = result
                self._last_rebalance = result.timestamp

            logger.info(
                f"HRP allocations computed: "
                + " ".join(f"{k}={v:.1%}" for k, v in bounded.items())
                + f" avg_corr={avg_corr:.3f}"
            )
            return bounded

        except Exception as e:
            logger.error(f"HRP computation failed: {e}", exc_info=True)
            return self._equal_weight(strategies)

    def should_rebalance(self) -> bool:
        """Check if enough time has passed since last rebalance."""
        with self._lock:
            if self._last_rebalance is None:
                return True
            elapsed = datetime.now(config.ET) - self._last_rebalance
            return elapsed >= timedelta(days=self._rebalance_days)

    # ------------------------------------------------------------------
    # HRP internals
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
        """Convert correlation matrix to a proper distance metric.

        d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
        """
        dist = np.sqrt(0.5 * (1.0 - corr.values))
        np.fill_diagonal(dist, 0.0)
        # Ensure symmetry (floating-point rounding)
        dist = (dist + dist.T) / 2.0
        return dist

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_items: list[str],
    ) -> dict[str, float]:
        """Allocate weights via top-down recursive bisection.

        At each step, split the sorted list in half and allocate between
        the two halves proportional to the inverse of their cluster variance.
        """
        weights = {s: 1.0 for s in sorted_items}
        clusters = [sorted_items]

        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Cluster variance = inverse-variance weighted
                left_var = self._cluster_variance(cov, left)
                right_var = self._cluster_variance(cov, right)

                total_var = left_var + right_var
                if total_var < 1e-12:
                    alpha = 0.5
                else:
                    # Allocate more to lower-variance cluster
                    alpha = 1.0 - left_var / total_var

                for s in left:
                    weights[s] *= alpha
                for s in right:
                    weights[s] *= (1.0 - alpha)

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            clusters = next_clusters

        return weights

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
        """Compute the inverse-variance portfolio variance for a cluster.

        w_i = (1/sigma_i^2) / sum(1/sigma_j^2)
        cluster_var = w' * Cov * w
        """
        sub_cov = cov.loc[items, items].values
        diag = np.diag(sub_cov)

        # Inverse variance weights
        inv_var = np.where(diag > 1e-12, 1.0 / diag, 0.0)
        total_inv = inv_var.sum()
        if total_inv < 1e-12:
            w = np.ones(len(items)) / len(items)
        else:
            w = inv_var / total_inv

        cluster_var = float(w @ sub_cov @ w)
        return max(cluster_var, 0.0)

    def _apply_bounds(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp weights to [min_alloc, max_alloc] and renormalize to sum=1."""
        bounded = {}
        for s, w in weights.items():
            bounded[s] = max(self._min_alloc, min(self._max_alloc, w))

        # Renormalize
        total = sum(bounded.values())
        if total > 0:
            bounded = {s: w / total for s, w in bounded.items()}
        return bounded

    @staticmethod
    def _equal_weight(strategies: list[str]) -> dict[str, float]:
        """Fallback: equal-weight allocation."""
        n = len(strategies)
        if n == 0:
            return {}
        w = 1.0 / n
        return {s: w for s in strategies}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> HRPResult | None:
        with self._lock:
            return self._last_result

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "last_rebalance": (
                    self._last_rebalance.isoformat() if self._last_rebalance else None
                ),
                "min_alloc": self._min_alloc,
                "max_alloc": self._max_alloc,
                "rebalance_days": self._rebalance_days,
                "current_allocations": (
                    self._last_result.allocations if self._last_result else None
                ),
                "avg_correlation": (
                    self._last_result.avg_correlation if self._last_result else None
                ),
            }
