"""
EDGE-005: Topological Data Analysis for Regime Detection
=========================================================

Uses persistent homology to detect market regime changes from the "shape"
of return distributions in sliding windows.

Core idea: embed a sliding window of returns into a point cloud (via
time-delay embedding), compute persistent homology, and use topological
features (Betti numbers, persistence diagrams, persistence landscapes)
to classify the current regime.

Regimes detected:
  - 0: Low volatility / mean-reverting
  - 1: Trending
  - 2: High volatility / crisis

If `ripser` or `giotto-tda` is available, uses Vietoris-Rips persistence.
Otherwise, falls back to a distance-matrix heuristic using scipy.

Conforms to AlphaModel interface:
    fit(X, y)     -- learn regime boundaries from labeled data
    predict(X)    -- detect current regime
    score(X, y)   -- regime classification accuracy
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt TDA library imports
# ---------------------------------------------------------------------------
_RIPSER_AVAILABLE = False
_GIOTTO_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    from ripser import ripser
    _RIPSER_AVAILABLE = True
    logger.debug("EDGE-005: ripser available.")
except ImportError:
    pass

if not _RIPSER_AVAILABLE:
    try:
        from gtda.homology import VietorisRipsPersistence
        _GIOTTO_AVAILABLE = True
        logger.debug("EDGE-005: giotto-tda available.")
    except ImportError:
        pass

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import fcluster, linkage
    _SCIPY_AVAILABLE = True
except ImportError:
    logger.info("EDGE-005: scipy not available. Using numpy-only fallback.")


# ===================================================================
# Time-delay embedding
# ===================================================================

def time_delay_embedding(x: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """Embed a 1-D time series into a higher-dimensional point cloud.

    Args:
        x: (n,) 1-D time series
        dim: embedding dimension
        tau: time delay

    Returns:
        point_cloud: (n - (dim-1)*tau, dim)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    n_points = n - (dim - 1) * tau
    if n_points <= 0:
        raise ValueError(
            f"Series length {n} too short for dim={dim}, tau={tau}."
        )
    cloud = np.empty((n_points, dim), dtype=np.float64)
    for d in range(dim):
        cloud[:, d] = x[d * tau: d * tau + n_points]
    return cloud


# ===================================================================
# Persistent homology computation
# ===================================================================

def compute_persistence_ripser(point_cloud: np.ndarray,
                                max_dim: int = 1) -> List[np.ndarray]:
    """Compute persistence diagrams using ripser."""
    result = ripser(point_cloud, maxdim=max_dim)
    return result["dgms"]


def compute_persistence_giotto(point_cloud: np.ndarray,
                                max_dim: int = 1) -> List[np.ndarray]:
    """Compute persistence diagrams using giotto-tda."""
    vr = VietorisRipsPersistence(homology_dimensions=list(range(max_dim + 1)))
    # giotto expects (n_samples, n_points, n_dims) for fit_transform
    diagrams = vr.fit_transform(point_cloud[np.newaxis, :, :])
    # Convert giotto format to list of arrays per dimension
    result = []
    for d in range(max_dim + 1):
        mask = diagrams[0, :, 2] == d
        if mask.any():
            result.append(diagrams[0, mask, :2])
        else:
            result.append(np.empty((0, 2)))
    return result


def compute_persistence_fallback(point_cloud: np.ndarray,
                                  max_dim: int = 1) -> List[np.ndarray]:
    """Approximate persistence features using distance statistics.

    Not true persistent homology, but captures similar structural information
    for regime detection purposes.
    """
    if _SCIPY_AVAILABLE:
        dists = pdist(point_cloud)
        dist_matrix = squareform(dists)
    else:
        # Pairwise Euclidean distances via numpy
        diff = point_cloud[:, np.newaxis, :] - point_cloud[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    # Approximate H0 (connected components):
    # births at 0, deaths at distances where components merge
    sorted_dists = np.sort(dists)
    n_points = point_cloud.shape[0]
    # Take evenly spaced "death" times
    step = max(1, len(sorted_dists) // n_points)
    deaths_h0 = sorted_dists[::step][:n_points]
    births_h0 = np.zeros_like(deaths_h0)
    dgm_h0 = np.column_stack([births_h0, deaths_h0])

    # Approximate H1 (loops): use distance distribution statistics
    # Births ~ median distance, deaths ~ high quantile
    if len(dists) > 10:
        q25, q50, q75, q90 = np.quantile(dists, [0.25, 0.5, 0.75, 0.9])
        dgm_h1 = np.array([[q25, q75], [q50, q90]])
    else:
        dgm_h1 = np.empty((0, 2))

    result = [dgm_h0]
    if max_dim >= 1:
        result.append(dgm_h1)
    return result


def compute_persistence(point_cloud: np.ndarray,
                         max_dim: int = 1) -> List[np.ndarray]:
    """Dispatch to best available persistence computation."""
    if _RIPSER_AVAILABLE:
        return compute_persistence_ripser(point_cloud, max_dim)
    elif _GIOTTO_AVAILABLE:
        return compute_persistence_giotto(point_cloud, max_dim)
    else:
        return compute_persistence_fallback(point_cloud, max_dim)


# ===================================================================
# Topological feature extraction
# ===================================================================

def extract_topo_features(diagrams: List[np.ndarray]) -> np.ndarray:
    """Extract numerical features from persistence diagrams.

    Features per homology dimension:
      - Number of features (Betti number proxy)
      - Mean persistence (death - birth)
      - Max persistence
      - Std of persistence
      - Persistence entropy
      - Total persistence (sum)

    Returns:
        feature_vector: (n_dims * 6,)
    """
    features = []
    for dgm in diagrams:
        if dgm.shape[0] == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        # Filter out infinite persistence
        finite_mask = np.isfinite(dgm[:, 1])
        dgm = dgm[finite_mask]
        if dgm.shape[0] == 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        persistence = dgm[:, 1] - dgm[:, 0]
        persistence = np.maximum(persistence, 0.0)

        n_features = float(len(persistence))
        mean_pers = float(np.mean(persistence))
        max_pers = float(np.max(persistence))
        std_pers = float(np.std(persistence)) if len(persistence) > 1 else 0.0

        # Persistence entropy
        total = np.sum(persistence)
        if total > 1e-12:
            p = persistence / total
            p = p[p > 0]
            entropy = -float(np.sum(p * np.log(p + 1e-12)))
        else:
            entropy = 0.0

        features.extend([n_features, mean_pers, max_pers, std_pers, entropy, total])

    return np.array(features, dtype=np.float64)


# ===================================================================
# Public API: TDARegimeDetector (AlphaModel interface)
# ===================================================================

class TDARegimeDetector:
    """Topological Data Analysis regime detector.

    Analyzes the topological structure of sliding windows of returns
    to classify the current market regime.

    Parameters
    ----------
    window_size : int
        Size of the sliding window (number of bars).
    embedding_dim : int
        Dimension for time-delay embedding.
    embedding_tau : int
        Time delay for embedding.
    max_homology_dim : int
        Maximum homology dimension to compute (0 or 1).
    n_regimes : int
        Number of distinct regimes to detect (default 3).
    normalize : bool
        Whether to z-score normalize windows before embedding.
    """

    # Regime labels
    REGIME_NAMES = {0: "low_vol", 1: "trending", 2: "crisis"}

    def __init__(self, *, window_size: int = 50, embedding_dim: int = 3,
                 embedding_tau: int = 1, max_homology_dim: int = 1,
                 n_regimes: int = 3, normalize: bool = True, **kwargs):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.embedding_tau = embedding_tau
        self.max_homology_dim = max_homology_dim
        self.n_regimes = n_regimes
        self.normalize = normalize

        # Learned regime centroids (from fit)
        self._centroids: Optional[np.ndarray] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._fitted = False

    def _returns_to_features(self, returns: np.ndarray) -> np.ndarray:
        """Convert a single window of returns to topological features."""
        if self.normalize:
            std = np.std(returns)
            if std > 1e-10:
                returns = (returns - np.mean(returns)) / std
            else:
                returns = returns - np.mean(returns)

        cloud = time_delay_embedding(returns, self.embedding_dim, self.embedding_tau)
        diagrams = compute_persistence(cloud, self.max_homology_dim)
        return extract_topo_features(diagrams)

    def _extract_windows(self, X: np.ndarray) -> np.ndarray:
        """Extract topological features from all windows in the data.

        Args:
            X: (n_samples,) 1-D returns or (n_samples, 1) or
               (n_windows, window_size) pre-windowed.

        Returns:
            features: (n_windows, n_topo_features)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            # Sliding window over 1-D series
            series = X.ravel()
            n = len(series)
            if n < self.window_size:
                logger.warning("EDGE-005: Series length %d < window_size %d.", n, self.window_size)
                return self._returns_to_features(series).reshape(1, -1)

            step = max(1, self.window_size // 4)
            windows = []
            for start in range(0, n - self.window_size + 1, step):
                w = series[start:start + self.window_size]
                windows.append(self._returns_to_features(w))
            return np.array(windows)

        elif X.ndim == 2 and X.shape[1] >= self.window_size:
            # Pre-windowed: each row is a window
            features = []
            for i in range(X.shape[0]):
                features.append(self._returns_to_features(X[i, :self.window_size]))
            return np.array(features)

        elif X.ndim == 2:
            # Assume each row is a feature vector already (pass-through for pre-computed)
            return X

        raise ValueError(f"EDGE-005: Unexpected input shape {X.shape}")

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TDARegimeDetector":
        """Learn regime centroids from historical data.

        If y is provided, uses supervised centroid computation.
        If y is None, uses unsupervised clustering (k-means-like).

        Args:
            X: returns series or pre-windowed data
            y: optional regime labels per window
        """
        features = self._extract_windows(X)
        n_windows, n_feat = features.shape

        # Normalize features
        self._feature_mean = features.mean(axis=0)
        self._feature_std = features.std(axis=0)
        self._feature_std[self._feature_std < 1e-10] = 1.0
        features_norm = (features - self._feature_mean) / self._feature_std

        if y is not None:
            # Supervised: compute centroid per label
            y = np.asarray(y, dtype=np.int64).ravel()
            if len(y) != n_windows:
                # If y corresponds to full series, subsample
                step = max(1, len(y) // n_windows)
                y = y[::step][:n_windows]
            centroids = []
            for regime in range(self.n_regimes):
                mask = y == regime
                if mask.any():
                    centroids.append(features_norm[mask].mean(axis=0))
                else:
                    centroids.append(np.zeros(n_feat))
            self._centroids = np.array(centroids)
        else:
            # Unsupervised: simple k-means
            self._centroids = self._kmeans(features_norm, self.n_regimes)

        self._fitted = True
        logger.info("EDGE-005: TDA regime detector fitted on %d windows, %d features.",
                     n_windows, n_feat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels.

        Args:
            X: returns series or windowed data

        Returns:
            regimes: (n_windows,) integer labels
        """
        if not self._fitted:
            logger.warning("EDGE-005: Not fitted. Returning regime 0 (low_vol).")
            X = np.asarray(X, dtype=np.float64)
            n = 1 if X.ndim <= 1 else X.shape[0]
            return np.zeros(n, dtype=np.int64)

        features = self._extract_windows(X)
        features_norm = (features - self._feature_mean) / self._feature_std

        # Assign to nearest centroid
        dists = np.linalg.norm(
            features_norm[:, np.newaxis, :] - self._centroids[np.newaxis, :, :],
            axis=-1
        )
        return np.argmin(dists, axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft regime probabilities via inverse-distance weighting.

        Returns:
            probs: (n_windows, n_regimes) softmax over negative distances
        """
        if not self._fitted:
            n = np.asarray(X).shape[0] if np.asarray(X).ndim > 1 else 1
            probs = np.ones((n, self.n_regimes)) / self.n_regimes
            return probs

        features = self._extract_windows(X)
        features_norm = (features - self._feature_mean) / self._feature_std

        dists = np.linalg.norm(
            features_norm[:, np.newaxis, :] - self._centroids[np.newaxis, :, :],
            axis=-1
        )
        # Softmax over negative distances
        neg_dists = -dists
        exp_d = np.exp(neg_dists - neg_dists.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)
        return probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy of regime prediction."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        # Align lengths
        n = min(len(preds), len(y))
        return float(np.mean(preds[:n] == y[:n]))

    def regime_name(self, label: int) -> str:
        """Get human-readable name for a regime label."""
        return self.REGIME_NAMES.get(label, f"regime_{label}")

    # ------------------------------------------------------------------
    # Internal k-means
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 100,
                seed: int = 42) -> np.ndarray:
        """Simple k-means clustering."""
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        idx = rng.choice(n, size=min(k, n), replace=False)
        centroids = X[idx].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(
                X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1
            )
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centroids[c] = X[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "embedding_dim": self.embedding_dim,
            "embedding_tau": self.embedding_tau,
            "max_homology_dim": self.max_homology_dim,
            "n_regimes": self.n_regimes,
            "backend": "ripser" if _RIPSER_AVAILABLE else
                       "giotto" if _GIOTTO_AVAILABLE else "fallback",
        }

    def __repr__(self) -> str:
        backend = ("ripser" if _RIPSER_AVAILABLE else
                   "giotto" if _GIOTTO_AVAILABLE else "fallback")
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"TDARegimeDetector(window={self.window_size}, "
            f"backend={backend}, {status})"
        )
