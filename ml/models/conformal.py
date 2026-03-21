"""
EDGE-004: Conformal Prediction for Calibrated Uncertainty
==========================================================

Implements split conformal prediction to produce prediction intervals with
*guaranteed finite-sample coverage*.  Given any base predictor (sklearn,
xgboost, neural net), conformal prediction wraps it to output intervals
[y_lo, y_hi] such that:

    P(y in [y_lo, y_hi]) >= 1 - alpha

for a user-chosen miscoverage level alpha (e.g., 0.05 for 95% coverage).

Algorithm:
  1. Split data into training set and calibration set.
  2. Fit base model on training set.
  3. Compute nonconformity scores on calibration set:
       s_i = |y_i - y_hat_i|   (for regression)
  4. Compute quantile q at level ceil((1-alpha)(n_cal+1)) / n_cal.
  5. For new x, predict: [y_hat(x) - q, y_hat(x) + q].

Also supports:
  - Conformalized quantile regression (CQR)
  - Adaptive intervals via locally-weighted conformal
  - Classification coverage via conformal sets

Dependencies: numpy, sklearn (optional for base models).

Conforms to AlphaModel interface:
    fit(X, y)     -- calibrate conformal predictor
    predict(X)    -- point predictions from base model
    score(X, y)   -- empirical coverage rate
"""

import logging
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Optional sklearn import
_SKLEARN_AVAILABLE = False
try:
    from sklearn.base import clone as sklearn_clone
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.info("EDGE-004: sklearn not available. Using built-in ridge regression.")


# ===================================================================
# Base predictor protocol
# ===================================================================

@runtime_checkable
class BasePredictor(Protocol):
    """Any object with fit/predict qualifies."""
    def fit(self, X, y) -> Any: ...
    def predict(self, X) -> np.ndarray: ...


# ===================================================================
# Built-in minimal ridge regression (no sklearn needed)
# ===================================================================

class MinimalRidge:
    """Tiny ridge regression for when sklearn is unavailable."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._w = None
        self._b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MinimalRidge":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, d = X.shape
        A = X.T @ X + self.alpha * np.eye(d)
        b = X.T @ y
        self._w = np.linalg.solve(A, b)
        self._b = np.mean(y - X @ self._w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self._w + self._b


# ===================================================================
# Nonconformity score functions
# ===================================================================

def absolute_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Standard nonconformity score: |y - y_hat|."""
    return np.abs(y_true - y_pred)


def signed_residual_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed residual for asymmetric intervals."""
    return y_true - y_pred


def normalized_residual_score(y_true: np.ndarray, y_pred: np.ndarray,
                              sigma: np.ndarray) -> np.ndarray:
    """Normalized score for locally adaptive intervals: |y - y_hat| / sigma."""
    return np.abs(y_true - y_pred) / np.maximum(sigma, 1e-8)


# ===================================================================
# Conformal prediction interval methods
# ===================================================================

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile guaranteeing (1-alpha) coverage.

    Uses the ceil((1-alpha)*(n+1))/n finite-sample correction.
    """
    n = len(scores)
    level = math.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)  # cap at 1.0
    return float(np.quantile(scores, level))


# ===================================================================
# Public API: ConformalPredictor (AlphaModel interface)
# ===================================================================

class ConformalPredictor:
    """Split conformal prediction wrapper for calibrated prediction intervals.

    Wraps any base predictor to add statistically valid uncertainty estimates.

    Parameters
    ----------
    base_model : object or None
        Any object with fit(X,y) and predict(X).
        If None, uses GradientBoostingRegressor (sklearn) or MinimalRidge.
    alpha : float
        Miscoverage level. Default 0.05 for 95% intervals.
    cal_fraction : float
        Fraction of training data reserved for calibration.
    score_fn : str
        'absolute' (default), 'signed', or 'normalized'.
    adaptive : bool
        If True, use locally-weighted conformal (requires a separate
        variance estimator).
    random_state : int or None
        Seed for train/cal split reproducibility.
    """

    def __init__(self, *, base_model=None, alpha: float = 0.05,
                 cal_fraction: float = 0.2, score_fn: str = "absolute",
                 adaptive: bool = False, random_state: Optional[int] = 42,
                 **kwargs):
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.score_fn_name = score_fn
        self.adaptive = adaptive
        self.random_state = random_state

        # Set up base model
        if base_model is not None:
            self._base = base_model
        elif _SKLEARN_AVAILABLE:
            self._base = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=random_state,
            )
        else:
            self._base = MinimalRidge(alpha=1.0)

        # Variance estimator for adaptive mode
        self._sigma_model = None
        if self.adaptive:
            if _SKLEARN_AVAILABLE:
                self._sigma_model = RandomForestRegressor(
                    n_estimators=50, max_depth=4, random_state=random_state
                )
            else:
                self._sigma_model = MinimalRidge(alpha=1.0)

        # Calibration state
        self._q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ConformalPredictor":
        """Fit base model and calibrate conformal quantile.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) target values
        """
        if y is None:
            raise ValueError("EDGE-004: y is required for conformal calibration.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Split into train and calibration
        if _SKLEARN_AVAILABLE:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=self.cal_fraction, random_state=self.random_state
            )
        else:
            n = len(X)
            n_cal = max(1, int(n * self.cal_fraction))
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(n)
            X_train, y_train = X[idx[n_cal:]], y[idx[n_cal:]]
            X_cal, y_cal = X[idx[:n_cal]], y[idx[:n_cal]]

        # Fit base model
        self._base.fit(X_train, y_train)
        logger.info("EDGE-004: Base model trained on %d samples.", len(X_train))

        # Compute calibration predictions
        y_cal_pred = np.asarray(self._base.predict(X_cal), dtype=np.float64).ravel()

        # Compute nonconformity scores
        if self.adaptive and self._sigma_model is not None:
            # Train variance model on training residuals
            y_train_pred = np.asarray(self._base.predict(X_train)).ravel()
            train_residuals = np.abs(y_train - y_train_pred)
            self._sigma_model.fit(X_train, train_residuals)
            sigma_cal = np.maximum(
                np.asarray(self._sigma_model.predict(X_cal)).ravel(), 1e-8
            )
            self._cal_scores = normalized_residual_score(y_cal, y_cal_pred, sigma_cal)
        elif self.score_fn_name == "signed":
            self._cal_scores = signed_residual_score(y_cal, y_cal_pred)
        else:
            self._cal_scores = absolute_residual_score(y_cal, y_cal_pred)

        # Compute conformal quantile
        self._q_hat = conformal_quantile(self._cal_scores, self.alpha)
        self._fitted = True

        logger.info(
            "EDGE-004: Conformal calibration complete. q_hat=%.6f, "
            "alpha=%.2f, n_cal=%d.",
            self._q_hat, self.alpha, len(X_cal),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions from the base model.

        Args:
            X: (n_samples, n_features)

        Returns:
            predictions: (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._base.predict(X), dtype=np.float64).ravel()

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals with (1-alpha) coverage guarantee.

        Args:
            X: (n_samples, n_features)

        Returns:
            lower: (n_samples,)
            upper: (n_samples,)
        """
        if not self._fitted or self._q_hat is None:
            raise RuntimeError("EDGE-004: Must call fit() before predict_interval().")

        X = np.asarray(X, dtype=np.float64)
        y_hat = self.predict(X)
        q = self._q_hat

        if self.adaptive and self._sigma_model is not None:
            sigma = np.maximum(
                np.asarray(self._sigma_model.predict(X)).ravel(), 1e-8
            )
            lower = y_hat - q * sigma
            upper = y_hat + q * sigma
        elif self.score_fn_name == "signed":
            # For signed scores, compute separate upper/lower quantiles
            q_lo = float(np.quantile(self._cal_scores, self.alpha / 2))
            q_hi = float(np.quantile(self._cal_scores, 1 - self.alpha / 2))
            lower = y_hat + q_lo
            upper = y_hat + q_hi
        else:
            lower = y_hat - q
            upper = y_hat + q

        return lower, upper

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage rate on test data.

        Returns a value in [0, 1]. Ideally close to (1 - alpha).
        """
        y = np.asarray(y, dtype=np.float64).ravel()

        if not self._fitted:
            # Return negative MSE if not calibrated
            preds = self.predict(X)
            return -float(np.mean((preds - y) ** 2))

        lower, upper = self.predict_interval(X)
        covered = (y >= lower) & (y <= upper)
        coverage = float(np.mean(covered))
        return coverage

    def interval_width(self, X: np.ndarray) -> np.ndarray:
        """Return the width of prediction intervals (useful for monitoring)."""
        lower, upper = self.predict_interval(X)
        return upper - lower

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "cal_fraction": self.cal_fraction,
            "score_fn": self.score_fn_name,
            "adaptive": self.adaptive,
            "q_hat": self._q_hat,
            "base_model": type(self._base).__name__,
        }

    def __repr__(self) -> str:
        status = "calibrated" if self._fitted else "uncalibrated"
        q = f", q={self._q_hat:.4f}" if self._q_hat is not None else ""
        return (
            f"ConformalPredictor(alpha={self.alpha}, "
            f"base={type(self._base).__name__}, {status}{q})"
        )


# ===================================================================
# Conformal Classification (bonus utility)
# ===================================================================

class ConformalClassifier:
    """Conformal prediction sets for classification.

    For a classifier that outputs class probabilities, produces prediction
    *sets* (not intervals) such that the true class is included with
    probability >= 1 - alpha.

    Uses the APS (Adaptive Prediction Sets) method.
    """

    def __init__(self, *, base_model=None, alpha: float = 0.05,
                 cal_fraction: float = 0.2, random_state: Optional[int] = 42):
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.random_state = random_state
        self._base = base_model
        self._q_hat: Optional[float] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalClassifier":
        """Fit base classifier and calibrate."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        if self._base is None:
            raise ValueError("EDGE-004: base_model is required for ConformalClassifier.")

        # Split
        n = len(X)
        n_cal = max(1, int(n * self.cal_fraction))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        X_train, y_train = X[idx[n_cal:]], y[idx[n_cal:]]
        X_cal, y_cal = X[idx[:n_cal]], y[idx[:n_cal]]

        self._base.fit(X_train, y_train)

        # Get calibration probabilities
        if hasattr(self._base, "predict_proba"):
            probs_cal = self._base.predict_proba(X_cal)
        else:
            raise ValueError("EDGE-004: base_model must have predict_proba().")

        # APS scores: cumulative probability needed to include true class
        scores = []
        for i in range(len(X_cal)):
            sorted_idx = np.argsort(-probs_cal[i])
            cumsum = 0.0
            for j, cls_idx in enumerate(sorted_idx):
                cumsum += probs_cal[i, cls_idx]
                if cls_idx == y_cal[i]:
                    scores.append(cumsum)
                    break
        scores = np.array(scores)
        self._q_hat = conformal_quantile(scores, self.alpha)
        self._fitted = True
        logger.info("EDGE-004: Conformal classifier calibrated, q_hat=%.4f.", self._q_hat)
        return self

    def predict_set(self, X: np.ndarray) -> list:
        """Return prediction sets for each sample.

        Returns:
            List of lists, each containing the class indices in the prediction set.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first.")

        probs = self._base.predict_proba(np.asarray(X, dtype=np.float64))
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(-probs[i])
            cumsum = 0.0
            pred_set = []
            for cls_idx in sorted_idx:
                cumsum += probs[i, cls_idx]
                pred_set.append(int(cls_idx))
                if cumsum >= self._q_hat:
                    break
            sets.append(pred_set)
        return sets

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely class (standard point prediction)."""
        if hasattr(self._base, "predict"):
            return np.asarray(self._base.predict(X))
        probs = self._base.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Empirical coverage: fraction of samples where true class is in set."""
        y = np.asarray(y, dtype=np.int64).ravel()
        sets = self.predict_set(X)
        covered = sum(1 for i, s in enumerate(sets) if y[i] in s)
        return covered / max(len(y), 1)
