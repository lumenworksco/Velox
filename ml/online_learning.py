"""ML-003: Online Learning with Concept Drift Detection.

Supports exponential decay weighting, drift detection via prediction error
monitoring, and shadow model comparison for safe model swaps.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conditional imports
_HAS_SKLEARN = False
try:
    from sklearn.metrics import mean_squared_error
    _HAS_SKLEARN = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a concept drift check."""
    drift_detected: bool = False
    drift_score: float = 0.0
    error_trend: float = 0.0        # positive = error increasing
    window_error: float = 0.0       # recent window error
    baseline_error: float = 0.0     # historical baseline error
    samples_since_retrain: int = 0
    recommendation: str = ""        # "retrain", "monitor", "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "drift_score": round(self.drift_score, 4),
            "error_trend": round(self.error_trend, 4),
            "window_error": round(self.window_error, 4),
            "baseline_error": round(self.baseline_error, 4),
            "samples_since_retrain": self.samples_since_retrain,
            "recommendation": self.recommendation,
        }


@dataclass
class ShadowModelStats:
    """Performance tracking for a shadow model."""
    model_id: str = ""
    predictions: int = 0
    cumulative_error: float = 0.0
    recent_errors: List[float] = field(default_factory=list)
    created_at: float = 0.0

    @property
    def avg_error(self) -> float:
        if self.predictions == 0:
            return float("inf")
        return self.cumulative_error / self.predictions

    @property
    def recent_avg_error(self) -> float:
        if not self.recent_errors:
            return float("inf")
        return float(np.mean(self.recent_errors[-100:]))


# ---------------------------------------------------------------------------
# Online Learner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Online learning wrapper with drift detection and shadow models.

    Wraps a production model and monitors its prediction quality over time.
    When concept drift is detected, it can trigger retraining.  Shadow models
    run alongside and are swapped in when they outperform production.

    Usage::

        learner = OnlineLearner(production_model, retrain_fn=my_retrain)
        for obs in new_data_stream:
            learner.update(obs)
            if learner.detect_drift():
                learner.retrain()
            if learner.should_swap_model():
                learner.swap_best_shadow()
    """

    def __init__(
        self,
        production_model: Any = None,
        retrain_fn: Optional[Callable] = None,
        decay_factor: float = 0.995,
        drift_threshold: float = 2.0,
        drift_window: int = 100,
        baseline_window: int = 500,
        min_samples_for_drift: int = 50,
        max_observations: int = 10000,
    ):
        """Initialize the online learner.

        Args:
            production_model: The current production model (must have ``predict``).
            retrain_fn: Callable that returns a new trained model when invoked.
            decay_factor: Exponential decay weight (0.99 = slow decay, 0.9 = fast).
            drift_threshold: Ratio of recent-to-baseline error that triggers drift.
            drift_window: Number of recent observations for drift computation.
            baseline_window: Number of observations for baseline error.
            min_samples_for_drift: Minimum observations before drift detection activates.
            max_observations: Maximum stored observations (FIFO eviction).
        """
        self.production_model = production_model
        self.retrain_fn = retrain_fn
        self.decay_factor = decay_factor
        self.drift_threshold = drift_threshold
        self.drift_window = drift_window
        self.baseline_window = baseline_window
        self.min_samples_for_drift = min_samples_for_drift

        # Observation storage
        self._observations: deque = deque(maxlen=max_observations)
        self._errors: deque = deque(maxlen=max_observations)
        self._weighted_errors: deque = deque(maxlen=max_observations)
        self._timestamps: deque = deque(maxlen=max_observations)

        # Counters
        self._total_updates: int = 0
        self._samples_since_retrain: int = 0
        self._last_retrain_time: float = 0.0
        self._last_drift_report: Optional[DriftReport] = None

        # Shadow models
        self._shadows: Dict[str, Any] = {}
        self._shadow_stats: Dict[str, ShadowModelStats] = {}

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------

    def update(self, observation: Dict[str, Any]) -> Optional[float]:
        """Process a new observation.

        An observation must contain:
            - ``features``: Dict[str, float] or pd.Series or np.ndarray
            - ``label``: float (the true outcome)
            - ``timestamp`` (optional): float epoch

        Args:
            observation: Dict with ``features`` and ``label``.

        Returns:
            Prediction error for this observation, or None if no model.
        """
        features = observation.get("features")
        label = observation.get("label")
        ts = observation.get("timestamp", time.time())

        if features is None or label is None:
            logger.debug("Skipping observation: missing features or label")
            return None

        self._total_updates += 1
        self._samples_since_retrain += 1
        self._timestamps.append(ts)
        self._observations.append(observation)

        error = None

        # --- Production model prediction + error ---
        if self.production_model is not None:
            try:
                pred = self._predict_single(self.production_model, features)
                error = float(abs(pred - label))

                # Exponential decay weight
                weight = self.decay_factor ** max(0, self._total_updates - 1)
                self._errors.append(error)
                self._weighted_errors.append(error * weight)
            except Exception as e:
                logger.warning("Production model prediction failed: %s", e)
                self._errors.append(0.0)
                self._weighted_errors.append(0.0)

        # --- Shadow model predictions ---
        for model_id, shadow_model in list(self._shadows.items()):
            try:
                shadow_pred = self._predict_single(shadow_model, features)
                shadow_error = float(abs(shadow_pred - label))
                stats = self._shadow_stats[model_id]
                stats.predictions += 1
                stats.cumulative_error += shadow_error
                stats.recent_errors.append(shadow_error)
                # Keep recent_errors bounded
                if len(stats.recent_errors) > 200:
                    stats.recent_errors = stats.recent_errors[-200:]
            except Exception as e:
                logger.debug("Shadow model %s prediction failed: %s", model_id, e)

        return error

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self) -> bool:
        """Detect concept drift by comparing recent vs baseline prediction error.

        Uses a sliding window comparison: if the ratio of recent error to
        baseline error exceeds ``drift_threshold``, drift is declared.

        Returns:
            True if drift is detected.
        """
        report = self._compute_drift_report()
        self._last_drift_report = report
        return report.drift_detected

    def get_drift_report(self) -> DriftReport:
        """Return the most recent drift report, computing if needed."""
        if self._last_drift_report is None:
            self.detect_drift()
        return self._last_drift_report  # type: ignore[return-value]

    def _compute_drift_report(self) -> DriftReport:
        """Internal drift computation."""
        report = DriftReport(samples_since_retrain=self._samples_since_retrain)

        if len(self._errors) < self.min_samples_for_drift:
            report.recommendation = "insufficient_data"
            return report

        errors = np.array(list(self._errors))

        # Recent window error
        recent = errors[-self.drift_window:]
        report.window_error = float(np.mean(recent))

        # Baseline error (earlier data)
        if len(errors) > self.drift_window:
            baseline_end = len(errors) - self.drift_window
            baseline_start = max(0, baseline_end - self.baseline_window)
            baseline = errors[baseline_start:baseline_end]
            report.baseline_error = float(np.mean(baseline))
        else:
            report.baseline_error = report.window_error
            report.recommendation = "monitor"
            return report

        # Drift score: ratio of recent to baseline error
        if report.baseline_error > 1e-8:
            report.drift_score = report.window_error / report.baseline_error
        else:
            report.drift_score = 1.0

        # Error trend (slope over recent window)
        if len(recent) >= 10:
            x = np.arange(len(recent), dtype=float)
            coeffs = np.polyfit(x, recent, 1)
            report.error_trend = float(coeffs[0])

        # Decision
        if report.drift_score >= self.drift_threshold:
            report.drift_detected = True
            report.recommendation = "retrain"
        elif report.drift_score >= self.drift_threshold * 0.75:
            report.recommendation = "monitor"
        else:
            report.recommendation = "stable"

        return report

    # ------------------------------------------------------------------
    # Shadow models
    # ------------------------------------------------------------------

    def add_shadow_model(self, model_id: str, model: Any) -> None:
        """Register a shadow model to run alongside production.

        Args:
            model_id: Unique identifier for the shadow model.
            model: Model object with a ``predict`` method.
        """
        self._shadows[model_id] = model
        self._shadow_stats[model_id] = ShadowModelStats(
            model_id=model_id,
            created_at=time.time(),
        )
        logger.info("Shadow model registered: %s", model_id)

    def remove_shadow_model(self, model_id: str) -> None:
        """Remove a shadow model."""
        self._shadows.pop(model_id, None)
        self._shadow_stats.pop(model_id, None)

    def should_swap_model(self, min_predictions: int = 50) -> bool:
        """Check if any shadow model outperforms production.

        A shadow must have at least ``min_predictions`` and a lower
        recent average error than production.

        Args:
            min_predictions: Minimum predictions before a shadow is eligible.

        Returns:
            True if a shadow model is outperforming production.
        """
        if not self._shadows or len(self._errors) < min_predictions:
            return False

        prod_recent = np.mean(list(self._errors)[-100:])

        for model_id, stats in self._shadow_stats.items():
            if stats.predictions < min_predictions:
                continue
            if stats.recent_avg_error < prod_recent * 0.9:  # 10% better
                logger.info(
                    "Shadow %s outperforming production: %.4f vs %.4f",
                    model_id, stats.recent_avg_error, prod_recent,
                )
                return True

        return False

    def get_best_shadow(self, min_predictions: int = 50) -> Optional[str]:
        """Return the model_id of the best-performing shadow, or None."""
        best_id = None
        best_error = float("inf")

        for model_id, stats in self._shadow_stats.items():
            if stats.predictions >= min_predictions:
                if stats.recent_avg_error < best_error:
                    best_error = stats.recent_avg_error
                    best_id = model_id

        return best_id

    def swap_best_shadow(self, min_predictions: int = 50) -> bool:
        """Swap the best shadow model into production.

        Returns:
            True if a swap occurred.
        """
        best_id = self.get_best_shadow(min_predictions)
        if best_id is None:
            return False

        prod_recent = np.mean(list(self._errors)[-100:]) if self._errors else float("inf")
        shadow_error = self._shadow_stats[best_id].recent_avg_error

        if shadow_error >= prod_recent:
            logger.info("Best shadow %s (%.4f) not better than production (%.4f) — no swap",
                        best_id, shadow_error, prod_recent)
            return False

        old_model = self.production_model
        self.production_model = self._shadows.pop(best_id)
        self._shadow_stats.pop(best_id, None)
        self._samples_since_retrain = 0
        self._last_retrain_time = time.time()

        # Optionally keep old production as shadow for comparison
        if old_model is not None:
            self.add_shadow_model("prev_production", old_model)

        logger.info("Swapped shadow %s into production (error: %.4f -> %.4f)",
                     best_id, prod_recent, shadow_error)
        return True

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def retrain(self) -> bool:
        """Trigger retraining using the stored retrain function.

        Returns:
            True if retraining succeeded and model was swapped.
        """
        if self.retrain_fn is None:
            logger.warning("No retrain_fn provided — cannot retrain")
            return False

        try:
            logger.info("Triggering retrain (samples_since=%d)", self._samples_since_retrain)
            new_model = self.retrain_fn()
            if new_model is not None:
                # Add as shadow first, let it prove itself
                shadow_id = f"retrained_{int(time.time())}"
                self.add_shadow_model(shadow_id, new_model)
                logger.info("Retrained model added as shadow: %s", shadow_id)
                return True
            else:
                logger.warning("Retrain function returned None")
                return False
        except Exception as e:
            logger.error("Retraining failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Exponential decay weighting
    # ------------------------------------------------------------------

    def get_sample_weights(self, n_samples: int) -> np.ndarray:
        """Generate exponential decay weights for the most recent n_samples.

        Most recent sample gets weight 1.0, older samples decay.

        Args:
            n_samples: Number of samples to weight.

        Returns:
            Array of weights, newest-first.
        """
        weights = np.array([
            self.decay_factor ** i for i in range(n_samples)
        ])
        # Normalize so they sum to n_samples (preserves effective sample size)
        weights = weights * n_samples / weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current online learning statistics."""
        stats: Dict[str, Any] = {
            "total_updates": self._total_updates,
            "samples_since_retrain": self._samples_since_retrain,
            "stored_observations": len(self._observations),
            "has_production_model": self.production_model is not None,
            "n_shadow_models": len(self._shadows),
        }

        if self._errors:
            errors = np.array(list(self._errors))
            stats["mean_error"] = float(np.mean(errors))
            stats["recent_mean_error"] = float(np.mean(errors[-100:]))
            stats["error_std"] = float(np.std(errors))

        if self._last_drift_report:
            stats["drift"] = self._last_drift_report.to_dict()

        shadow_info = {}
        for model_id, s in self._shadow_stats.items():
            shadow_info[model_id] = {
                "predictions": s.predictions,
                "avg_error": round(s.avg_error, 4),
                "recent_avg_error": round(s.recent_avg_error, 4),
            }
        stats["shadows"] = shadow_info

        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict_single(self, model: Any, features: Any) -> float:
        """Make a single prediction from various feature formats."""
        if isinstance(features, dict):
            # Convert dict to DataFrame row
            df = pd.DataFrame([features])
            pred = model.predict(df)
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features.to_dict()])
            pred = model.predict(df)
        elif isinstance(features, np.ndarray):
            pred = model.predict(features.reshape(1, -1))
        else:
            pred = model.predict(features)

        if isinstance(pred, (np.ndarray, list)):
            return float(pred[0]) if len(pred) > 0 else 0.0
        return float(pred)
