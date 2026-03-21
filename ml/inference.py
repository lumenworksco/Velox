"""PROD-002: Batch ML inference module — predict across all symbols per scan cycle.

Loads a trained model (sklearn-compatible or joblib-serialized) and runs
batch predictions to minimize per-symbol overhead. Supports graceful
degradation: if the model is unavailable or prediction fails, returns
neutral scores so that the trading engine continues without ML signals.

Usage:
    from ml.inference import BatchInferenceEngine

    engine = BatchInferenceEngine(model_path="models/ensemble_v1.joblib")
    predictions = engine.predict_batch(feature_df)
    # predictions: dict[str, float] mapping symbol -> score
"""

import logging
import os
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """Batch prediction engine for ML-scored trading signals.

    Loads a trained model once and reuses it across scan cycles.
    All operations are fail-open: errors return neutral predictions
    rather than raising exceptions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        default_score: float = 0.5,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to a joblib/pickle model file.
            feature_columns: Expected feature column names (for validation).
            default_score: Score returned when prediction fails (0.5 = neutral).
        """
        self._model: Any = None
        self._model_path = model_path
        self._feature_columns = feature_columns
        self._default_score = default_score
        self._loaded = False
        self._load_error: Optional[str] = None

        # Performance tracking
        self._total_predictions = 0
        self._total_errors = 0
        self._last_inference_time_ms: float = 0.0

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> bool:
        """Load a trained model from disk.

        Supports joblib and pickle formats. Returns True on success.
        """
        try:
            if not os.path.exists(path):
                self._load_error = f"Model file not found: {path}"
                logger.warning("PROD-002: %s", self._load_error)
                return False

            import joblib
            self._model = joblib.load(path)
            self._loaded = True
            self._load_error = None
            logger.info(
                "PROD-002: Loaded model from %s (type=%s)",
                path, type(self._model).__name__,
            )
            return True
        except ImportError:
            # Fall back to pickle if joblib not installed
            try:
                import pickle
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
                self._loaded = True
                self._load_error = None
                logger.info("PROD-002: Loaded model via pickle from %s", path)
                return True
            except Exception as e:
                self._load_error = f"Failed to load model: {e}"
                logger.error("PROD-002: %s", self._load_error)
                return False
        except Exception as e:
            self._load_error = f"Failed to load model: {e}"
            logger.error("PROD-002: %s", self._load_error)
            return False

    def reload_model(self, path: Optional[str] = None) -> bool:
        """Reload the model (e.g., after retraining).

        Args:
            path: Optional new model path. Uses original path if not provided.

        Returns:
            True if model loaded successfully.
        """
        target = path or self._model_path
        if not target:
            logger.warning("PROD-002: No model path specified for reload")
            return False
        self._model_path = target
        return self._load_model(target)

    def predict_batch(
        self,
        features: pd.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run batch prediction across all symbols.

        Args:
            features: DataFrame where each row is a symbol and columns are features.
                      Index should be symbol names, or provide `symbols` list.
            symbols: Optional symbol list (if features.index is not symbol names).

        Returns:
            Dict mapping symbol -> predicted score. Returns default_score for
            any symbol where prediction fails.
        """
        if symbols is None:
            symbols = list(features.index)

        # Graceful degradation: return neutral scores if model not loaded
        if not self._loaded or self._model is None:
            logger.debug(
                "PROD-002: Model not loaded, returning default scores for %d symbols",
                len(symbols),
            )
            return {sym: self._default_score for sym in symbols}

        try:
            start = _time.perf_counter()

            # Validate feature columns if specified
            if self._feature_columns:
                missing = set(self._feature_columns) - set(features.columns)
                if missing:
                    logger.warning(
                        "PROD-002: Missing %d feature columns: %s",
                        len(missing), list(missing)[:5],
                    )
                    # Fill missing columns with NaN
                    for col in missing:
                        features[col] = np.nan
                # Reorder columns to match expected order
                features = features.reindex(columns=self._feature_columns)

            # Handle NaN values
            features_clean = features.fillna(0.0)

            # Batch predict
            if hasattr(self._model, "predict_proba"):
                # Classification model — use probability of positive class
                probs = self._model.predict_proba(features_clean)
                scores = probs[:, 1] if probs.ndim > 1 else probs
            elif hasattr(self._model, "predict"):
                # Regression model
                scores = self._model.predict(features_clean)
            else:
                logger.error("PROD-002: Model has no predict or predict_proba method")
                return {sym: self._default_score for sym in symbols}

            elapsed_ms = (_time.perf_counter() - start) * 1000
            self._last_inference_time_ms = elapsed_ms
            self._total_predictions += len(symbols)

            results = {}
            for i, sym in enumerate(symbols):
                try:
                    results[sym] = float(scores[i])
                except (IndexError, ValueError):
                    results[sym] = self._default_score

            logger.info(
                "PROD-002: Batch inference complete — %d symbols in %.1fms (avg %.2fms/sym)",
                len(symbols), elapsed_ms, elapsed_ms / max(len(symbols), 1),
            )
            return results

        except Exception as e:
            self._total_errors += 1
            logger.error("PROD-002: Batch inference failed: %s", e, exc_info=True)
            return {sym: self._default_score for sym in symbols}

    def predict_single(self, symbol: str, features: pd.Series) -> float:
        """Predict for a single symbol (convenience wrapper).

        Args:
            symbol: Ticker symbol.
            features: Feature values as a pandas Series.

        Returns:
            Predicted score, or default_score on failure.
        """
        try:
            df = pd.DataFrame([features])
            df.index = [symbol]
            result = self.predict_batch(df, symbols=[symbol])
            return result.get(symbol, self._default_score)
        except Exception as e:
            logger.warning("PROD-002: Single prediction failed for %s: %s", symbol, e)
            return self._default_score

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded and ready for inference."""
        return self._loaded and self._model is not None

    def stats(self) -> dict:
        """Return inference engine statistics."""
        return {
            "model_loaded": self._loaded,
            "model_path": self._model_path,
            "model_type": type(self._model).__name__ if self._model else None,
            "load_error": self._load_error,
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "last_inference_time_ms": self._last_inference_time_ms,
        }
