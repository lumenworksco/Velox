"""V3: ML Signal Filter — XGBoost classifier trained on historical trade outcomes."""

import json
import logging
import joblib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import database

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "hour", "minute", "day_of_week",
    "volume_ratio", "rsi_at_signal", "distance_from_vwap_pct", "atr_pct",
    "spy_day_pct", "regime_encoded",
    "orb_range_pct", "breakout_strength",
    "vwap_band_distance", "intraday_move_pct",
]


class MLSignalFilter:
    """XGBoost (or RandomForest fallback) classifier for signal quality."""

    def __init__(self):
        self._models: dict = {}  # strategy -> trained model
        self._active: dict = {}  # strategy -> bool (meets precision threshold)
        self._loaded = False

    def load_models(self):
        """Load pre-trained models from disk."""
        for strategy in ("ORB", "VWAP", "MOMENTUM", "GAP_GO"):
            path = MODELS_DIR / f"signal_filter_{strategy}.pkl"
            if path.exists():
                try:
                    data = joblib.load(path)
                    self._models[strategy] = data["model"]
                    self._active[strategy] = data.get("active", False)
                    logger.info(
                        f"ML filter loaded for {strategy}: "
                        f"precision={data.get('precision', 0):.3f}, "
                        f"active={self._active[strategy]}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load ML model for {strategy}: {e}")
        self._loaded = True

    def should_trade(self, strategy: str, features: dict) -> float:
        """Return probability that this signal will be profitable.

        Returns 1.0 (always trade) if:
        - ML filter is disabled
        - No model exists for this strategy
        - Model didn't meet precision threshold
        """
        if not config.USE_ML_FILTER:
            return 1.0
        if not self._loaded:
            self.load_models()
        if strategy not in self._models or not self._active.get(strategy, False):
            return 1.0

        try:
            model = self._models[strategy]
            X = self._features_to_array(features)
            proba = model.predict_proba(X)[0]
            # Return probability of class 1 (profitable)
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            logger.error(f"ML prediction failed for {strategy}: {e}")
            return 1.0  # Fail open — allow trade

    def train(self, strategy: str) -> dict | None:
        """Train model for a specific strategy using historical data.

        Returns metrics dict or None if insufficient data.
        """
        # Get labeled data from DB
        all_data = database.get_signals_with_outcomes()
        strategy_data = [r for r in all_data if r["strategy"] == strategy]

        if len(strategy_data) < config.ML_MIN_TRADES:
            logger.info(
                f"ML filter: {strategy} has {len(strategy_data)} trades "
                f"(need {config.ML_MIN_TRADES}), skipping training"
            )
            return None

        logger.info(f"ML filter: Training {strategy} on {len(strategy_data)} trades...")

        # Build feature matrix
        df = pd.DataFrame(strategy_data)
        features_df = self._build_features_from_db(df, strategy)
        if features_df is None or len(features_df) < config.ML_MIN_TRADES:
            logger.warning(f"ML filter: Not enough valid features for {strategy}")
            return None

        X = features_df[FEATURE_NAMES].values
        y = features_df["profitable"].values

        # Chronological split (80/20) — NO random split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_test) < 10:
            logger.warning(f"ML filter: Test set too small for {strategy}")
            return None

        # Train model
        model = self._create_model()
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        active = precision >= config.ML_MIN_PRECISION
        version = datetime.now(config.ET).strftime("%Y%m%d_%H%M")

        logger.info(
            f"ML filter {strategy}: precision={precision:.3f}, "
            f"recall={recall:.3f}, f1={f1:.3f}, "
            f"active={'YES' if active else 'NO (below threshold)'}"
        )

        # Save model
        model_data = {
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "active": active,
            "version": version,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_NAMES,
        }
        path = MODELS_DIR / f"signal_filter_{strategy}.pkl"
        joblib.dump(model_data, path)

        self._models[strategy] = model
        self._active[strategy] = active

        # Log to DB
        database.log_model_performance(
            strategy=strategy,
            train_samples=len(X_train),
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            features_used=FEATURE_NAMES,
            model_version=version,
        )

        return {
            "strategy": strategy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "active": active,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def retrain_all(self):
        """Retrain models for all strategies. Called Sunday midnight."""
        results = {}
        for strategy in ("ORB", "VWAP", "MOMENTUM", "GAP_GO"):
            result = self.train(strategy)
            if result:
                results[strategy] = result
        return results

    def _create_model(self):
        """Create classifier — XGBoost preferred, RandomForest fallback."""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
        except ImportError:
            logger.info("XGBoost not available, using RandomForest fallback")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
            )

    def _features_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array in correct order."""
        row = []
        for name in FEATURE_NAMES:
            val = features.get(name, 0.0)
            row.append(float(val) if val is not None else 0.0)
        return np.array([row])

    def _build_features_from_db(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame | None:
        """Build feature matrix from database records.

        Extracts time features from timestamps and fills strategy-specific
        features with defaults where not applicable.
        """
        try:
            records = []
            for _, row in df.iterrows():
                ts = pd.Timestamp(row["timestamp"])
                entry_price = row.get("entry_price", 0) or 0

                feat = {
                    "hour": ts.hour,
                    "minute": ts.minute,
                    "day_of_week": ts.weekday(),
                    # These will be approximate from available data
                    "volume_ratio": 1.0,  # Default — exact value not stored in signals
                    "rsi_at_signal": 50.0,  # Default
                    "distance_from_vwap_pct": 0.0,
                    "atr_pct": 1.0,  # Default 1%
                    "spy_day_pct": 0.0,
                    "regime_encoded": 1,  # Default bearish
                    "orb_range_pct": 0.0,
                    "breakout_strength": 0.0,
                    "vwap_band_distance": 0.0,
                    "intraday_move_pct": 0.0,
                    "profitable": row.get("profitable", 0),
                }
                records.append(feat)

            if not records:
                return None
            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None


def extract_live_features(signal, regime: str, market_data: dict | None = None) -> dict:
    """Extract features from a live signal for ML prediction.

    Args:
        signal: Signal dataclass from strategy
        regime: Current market regime string
        market_data: Optional dict with live market context
    """
    from datetime import datetime as dt
    now = dt.now(config.ET)
    md = market_data or {}

    regime_map = {"BULLISH": 2, "BEARISH": 1, "UNKNOWN": 0}

    features = {
        "hour": now.hour,
        "minute": now.minute,
        "day_of_week": now.weekday(),
        "volume_ratio": md.get("volume_ratio", 1.0),
        "rsi_at_signal": md.get("rsi", 50.0),
        "distance_from_vwap_pct": md.get("vwap_distance_pct", 0.0),
        "atr_pct": md.get("atr_pct", 1.0),
        "spy_day_pct": md.get("spy_day_pct", 0.0),
        "regime_encoded": regime_map.get(regime, 0),
        "orb_range_pct": md.get("orb_range_pct", 0.0),
        "breakout_strength": md.get("breakout_strength", 0.0),
        "vwap_band_distance": md.get("vwap_band_distance", 0.0),
        "intraday_move_pct": md.get("intraday_move_pct", 0.0),
    }
    return features


# Module-level singleton
ml_filter = MLSignalFilter()
