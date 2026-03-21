"""LPRADO-002: Triple-Barrier Meta-Labeling.

Implements the triple-barrier labeling method and meta-labeling framework
from Lopez de Prado's *Advances in Financial Machine Learning* (Chapter 3).

Triple-barrier labeling assigns each trade a label based on which of three
barriers is hit first:
    1. Upper barrier (take-profit): price rises by tp_mult * volatility
    2. Lower barrier (stop-loss): price falls by sl_mult * volatility
    3. Vertical barrier (time expiry): max_hold bars elapse

Meta-labeling trains a secondary model to predict the probability that a
primary model's signal will be profitable, enabling position sizing and
filtering of low-confidence trades.

Usage:
    ml = MetaLabeler()
    labels = ml.generate_labels(prices, vol, tp_mult=2, sl_mult=2, max_hold=10)
    meta_model = ml.train_meta_model(signals, features)
    proba = meta_model.predict_proba(new_features)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

_HAS_SKLEARN = False
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    logger.debug("scikit-learn not available — meta-model training disabled")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BarrierEvent:
    """A single triple-barrier outcome."""
    entry_idx: int = 0
    entry_price: float = 0.0
    exit_idx: int = 0
    exit_price: float = 0.0
    return_pct: float = 0.0
    label: int = 0          # +1 = TP hit, -1 = SL hit, 0 = time expiry
    barrier_hit: str = ""   # "upper", "lower", "vertical"
    holding_period: int = 0


@dataclass
class MetaModel:
    """Container for a trained meta-labeling model."""
    model: Any = None
    scaler: Any = None
    feature_names: List[str] = field(default_factory=list)
    threshold: float = 0.5
    metrics: Dict[str, float] = field(default_factory=dict)
    trained_at: float = 0.0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability that the primary signal is profitable.

        Args:
            features: Feature DataFrame aligned to ``self.feature_names``.

        Returns:
            Array of shape (n_samples, 2) with [P(unprofitable), P(profitable)].
        """
        if self.model is None:
            raise RuntimeError("MetaModel has not been trained")

        X = self._prepare_features(features)
        return self.model.predict_proba(X)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Binary prediction: 1 = take the trade, 0 = skip."""
        proba = self.predict_proba(features)
        return (proba[:, 1] >= self.threshold).astype(int)

    def get_bet_size(self, features: pd.DataFrame) -> np.ndarray:
        """Position sizing based on meta-model confidence.

        Returns values in [0, 1] that can be multiplied by the base
        position size. Higher confidence = larger position.

        Args:
            features: Feature DataFrame.

        Returns:
            Array of bet sizes in [0, 1].
        """
        proba = self.predict_proba(features)
        # Map probability to bet size: 2*(p - 0.5) clipped to [0, 1]
        raw = 2.0 * (proba[:, 1] - 0.5)
        return np.clip(raw, 0.0, 1.0)

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Align and scale features for prediction."""
        if self.feature_names:
            missing = set(self.feature_names) - set(features.columns)
            if missing:
                logger.warning("Missing meta-model features filled with 0: %s", missing)
                features = features.copy()
                for col in missing:
                    features[col] = 0.0
            X = features[self.feature_names].values.astype(np.float64)
        else:
            X = features.values.astype(np.float64)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X


# ---------------------------------------------------------------------------
# Meta-Labeler
# ---------------------------------------------------------------------------

class MetaLabeler:
    """Triple-barrier labeling and meta-model training.

    Usage::

        ml = MetaLabeler()
        labels = ml.generate_labels(close_prices, volatility)
        meta = ml.train_meta_model(primary_signals, features)
        bet_size = meta.get_bet_size(new_features)
    """

    # ------------------------------------------------------------------
    # Triple-barrier labeling
    # ------------------------------------------------------------------

    @staticmethod
    def generate_labels(
        prices: pd.Series,
        vol: pd.Series,
        tp_mult: float = 2.0,
        sl_mult: float = 2.0,
        max_hold: int = 10,
        min_return: float = 0.0,
    ) -> pd.Series:
        """Apply the triple-barrier method to generate trade labels.

        For each bar, computes which barrier is hit first:
        - Upper barrier at entry_price * (1 + tp_mult * vol)
        - Lower barrier at entry_price * (1 - sl_mult * vol)
        - Vertical barrier at entry + max_hold bars

        Args:
            prices: Close price series (datetime-indexed preferred).
            vol: Volatility series (e.g., rolling std of returns) aligned
                 with prices. Used to set dynamic barrier widths.
            tp_mult: Take-profit multiplier on volatility. Default 2.0.
            sl_mult: Stop-loss multiplier on volatility. Default 2.0.
            max_hold: Maximum holding period in bars. Default 10.
            min_return: Minimum absolute return to classify as +1/-1.
                        Returns within [-min_return, min_return] are labeled 0.
                        Default 0.0 (any direction counts).

        Returns:
            Series of labels: +1 (profitable), -1 (loss), 0 (neutral/expired).
            Indexed to match the input prices. Final max_hold entries are NaN
            (insufficient forward data).
        """
        close = prices.values.astype(np.float64)
        volatility = vol.values.astype(np.float64)
        n = len(close)

        labels = np.full(n, np.nan, dtype=np.float64)

        for i in range(n - 1):
            entry_price = close[i]
            sigma = volatility[i]

            if np.isnan(sigma) or sigma < 1e-10:
                # Skip bars with no volatility estimate
                labels[i] = 0.0
                continue

            upper = entry_price * (1.0 + tp_mult * sigma)
            lower = entry_price * (1.0 - sl_mult * sigma)
            horizon = min(i + max_hold, n - 1)

            label = 0
            for j in range(i + 1, horizon + 1):
                if close[j] >= upper:
                    label = 1
                    break
                elif close[j] <= lower:
                    label = -1
                    break
            else:
                # Vertical barrier hit: label by final return
                final_return = (close[horizon] - entry_price) / entry_price
                if abs(final_return) > min_return:
                    label = 1 if final_return > 0 else -1
                else:
                    label = 0

            labels[i] = label

        return pd.Series(labels, index=prices.index, name="triple_barrier_label")

    @staticmethod
    def generate_barrier_events(
        prices: pd.Series,
        vol: pd.Series,
        tp_mult: float = 2.0,
        sl_mult: float = 2.0,
        max_hold: int = 10,
    ) -> List[BarrierEvent]:
        """Generate detailed barrier event records.

        Like ``generate_labels`` but returns full event details including
        entry/exit prices, holding periods, and which barrier was hit.

        Args:
            prices: Close price series.
            vol: Volatility series.
            tp_mult: Take-profit multiplier.
            sl_mult: Stop-loss multiplier.
            max_hold: Maximum holding period.

        Returns:
            List of BarrierEvent dataclass instances.
        """
        close = prices.values.astype(np.float64)
        volatility = vol.values.astype(np.float64)
        n = len(close)
        events = []

        for i in range(n - 1):
            entry_price = close[i]
            sigma = volatility[i]

            if np.isnan(sigma) or sigma < 1e-10:
                continue

            upper = entry_price * (1.0 + tp_mult * sigma)
            lower = entry_price * (1.0 - sl_mult * sigma)
            horizon = min(i + max_hold, n - 1)

            exit_idx = horizon
            exit_price = close[horizon]
            barrier_hit = "vertical"
            label = 0

            for j in range(i + 1, horizon + 1):
                if close[j] >= upper:
                    exit_idx = j
                    exit_price = close[j]
                    barrier_hit = "upper"
                    label = 1
                    break
                elif close[j] <= lower:
                    exit_idx = j
                    exit_price = close[j]
                    barrier_hit = "lower"
                    label = -1
                    break
            else:
                final_return = (close[horizon] - entry_price) / entry_price
                label = 1 if final_return > 0 else (-1 if final_return < 0 else 0)

            events.append(BarrierEvent(
                entry_idx=i,
                entry_price=entry_price,
                exit_idx=exit_idx,
                exit_price=exit_price,
                return_pct=(exit_price - entry_price) / entry_price,
                label=label,
                barrier_hit=barrier_hit,
                holding_period=exit_idx - i,
            ))

        return events

    # ------------------------------------------------------------------
    # Meta-model training
    # ------------------------------------------------------------------

    def train_meta_model(
        self,
        signals: pd.Series,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        vol: Optional[pd.Series] = None,
        tp_mult: float = 2.0,
        sl_mult: float = 2.0,
        max_hold: int = 10,
        model_type: str = "random_forest",
        n_cv_splits: int = 5,
    ) -> MetaModel:
        """Train a meta-labeling model.

        The meta-model learns to predict P(primary signal is profitable)
        conditioned on features. Only bars where the primary model issues
        a signal (signal != 0) are used for training.

        Args:
            signals: Primary model signals (+1 = long, -1 = short, 0 = flat).
            features: Feature DataFrame aligned with signals.
            labels: Pre-computed triple-barrier labels. If None, computed
                    from prices and vol.
            prices: Close prices (required if labels is None).
            vol: Volatility (required if labels is None).
            tp_mult: Take-profit multiplier for label generation.
            sl_mult: Stop-loss multiplier for label generation.
            max_hold: Max holding period for label generation.
            model_type: Base classifier: "random_forest", "gradient_boosting",
                        or "logistic". Default "random_forest".
            n_cv_splits: Number of time-series CV splits. Default 5.

        Returns:
            Trained MetaModel.
        """
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required for meta-model training. "
                "Install with: pip install scikit-learn"
            )

        # Generate labels if not provided
        if labels is None:
            if prices is None or vol is None:
                raise ValueError(
                    "Either 'labels' or both 'prices' and 'vol' must be provided"
                )
            labels = self.generate_labels(prices, vol, tp_mult, sl_mult, max_hold)

        # Align everything on common index
        common_idx = signals.index.intersection(features.index).intersection(labels.index)
        signals_aligned = signals.loc[common_idx]
        features_aligned = features.loc[common_idx]
        labels_aligned = labels.loc[common_idx]

        # Filter to bars where primary model has a signal
        signal_mask = signals_aligned != 0
        if signal_mask.sum() < 20:
            raise ValueError(
                f"Only {signal_mask.sum()} signal bars — need at least 20 "
                f"for meta-model training"
            )

        X_full = features_aligned.loc[signal_mask]
        y_raw = labels_aligned.loc[signal_mask]
        signal_dir = signals_aligned.loc[signal_mask]

        # Meta-label: 1 if primary signal direction matches outcome, 0 otherwise
        # For long signals (+1): profitable if label == +1
        # For short signals (-1): profitable if label == -1
        y_meta = (((signal_dir > 0) & (y_raw > 0)) | ((signal_dir < 0) & (y_raw < 0))).astype(int)

        feature_names = list(X_full.columns)
        X = X_full.values.astype(np.float64)
        y = y_meta.values.astype(np.float64)

        # Clean NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select classifier
        clf = self._build_classifier(model_type)

        # Time-series CV for evaluation
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        cv_metrics = {
            "accuracy": [], "precision": [], "recall": [],
            "f1": [], "roc_auc": [],
        }

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf_cv = self._build_classifier(model_type)
            clf_cv.fit(X_train, y_train)

            y_pred = clf_cv.predict(X_test)
            y_proba = clf_cv.predict_proba(X_test)[:, 1]

            cv_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            cv_metrics["precision"].append(
                precision_score(y_test, y_pred, zero_division=0)
            )
            cv_metrics["recall"].append(
                recall_score(y_test, y_pred, zero_division=0)
            )
            cv_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))

            try:
                cv_metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))
            except ValueError:
                cv_metrics["roc_auc"].append(0.5)

        # Train final model on all data
        clf.fit(X_scaled, y)

        avg_metrics = {k: float(np.mean(v)) for k, v in cv_metrics.items()}

        meta = MetaModel(
            model=clf,
            scaler=scaler,
            feature_names=feature_names,
            threshold=0.5,
            metrics=avg_metrics,
            trained_at=time.time(),
        )

        logger.info(
            "Meta-model trained: %d samples, %d features, "
            "CV AUC=%.4f, CV F1=%.4f, positive rate=%.2f",
            len(y), len(feature_names),
            avg_metrics["roc_auc"], avg_metrics["f1"],
            float(y.mean()),
        )

        return meta

    # ------------------------------------------------------------------
    # Label statistics
    # ------------------------------------------------------------------

    @staticmethod
    def label_statistics(labels: pd.Series) -> Dict[str, Any]:
        """Compute descriptive statistics for triple-barrier labels.

        Args:
            labels: Series of labels from ``generate_labels``.

        Returns:
            Dict with counts, fractions, and balance metrics.
        """
        clean = labels.dropna()
        n = len(clean)

        if n == 0:
            return {"n_labels": 0}

        n_pos = int((clean > 0).sum())
        n_neg = int((clean < 0).sum())
        n_neutral = int((clean == 0).sum())

        return {
            "n_labels": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_neutral": n_neutral,
            "frac_positive": n_pos / n,
            "frac_negative": n_neg / n,
            "frac_neutral": n_neutral / n,
            "balance_ratio": min(n_pos, n_neg) / max(n_pos, n_neg, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_classifier(model_type: str) -> Any:
        """Instantiate a classifier by name."""
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )
        elif model_type == "logistic":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: random_forest, gradient_boosting, logistic"
            )
