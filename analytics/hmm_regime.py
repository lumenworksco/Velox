"""Hidden Markov Model regime detection — probabilistic market state classification.

Replaces the simple EMA20 slope regime detector with a 5-state HMM that
identifies distinct market regimes using daily returns, realized volatility,
volume, and VIX. Detects regime transitions BEFORE they become obvious.

Features fed to HMM (computed daily from SPY):
  - Daily return (close-to-close)
  - 5-day realized volatility (fast vol)
  - 20-day realized volatility (slow vol)
  - Volume ratio (today / 20-day average)
  - VIX level

Training: Fit on 3 years of daily SPY data. Retrain weekly on Sunday.
Model persisted to models/hmm_regime.pkl.
"""

import logging
import joblib
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Model persistence path
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hmm_regime.pkl"


class MarketRegimeState(Enum):
    """Five distinct market regimes identified by the HMM."""
    LOW_VOL_BULL = "low_vol_bull"        # Trending up, low volatility — best for momentum/breakouts
    HIGH_VOL_BULL = "high_vol_bull"      # Trending up, high volatility — reduce size, widen stops
    LOW_VOL_BEAR = "low_vol_bear"        # Grinding down, low vol — best for mean reversion
    HIGH_VOL_BEAR = "high_vol_bear"      # Crisis/crash — halt most trading
    MEAN_REVERTING = "mean_reverting"    # Range-bound, choppy — best for StatMR/VWAP/Pairs


# Strategy-regime affinity multipliers: how well each strategy performs in each regime
STRATEGY_REGIME_AFFINITY = {
    "STAT_MR": {
        MarketRegimeState.LOW_VOL_BULL: 0.7,
        MarketRegimeState.HIGH_VOL_BULL: 0.5,
        MarketRegimeState.LOW_VOL_BEAR: 1.0,
        MarketRegimeState.HIGH_VOL_BEAR: 0.2,
        MarketRegimeState.MEAN_REVERTING: 1.3,
    },
    "VWAP": {
        MarketRegimeState.LOW_VOL_BULL: 0.6,
        MarketRegimeState.HIGH_VOL_BULL: 0.4,
        MarketRegimeState.LOW_VOL_BEAR: 1.0,
        MarketRegimeState.HIGH_VOL_BEAR: 0.2,
        MarketRegimeState.MEAN_REVERTING: 1.2,
    },
    "KALMAN_PAIRS": {
        MarketRegimeState.LOW_VOL_BULL: 0.8,
        MarketRegimeState.HIGH_VOL_BULL: 0.6,
        MarketRegimeState.LOW_VOL_BEAR: 0.9,
        MarketRegimeState.HIGH_VOL_BEAR: 0.3,
        MarketRegimeState.MEAN_REVERTING: 1.1,
    },
    "ORB": {
        MarketRegimeState.LOW_VOL_BULL: 1.3,
        MarketRegimeState.HIGH_VOL_BULL: 0.8,
        MarketRegimeState.LOW_VOL_BEAR: 0.4,
        MarketRegimeState.HIGH_VOL_BEAR: 0.3,
        MarketRegimeState.MEAN_REVERTING: 0.5,
    },
    "MICRO_MOM": {
        MarketRegimeState.LOW_VOL_BULL: 1.0,
        MarketRegimeState.HIGH_VOL_BULL: 1.3,
        MarketRegimeState.LOW_VOL_BEAR: 0.6,
        MarketRegimeState.HIGH_VOL_BEAR: 0.8,
        MarketRegimeState.MEAN_REVERTING: 0.7,
    },
    "PEAD": {
        MarketRegimeState.LOW_VOL_BULL: 1.2,
        MarketRegimeState.HIGH_VOL_BULL: 0.8,
        MarketRegimeState.LOW_VOL_BEAR: 0.7,
        MarketRegimeState.HIGH_VOL_BEAR: 0.4,
        MarketRegimeState.MEAN_REVERTING: 1.0,
    },
}


def _compute_raw_features(df: pd.DataFrame) -> np.ndarray:
    """Compute raw HMM feature matrix from daily OHLCV data (unnormalized).

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        np.ndarray of shape (n_samples, 5) with columns:
        [daily_return, vol_5d, vol_20d, volume_ratio, vix_proxy]
        NaN rows are already dropped.
    """
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Daily returns (log)
    log_returns = np.diff(np.log(np.maximum(close, 1e-8)))

    # Realized volatility (5-day and 20-day rolling std of returns)
    n = len(log_returns)
    vol_5d = np.full(n, np.nan)
    vol_20d = np.full(n, np.nan)
    for i in range(4, n):
        vol_5d[i] = np.std(log_returns[i - 4 : i + 1])
    for i in range(19, n):
        vol_20d[i] = np.std(log_returns[i - 19 : i + 1])

    # Volume ratio (today / 20-day average)
    vol_shifted = volume[1:]  # Align with returns
    vol_ratio = np.full(n, np.nan)
    for i in range(19, n):
        avg_vol = np.mean(vol_shifted[i - 19 : i])
        if avg_vol > 0:
            vol_ratio[i] = vol_shifted[i] / avg_vol
        else:
            vol_ratio[i] = 1.0

    # VIX proxy: 20-day annualized vol (since we may not have VIX in historical data)
    vix_proxy = vol_20d * np.sqrt(252) * 100  # Convert to VIX-like scale

    # Stack features
    features = np.column_stack([log_returns, vol_5d, vol_20d, vol_ratio, vix_proxy])

    # Drop rows with NaN (first ~20 rows)
    valid_mask = ~np.isnan(features).any(axis=1)
    return features[valid_mask]


def _normalize_features(features: np.ndarray,
                         means: np.ndarray | None = None,
                         stds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features. Returns (normalized, means, stds)."""
    if means is None:
        means = features.mean(axis=0)
    if stds is None:
        stds = features.std(axis=0)
    stds_safe = stds.copy()
    stds_safe[stds_safe < 1e-10] = 1.0
    normalized = (features - means) / stds_safe
    return normalized, means, stds


def _compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute normalized HMM features. Convenience wrapper for backward compat."""
    raw = _compute_raw_features(df)
    if len(raw) == 0:
        return raw
    normalized, _, _ = _normalize_features(raw)
    return normalized


def _label_states(model, features: np.ndarray) -> dict[int, MarketRegimeState]:
    """Map HMM state indices to MarketRegimeState enum based on fitted means.

    Labeling logic:
    - Compute each state's mean return and mean volatility
    - Sort by return (ascending): lowest = bear, highest = bull
    - Within bears, high vol = HIGH_VOL_BEAR, low vol = LOW_VOL_BEAR
    - Within bulls, high vol = HIGH_VOL_BULL, low vol = LOW_VOL_BULL
    - Middle state = MEAN_REVERTING
    """
    n_states = model.n_components
    means = model.means_  # shape (n_states, n_features)

    # Feature columns: [daily_return, vol_5d, vol_20d, volume_ratio, vix_proxy]
    state_return = means[:, 0]  # Mean daily return
    state_vol = means[:, 2]     # Mean 20-day vol

    # Sort states by return
    sorted_by_return = np.argsort(state_return)

    label_map = {}

    if n_states >= 5:
        # Bottom 2 = bears, top 2 = bulls, middle = mean-reverting
        bears = sorted_by_return[:2]
        bulls = sorted_by_return[-2:]
        mid = sorted_by_return[2]

        # Bears: sort by vol
        if state_vol[bears[0]] > state_vol[bears[1]]:
            label_map[bears[0]] = MarketRegimeState.HIGH_VOL_BEAR
            label_map[bears[1]] = MarketRegimeState.LOW_VOL_BEAR
        else:
            label_map[bears[0]] = MarketRegimeState.LOW_VOL_BEAR
            label_map[bears[1]] = MarketRegimeState.HIGH_VOL_BEAR

        # Bulls: sort by vol
        if state_vol[bulls[0]] > state_vol[bulls[1]]:
            label_map[bulls[0]] = MarketRegimeState.HIGH_VOL_BULL
            label_map[bulls[1]] = MarketRegimeState.LOW_VOL_BULL
        else:
            label_map[bulls[0]] = MarketRegimeState.LOW_VOL_BULL
            label_map[bulls[1]] = MarketRegimeState.HIGH_VOL_BULL

        label_map[mid] = MarketRegimeState.MEAN_REVERTING
    else:
        # Fallback for fewer states
        for i in range(n_states):
            if state_return[i] > 0 and state_vol[i] < np.median(state_vol):
                label_map[i] = MarketRegimeState.LOW_VOL_BULL
            elif state_return[i] > 0:
                label_map[i] = MarketRegimeState.HIGH_VOL_BULL
            elif state_return[i] < 0 and state_vol[i] > np.median(state_vol):
                label_map[i] = MarketRegimeState.HIGH_VOL_BEAR
            elif state_return[i] < 0:
                label_map[i] = MarketRegimeState.LOW_VOL_BEAR
            else:
                label_map[i] = MarketRegimeState.MEAN_REVERTING

    return label_map


class HMMRegimeDetector:
    """Hidden Markov Model-based market regime detector.

    Uses GaussianHMM to identify 5 distinct market states from daily
    SPY data. Provides probabilistic regime classification and transition
    matrix for forward-looking regime estimation.
    """

    def __init__(self, n_states: int | None = None):
        self.n_states = n_states or getattr(config, "HMM_N_STATES", 5)
        self.model = None
        self._label_map: dict[int, MarketRegimeState] = {}
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._current_regime = MarketRegimeState.MEAN_REVERTING
        self._regime_probabilities: dict[MarketRegimeState, float] = {
            r: 0.2 for r in MarketRegimeState
        }
        self._fitted = False

        # Try to load saved model
        self._load_model()

    def _load_model(self) -> bool:
        """Load a previously fitted model from disk."""
        try:
            if MODEL_PATH.exists():
                saved = joblib.load(MODEL_PATH)
                self.model = saved["model"]
                self._label_map = saved["label_map"]
                self._feature_means = saved.get("feature_means")
                self._feature_stds = saved.get("feature_stds")
                self._fitted = True
                logger.info("HMM regime model loaded from disk")
                return True
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")
        return False

    def _save_model(self):
        """Save fitted model to disk."""
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self.model,
                "label_map": self._label_map,
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
            }, MODEL_PATH)
            logger.info(f"HMM regime model saved to {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save HMM model: {e}")

    def fit(self, market_data: pd.DataFrame) -> bool:
        """Fit the HMM on historical daily SPY data.

        Args:
            market_data: DataFrame with columns [open, high, low, close, volume].
                Should contain 2+ years of daily data for robust fitting.

        Returns:
            True if fitting succeeded, False otherwise.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed — HMM regime detection unavailable")
            return False

        raw_features = _compute_raw_features(market_data)
        if len(raw_features) < 100:
            logger.warning(f"Insufficient data for HMM fitting: {len(raw_features)} samples (need 100+)")
            return False

        try:
            features, means, stds = _normalize_features(raw_features)

            # V10 BUG-017: Train/validation split (80/20) to detect overfitting
            split_idx = int(len(features) * 0.8)
            train_features = features[:split_idx]
            val_features = features[split_idx:]

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            model.fit(train_features)

            # Validate: check that validation score is not drastically worse
            train_score = model.score(train_features)
            val_score = model.score(val_features) if len(val_features) >= 10 else train_score
            score_ratio = val_score / train_score if train_score != 0 else 1.0

            if score_ratio < 0.5:
                logger.warning(
                    f"HMM overfitting detected: train_score={train_score:.1f}, "
                    f"val_score={val_score:.1f} (ratio={score_ratio:.2f})"
                )

            self.model = model
            self._label_map = _label_states(model, features)
            self._feature_means = means
            self._feature_stds = stds
            self._fitted = True
            self._save_model()

            logger.info(
                f"HMM fitted on {len(train_features)} train + {len(val_features)} val samples, "
                f"{self.n_states} states, train_score={train_score:.1f}, val_score={val_score:.1f}"
            )
            return True

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False

    def predict_regime(
        self, recent_data: pd.DataFrame
    ) -> tuple[MarketRegimeState, dict[MarketRegimeState, float]]:
        """Predict current regime from recent market data.

        Args:
            recent_data: DataFrame with columns [open, high, low, close, volume].
                Should contain at least 25 rows (20 for vol computation + 5 buffer).

        Returns:
            Tuple of (most_likely_regime, probability_distribution).
        """
        if not self._fitted or self.model is None:
            return self._current_regime, self._regime_probabilities

        try:
            raw = _compute_raw_features(recent_data)
            if len(raw) == 0:
                return self._current_regime, self._regime_probabilities
            features, _, _ = _normalize_features(raw, self._feature_means, self._feature_stds)
            if len(features) == 0:
                return self._current_regime, self._regime_probabilities

            # Get state probabilities for the last observation
            log_prob, posteriors = self.model.score_samples(features)
            last_posteriors = posteriors[-1]  # Probabilities for most recent day

            # Map HMM states to regime labels
            probs: dict[MarketRegimeState, float] = {}
            for state_idx, prob in enumerate(last_posteriors):
                regime = self._label_map.get(state_idx, MarketRegimeState.MEAN_REVERTING)
                probs[regime] = probs.get(regime, 0.0) + prob

            # Determine most likely regime
            best_regime = max(probs, key=lambda r: probs[r])

            self._current_regime = best_regime
            self._regime_probabilities = probs

            return best_regime, probs

        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._current_regime, self._regime_probabilities

    def get_transition_matrix(self) -> np.ndarray | None:
        """Return the fitted transition probability matrix.

        Returns:
            np.ndarray of shape (n_states, n_states) or None if not fitted.
        """
        if self._fitted and self.model is not None:
            return self.model.transmat_
        return None

    @property
    def current_regime(self) -> MarketRegimeState:
        """Current most-likely regime."""
        return self._current_regime

    @property
    def regime_probabilities(self) -> dict[MarketRegimeState, float]:
        """Full probability distribution across regimes."""
        return dict(self._regime_probabilities)

    @property
    def is_fitted(self) -> bool:
        """Whether the HMM has been fitted."""
        return self._fitted


def get_strategy_regime_affinity() -> dict[str, dict[MarketRegimeState, float]]:
    """Return strategy-regime affinity mapping."""
    return STRATEGY_REGIME_AFFINITY


def get_regime_size_multiplier(strategy: str, regime: MarketRegimeState,
                                probabilities: dict[MarketRegimeState, float] | None = None) -> float:
    """Compute a position size multiplier based on regime and strategy.

    If probabilities are provided, computes a weighted average across all
    regimes. Otherwise uses the single regime's affinity score.

    Args:
        strategy: Strategy name (e.g., "STAT_MR").
        regime: Current most-likely regime.
        probabilities: Optional probability distribution across regimes.

    Returns:
        Multiplier in range [0.2, 1.5].
    """
    affinity = STRATEGY_REGIME_AFFINITY.get(strategy)
    if affinity is None:
        return 1.0

    if probabilities:
        # Weighted average across all regime probabilities
        multiplier = sum(
            affinity.get(r, 1.0) * p
            for r, p in probabilities.items()
        )
    else:
        multiplier = affinity.get(regime, 1.0)

    # Clamp to reasonable range
    return max(0.2, min(1.5, multiplier))


def map_hmm_to_legacy(regime: MarketRegimeState) -> str:
    """Map HMM MarketRegimeState to legacy string regime for backward compatibility.

    The old regime detector returned "BULLISH", "BEARISH", or "UNKNOWN".
    This maps the 5-state HMM output to those strings so existing code
    that checks `regime == "BEARISH"` continues to work.
    """
    if regime in (MarketRegimeState.LOW_VOL_BULL, MarketRegimeState.HIGH_VOL_BULL):
        return "BULLISH"
    elif regime in (MarketRegimeState.LOW_VOL_BEAR, MarketRegimeState.HIGH_VOL_BEAR):
        return "BEARISH"
    else:
        return "UNKNOWN"
