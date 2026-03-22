"""T5-011: Temporal Fusion Transformer for Multi-Horizon Return Prediction.

Lightweight TFT implementation (2-layer, 64 hidden units) for predicting
return distributions at 30min, 2h, and EOD horizons.

Inputs: price features (OHLCV ratios), volume profile, VIX level,
        sector ETF return, VPIN score, OU z-score, RSI.
Outputs: predicted return distribution (mean + std) at each horizon.

Training: walk-forward on rolling 252-day windows.
Gated behind ``TFT_ENABLED=True`` config flag.
Falls back to existing ensemble if PyTorch unavailable.

Usage::

    tft = TemporalFusionTransformer()
    tft.fit(features_df, targets_df)
    preds = tft.predict(current_features)
    # preds = {"30min": (mean, std), "2h": (mean, std), "eod": (mean, std)}
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Conditional PyTorch import (fail-open to numpy fallback)
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    logger.info("T5-011: PyTorch not available — TFT will use numpy fallback")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Feature names expected by the model
FEATURE_NAMES = [
    "open_close_ratio",     # Open / Close
    "high_low_ratio",       # High / Low
    "close_prev_ratio",     # Close / Prev Close
    "volume_sma_ratio",     # Volume / SMA(Volume, 20)
    "volume_profile",       # Volume concentration at price level
    "vix_level",            # Current VIX
    "sector_etf_return",    # Sector ETF intraday return
    "vpin_score",           # VPIN toxicity score
    "ou_zscore",            # Ornstein-Uhlenbeck z-score
    "rsi_14",               # RSI(14)
]

HORIZONS = ["30min", "2h", "eod"]
N_FEATURES = len(FEATURE_NAMES)
HIDDEN_DIM = 64
N_LAYERS = 2
WALK_FORWARD_WINDOW = 252  # Trading days


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TFTPrediction:
    """Prediction output for a single symbol."""
    symbol: str
    horizon_30min: tuple[float, float]   # (mean, std)
    horizon_2h: tuple[float, float]
    horizon_eod: tuple[float, float]
    confidence: float                     # Overall confidence 0-1

    def best_horizon(self) -> str:
        """Return the horizon with highest expected return / risk."""
        ratios = {
            "30min": self.horizon_30min[0] / max(self.horizon_30min[1], 1e-8),
            "2h": self.horizon_2h[0] / max(self.horizon_2h[1], 1e-8),
            "eod": self.horizon_eod[0] / max(self.horizon_eod[1], 1e-8),
        }
        return max(ratios, key=lambda k: abs(ratios[k]))


# ---------------------------------------------------------------------------
# PyTorch TFT Module (conditional)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _GatedResidualNetwork(nn.Module):
        """Gated Residual Network — core building block of TFT."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     dropout: float = 0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.gate = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()
            self.layer_norm = nn.LayerNorm(output_dim)
            # Skip connection projection if dimensions differ
            self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.skip(x)
            h = self.elu(self.fc1(x))
            h = self.dropout(h)
            output = self.fc2(h)
            gate = self.sigmoid(self.gate(h))
            return self.layer_norm(gate * output + residual)

    class _VariableSelectionNetwork(nn.Module):
        """Variable Selection Network — learns which features matter."""

        def __init__(self, n_features: int, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.grns = nn.ModuleList([
                _GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
                for _ in range(n_features)
            ])
            self.softmax_grn = _GatedResidualNetwork(
                n_features * hidden_dim, hidden_dim, n_features, dropout
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x shape: (batch, n_features)
            processed = []
            for i, grn in enumerate(self.grns):
                processed.append(grn(x[:, i:i+1]))

            stacked = torch.stack(processed, dim=1)  # (batch, n_features, hidden)
            flat = stacked.reshape(x.shape[0], -1)
            weights = self.softmax(self.softmax_grn(flat))  # (batch, n_features)

            # Weighted sum
            selected = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
            return selected, weights

    class _TFTCore(nn.Module):
        """Lightweight Temporal Fusion Transformer core.

        2-layer, 64 hidden units. Outputs mean and std for 3 horizons.
        """

        def __init__(self, n_features: int = N_FEATURES,
                     hidden_dim: int = HIDDEN_DIM,
                     n_horizons: int = 3,
                     dropout: float = 0.1):
            super().__init__()
            self.vsn = _VariableSelectionNetwork(n_features, hidden_dim, dropout)

            # Temporal processing (simplified — single-step for real-time)
            self.temporal_layers = nn.ModuleList([
                _GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(N_LAYERS)
            ])

            # Output heads: mean and std for each horizon
            self.mean_head = nn.Linear(hidden_dim, n_horizons)
            self.std_head = nn.Sequential(
                nn.Linear(hidden_dim, n_horizons),
                nn.Softplus(),  # Ensure positive std
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                x: (batch, n_features) input features.

            Returns:
                (means, stds, feature_weights) where means/stds are (batch, 3).
            """
            selected, weights = self.vsn(x)

            h = selected
            for layer in self.temporal_layers:
                h = layer(h)

            means = self.mean_head(h)
            stds = self.std_head(h) + 1e-6  # Floor at small positive

            return means, stds, weights


# ---------------------------------------------------------------------------
# Numpy Fallback Model
# ---------------------------------------------------------------------------

class _NumpyFallbackTFT:
    """Simple linear regression fallback when PyTorch is not available.

    Uses ridge regression per horizon for mean prediction and
    residual std estimation.
    """

    def __init__(self):
        self._weights: dict[str, np.ndarray] = {}
        self._bias: dict[str, float] = {}
        self._residual_std: dict[str, float] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
        """Fit ridge regression for each horizon.

        Args:
            X: (n_samples, n_features)
            Y: (n_samples, 3) — returns at 30min, 2h, eod
        """
        n_features = X.shape[1]
        for i, horizon in enumerate(HORIZONS):
            y = Y[:, i]
            # Ridge regression: (X'X + alpha*I)^-1 X'y
            XtX = X.T @ X + alpha * np.eye(n_features)
            Xty = X.T @ y
            try:
                w = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                w = np.zeros(n_features)
            self._weights[horizon] = w
            self._bias[horizon] = float(np.mean(y) - np.mean(X @ w))

            # Residual std
            preds = X @ w + self._bias[horizon]
            residuals = y - preds
            self._residual_std[horizon] = max(float(np.std(residuals)), 1e-6)

        self._fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for each horizon.

        Returns:
            (means, stds) each of shape (n_samples, 3)
        """
        if not self._fitted:
            n = X.shape[0] if X.ndim > 1 else 1
            return np.zeros((n, 3)), np.ones((n, 3)) * 0.01

        means = []
        stds = []
        for horizon in HORIZONS:
            w = self._weights[horizon]
            b = self._bias[horizon]
            pred = X @ w + b if X.ndim > 1 else X.dot(w) + b
            means.append(pred)
            stds.append(np.full_like(pred, self._residual_std[horizon]))

        return np.column_stack(means), np.column_stack(stds)


# ---------------------------------------------------------------------------
# Main Public Class
# ---------------------------------------------------------------------------

class TemporalFusionTransformer:
    """T5-011: Multi-horizon return prediction using Temporal Fusion Transformer.

    Gated behind ``config.TFT_ENABLED``. Falls back to numpy linear model
    if PyTorch is unavailable.

    Usage::

        tft = TemporalFusionTransformer()
        tft.fit(features_array, targets_array)
        pred = tft.predict_symbol("AAPL", current_features)
    """

    def __init__(self):
        import config as _cfg
        self._enabled = getattr(_cfg, "TFT_ENABLED", False)
        self._lock = threading.Lock()
        self._fitted = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        if _TORCH_AVAILABLE and self._enabled:
            self._model = _TFTCore()
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
            self._use_torch = True
            logger.info("T5-011: TFT initialized with PyTorch backend (hidden=%d, layers=%d)",
                        HIDDEN_DIM, N_LAYERS)
        else:
            self._fallback = _NumpyFallbackTFT()
            self._use_torch = False
            if self._enabled:
                logger.info("T5-011: TFT initialized with numpy fallback (PyTorch unavailable)")
            else:
                logger.info("T5-011: TFT disabled (TFT_ENABLED=False)")

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        walk_forward: bool = True,
    ) -> dict:
        """Train the TFT on historical data.

        Args:
            X: (n_samples, n_features) feature matrix.
            Y: (n_samples, 3) target returns at [30min, 2h, eod].
            epochs: Training epochs (PyTorch only).
            batch_size: Batch size (PyTorch only).
            walk_forward: If True, use rolling 252-day windows.

        Returns:
            Training metrics dict.
        """
        if not self._enabled:
            return {"status": "disabled"}

        if X.shape[0] < 30:
            logger.warning("T5-011: Insufficient training data (%d samples)", X.shape[0])
            return {"status": "insufficient_data", "samples": X.shape[0]}

        # Standardize features
        self._feature_means = np.mean(X, axis=0)
        self._feature_stds = np.std(X, axis=0) + 1e-8
        X_norm = (X - self._feature_means) / self._feature_stds

        if walk_forward and X.shape[0] > WALK_FORWARD_WINDOW:
            return self._fit_walk_forward(X_norm, Y, epochs, batch_size)

        return self._fit_single(X_norm, Y, epochs, batch_size)

    def _fit_single(self, X: np.ndarray, Y: np.ndarray,
                    epochs: int, batch_size: int) -> dict:
        """Fit on the full dataset."""
        with self._lock:
            if self._use_torch:
                metrics = self._fit_torch(X, Y, epochs, batch_size)
            else:
                self._fallback.fit(X, Y)
                metrics = {"status": "fitted", "backend": "numpy", "samples": X.shape[0]}

            self._fitted = True
        return metrics

    def _fit_walk_forward(self, X: np.ndarray, Y: np.ndarray,
                          epochs: int, batch_size: int) -> dict:
        """Walk-forward training on rolling 252-day windows."""
        n = X.shape[0]
        window = WALK_FORWARD_WINDOW
        oos_errors = []

        for start in range(0, n - window, window // 4):
            end = min(start + window, n)
            train_end = start + int(0.8 * (end - start))
            if train_end >= end - 5:
                continue

            X_train, Y_train = X[start:train_end], Y[start:train_end]
            X_test, Y_test = X[train_end:end], Y[train_end:end]

            if self._use_torch:
                self._fit_torch(X_train, Y_train, epochs=epochs // 2, batch_size=batch_size)
                means, _ = self._predict_torch(X_test)
                oos_mse = float(np.mean((means - Y_test) ** 2))
            else:
                self._fallback.fit(X_train, Y_train)
                means, _ = self._fallback.predict(X_test)
                oos_mse = float(np.mean((means - Y_test) ** 2))

            oos_errors.append(oos_mse)

        # Final fit on most recent window
        final_start = max(0, n - window)
        if self._use_torch:
            self._fit_torch(X[final_start:], Y[final_start:], epochs, batch_size)
        else:
            self._fallback.fit(X[final_start:], Y[final_start:])

        with self._lock:
            self._fitted = True

        avg_oos_mse = float(np.mean(oos_errors)) if oos_errors else 0.0
        logger.info("T5-011: Walk-forward training complete. OOS MSE=%.6f (%d windows)",
                     avg_oos_mse, len(oos_errors))
        return {
            "status": "fitted",
            "backend": "torch" if self._use_torch else "numpy",
            "walk_forward_windows": len(oos_errors),
            "avg_oos_mse": avg_oos_mse,
        }

    def _fit_torch(self, X: np.ndarray, Y: np.ndarray,
                   epochs: int, batch_size: int) -> dict:
        """Train the PyTorch TFT model."""
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        n = X_t.shape[0]

        self._model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x_batch = X_t[idx]
                y_batch = Y_t[idx]

                means, stds, _ = self._model(x_batch)

                # Gaussian NLL loss
                loss = 0.5 * (
                    torch.log(stds ** 2) + ((y_batch - means) ** 2) / (stds ** 2)
                ).mean()

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            total_loss = epoch_loss / max(n_batches, 1)

        return {"status": "fitted", "backend": "torch", "final_loss": total_loss,
                "epochs": epochs, "samples": n}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for all horizons.

        Args:
            X: (n_samples, n_features) or (n_features,) for single sample.

        Returns:
            (means, stds) each of shape (n_samples, 3) for [30min, 2h, eod].
        """
        if not self._fitted:
            n = X.shape[0] if X.ndim > 1 else 1
            return np.zeros((n, 3)), np.ones((n, 3)) * 0.01

        # Normalize
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._feature_means is not None:
            X = (X - self._feature_means) / self._feature_stds

        with self._lock:
            if self._use_torch:
                return self._predict_torch(X)
            else:
                return self._fallback.predict(X)

    def _predict_torch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict using PyTorch model."""
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            means, stds, _ = self._model(X_t)
            return means.numpy(), stds.numpy()

    def predict_symbol(self, symbol: str, features: np.ndarray) -> TFTPrediction:
        """Predict return distribution for a specific symbol.

        Args:
            symbol: Ticker symbol.
            features: (n_features,) feature vector.

        Returns:
            TFTPrediction with per-horizon (mean, std).
        """
        means, stds = self.predict(features)
        m, s = means[0], stds[0]

        # Confidence: inverse of average std, normalized to 0-1
        avg_std = float(np.mean(s))
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_std * 100)))

        return TFTPrediction(
            symbol=symbol,
            horizon_30min=(float(m[0]), float(s[0])),
            horizon_2h=(float(m[1]), float(s[1])),
            horizon_eod=(float(m[2]), float(s[2])),
            confidence=confidence,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores from the Variable Selection Network.

        Only available with PyTorch backend after fitting. Returns empty dict
        on fallback or if not fitted.
        """
        if not self._fitted or not self._use_torch:
            return {}

        try:
            # Generate dummy input to extract VSN weights
            self._model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, N_FEATURES)
                _, _, weights = self._model(dummy)
                w = weights[0].numpy()
                return {name: float(w[i]) for i, name in enumerate(FEATURE_NAMES)}
        except Exception as e:
            logger.debug("T5-011: Feature importance extraction failed: %s", e)
            return {}


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_tft_instance: Optional[TemporalFusionTransformer] = None
_tft_lock = threading.Lock()


def get_tft() -> TemporalFusionTransformer:
    """Get or create the global TFT singleton."""
    global _tft_instance
    if _tft_instance is None:
        with _tft_lock:
            if _tft_instance is None:
                _tft_instance = TemporalFusionTransformer()
    return _tft_instance
