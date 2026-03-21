"""EDGE-011: Liquid Neural Networks with Adaptive Time Constants.

Implements a Liquid Time-Constant (LTC) network based on ODE-driven neurons
whose dynamics adapt to the input time series.  Advantages for finance:

  - Handles non-stationary data by adapting internal time constants
  - Compact parameter count relative to LSTMs / Transformers
  - Continuous-time dynamics suit irregular market data

Uses PyTorch when available; falls back to a numpy-based Euler-method
simulation of simplified LTC dynamics.

All ML library imports are conditional — the bot runs without them.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional PyTorch import
# ---------------------------------------------------------------------------

_HAS_TORCH = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LiquidNNConfig:
    """Configuration for Liquid Neural Networks."""

    hidden_dim: int = 64
    n_neurons: int = 32       # number of liquid neurons
    output_dim: int = 1
    tau_min: float = 0.1      # minimum time constant
    tau_max: float = 10.0     # maximum time constant
    ode_steps: int = 5        # ODE integration sub-steps per time step
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    seq_len: int = 20
    dropout: float = 0.1
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# PyTorch LTC cell & network
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class _LTCCell(nn.Module):
        """Liquid Time-Constant cell.

        Implements the ODE:
            tau * dx/dt = -x + f(W_in * input + W_rec * x + bias)

        where tau is a learnable, input-dependent time constant.
        """

        def __init__(self, input_dim: int, n_neurons: int, cfg: LiquidNNConfig):
            super().__init__()
            self.n_neurons = n_neurons
            self.ode_steps = cfg.ode_steps

            # Input-to-hidden
            self.W_in = nn.Linear(input_dim, n_neurons)
            # Recurrent
            self.W_rec = nn.Linear(n_neurons, n_neurons, bias=False)
            # Time-constant network: tau = sigmoid(fc(x, input)) * (max-min) + min
            self.tau_net = nn.Linear(input_dim + n_neurons, n_neurons)
            self.tau_min = cfg.tau_min
            self.tau_range = cfg.tau_max - cfg.tau_min

        def forward(
            self, x_input: torch.Tensor, h: torch.Tensor
        ) -> torch.Tensor:
            """Advance one time step.

            Parameters
            ----------
            x_input : (batch, input_dim)
            h : (batch, n_neurons) — hidden state

            Returns
            -------
            h_new : (batch, n_neurons)
            """
            dt = 1.0 / self.ode_steps
            for _ in range(self.ode_steps):
                tau_input = torch.cat([x_input, h], dim=-1)
                tau = torch.sigmoid(self.tau_net(tau_input)) * self.tau_range + self.tau_min
                pre = self.W_in(x_input) + self.W_rec(h)
                dh = (-h + torch.tanh(pre)) / tau
                h = h + dt * dh
            return h

    class _LiquidNetwork(nn.Module):
        """Full Liquid Neural Network for sequence-to-value prediction."""

        def __init__(self, input_dim: int, cfg: LiquidNNConfig):
            super().__init__()
            self.cell = _LTCCell(input_dim, cfg.n_neurons, cfg)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Sequential(
                nn.Linear(cfg.n_neurons, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.output_dim),
            )
            self.n_neurons = cfg.n_neurons

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Process a sequence (batch, seq_len, input_dim) -> (batch, output_dim)."""
            batch = x.size(0)
            h = torch.zeros(batch, self.n_neurons, device=x.device)
            for t in range(x.size(1)):
                h = self.cell(x[:, t, :], h)
                h = self.dropout(h)
            return self.head(h)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LiquidNN:
    """Liquid Neural Network for financial time series.

    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : LiquidNNConfig, optional
        Model configuration.
    """

    def __init__(self, config: Optional[LiquidNNConfig] = None):
        self.config = config or LiquidNNConfig()
        self._model = None
        self._fitted = False
        self._use_torch = _HAS_TORCH
        self._input_dim: int = 1
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        # Fallback state
        self._fallback_weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "LiquidNN":
        """Train the liquid neural network.

        Parameters
        ----------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_len, features) or
            (n_samples, features) which will be auto-windowed.
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, output_dim).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 2:
            X = self._make_sequences(X, self.config.seq_len)
            y = y[self.config.seq_len - 1 :]

        self._input_dim = X.shape[2]
        self._mean = X.reshape(-1, X.shape[2]).mean(axis=0)
        self._std = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        if self._use_torch:
            self._fit_torch(X_norm, y_norm, **kwargs)
        else:
            logger.info("PyTorch not available — using linear fallback")
            self._fit_fallback(X_norm, y_norm)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input sequences."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = self._make_sequences(X, self.config.seq_len)

        X_norm = (X - self._mean) / self._std  # type: ignore[operator]

        if self._use_torch and self._model is not None:
            preds_norm = self._predict_torch(X_norm)
        else:
            preds_norm = self._predict_fallback(X_norm)

        return preds_norm * self._y_std + self._y_mean

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate prediction quality."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).flatten()
        # Align lengths after sequencing
        n = min(len(preds), len(y))
        preds = preds[:n]
        y = y[-n:]

        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))
        if len(preds) > 2:
            corr = float(np.corrcoef(preds, y)[0, 1])
        else:
            corr = 0.0
        directional = float(np.mean(np.sign(preds) == np.sign(y)))
        return {
            "mse": mse,
            "mae": mae,
            "correlation": corr,
            "directional_accuracy": directional,
        }

    # ------------------------------------------------------------------
    # PyTorch training & inference
    # ------------------------------------------------------------------

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        cfg = self.config
        device = torch.device(cfg.device)
        self._model = _LiquidNetwork(self._input_dim, cfg).to(device)
        optimizer = optim.Adam(self._model.parameters(), lr=cfg.learning_rate)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y.reshape(-1, cfg.output_dim), dtype=torch.float32, device=device)

        logger.info("Training Liquid NN (%d epochs, %d samples) ...", cfg.epochs, len(X))
        self._model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(len(X_t))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_t), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                out = self._model(X_t[idx])
                loss = ((out - y_t[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
                logger.info(
                    "  epoch %d/%d  loss=%.6f",
                    epoch + 1, cfg.epochs, total_loss / max(n_batches, 1),
                )

    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config
        device = torch.device(cfg.device)
        self._model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            preds = self._model(X_t).cpu().numpy()  # type: ignore[misc]
        return preds.flatten()

    # ------------------------------------------------------------------
    # Numpy fallback (exponentially-weighted linear model)
    # ------------------------------------------------------------------

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a simple exponentially-weighted linear model."""
        # Flatten sequences: use exponentially-decayed feature averages
        X_flat = self._flatten_sequences(X)
        y_flat = y.flatten()
        # Ridge regression via normal equations
        lam = 1.0
        XtX = X_flat.T @ X_flat + lam * np.eye(X_flat.shape[1])
        Xty = X_flat.T @ y_flat
        try:
            self._fallback_weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._fallback_weights = np.zeros(X_flat.shape[1])
            logger.warning("Fallback fit failed — using zero weights")

    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten_sequences(X)
        if self._fallback_weights is None:
            return np.zeros(len(X_flat))
        return X_flat @ self._fallback_weights

    @staticmethod
    def _flatten_sequences(X: np.ndarray) -> np.ndarray:
        """Flatten sequences using exponential decay weighting."""
        n, seq_len, dim = X.shape
        weights = np.exp(np.linspace(-2, 0, seq_len))
        weights /= weights.sum()
        return np.einsum("ijk,j->ik", X, weights)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
        """Convert 2-D data into sliding-window sequences."""
        if len(data) < seq_len:
            padded = np.zeros((seq_len, data.shape[1]))
            padded[-len(data) :] = data
            return padded[np.newaxis, :, :]
        seqs = []
        for i in range(len(data) - seq_len + 1):
            seqs.append(data[i : i + seq_len])
        return np.array(seqs)

    def get_time_constants(self) -> Optional[np.ndarray]:
        """Extract learned time constants (PyTorch backend only).

        Returns None if using fallback or model not fitted.
        """
        if not self._use_torch or self._model is None:
            return None
        try:
            tau_bias = self._model.cell.tau_net.bias.detach().cpu().numpy()
            tau = 1.0 / (1.0 + np.exp(-tau_bias)) * (self.config.tau_max - self.config.tau_min) + self.config.tau_min
            return tau
        except Exception as exc:
            logger.warning("Could not extract time constants: %s", exc)
            return None
