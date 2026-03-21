"""
EDGE-001: Mamba State Space Model for Time Series Prediction
=============================================================

Implements a selective state space model (S4/Mamba architecture) for predicting
next-bar returns from feature vectors. The Mamba architecture replaces attention
with a hardware-efficient selective scan, achieving linear-time sequence modeling.

Key components:
  - Selective scan mechanism with input-dependent gating
  - Discretized state space layers (A, B, C, D parameterization)
  - Causal 1D convolution for local context
  - MambaTimeSeriesPredictor: high-level API conforming to AlphaModel interface

Requires PyTorch. Falls back to a zero-prediction stub if unavailable.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt PyTorch import with graceful fallback
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "EDGE-001: PyTorch not available. MambaTimeSeriesPredictor will "
        "return zero predictions. Install with: pip install torch"
    )


# ===================================================================
# PyTorch components (only defined when torch is available)
# ===================================================================
if _TORCH_AVAILABLE:

    class SelectiveScanLayer(nn.Module):
        """Core selective scan operation.

        Given input x of shape (B, L, D), produces output y of the same shape
        by running a discretized state space recurrence where the transition
        matrices are *input-dependent* (selective).
        """

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                     expand: int = 2, dt_rank: str = "auto"):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = d_model * expand
            self.dt_rank = d_model if dt_rank == "auto" else int(dt_rank)

            # Input projection: x -> (z, x_proj) split
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

            # Causal 1D convolution (local context)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

            # SSM parameter projections (input-dependent)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

            # dt projection
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

            # Learnable SSM parameters
            # A is initialized as a structured matrix (log-space for stability)
            A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            # Output projection
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, seq_len, d_model)

            Returns:
                (batch, seq_len, d_model)
            """
            B, L, _ = x.shape

            # Project and split into two streams
            xz = self.in_proj(x)  # (B, L, 2*d_inner)
            x_stream, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

            # Causal convolution on x stream
            x_conv = x_stream.transpose(1, 2)  # (B, d_inner, L)
            x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim future
            x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
            x_conv = F.silu(x_conv)

            # Compute input-dependent SSM parameters
            ssm_input = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
            dt, B_param, C_param = torch.split(
                ssm_input, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

            # Recover A from log-space
            A = -torch.exp(self.A_log)  # (d_inner, d_state)

            # Discretize and run the scan (simplified sequential version)
            y = self._selective_scan(x_conv, dt, A, B_param, C_param)

            # Gating with z branch
            y = y * F.silu(z)

            return self.out_proj(y)

        def _selective_scan(self, x, dt, A, B_param, C_param):
            """Run selective scan recurrence.

            This is a sequential implementation. A fused CUDA kernel would be
            used in production for O(1) memory and hardware efficiency.
            """
            B_size, L, d_inner = x.shape
            d_state = self.d_state

            # Initialize hidden state
            h = torch.zeros(B_size, d_inner, d_state, device=x.device, dtype=x.dtype)
            outputs = []

            for t in range(L):
                # Discretize: dA = exp(A * dt), dB = dt * B
                dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
                dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, d_inner, d_state)
                dB = dt_t * B_param[:, t, :].unsqueeze(1)  # (B, d_inner, d_state)

                # State update: h = dA * h + dB * x
                x_t = x[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
                h = dA * h + dB * x_t

                # Output: y = C * h + D * x
                C_t = C_param[:, t, :].unsqueeze(1)  # (B, 1, d_state)
                y_t = (h * C_t).sum(dim=-1)  # (B, d_inner)
                y_t = y_t + self.D * x[:, t, :]

                outputs.append(y_t)

            return torch.stack(outputs, dim=1)  # (B, L, d_inner)


    class MambaBlock(nn.Module):
        """Single Mamba block: LayerNorm -> SelectiveScan -> Residual."""

        def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                     expand: int = 2):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.ssm = SelectiveScanLayer(d_model, d_state, d_conv, expand)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.ssm(self.norm(x))


    class MambaNetwork(nn.Module):
        """Stack of Mamba blocks for sequence-to-one regression."""

        def __init__(self, n_features: int, d_model: int = 64, n_layers: int = 4,
                     d_state: int = 16, d_conv: int = 4, expand: int = 2,
                     dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, n_features)
            Returns:
                predictions: (batch,) — predicted next-bar return
            """
            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            x = x[:, -1, :]  # take last time step
            x = self.dropout(x)
            return self.head(x).squeeze(-1)


# ===================================================================
# Public API: MambaTimeSeriesPredictor (AlphaModel interface)
# ===================================================================

class MambaTimeSeriesPredictor:
    """Mamba-based time-series predictor for next-bar return forecasting.

    Conforms to the AlphaModel interface:
        fit(X, y)     — train on (sequences, targets)
        predict(X)    — predict returns
        score(X, y)   — evaluate via negative MSE

    Parameters
    ----------
    seq_len : int
        Number of bars in each input sequence.
    n_features : int
        Dimensionality of feature vectors per bar.
    d_model : int
        Hidden dimension of the Mamba blocks.
    n_layers : int
        Number of stacked Mamba blocks.
    d_state : int
        State dimension for the SSM recurrence.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, *, seq_len: int = 20, n_features: int = 10,
                 d_model: int = 64, n_layers: int = 4, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 64,
                 device: str = "cpu", **kwargs):
        self.params: Dict[str, Any] = {
            "seq_len": seq_len, "n_features": n_features,
            "d_model": d_model, "n_layers": n_layers, "d_state": d_state,
            "d_conv": d_conv, "expand": expand, "dropout": dropout,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "device": device,
        }
        self._model = None
        self._fitted = False

        if not _TORCH_AVAILABLE:
            logger.warning(
                "EDGE-001: Running in stub mode (no PyTorch). "
                "All predictions will be zero."
            )

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MambaTimeSeriesPredictor":
        """Train the Mamba model.

        Args:
            X: shape (n_samples, seq_len, n_features) or (n_samples, n_features).
               If 2-D, sequences are constructed using a sliding window.
            y: shape (n_samples,) — target next-bar returns.

        Returns:
            self
        """
        if not _TORCH_AVAILABLE:
            logger.info("EDGE-001 stub: fit() is a no-op without PyTorch.")
            self._fitted = True
            return self

        p = self.params
        X, y = self._validate_inputs(X, y)
        n_features = X.shape[-1]

        # Build network
        self._model = MambaNetwork(
            n_features=n_features, d_model=p["d_model"],
            n_layers=p["n_layers"], d_state=p["d_state"],
            d_conv=p["d_conv"], expand=p["expand"], dropout=p["dropout"],
        ).to(p["device"])

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=p["lr"],
                                       weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=p["epochs"]
        )
        criterion = nn.MSELoss()

        # Data loader
        X_t = torch.tensor(X, dtype=torch.float32).to(p["device"])
        y_t = torch.tensor(y, dtype=torch.float32).to(p["device"])
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True)

        # Training loop
        self._model.train()
        for epoch in range(p["epochs"]):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(dataset)
                logger.debug("EDGE-001 epoch %d/%d  loss=%.6f", epoch + 1, p["epochs"], avg)

        self._fitted = True
        logger.info("EDGE-001: Mamba model trained on %d samples.", len(dataset))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-bar returns.

        Args:
            X: shape (n_samples, seq_len, n_features) or (n_samples, n_features)

        Returns:
            predictions: shape (n_samples,)
        """
        if not _TORCH_AVAILABLE or self._model is None:
            n = X.shape[0] if X.ndim >= 1 else 1
            logger.debug("EDGE-001 stub: returning %d zero predictions.", n)
            return np.zeros(n, dtype=np.float64)

        X = self._ensure_3d(X)
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.params["device"])
            preds = self._model(X_t).cpu().numpy()
        return preds.astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return negative MSE (higher is better).

        Args:
            X: feature array
            y: true next-bar returns

        Returns:
            Negative mean squared error.
        """
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        mse = float(np.mean((preds - y) ** 2))
        return -mse

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        X = self._ensure_3d(X)
        if y is not None:
            y = np.asarray(y, dtype=np.float32).ravel()
            if len(y) != X.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} samples but y has {len(y)}"
                )
        return X, y

    def _ensure_3d(self, X):
        """If X is 2-D (samples, features), reshape to (samples, 1, features)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if X.ndim != 3:
            raise ValueError(f"Expected 2-D or 3-D input, got {X.ndim}-D")
        return X

    def get_params(self) -> Dict[str, Any]:
        """Return model hyper-parameters."""
        return dict(self.params)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"MambaTimeSeriesPredictor(d_model={self.params['d_model']}, "
            f"n_layers={self.params['n_layers']}, {status})"
        )
