"""EDGE-010: LSTM Autoencoder for Market Anomaly Detection.

Trains on "normal" market behaviour and flags anomalies when the
reconstruction error exceeds a learned threshold.  Useful for:

  - Detecting regime changes and market dislocations
  - Flagging unusual price/volume patterns before they cause losses
  - Filtering out abnormal data from model training sets

Uses PyTorch when available; falls back to a simple PCA-based
autoencoder implemented with numpy.

All ML library imports are conditional — the bot runs without them.
"""

import logging
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
class AutoencoderConfig:
    """Configuration for the LSTM autoencoder."""

    hidden_dim: int = 64
    latent_dim: int = 16
    n_layers: int = 2
    dropout: float = 0.1
    seq_len: int = 20          # lookback window
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    threshold_percentile: float = 95.0
    device: str = "cpu"
    seed: int = 42
    # PCA fallback
    pca_components: int = 5


# ---------------------------------------------------------------------------
# PyTorch LSTM Autoencoder
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class _Encoder(nn.Module):
        def __init__(self, input_dim: int, cfg: AutoencoderConfig):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, cfg.hidden_dim, cfg.n_layers,
                batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
            _, (h, c) = self.lstm(x)
            latent = self.fc(h[-1])
            return latent, (h, c)

    class _Decoder(nn.Module):
        def __init__(self, input_dim: int, cfg: AutoencoderConfig):
            super().__init__()
            self.fc = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
            self.lstm = nn.LSTM(
                cfg.hidden_dim, cfg.hidden_dim, cfg.n_layers,
                batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            )
            self.output = nn.Linear(cfg.hidden_dim, input_dim)
            self.seq_len = cfg.seq_len

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            h = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(h)
            return self.output(out)

    class _LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim: int, cfg: AutoencoderConfig):
            super().__init__()
            self.encoder = _Encoder(input_dim, cfg)
            self.decoder = _Decoder(input_dim, cfg)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent, _ = self.encoder(x)
            return self.decoder(latent)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LSTMAutoencoder:
    """LSTM Autoencoder for anomaly detection in market data.

    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : AutoencoderConfig, optional
        Model configuration.
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None):
        self.config = config or AutoencoderConfig()
        self._model = None
        self._fitted = False
        self._use_torch = _HAS_TORCH
        self._threshold: float = 0.0
        self._input_dim: int = 1
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        # PCA fallback state
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._train_errors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def fit(self, normal_data: np.ndarray, **kwargs: Any) -> "LSTMAutoencoder":
        """Train on normal (non-anomalous) market data.

        Parameters
        ----------
        normal_data : np.ndarray
            2-D array (n_samples, n_features) of normal market observations.
        """
        data = np.asarray(normal_data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._input_dim = data.shape[1]

        # Normalize
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0) + 1e-8
        data_norm = (data - self._mean) / self._std

        if self._use_torch:
            self._fit_torch(data_norm, **kwargs)
        else:
            logger.info("PyTorch not available — using PCA-based autoencoder fallback")
            self._fit_pca(data_norm)

        # Compute threshold from training reconstruction errors
        train_errors = self._compute_errors(data_norm)
        self._train_errors = train_errors
        self._threshold = float(np.percentile(train_errors, self.config.threshold_percentile))
        logger.info(
            "Anomaly threshold set at %.6f (p%d of training errors)",
            self._threshold, int(self.config.threshold_percentile),
        )
        self._fitted = True
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels (1 = anomaly, 0 = normal)."""
        scores = self.detect_anomalies(data)
        return (scores > self._threshold).astype(int)

    def score(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate anomaly detection quality.

        If labels are provided (1=anomaly), computes precision/recall.
        Otherwise returns reconstruction error statistics.
        """
        scores = self.detect_anomalies(data)
        preds = (scores > self._threshold).astype(int)
        result: Dict[str, float] = {
            "mean_error": float(np.mean(scores)),
            "max_error": float(np.max(scores)),
            "threshold": self._threshold,
            "anomaly_rate": float(np.mean(preds)),
        }
        if labels is not None:
            labels = np.asarray(labels).flatten()
            tp = float(np.sum((preds == 1) & (labels == 1)))
            fp = float(np.sum((preds == 1) & (labels == 0)))
            fn = float(np.sum((preds == 0) & (labels == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            result.update({"precision": precision, "recall": recall, "f1": f1})
        return result

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores (reconstruction error).

        Higher scores indicate more anomalous observations.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before detecting anomalies")

        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_norm = (data - self._mean) / self._std  # type: ignore[operator]
        return self._compute_errors(data_norm)

    def get_threshold(self, percentile: float = 95.0) -> float:
        """Return the anomaly threshold at the given percentile.

        Uses the training error distribution.
        """
        if self._train_errors is None:
            raise RuntimeError("Model must be fitted first")
        return float(np.percentile(self._train_errors, percentile))

    # ------------------------------------------------------------------
    # PyTorch training
    # ------------------------------------------------------------------

    def _fit_torch(self, data: np.ndarray, **kwargs: Any) -> None:
        """Train the LSTM autoencoder with PyTorch."""
        cfg = self.config
        device = torch.device(cfg.device)
        self._model = _LSTMAutoencoder(self._input_dim, cfg).to(device)
        optimizer = optim.Adam(self._model.parameters(), lr=cfg.learning_rate)

        sequences = self._make_sequences(data, cfg.seq_len)
        dataset = torch.tensor(sequences, dtype=torch.float32, device=device)

        logger.info(
            "Training LSTM autoencoder (%d epochs, %d sequences) ...",
            cfg.epochs, len(sequences),
        )
        self._model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(len(dataset))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(dataset), cfg.batch_size):
                batch = dataset[perm[i : i + cfg.batch_size]]
                recon = self._model(batch)
                loss = ((recon - batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
                logger.info(
                    "  epoch %d/%d  loss=%.6f",
                    epoch + 1, cfg.epochs, total_loss / max(n_batches, 1),
                )

    def _compute_errors_torch(self, data: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors using the LSTM model."""
        cfg = self.config
        device = torch.device(cfg.device)
        self._model.eval()  # type: ignore[union-attr]

        sequences = self._make_sequences(data, cfg.seq_len)
        if len(sequences) == 0:
            return np.array([0.0])

        errors = []
        with torch.no_grad():
            dataset = torch.tensor(sequences, dtype=torch.float32, device=device)
            for i in range(0, len(dataset), cfg.batch_size):
                batch = dataset[i : i + cfg.batch_size]
                recon = self._model(batch)  # type: ignore[misc]
                batch_errors = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.append(batch_errors)
        return np.concatenate(errors) if errors else np.array([0.0])

    # ------------------------------------------------------------------
    # PCA fallback
    # ------------------------------------------------------------------

    def _fit_pca(self, data: np.ndarray) -> None:
        """Fit a PCA-based autoencoder (numpy only)."""
        self._pca_mean = data.mean(axis=0)
        centered = data - self._pca_mean
        n_comp = min(self.config.pca_components, data.shape[1], data.shape[0])
        # SVD-based PCA
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = Vt[:n_comp]
        except np.linalg.LinAlgError:
            logger.warning("SVD failed — using identity projection")
            self._pca_components = np.eye(data.shape[1])[:n_comp]
        logger.info("PCA autoencoder fitted with %d components", n_comp)

    def _compute_errors_pca(self, data: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors using PCA."""
        centered = data - self._pca_mean  # type: ignore[operator]
        projected = centered @ self._pca_components.T  # type: ignore[union-attr]
        reconstructed = projected @ self._pca_components  # type: ignore[union-attr]
        errors = np.mean((centered - reconstructed) ** 2, axis=1)
        return errors

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _compute_errors(self, data_norm: np.ndarray) -> np.ndarray:
        """Route to the correct error computation backend."""
        if self._use_torch and self._model is not None:
            return self._compute_errors_torch(data_norm)
        if self._pca_components is not None:
            return self._compute_errors_pca(data_norm)
        return np.zeros(len(data_norm))

    @staticmethod
    def _make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
        """Convert a 2-D array into overlapping sequences.

        Returns shape (n_sequences, seq_len, n_features).
        """
        if len(data) < seq_len:
            # Pad with zeros
            padded = np.zeros((seq_len, data.shape[1]))
            padded[-len(data) :] = data
            return padded[np.newaxis, :, :]
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i : i + seq_len])
        return np.array(sequences)
