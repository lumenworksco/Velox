"""COMP-014: Transformer-based non-Markovian regime classifier.

Unlike HMM-based regime detection (which is Markovian — future depends
only on the current state), this module uses self-attention to capture
long-range dependencies in regime dynamics.  A regime that started with
a specific pattern 50 steps ago can influence the current classification.

Architecture:
    1. Input embedding: project market features to model dimension.
    2. Positional encoding: sinusoidal, so the model knows temporal order.
    3. Multi-head self-attention (K layers): each time step attends to
       all other time steps, capturing non-Markovian dependencies.
    4. Classification head: project to regime probabilities.

Uses PyTorch if available; otherwise provides a numpy-based simplified
attention mechanism (single-head, single-layer).

Usage:
    classifier = AttentionRegimeClassifier(
        n_features=8, n_regimes=4, seq_len=60,
    )
    features = np.random.randn(60, 8)  # 60 time steps, 8 features
    probs = classifier.classify(features)  # (60, 4) regime probabilities

Dependencies: numpy (required), torch (optional).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Numpy fallback: simplified single-head attention
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encoding matrix.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Model dimension.

    Returns
    -------
    np.ndarray
        (seq_len, d_model) positional encoding matrix.
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])

    return pe


class NumpyAttentionLayer:
    """Single-head self-attention layer using numpy.

    Computes scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
    """

    def __init__(self, d_model: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        scale = math.sqrt(2.0 / d_model)
        self.W_q = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_k = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_v = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.W_o = rng.randn(d_model, d_model).astype(np.float64) * scale
        self.d_model = d_model

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply self-attention.

        Parameters
        ----------
        X : np.ndarray
            (seq_len, d_model) input sequence.

        Returns
        -------
        np.ndarray
            (seq_len, d_model) attended output.
        """
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        scores = Q @ K.T / math.sqrt(self.d_model)
        attn = _softmax(scores, axis=-1)
        out = attn @ V
        return out @ self.W_o


class NumpyAttentionRegime:
    """Numpy-based simplified attention regime classifier.

    Single-layer, single-head attention with linear classification head.

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_regimes : int
        Number of regime classes.
    d_model : int
        Internal model dimension.
    seq_len : int
        Expected input sequence length.
    """

    def __init__(
        self,
        n_features: int = 8,
        n_regimes: int = 4,
        d_model: int = 32,
        seq_len: int = 60,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        scale = math.sqrt(2.0 / n_features)

        # Input projection
        self.W_in = rng.randn(n_features, d_model).astype(np.float64) * scale
        self.b_in = np.zeros(d_model, dtype=np.float64)

        # Attention layer
        self.attention = NumpyAttentionLayer(d_model, seed=seed)

        # Classification head
        scale_out = math.sqrt(2.0 / d_model)
        self.W_out = rng.randn(d_model, n_regimes).astype(np.float64) * scale_out
        self.b_out = np.zeros(n_regimes, dtype=np.float64)

        # Positional encoding
        self.pe = _sinusoidal_encoding(seq_len, d_model)
        self.seq_len = seq_len
        self.n_regimes = n_regimes

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        X : np.ndarray
            (seq_len, n_features) input features.

        Returns
        -------
        np.ndarray
            (seq_len, n_regimes) regime probabilities.
        """
        T = X.shape[0]

        # Input projection
        h = X @ self.W_in + self.b_in  # (T, d_model)

        # Add positional encoding
        pe = self.pe[:T] if T <= self.seq_len else _sinusoidal_encoding(T, h.shape[1])
        h = h + pe

        # Self-attention
        h = h + self.attention.forward(h)  # residual connection

        # Layer norm (simplified: mean/std normalization)
        mean = h.mean(axis=-1, keepdims=True)
        std = h.std(axis=-1, keepdims=True) + 1e-8
        h = (h - mean) / std

        # Classification head
        logits = h @ self.W_out + self.b_out  # (T, n_regimes)
        probs = _softmax(logits, axis=-1)
        return probs


# ---------------------------------------------------------------------------
# PyTorch attention regime classifier
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer."""

        def __init__(self, d_model: int, max_len: int = 500) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class TorchAttentionRegime(nn.Module):
        """Multi-head self-attention regime classifier in PyTorch.

        Parameters
        ----------
        n_features : int
            Input feature dimension.
        n_regimes : int
            Number of regime classes.
        d_model : int
            Transformer model dimension.
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer encoder layers.
        dropout : float
            Dropout rate.
        """

        def __init__(
            self,
            n_features: int = 8,
            n_regimes: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers,
            )
            self.classifier = nn.Linear(d_model, n_regimes)
            self.n_regimes = n_regimes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                (batch, seq_len, n_features) or (seq_len, n_features).

            Returns
            -------
            torch.Tensor
                (batch, seq_len, n_regimes) regime probabilities.
            """
            if x.dim() == 2:
                x = x.unsqueeze(0)

            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.transformer(h)
            logits = self.classifier(h)
            return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


class AttentionRegimeClassifier:
    """Transformer-based non-Markovian regime classifier.

    Captures long-range temporal dependencies in regime dynamics using
    self-attention, unlike Markovian HMM approaches.

    Parameters
    ----------
    n_features : int
        Number of input market features per time step.
    n_regimes : int
        Number of regime classes.
    d_model : int
        Internal model dimension.
    n_heads : int
        Number of attention heads (PyTorch only).
    n_layers : int
        Number of transformer layers (PyTorch only).
    seq_len : int
        Expected sequence length.
    backend : str
        ``"auto"`` selects torch if available, else numpy.
    """

    REGIME_NAMES = {
        0: "low_volatility",
        1: "high_volatility",
        2: "trending",
        3: "mean_reverting",
    }

    def __init__(
        self,
        n_features: int = 8,
        n_regimes: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 60,
        backend: str = "auto",
    ) -> None:
        self.n_features = n_features
        self.n_regimes = n_regimes
        self.seq_len = seq_len

        if backend == "auto" and _HAS_TORCH:
            self._backend = "torch"
            self._model = TorchAttentionRegime(
                n_features=n_features,
                n_regimes=n_regimes,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
            )
            logger.info("AttentionRegimeClassifier using PyTorch backend.")
        else:
            self._backend = "numpy"
            self._model = NumpyAttentionRegime(
                n_features=n_features,
                n_regimes=n_regimes,
                d_model=min(d_model, 32),
                seq_len=seq_len,
            )
            logger.info("AttentionRegimeClassifier using numpy fallback.")

    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify regime at each time step.

        Parameters
        ----------
        features : np.ndarray
            (seq_len, n_features) market features over time.

        Returns
        -------
        np.ndarray
            (seq_len, n_regimes) regime probabilities.
        """
        try:
            features = np.asarray(features, dtype=np.float64)
            if features.ndim == 1:
                features = features.reshape(-1, 1)

            if self._backend == "numpy":
                return self._model.forward(features)
            else:
                return self._classify_torch(features)

        except Exception as e:
            logger.error("Regime classification failed: %s — returning uniform", e)
            T = features.shape[0] if features.ndim >= 1 else 1
            return np.ones((T, self.n_regimes)) / self.n_regimes

    def _classify_torch(self, features: np.ndarray) -> np.ndarray:
        """Run classification with PyTorch model."""
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            probs = self._model(x)
        return probs.squeeze(0).cpu().numpy()

    def get_current_regime(self, features: np.ndarray) -> Tuple[int, float, str]:
        """Get the most likely regime for the last time step.

        Parameters
        ----------
        features : np.ndarray
            (seq_len, n_features) market features.

        Returns
        -------
        Tuple[int, float, str]
            (regime_id, probability, regime_name)
        """
        probs = self.classify(features)
        last_probs = probs[-1]
        regime_id = int(np.argmax(last_probs))
        prob = float(last_probs[regime_id])
        name = self.REGIME_NAMES.get(regime_id, f"regime_{regime_id}")
        return regime_id, prob, name

    def get_attention_weights(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Extract attention weights for interpretability.

        Only available with numpy backend (torch would need hooks).

        Returns
        -------
        np.ndarray or None
            (seq_len, seq_len) attention weight matrix, or None.
        """
        if self._backend != "numpy":
            logger.debug("Attention weight extraction not supported for torch backend.")
            return None

        try:
            features = np.asarray(features, dtype=np.float64)
            h = features @ self._model.W_in + self._model.b_in
            T = h.shape[0]
            pe = self._model.pe[:T]
            h = h + pe

            Q = h @ self._model.attention.W_q
            K = h @ self._model.attention.W_k
            scores = Q @ K.T / math.sqrt(self._model.attention.d_model)
            return _softmax(scores, axis=-1)

        except Exception as e:
            logger.warning("Failed to extract attention weights: %s", e)
            return None

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "backend": self._backend,
            "n_features": self.n_features,
            "n_regimes": self.n_regimes,
            "seq_len": self.seq_len,
            "has_torch": _HAS_TORCH,
            "regime_names": dict(self.REGIME_NAMES),
        }
