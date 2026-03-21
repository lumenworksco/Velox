"""COMP-010: Graph Neural Network for stock relationship modeling.

Models stock-to-stock relationships as a graph where edges represent
correlation, sector membership, or supply-chain linkages.  Uses message
passing to propagate information across the graph and predicts returns
using neighborhood aggregation.

Architecture:
    1. Build adjacency matrix from correlation, sector, or custom edges.
    2. K rounds of message passing (neighborhood aggregation).
    3. Node-level readout produces per-stock return predictions.

Uses PyTorch Geometric if available; otherwise falls back to a simple
adjacency-matrix-based message-passing implementation using numpy.

Usage:
    gnn = StockGNN(n_stocks=50, feature_dim=10, hidden_dim=32)
    adj = gnn.build_correlation_adjacency(returns_matrix, threshold=0.5)
    predictions = gnn.predict(features, adj)

Dependencies: numpy (required), torch + torch_geometric (optional).
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
_HAS_PYG = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    pass

try:
    import torch_geometric  # noqa: F401

    _HAS_PYG = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Adjacency construction helpers
# ---------------------------------------------------------------------------


def build_correlation_adjacency(
    returns: np.ndarray,
    threshold: float = 0.5,
    abs_value: bool = True,
) -> np.ndarray:
    """Build an adjacency matrix from a returns matrix.

    Parameters
    ----------
    returns : np.ndarray
        (T, N) matrix of asset returns — T observations, N assets.
    threshold : float
        Minimum |correlation| to create an edge.  Default 0.5.
    abs_value : bool
        If True, use absolute correlation for thresholding.

    Returns
    -------
    np.ndarray
        (N, N) adjacency matrix with 0/1 entries (self-loops excluded).
    """
    try:
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2 or returns.shape[0] < 2:
            logger.warning("Returns matrix needs shape (T>=2, N). Returning identity.")
            n = returns.shape[1] if returns.ndim == 2 else 1
            return np.eye(n)

        corr = np.corrcoef(returns, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        values = np.abs(corr) if abs_value else corr
        adj = (values >= threshold).astype(np.float64)
        np.fill_diagonal(adj, 0.0)

        n_edges = int(adj.sum())
        logger.info(
            "Correlation adjacency: %d nodes, %d edges (threshold=%.2f)",
            adj.shape[0], n_edges, threshold,
        )
        return adj
    except Exception as e:
        logger.error("Failed to build correlation adjacency: %s", e)
        n = returns.shape[1] if returns.ndim == 2 else 1
        return np.zeros((n, n))


def build_sector_adjacency(
    sector_labels: List[str],
) -> np.ndarray:
    """Build adjacency from sector membership — same-sector stocks are connected.

    Parameters
    ----------
    sector_labels : list of str
        Sector label for each stock (length N).

    Returns
    -------
    np.ndarray
        (N, N) binary adjacency matrix.
    """
    n = len(sector_labels)
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if sector_labels[i] == sector_labels[j]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    logger.info("Sector adjacency: %d nodes, %d edges", n, int(adj.sum()))
    return adj


def combine_adjacencies(
    adjacencies: List[np.ndarray],
    weights: Optional[List[float]] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Combine multiple adjacency matrices with optional weights.

    Parameters
    ----------
    adjacencies : list of np.ndarray
        List of (N, N) adjacency matrices.
    weights : list of float, optional
        Weights for each adjacency.  Defaults to equal weights.
    threshold : float
        Threshold on the weighted sum to produce binary adjacency.

    Returns
    -------
    np.ndarray
        (N, N) combined binary adjacency matrix.
    """
    if not adjacencies:
        return np.array([[]])
    if weights is None:
        weights = [1.0 / len(adjacencies)] * len(adjacencies)
    combined = sum(w * a for w, a in zip(weights, adjacencies))
    binary = (combined >= threshold).astype(np.float64)
    np.fill_diagonal(binary, 0.0)
    return binary


# ---------------------------------------------------------------------------
# Numpy message-passing GNN (fallback)
# ---------------------------------------------------------------------------


class NumpyGNNLayer:
    """Single message-passing layer using adjacency matrix multiplication.

    Implements: h' = ReLU(D^{-1} A X W + b)
    where D is degree matrix, A is adjacency, X is features, W is weight.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        scale = math.sqrt(2.0 / in_dim)  # He initialization
        self.W = rng.randn(in_dim, out_dim).astype(np.float64) * scale
        self.b = np.zeros(out_dim, dtype=np.float64)

    def forward(
        self, X: np.ndarray, adj: np.ndarray, add_self_loops: bool = True,
    ) -> np.ndarray:
        """Forward pass: aggregate neighborhood features.

        Parameters
        ----------
        X : np.ndarray
            (N, in_dim) node feature matrix.
        adj : np.ndarray
            (N, N) adjacency matrix.
        add_self_loops : bool
            Whether to add self-loops for self-aggregation.

        Returns
        -------
        np.ndarray
            (N, out_dim) updated node features.
        """
        A = adj.copy()
        if add_self_loops:
            A = A + np.eye(A.shape[0])

        # Degree normalization: D^{-1} A
        degree = A.sum(axis=1, keepdims=True)
        degree = np.maximum(degree, 1e-12)
        A_norm = A / degree

        # Message passing + linear transform
        aggregated = A_norm @ X  # (N, in_dim)
        out = aggregated @ self.W + self.b  # (N, out_dim)

        # ReLU activation
        return np.maximum(out, 0.0)


class NumpyGNN:
    """Multi-layer message-passing GNN using numpy.

    Parameters
    ----------
    n_features : int
        Input feature dimension per node.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output dimension (e.g. 1 for return prediction).
    n_layers : int
        Number of message-passing layers.
    seed : int
        Random seed for weight initialization.
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 1,
        n_layers: int = 2,
        seed: int = 42,
    ) -> None:
        self.layers: List[NumpyGNNLayer] = []
        in_dim = n_features
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            self.layers.append(NumpyGNNLayer(in_dim, out_dim, seed=seed + i))
            in_dim = out_dim

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        logger.info(
            "NumpyGNN: %d layers, %d -> %d -> %d",
            n_layers, n_features, hidden_dim, output_dim,
        )

    def forward(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Forward pass through all layers.

        Parameters
        ----------
        X : np.ndarray
            (N, n_features) node features.
        adj : np.ndarray
            (N, N) adjacency matrix.

        Returns
        -------
        np.ndarray
            (N, output_dim) predictions per node.
        """
        h = X
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, adj)
            # No activation on last layer
            if i < len(self.layers) - 1:
                pass  # ReLU already applied in layer
        return h


# ---------------------------------------------------------------------------
# StockGNN — unified interface
# ---------------------------------------------------------------------------


class StockGNN:
    """Graph Neural Network for stock return prediction.

    Automatically selects PyTorch Geometric backend if available,
    otherwise uses numpy message-passing fallback.

    Parameters
    ----------
    n_features : int
        Number of input features per stock.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output dimension (default 1 for return prediction).
    n_layers : int
        Number of message-passing rounds.
    backend : str
        ``"auto"`` selects best available; ``"numpy"`` forces fallback.
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 1,
        n_layers: int = 2,
        backend: str = "auto",
    ) -> None:
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backend = backend

        if backend == "auto" and _HAS_TORCH and _HAS_PYG:
            self._backend = "pyg"
            self._model = self._build_pyg_model(
                n_features, hidden_dim, output_dim, n_layers,
            )
            logger.info("StockGNN using PyTorch Geometric backend.")
        elif backend == "auto" and _HAS_TORCH:
            self._backend = "torch"
            self._model = self._build_torch_model(
                n_features, hidden_dim, output_dim, n_layers,
            )
            logger.info("StockGNN using PyTorch (no PYG) backend.")
        else:
            self._backend = "numpy"
            self._model = NumpyGNN(
                n_features, hidden_dim, output_dim, n_layers,
            )
            logger.info("StockGNN using numpy fallback backend.")

    def predict(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """Predict per-stock returns.

        Parameters
        ----------
        features : np.ndarray
            (N, n_features) matrix of stock features.
        adjacency : np.ndarray
            (N, N) adjacency matrix.

        Returns
        -------
        np.ndarray
            (N,) predicted return for each stock.
        """
        try:
            features = np.asarray(features, dtype=np.float64)
            adjacency = np.asarray(adjacency, dtype=np.float64)

            if features.shape[0] != adjacency.shape[0]:
                raise ValueError(
                    f"Feature rows ({features.shape[0]}) != adjacency size "
                    f"({adjacency.shape[0]})"
                )

            if self._backend == "numpy":
                out = self._model.forward(features, adjacency)
                return out.ravel()

            elif self._backend in ("torch", "pyg"):
                return self._predict_torch(features, adjacency)

            else:
                logger.error("Unknown backend %r", self._backend)
                return np.zeros(features.shape[0])

        except Exception as e:
            logger.error("StockGNN prediction failed: %s — returning zeros", e)
            return np.zeros(features.shape[0])

    def _predict_torch(
        self, features: np.ndarray, adjacency: np.ndarray,
    ) -> np.ndarray:
        """Run prediction with torch-based models."""
        x = torch.tensor(features, dtype=torch.float32)
        adj = torch.tensor(adjacency, dtype=torch.float32)

        self._model.eval()
        with torch.no_grad():
            out = self._model(x, adj)
        return out.cpu().numpy().ravel()

    @staticmethod
    def _build_pyg_model(
        n_features: int, hidden_dim: int, output_dim: int, n_layers: int,
    ):
        """Build a simple GCN model using PyTorch Geometric."""
        from torch_geometric.nn import GCNConv

        class PYGModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                in_d = n_features
                for i in range(n_layers):
                    out_d = hidden_dim if i < n_layers - 1 else output_dim
                    self.convs.append(GCNConv(in_d, out_d))
                    in_d = out_d

            def forward(self, x, adj):
                # Convert dense adjacency to edge_index
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                h = x
                for i, conv in enumerate(self.convs):
                    h = conv(h, edge_index)
                    if i < len(self.convs) - 1:
                        h = F.relu(h)
                return h

        return PYGModel()

    @staticmethod
    def _build_torch_model(
        n_features: int, hidden_dim: int, output_dim: int, n_layers: int,
    ):
        """Build a simple adjacency-based GNN using plain PyTorch."""

        class TorchGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList()
                in_d = n_features
                for i in range(n_layers):
                    out_d = hidden_dim if i < n_layers - 1 else output_dim
                    self.linears.append(nn.Linear(in_d, out_d))
                    in_d = out_d

            def forward(self, x, adj):
                A = adj + torch.eye(adj.shape[0], device=adj.device)
                deg = A.sum(dim=1, keepdim=True).clamp(min=1e-12)
                A_norm = A / deg
                h = x
                for i, lin in enumerate(self.linears):
                    h = A_norm @ h
                    h = lin(h)
                    if i < len(self.linears) - 1:
                        h = F.relu(h)
                return h

        return TorchGNN()

    # --- convenience adjacency builders ------------------------------------

    build_correlation_adjacency = staticmethod(build_correlation_adjacency)
    build_sector_adjacency = staticmethod(build_sector_adjacency)
    combine_adjacencies = staticmethod(combine_adjacencies)

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "backend": self._backend,
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "has_torch": _HAS_TORCH,
            "has_pyg": _HAS_PYG,
        }
