"""EDGE-008: Diffusion Models for Synthetic Market Data Generation.

Implements a simplified Denoising Diffusion Probabilistic Model (DDPM) that
can generate realistic synthetic price return paths.  Useful for:

  - Data augmentation when historical data is scarce
  - Scenario generation for stress testing
  - Training other models on augmented datasets

Uses PyTorch when available; falls back to a simple numpy-based AR(1)+noise
generator that preserves basic statistical properties.

All ML library imports are conditional — the bot runs without them.
"""

import logging
import math
from dataclasses import dataclass, field
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
class DiffusionConfig:
    """Configuration for the diffusion model."""

    n_timesteps: int = 200          # number of diffusion steps
    hidden_dim: int = 128           # hidden layer size for denoiser
    n_layers: int = 3               # depth of denoiser MLP
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    beta_start: float = 1e-4        # noise schedule start
    beta_end: float = 0.02          # noise schedule end
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# PyTorch denoiser network
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class _SinusoidalPosEmb(nn.Module):
        """Sinusoidal positional embedding for timestep conditioning."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            emb = math.log(10_000) / (half - 1)
            emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
            emb = t[:, None].float() * emb[None, :]
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    class _DenoiserMLP(nn.Module):
        """Simple MLP denoiser conditioned on diffusion timestep."""

        def __init__(self, data_dim: int, cfg: DiffusionConfig):
            super().__init__()
            self.time_emb = _SinusoidalPosEmb(cfg.hidden_dim)
            layers: list = [nn.Linear(data_dim + cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
            for _ in range(cfg.n_layers - 1):
                layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
            layers.append(nn.Linear(cfg.hidden_dim, data_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = self.time_emb(t)
            return self.net(torch.cat([x, t_emb], dim=-1))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DiffusionMarketGenerator:
    """Denoising Diffusion model for synthetic market data.

    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : DiffusionConfig, optional
        Model configuration.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self._model = None
        self._fitted = False
        self._data_dim: int = 1
        self._data_mean: float = 0.0
        self._data_std: float = 1.0
        self._betas: Optional[np.ndarray] = None
        self._alphas: Optional[np.ndarray] = None
        self._alpha_bars: Optional[np.ndarray] = None
        self._use_torch = _HAS_TORCH
        self._setup_schedule()

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _setup_schedule(self) -> None:
        """Pre-compute the linear noise schedule."""
        T = self.config.n_timesteps
        self._betas = np.linspace(self.config.beta_start, self.config.beta_end, T)
        self._alphas = 1.0 - self._betas
        self._alpha_bars = np.cumprod(self._alphas)

    # ------------------------------------------------------------------
    # Forward process (add noise)
    # ------------------------------------------------------------------

    def forward_process(
        self, x0: np.ndarray, t: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to data x0 at timestep t.

        Returns (x_t, noise) tuple.
        """
        alpha_bar_t = self._alpha_bars[t]  # type: ignore[index]
        noise = np.random.randn(*x0.shape)
        x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def fit(self, historical_returns: np.ndarray, **kwargs: Any) -> "DiffusionMarketGenerator":
        """Train the diffusion model on historical return data.

        Parameters
        ----------
        historical_returns : np.ndarray
            2-D array of shape (n_samples, features) or 1-D array of returns.
        """
        data = np.asarray(historical_returns, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._data_dim = data.shape[1]
        self._data_mean = float(data.mean())
        self._data_std = float(data.std()) + 1e-8
        data_norm = (data - self._data_mean) / self._data_std

        if self._use_torch:
            self._fit_torch(data_norm, **kwargs)
        else:
            logger.info("PyTorch not available — using AR(1) fallback fit")
            self._fit_fallback(data_norm)

        self._fitted = True
        return self

    def predict(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic samples (alias for generate_paths)."""
        return self.generate_paths(n_paths=n_samples, length=1).squeeze()

    def score(self, real_data: np.ndarray) -> Dict[str, float]:
        """Compare generated data distribution to real data."""
        synthetic = self.generate_paths(n_paths=len(real_data), length=1).squeeze()
        real = np.asarray(real_data).flatten()
        synthetic = synthetic.flatten()[: len(real)]
        return {
            "mean_diff": abs(float(np.mean(synthetic) - np.mean(real))),
            "std_diff": abs(float(np.std(synthetic) - np.std(real))),
            "ks_statistic": float(self._ks_stat(real, synthetic)),
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_paths(self, n_paths: int = 100, length: int = 252) -> np.ndarray:
        """Generate synthetic price return paths.

        Parameters
        ----------
        n_paths : int
            Number of paths to generate.
        length : int
            Length of each path (time steps).

        Returns
        -------
        np.ndarray of shape (n_paths, length, data_dim).
        """
        if not self._fitted:
            logger.warning("Model not fitted — generating pure noise paths")
            return np.random.randn(n_paths, length, self._data_dim)

        if self._use_torch and self._model is not None:
            return self._generate_torch(n_paths, length)
        return self._generate_fallback(n_paths, length)

    # ------------------------------------------------------------------
    # PyTorch training & generation
    # ------------------------------------------------------------------

    def _fit_torch(self, data: np.ndarray, **kwargs: Any) -> None:
        """Train the PyTorch denoiser."""
        cfg = self.config
        device = torch.device(cfg.device)
        self._model = _DenoiserMLP(self._data_dim, cfg).to(device)
        optimizer = optim.Adam(self._model.parameters(), lr=cfg.learning_rate)
        dataset = torch.tensor(data, dtype=torch.float32, device=device)

        T = cfg.n_timesteps
        alpha_bars_t = torch.tensor(self._alpha_bars, dtype=torch.float32, device=device)

        logger.info("Training diffusion model (%d epochs, %d samples) ...", cfg.epochs, len(data))
        self._model.train()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(len(dataset))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(dataset), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                x0 = dataset[idx]
                t = torch.randint(0, T, (len(x0),), device=device)
                ab = alpha_bars_t[t].unsqueeze(-1)
                noise = torch.randn_like(x0)
                x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

                pred_noise = self._model(x_t, t)
                loss = ((pred_noise - noise) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
                logger.info("  epoch %d/%d  loss=%.6f", epoch + 1, cfg.epochs, total_loss / n_batches)

    def _generate_torch(self, n_paths: int, length: int) -> np.ndarray:
        """Reverse diffusion to generate samples (PyTorch)."""
        cfg = self.config
        device = torch.device(cfg.device)
        self._model.eval()  # type: ignore[union-attr]
        all_paths = []

        betas_t = torch.tensor(self._betas, dtype=torch.float32, device=device)
        alphas_t = 1.0 - betas_t
        alpha_bars_t = torch.cumprod(alphas_t, dim=0)

        with torch.no_grad():
            for _ in range(length):
                x = torch.randn(n_paths, self._data_dim, device=device)
                for t_idx in reversed(range(cfg.n_timesteps)):
                    t_batch = torch.full((n_paths,), t_idx, device=device, dtype=torch.long)
                    pred = self._model(x, t_batch)  # type: ignore[misc]
                    alpha = alphas_t[t_idx]
                    alpha_bar = alpha_bars_t[t_idx]
                    x = (1 / alpha.sqrt()) * (x - (betas_t[t_idx] / (1 - alpha_bar).sqrt()) * pred)
                    if t_idx > 0:
                        x += betas_t[t_idx].sqrt() * torch.randn_like(x)
                all_paths.append(x.cpu().numpy())

        result = np.stack(all_paths, axis=1)  # (n_paths, length, dim)
        return result * self._data_std + self._data_mean

    # ------------------------------------------------------------------
    # Numpy fallback (AR(1) + noise)
    # ------------------------------------------------------------------

    def _fit_fallback(self, data: np.ndarray) -> None:
        """Fit a simple AR(1) model as fallback."""
        self._ar_coefs = []
        for d in range(self._data_dim):
            col = data[:, d]
            if len(col) > 1:
                phi = float(np.corrcoef(col[:-1], col[1:])[0, 1])
                phi = np.clip(phi, -0.99, 0.99)
            else:
                phi = 0.0
            self._ar_coefs.append(phi)
        self._residual_std = float(np.std(data))

    def _generate_fallback(self, n_paths: int, length: int) -> np.ndarray:
        """Generate paths using AR(1) fallback."""
        paths = np.zeros((n_paths, length, self._data_dim))
        for d in range(self._data_dim):
            phi = self._ar_coefs[d] if hasattr(self, "_ar_coefs") else 0.0
            noise = np.random.randn(n_paths, length) * self._residual_std
            for t in range(1, length):
                paths[:, t, d] = phi * paths[:, t - 1, d] + noise[:, t]
        return paths * self._data_std + self._data_mean

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
        """Two-sample Kolmogorov-Smirnov statistic (no scipy required)."""
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
        cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)
        return float(np.max(np.abs(cdf_a - cdf_b)))
