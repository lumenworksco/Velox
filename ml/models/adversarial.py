"""EDGE-009: Adversarial Robustness Testing for Trading Models.

Implements FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient
Descent) attacks to test whether trading models are robust to small
perturbations in their input features.

Use-cases:
  - Verify that alpha signals don't flip on tiny feature noise
  - Stress-test model predictions under adversarial conditions
  - Quantify worst-case prediction shifts for risk management

Requires PyTorch for gradient-based attacks.  Falls back to random
perturbation testing when PyTorch is not available.

All ML library imports are conditional — the bot runs without them.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional PyTorch import
# ---------------------------------------------------------------------------

_HAS_TORCH = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdversarialConfig:
    """Configuration for adversarial robustness testing."""

    epsilon: float = 0.01       # perturbation budget (L-inf norm)
    pgd_steps: int = 10         # number of PGD iterations
    pgd_step_size: float = 0.003  # per-step size for PGD
    n_random_restarts: int = 3  # random restarts for PGD
    loss_fn: str = "mse"        # "mse" or "ce" (cross-entropy)
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AdversarialRobustness:
    """Adversarial robustness evaluator for trading models.

    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : AdversarialConfig, optional
        Attack configuration.
    """

    def __init__(self, config: Optional[AdversarialConfig] = None):
        self.config = config or AdversarialConfig()
        self._fitted = False
        self._use_torch = _HAS_TORCH
        self._results_cache: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def fit(self, model: Any = None, X: Any = None, y: Any = None) -> "AdversarialRobustness":
        """Store reference model and data for later evaluation.

        Parameters
        ----------
        model : torch.nn.Module or callable
            The model to attack.
        X : array-like
            Input features.
        y : array-like
            True labels / targets.
        """
        self._model = model
        self._X = X
        self._y = y
        self._fitted = True
        return self

    def predict(self, X: Any = None) -> np.ndarray:
        """Return adversarial perturbation magnitudes for each sample."""
        X = X if X is not None else self._X
        if X is None:
            raise ValueError("No input data provided")
        report = self.evaluate_robustness(self._model, X, self._y)
        return np.full(len(X) if hasattr(X, "__len__") else 1, report["mean_pred_shift"])

    def score(self, model: Any = None, X: Any = None, y: Any = None) -> Dict[str, float]:
        """Alias for evaluate_robustness with defaults from fit()."""
        model = model or self._model
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        return self.evaluate_robustness(model, X, y)

    # ------------------------------------------------------------------
    # FGSM attack
    # ------------------------------------------------------------------

    def fgsm_attack(
        self,
        model: Any,
        X: Any,
        epsilon: Optional[float] = None,
        y: Optional[Any] = None,
    ) -> np.ndarray:
        """Fast Gradient Sign Method attack.

        Parameters
        ----------
        model : torch.nn.Module or callable
            Target model.
        X : array-like
            Input features (n_samples, n_features).
        epsilon : float, optional
            L-inf perturbation budget.  Defaults to config value.
        y : array-like, optional
            Targets.  If None, uses model predictions as pseudo-labels.

        Returns
        -------
        np.ndarray
            Adversarial examples of same shape as X.
        """
        eps = epsilon if epsilon is not None else self.config.epsilon

        if self._use_torch and isinstance(model, nn.Module):
            return self._fgsm_torch(model, X, eps, y)
        return self._fgsm_random(X, eps)

    def _fgsm_torch(
        self, model: nn.Module, X: Any, epsilon: float, y: Optional[Any]
    ) -> np.ndarray:
        """PyTorch FGSM implementation."""
        device = torch.device(self.config.device)
        model.to(device).eval()

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        X_t.requires_grad_(True)

        if y is not None:
            y_t = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                y_t = model(X_t).detach()

        loss_fn = self._get_loss_fn()
        output = model(X_t)
        loss = loss_fn(output, y_t)
        loss.backward()

        grad_sign = X_t.grad.sign()  # type: ignore[union-attr]
        X_adv = X_t + epsilon * grad_sign
        return X_adv.detach().cpu().numpy()

    def _fgsm_random(self, X: Any, epsilon: float) -> np.ndarray:
        """Random perturbation fallback (no gradients)."""
        X_arr = np.asarray(X, dtype=np.float64)
        perturbation = np.sign(np.random.randn(*X_arr.shape)) * epsilon
        return X_arr + perturbation

    # ------------------------------------------------------------------
    # PGD attack
    # ------------------------------------------------------------------

    def pgd_attack(
        self,
        model: Any,
        X: Any,
        epsilon: Optional[float] = None,
        steps: Optional[int] = None,
        y: Optional[Any] = None,
    ) -> np.ndarray:
        """Projected Gradient Descent attack.

        Iterative version of FGSM with random starts and projection back
        into the epsilon-ball after each step.

        Parameters
        ----------
        model : torch.nn.Module or callable
            Target model.
        X : array-like
            Input features.
        epsilon : float, optional
            Perturbation budget.
        steps : int, optional
            Number of PGD steps.
        y : array-like, optional
            True targets.

        Returns
        -------
        np.ndarray
            Adversarial examples.
        """
        eps = epsilon if epsilon is not None else self.config.epsilon
        n_steps = steps if steps is not None else self.config.pgd_steps

        if self._use_torch and isinstance(model, nn.Module):
            return self._pgd_torch(model, X, eps, n_steps, y)
        return self._pgd_random(X, eps, n_steps)

    def _pgd_torch(
        self,
        model: nn.Module,
        X: Any,
        epsilon: float,
        steps: int,
        y: Optional[Any],
    ) -> np.ndarray:
        """PyTorch PGD implementation with random restarts."""
        device = torch.device(self.config.device)
        model.to(device).eval()

        X_orig = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        if y is not None:
            y_t = torch.tensor(np.asarray(y), dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                y_t = model(X_orig).detach()

        loss_fn = self._get_loss_fn()
        best_adv = X_orig.clone()
        best_loss = torch.full((len(X_orig),), -float("inf"), device=device)
        step_size = self.config.pgd_step_size

        for _restart in range(self.config.n_random_restarts):
            delta = torch.empty_like(X_orig).uniform_(-epsilon, epsilon)
            delta.requires_grad_(True)

            for _step in range(steps):
                X_adv = X_orig + delta
                output = model(X_adv)
                loss = loss_fn(output, y_t)
                loss.backward()

                with torch.no_grad():
                    delta_grad = delta.grad.sign()  # type: ignore[union-attr]
                    delta.data = delta.data + step_size * delta_grad
                    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                    delta.grad.zero_()  # type: ignore[union-attr]

            # Keep the adversarial that achieves the highest per-sample loss
            with torch.no_grad():
                X_adv = X_orig + delta
                output = model(X_adv)
                per_sample = self._per_sample_loss(output, y_t, loss_fn)
                improved = per_sample > best_loss
                best_loss[improved] = per_sample[improved]
                best_adv[improved] = X_adv[improved]

        return best_adv.detach().cpu().numpy()

    def _pgd_random(self, X: Any, epsilon: float, steps: int) -> np.ndarray:
        """Iterative random search fallback (no gradients)."""
        X_arr = np.asarray(X, dtype=np.float64)
        best = X_arr.copy()
        for _ in range(steps):
            delta = np.random.uniform(-epsilon, epsilon, size=X_arr.shape)
            candidate = X_arr + delta
            # No gradient info; just use the last random perturbation
            best = candidate
        return best

    # ------------------------------------------------------------------
    # Robustness evaluation
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Run full robustness evaluation with FGSM and PGD.

        Returns
        -------
        dict
            Keys: clean_loss, fgsm_loss, pgd_loss, mean_pred_shift,
            max_pred_shift, robustness_score (0-1, higher = more robust).
        """
        X_arr = np.asarray(X, dtype=np.float64)

        # Clean predictions
        clean_preds = self._get_predictions(model, X_arr)

        # FGSM adversarial
        X_fgsm = self.fgsm_attack(model, X_arr, y=y)
        fgsm_preds = self._get_predictions(model, X_fgsm)

        # PGD adversarial
        X_pgd = self.pgd_attack(model, X_arr, y=y)
        pgd_preds = self._get_predictions(model, X_pgd)

        # Compute shifts
        fgsm_shift = np.abs(fgsm_preds - clean_preds)
        pgd_shift = np.abs(pgd_preds - clean_preds)

        # Losses
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float64).flatten()
            clean_loss = float(np.mean((clean_preds - y_arr) ** 2))
            fgsm_loss = float(np.mean((fgsm_preds - y_arr) ** 2))
            pgd_loss = float(np.mean((pgd_preds - y_arr) ** 2))
        else:
            clean_loss = 0.0
            fgsm_loss = float(np.mean(fgsm_shift ** 2))
            pgd_loss = float(np.mean(pgd_shift ** 2))

        mean_shift = float(np.mean(pgd_shift))
        max_shift = float(np.max(pgd_shift))
        # Robustness: 1.0 = perfectly robust, 0.0 = extremely fragile
        robustness = float(np.clip(1.0 - mean_shift / (self.config.epsilon + 1e-8), 0.0, 1.0))

        result = {
            "clean_loss": clean_loss,
            "fgsm_loss": fgsm_loss,
            "pgd_loss": pgd_loss,
            "mean_pred_shift": mean_shift,
            "max_pred_shift": max_shift,
            "robustness_score": robustness,
            "epsilon": self.config.epsilon,
        }
        self._results_cache.append(result)
        logger.info(
            "Robustness eval: score=%.3f  mean_shift=%.6f  max_shift=%.6f",
            robustness, mean_shift, max_shift,
        )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model predictions as numpy array."""
        if self._use_torch and isinstance(model, nn.Module):
            device = torch.device(self.config.device)
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32, device=device)
                preds = model(X_t).cpu().numpy()
            return preds.flatten()
        elif callable(model):
            return np.asarray(model(X)).flatten()
        raise TypeError(f"Unsupported model type: {type(model)}")

    def _get_loss_fn(self) -> Any:
        """Return the appropriate PyTorch loss function."""
        if self.config.loss_fn == "ce":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    @staticmethod
    def _per_sample_loss(output: Any, target: Any, loss_fn: Any) -> Any:
        """Compute per-sample loss for best-restart selection."""
        if output.dim() == 1:
            return (output - target.flatten()) ** 2
        return ((output - target) ** 2).mean(dim=-1)
