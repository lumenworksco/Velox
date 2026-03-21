"""
EDGE-006: Knowledge Distillation for Model Compression
========================================================

Trains a small, fast "student" model to mimic the predictions of a large
"teacher" ensemble.  The student learns from *soft labels* (probability
distributions / continuous outputs) rather than hard targets, capturing
inter-class relationships and dark knowledge.

Key techniques:
  - Temperature scaling: soften teacher output probabilities with T > 1
  - Loss blending: alpha * soft_loss + (1 - alpha) * hard_loss
  - Works with both classification (KL divergence) and regression (MSE)
  - Compatible with sklearn estimators and PyTorch models

Conforms to AlphaModel interface:
    fit(X, y)     -- distill teacher into student
    predict(X)    -- fast inference via student
    score(X, y)   -- student performance metric
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
try:
    from sklearn.base import clone as sklearn_clone
    from sklearn.ensemble import (
        GradientBoostingRegressor, GradientBoostingClassifier,
        RandomForestRegressor, RandomForestClassifier,
    )
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Temperature-scaled softmax
# ===================================================================

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature scaling.

    Higher temperature -> softer (more uniform) distribution.

    Args:
        logits: (n_samples, n_classes) or (n_samples,)
        temperature: T >= 1.0

    Returns:
        probabilities with same shape as input
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    scaled = logits / max(temperature, 1e-8)
    exp_scaled = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)


# ===================================================================
# Teacher wrapper: unified interface for ensembles
# ===================================================================

class TeacherEnsemble:
    """Wraps multiple models into a single teacher that produces soft labels.

    For regression: averages predictions and provides per-model outputs.
    For classification: averages predicted probabilities.
    """

    def __init__(self, models: List, task: str = "regression"):
        """
        Args:
            models: list of fitted model objects with predict() (and
                    predict_proba() for classification).
            task: 'regression' or 'classification'.
        """
        self.models = models
        self.task = task
        if not models:
            raise ValueError("EDGE-006: Teacher ensemble requires at least one model.")

    def predict_soft(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Generate soft labels from the teacher ensemble.

        For regression: returns (n_samples,) mean predictions.
        For classification: returns (n_samples, n_classes) averaged soft probs.
        """
        X = np.asarray(X, dtype=np.float64)

        if self.task == "regression":
            preds = np.array([m.predict(X) for m in self.models])
            return preds.mean(axis=0)

        # Classification: average probability distributions
        all_probs = []
        for m in self.models:
            if hasattr(m, "predict_proba"):
                probs = m.predict_proba(X)
            elif hasattr(m, "decision_function"):
                logits = m.decision_function(X)
                probs = softmax_with_temperature(logits, temperature)
            else:
                # One-hot from hard predictions
                preds = m.predict(X)
                n_classes = len(np.unique(preds))
                probs = np.eye(n_classes)[preds.astype(int)]
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        # Apply temperature scaling
        if temperature != 1.0:
            log_probs = np.log(avg_probs + 1e-12)
            avg_probs = softmax_with_temperature(log_probs, temperature)
        return avg_probs

    def predict_individual(self, X: np.ndarray) -> np.ndarray:
        """Return each teacher's prediction. Shape (n_models, n_samples, ...)."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([m.predict(X) for m in self.models])


# ===================================================================
# Sklearn-based distillation (no PyTorch needed)
# ===================================================================

class SklearnDistiller:
    """Distillation using sklearn models only.

    Trains the student on a blend of hard targets and teacher soft labels.
    For regression, soft labels are teacher mean predictions.
    """

    def __init__(self, student, teacher_ensemble: TeacherEnsemble,
                 alpha: float = 0.5, temperature: float = 3.0):
        self.student = student
        self.teacher = teacher_ensemble
        self.alpha = alpha
        self.temperature = temperature

    def fit(self, X: np.ndarray, y: np.ndarray):
        soft_labels = self.teacher.predict_soft(X, self.temperature)

        if self.teacher.task == "regression":
            # Blend: alpha * soft + (1-alpha) * hard
            blended_y = self.alpha * soft_labels + (1 - self.alpha) * y
            self.student.fit(X, blended_y)
        else:
            # For classification with sklearn, train on soft-label argmax
            # (sklearn classifiers don't natively support soft targets)
            hard_targets = y.astype(int)
            soft_targets = np.argmax(soft_labels, axis=1)
            # Blend by randomly choosing teacher vs true label
            rng = np.random.RandomState(42)
            use_soft = rng.rand(len(y)) < self.alpha
            blended = np.where(use_soft, soft_targets, hard_targets)
            self.student.fit(X, blended)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.student.predict(X)


# ===================================================================
# PyTorch-based distillation (full soft-label KD)
# ===================================================================

if _TORCH_AVAILABLE:

    class StudentNetwork(nn.Module):
        """Small feedforward network for the student."""

        def __init__(self, n_features: int, n_hidden: int = 64,
                     n_layers: int = 2, n_outputs: int = 1,
                     task: str = "regression"):
            super().__init__()
            self.task = task
            layers = [nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Dropout(0.1)]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(0.1)])
            layers.append(nn.Linear(n_hidden, n_outputs))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TorchDistiller:
        """Full knowledge distillation with temperature-scaled KL divergence.

        Loss = alpha * T^2 * KL(soft_teacher || soft_student) + (1-alpha) * hard_loss
        """

        def __init__(self, teacher_ensemble: TeacherEnsemble,
                     n_features: int, n_hidden: int = 64, n_layers: int = 2,
                     n_outputs: int = 1, alpha: float = 0.7,
                     temperature: float = 4.0, lr: float = 1e-3,
                     epochs: int = 100, batch_size: int = 64,
                     device: str = "cpu"):
            self.teacher = teacher_ensemble
            self.alpha = alpha
            self.temperature = temperature
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.device = device
            self.task = teacher_ensemble.task

            n_out = n_outputs
            self.student = StudentNetwork(
                n_features, n_hidden, n_layers, n_out, self.task
            ).to(device)

        def fit(self, X: np.ndarray, y: np.ndarray):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            # Get teacher soft labels
            soft_labels = self.teacher.predict_soft(X, self.temperature).astype(np.float32)

            X_t = torch.tensor(X, device=self.device)
            y_t = torch.tensor(y, device=self.device)
            soft_t = torch.tensor(soft_labels, device=self.device)

            dataset = TensorDataset(X_t, y_t, soft_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
            self.student.train()

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for xb, yb, sb in loader:
                    optimizer.zero_grad()
                    student_out = self.student(xb)

                    if self.task == "regression":
                        hard_loss = F.mse_loss(student_out.squeeze(), yb)
                        soft_loss = F.mse_loss(student_out.squeeze(), sb.squeeze())
                    else:
                        # Classification: KL divergence on temperature-scaled logits
                        T = self.temperature
                        student_log_soft = F.log_softmax(student_out / T, dim=-1)
                        teacher_soft = F.softmax(sb / T, dim=-1)
                        soft_loss = F.kl_div(
                            student_log_soft, teacher_soft,
                            reduction="batchmean"
                        ) * (T * T)
                        hard_loss = F.cross_entropy(student_out, yb.long())

                    loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)

                if (epoch + 1) % 20 == 0:
                    avg = epoch_loss / len(dataset)
                    logger.debug("EDGE-006 epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, avg)

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            self.student.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, device=self.device)
                out = self.student(X_t).cpu().numpy()
            if self.task == "regression":
                return out.squeeze()
            else:
                return np.argmax(out, axis=-1)


# ===================================================================
# Public API: DistillationModel (AlphaModel interface)
# ===================================================================

class DistillationModel:
    """Knowledge Distillation: compress a teacher ensemble into a fast student.

    Parameters
    ----------
    teacher_models : list or None
        List of fitted models forming the teacher ensemble.
        If None, a default ensemble is created and trained during fit().
    task : str
        'regression' or 'classification'.
    alpha : float
        Blending weight for soft loss (0 = all hard, 1 = all soft).
    temperature : float
        Softmax temperature for soft labels (higher = softer).
    use_torch : bool or None
        Force PyTorch (True) or sklearn (False) distillation.
        None = auto-detect.
    student_hidden : int
        Hidden layer size for PyTorch student.
    student_layers : int
        Number of hidden layers for PyTorch student.
    lr : float
        Learning rate (PyTorch only).
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    n_teacher_models : int
        Number of models in default teacher ensemble.
    """

    def __init__(self, *, teacher_models: Optional[List] = None,
                 task: str = "regression", alpha: float = 0.7,
                 temperature: float = 4.0, use_torch: Optional[bool] = None,
                 student_hidden: int = 64, student_layers: int = 2,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 64,
                 device: str = "cpu", n_teacher_models: int = 5, **kwargs):
        self.task = task
        self.alpha = alpha
        self.temperature = temperature
        self.use_torch = use_torch if use_torch is not None else _TORCH_AVAILABLE
        self.student_hidden = student_hidden
        self.student_layers = student_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.n_teacher_models = n_teacher_models

        self._teacher_models = teacher_models
        self._teacher: Optional[TeacherEnsemble] = None
        self._distiller = None
        self._fitted = False

    def _build_default_teachers(self, X: np.ndarray, y: np.ndarray) -> List:
        """Build and train a default teacher ensemble."""
        if not _SKLEARN_AVAILABLE:
            logger.warning(
                "EDGE-006: sklearn not available. Cannot build default teachers. "
                "Provide pre-fitted teacher_models."
            )
            return []

        models = []
        if self.task == "regression":
            configs = [
                GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05),
                GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1),
                RandomForestRegressor(n_estimators=200, max_depth=8),
                Ridge(alpha=1.0),
                Ridge(alpha=10.0),
            ]
        else:
            configs = [
                GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05),
                GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1),
                RandomForestClassifier(n_estimators=200, max_depth=8),
                LogisticRegression(max_iter=500, C=1.0),
                LogisticRegression(max_iter=500, C=0.1),
            ]

        for i, model in enumerate(configs[:self.n_teacher_models]):
            try:
                model.fit(X, y)
                models.append(model)
                logger.debug("EDGE-006: Teacher %d/%d trained (%s).",
                             i + 1, self.n_teacher_models, type(model).__name__)
            except Exception as e:
                logger.warning("EDGE-006: Teacher %d failed: %s", i + 1, e)

        return models

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DistillationModel":
        """Train teacher (if needed) and distill into student.

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) targets
        """
        if y is None:
            raise ValueError("EDGE-006: y is required for distillation training.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Build/use teacher ensemble
        if self._teacher_models is not None and len(self._teacher_models) > 0:
            teachers = self._teacher_models
        else:
            logger.info("EDGE-006: Training default teacher ensemble...")
            teachers = self._build_default_teachers(X, y)

        if not teachers:
            logger.error("EDGE-006: No teacher models available. Distillation aborted.")
            self._fitted = False
            return self

        self._teacher = TeacherEnsemble(teachers, self.task)

        # Log teacher performance
        teacher_preds = self._teacher.predict_soft(X, temperature=1.0)
        if self.task == "regression":
            teacher_mse = float(np.mean((teacher_preds - y) ** 2))
            logger.info("EDGE-006: Teacher ensemble MSE=%.6f", teacher_mse)
        else:
            teacher_acc = float(np.mean(np.argmax(teacher_preds, axis=1) == y.astype(int)))
            logger.info("EDGE-006: Teacher ensemble accuracy=%.4f", teacher_acc)

        # Distill
        n_features = X.shape[1]
        if self.use_torch and _TORCH_AVAILABLE:
            n_outputs = 1 if self.task == "regression" else int(np.max(y) + 1)
            self._distiller = TorchDistiller(
                teacher_ensemble=self._teacher,
                n_features=n_features, n_hidden=self.student_hidden,
                n_layers=self.student_layers, n_outputs=n_outputs,
                alpha=self.alpha, temperature=self.temperature,
                lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                device=self.device,
            )
            logger.info("EDGE-006: Distilling with PyTorch (T=%.1f, alpha=%.2f)...",
                         self.temperature, self.alpha)
        elif _SKLEARN_AVAILABLE:
            if self.task == "regression":
                student = Ridge(alpha=1.0)
            else:
                student = LogisticRegression(max_iter=500)
            self._distiller = SklearnDistiller(
                student=student, teacher_ensemble=self._teacher,
                alpha=self.alpha, temperature=self.temperature,
            )
            logger.info("EDGE-006: Distilling with sklearn (T=%.1f, alpha=%.2f)...",
                         self.temperature, self.alpha)
        else:
            logger.error("EDGE-006: Neither PyTorch nor sklearn available for student.")
            self._fitted = False
            return self

        self._distiller.fit(X, y)
        self._fitted = True

        # Log student performance
        student_preds = self._distiller.predict(X)
        if self.task == "regression":
            student_mse = float(np.mean((student_preds - y) ** 2))
            logger.info("EDGE-006: Student MSE=%.6f (distilled).", student_mse)
        else:
            student_acc = float(np.mean(student_preds.astype(int) == y.astype(int)))
            logger.info("EDGE-006: Student accuracy=%.4f (distilled).", student_acc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fast inference through the distilled student model.

        Args:
            X: (n_samples, n_features)

        Returns:
            predictions: (n_samples,)
        """
        if not self._fitted or self._distiller is None:
            n = np.asarray(X).shape[0]
            logger.warning("EDGE-006: Not fitted. Returning zeros.")
            return np.zeros(n, dtype=np.float64)

        return np.asarray(self._distiller.predict(X), dtype=np.float64)

    def predict_teacher(self, X: np.ndarray) -> np.ndarray:
        """Get teacher ensemble predictions (for comparison)."""
        if self._teacher is None:
            raise RuntimeError("EDGE-006: No teacher available. Call fit() first.")
        soft = self._teacher.predict_soft(X, temperature=1.0)
        if self.task == "regression":
            return soft
        return np.argmax(soft, axis=1).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate student performance.

        For regression: negative MSE.
        For classification: accuracy.
        """
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.task == "regression":
            return -float(np.mean((preds - y) ** 2))
        else:
            return float(np.mean(preds.astype(int) == y.astype(int)))

    def compression_ratio(self) -> Dict[str, Any]:
        """Estimate the compression achieved by distillation."""
        if self._teacher is None:
            return {"status": "no teacher"}
        n_teachers = len(self._teacher.models)
        teacher_type = type(self._teacher.models[0]).__name__
        student_type = "TorchStudent" if (self.use_torch and _TORCH_AVAILABLE) else "SklearnStudent"
        return {
            "n_teacher_models": n_teachers,
            "teacher_type": teacher_type,
            "student_type": student_type,
            "alpha": self.alpha,
            "temperature": self.temperature,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "use_torch": self.use_torch,
            "student_hidden": self.student_hidden,
            "student_layers": self.student_layers,
            "epochs": self.epochs,
        }

    def __repr__(self) -> str:
        backend = "torch" if (self.use_torch and _TORCH_AVAILABLE) else "sklearn"
        status = "distilled" if self._fitted else "undistilled"
        return (
            f"DistillationModel(task={self.task}, T={self.temperature}, "
            f"alpha={self.alpha}, backend={backend}, {status})"
        )
