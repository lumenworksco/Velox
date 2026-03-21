"""PROD-007: Model Registry — version tracking, performance metrics, and rollback support.

Provides a centralized registry for trained ML models. Each model version
is stored with metadata (Sharpe, accuracy, PBO, training date, features used)
in a JSON sidecar file. Supports rollback to previous versions.

Usage:
    from ml.model_registry import ModelRegistry

    registry = ModelRegistry(base_dir="models/")
    registry.register("ensemble_v2", model_path="models/ensemble_v2.joblib",
                       metrics={"sharpe": 1.2, "accuracy": 0.65, "pbo": 0.15})
    best = registry.get_best("ensemble", metric="sharpe")
    registry.rollback("ensemble", version=1)
"""

import json
import logging
import os
import shutil
import time as _time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default registry metadata filename
_REGISTRY_FILE = "model_registry.json"


@dataclass
class ModelVersion:
    """Metadata for a single model version."""
    name: str
    version: int
    model_path: str
    registered_at: str  # ISO timestamp
    metrics: Dict[str, float] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    training_samples: int = 0
    description: str = ""
    is_active: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """Centralized model version registry with rollback support.

    Stores model metadata in a JSON file alongside model files.
    Each model "name" can have multiple versions.
    """

    def __init__(self, base_dir: str = "models"):
        """Initialize the model registry.

        Args:
            base_dir: Directory where model files and registry JSON are stored.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._base_dir / _REGISTRY_FILE
        self._models: Dict[str, List[ModelVersion]] = {}
        self._load()

    def _load(self):
        """Load registry from JSON file."""
        if self._registry_path.exists():
            try:
                data = json.loads(self._registry_path.read_text())
                for name, versions in data.get("models", {}).items():
                    self._models[name] = [
                        ModelVersion(**v) for v in versions
                    ]
                logger.info(
                    "PROD-007: Loaded model registry (%d models, %d total versions)",
                    len(self._models),
                    sum(len(v) for v in self._models.values()),
                )
            except Exception as e:
                logger.error("PROD-007: Failed to load model registry: %s", e)
                self._models = {}
        else:
            logger.info("PROD-007: No existing model registry found, starting fresh")

    def _save(self):
        """Persist registry to JSON file."""
        try:
            data = {
                "models": {
                    name: [asdict(v) for v in versions]
                    for name, versions in self._models.items()
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._registry_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("PROD-007: Failed to save model registry: %s", e)

    def register(
        self,
        name: str,
        model_path: str,
        metrics: Optional[Dict[str, float]] = None,
        features: Optional[List[str]] = None,
        training_samples: int = 0,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        set_active: bool = True,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            name: Model family name (e.g., "ensemble", "meta_labeler").
            model_path: Path to the serialized model file.
            metrics: Performance metrics (e.g., {"sharpe": 1.2, "accuracy": 0.65}).
            features: List of feature names used by the model.
            training_samples: Number of training samples.
            description: Human-readable description.
            tags: Arbitrary key-value metadata.
            set_active: Whether to mark this as the active version.

        Returns:
            The registered ModelVersion.
        """
        if name not in self._models:
            self._models[name] = []

        # Determine next version number
        existing = self._models[name]
        next_version = max((v.version for v in existing), default=0) + 1

        version = ModelVersion(
            name=name,
            version=next_version,
            model_path=model_path,
            registered_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics or {},
            features=features or [],
            training_samples=training_samples,
            description=description,
            is_active=False,
            tags=tags or {},
        )

        self._models[name].append(version)

        if set_active:
            self._set_active(name, next_version)

        self._save()
        logger.info(
            "PROD-007: Registered %s v%d (metrics=%s, active=%s)",
            name, next_version, metrics, set_active,
        )
        return version

    def _set_active(self, name: str, version: int):
        """Set a specific version as active (deactivate all others)."""
        for v in self._models.get(name, []):
            v.is_active = (v.version == version)

    def get_active(self, name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model name.

        Returns:
            The active ModelVersion, or None if no active version.
        """
        for v in self._models.get(name, []):
            if v.is_active:
                return v
        return None

    def get_version(self, name: str, version: int) -> Optional[ModelVersion]:
        """Get a specific version of a model.

        Returns:
            The ModelVersion, or None if not found.
        """
        for v in self._models.get(name, []):
            if v.version == version:
                return v
        return None

    def get_best(self, name: str, metric: str = "sharpe") -> Optional[ModelVersion]:
        """Get the version with the best value of a given metric.

        Args:
            name: Model family name.
            metric: Metric key to maximize (e.g., "sharpe", "accuracy").

        Returns:
            The best ModelVersion by the specified metric, or None.
        """
        versions = self._models.get(name, [])
        if not versions:
            return None

        valid = [v for v in versions if metric in v.metrics]
        if not valid:
            return None

        return max(valid, key=lambda v: v.metrics[metric])

    def rollback(self, name: str, version: Optional[int] = None) -> Optional[ModelVersion]:
        """Rollback to a previous model version.

        Args:
            name: Model family name.
            version: Specific version to rollback to. If None, rolls back
                     to the previous version (current - 1).

        Returns:
            The newly active ModelVersion, or None if rollback failed.
        """
        versions = self._models.get(name, [])
        if not versions:
            logger.warning("PROD-007: Cannot rollback %s — no versions found", name)
            return None

        if version is not None:
            target = self.get_version(name, version)
            if not target:
                logger.warning("PROD-007: Cannot rollback %s to v%d — not found", name, version)
                return None
        else:
            # Find current active, roll back to previous
            current_active = self.get_active(name)
            if current_active and current_active.version > 1:
                target = self.get_version(name, current_active.version - 1)
            else:
                target = versions[0] if versions else None

        if target is None:
            logger.warning("PROD-007: No valid rollback target for %s", name)
            return None

        # Verify model file exists
        if not os.path.exists(target.model_path):
            logger.error(
                "PROD-007: Rollback target %s v%d model file missing: %s",
                name, target.version, target.model_path,
            )
            return None

        self._set_active(name, target.version)
        self._save()
        logger.info("PROD-007: Rolled back %s to v%d", name, target.version)
        return target

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model, newest first."""
        return sorted(
            self._models.get(name, []),
            key=lambda v: v.version,
            reverse=True,
        )

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def delete_version(self, name: str, version: int, delete_file: bool = False) -> bool:
        """Delete a model version from the registry.

        Args:
            name: Model family name.
            version: Version number to delete.
            delete_file: Also delete the model file from disk.

        Returns:
            True if the version was deleted.
        """
        versions = self._models.get(name, [])
        target = None
        for i, v in enumerate(versions):
            if v.version == version:
                target = versions.pop(i)
                break

        if target is None:
            return False

        if delete_file and os.path.exists(target.model_path):
            try:
                os.remove(target.model_path)
                logger.info("PROD-007: Deleted model file %s", target.model_path)
            except Exception as e:
                logger.warning("PROD-007: Failed to delete model file: %s", e)

        self._save()
        logger.info("PROD-007: Deleted %s v%d from registry", name, version)
        return True

    def stats(self) -> dict:
        """Return registry statistics."""
        return {
            "total_models": len(self._models),
            "total_versions": sum(len(v) for v in self._models.values()),
            "models": {
                name: {
                    "versions": len(versions),
                    "active_version": next(
                        (v.version for v in versions if v.is_active), None
                    ),
                }
                for name, versions in self._models.items()
            },
        }
