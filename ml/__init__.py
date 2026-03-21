"""ML Pipeline — feature engineering, model training, online learning, and validation.

Modules:
    features         — ML-001: 200+ feature engineering framework
    training         — ML-002: Ensemble model training with purged CV
    online_learning  — ML-003: Online learning with drift detection
    validation       — ML-004: Overfitting prevention (CPCV, DSR, PBO)
    fracdiff         — LPRADO-001: Fractional differentiation
    meta_labeling    — LPRADO-002: Triple-barrier meta-labeling
    information_bars — LPRADO-003: Information-driven bars (TIB, VIB, DIB)
    entropy          — LPRADO-004: Entropy features (Shannon, SampEn, PermEn)
    sample_weights   — LPRADO-005: Sample weights by uniqueness
    covariance_cleaning — ADVML-002: RMT Marchenko-Pastur covariance denoising
    change_point        — ADVML-003: Bayesian Online Change-Point Detection
    explainability      — ADVML-004: XAI / SHAP explanation layer
    synthetic_data      — ADVML-005: Synthetic data generation (GARCH + bootstrap)
    inference           — PROD-002: Batch inference engine
    model_registry      — PROD-007: Model version registry with rollback
"""

__all__ = [
    "features",
    "training",
    "online_learning",
    "validation",
    "fracdiff",
    "meta_labeling",
    "information_bars",
    "entropy",
    "sample_weights",
    "covariance_cleaning",
    "change_point",
    "explainability",
    "synthetic_data",
    "inference",
    "model_registry",
]
