"""
Velox ML Alpha Models (EDGE-001 through EDGE-006)
===================================================

Bleeding-edge machine learning models for alpha generation, risk management,
and market microstructure analysis.  All models conform to the AlphaModel
interface:

    model.fit(X, y)       -- train / calibrate
    model.predict(X)      -- generate predictions
    model.score(X, y)     -- evaluate performance

Each module is designed with fail-open semantics: if optional dependencies
(PyTorch, ripser, etc.) are unavailable, models degrade gracefully to
simpler fallbacks rather than raising import errors.

Models
------
EDGE-001  MambaTimeSeriesPredictor   Mamba state space model for return prediction
EDGE-002  MARLTradingModel           Multi-agent RL (alpha + risk + execution)
EDGE-003  NeuralHawkesPredictor      Neural Hawkes process for order flow
EDGE-004  ConformalPredictor         Conformal prediction for calibrated intervals
EDGE-005  TDARegimeDetector          Topological data analysis for regime detection
EDGE-006  DistillationModel          Knowledge distillation (teacher -> student)
"""

import logging

logger = logging.getLogger(__name__)

# EDGE-001: Mamba State Space Model
from .mamba_ssm import MambaTimeSeriesPredictor

# EDGE-002: Multi-Agent Reinforcement Learning
from .marl import MARLTradingModel

# EDGE-003: Neural Hawkes Process
from .neural_hawkes import NeuralHawkesPredictor

# EDGE-004: Conformal Prediction
from .conformal import ConformalPredictor, ConformalClassifier

# EDGE-005: Topological Data Analysis
from .tda_regime import TDARegimeDetector

# EDGE-006: Knowledge Distillation
from .distillation import DistillationModel, TeacherEnsemble

__all__ = [
    # EDGE-001
    "MambaTimeSeriesPredictor",
    # EDGE-002
    "MARLTradingModel",
    # EDGE-003
    "NeuralHawkesPredictor",
    # EDGE-004
    "ConformalPredictor",
    "ConformalClassifier",
    # EDGE-005
    "TDARegimeDetector",
    # EDGE-006
    "DistillationModel",
    "TeacherEnsemble",
]

logger.debug("Velox ML models loaded: %s", ", ".join(__all__))
