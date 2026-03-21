"""COMP-016: Macro-regime-based factor exposure timing.

Adjusts factor exposures (value, momentum, quality, size, low-volatility)
based on the current macro regime.  Integrates with HMM regime state and
macro indicators to determine which factors to overweight or underweight.

Factor behavior across regimes:
    - **Expansion**: Momentum + Size dominate (risk-on, small-cap leadership).
    - **Late cycle**: Quality + Low-Vol dominate (flight to safety begins).
    - **Contraction**: Value + Quality dominate (bargain hunting, resilience).
    - **Recovery**: Value + Momentum dominate (mean reversion + early trend).

Usage:
    timer = FactorTimer()
    regime = "expansion"  # from HMM or macro model
    weights = timer.get_factor_weights(regime)
    adjusted = timer.adjust_portfolio_exposures(current_exposures, regime)

Dependencies: numpy, pandas (required).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factor regime mapping
# ---------------------------------------------------------------------------

# Default factor-regime weight matrix.
# Rows: regimes.  Columns: factors.
# Values represent target overweight (+) or underweight (-) signals.
# Neutral = 0.  Range [-1, 1].

FACTORS = ["value", "momentum", "quality", "size", "low_volatility"]

REGIME_FACTOR_MATRIX: Dict[str, Dict[str, float]] = {
    "expansion": {
        "value": -0.2,
        "momentum": 0.6,
        "quality": 0.0,
        "size": 0.5,       # small cap outperforms
        "low_volatility": -0.3,
    },
    "late_cycle": {
        "value": 0.1,
        "momentum": -0.1,
        "quality": 0.5,
        "size": -0.3,      # shift to large cap
        "low_volatility": 0.5,
    },
    "contraction": {
        "value": 0.5,
        "momentum": -0.4,
        "quality": 0.5,
        "size": -0.5,      # large cap safety
        "low_volatility": 0.6,
    },
    "recovery": {
        "value": 0.6,
        "momentum": 0.4,
        "quality": 0.0,
        "size": 0.3,
        "low_volatility": -0.2,
    },
    # HMM-compatible regime names
    "low_vol_bull": {
        "value": -0.1,
        "momentum": 0.5,
        "quality": 0.1,
        "size": 0.3,
        "low_volatility": -0.2,
    },
    "high_vol_bull": {
        "value": 0.2,
        "momentum": 0.3,
        "quality": 0.3,
        "size": 0.0,
        "low_volatility": 0.1,
    },
    "low_vol_bear": {
        "value": 0.3,
        "momentum": -0.2,
        "quality": 0.4,
        "size": -0.2,
        "low_volatility": 0.3,
    },
    "high_vol_bear": {
        "value": 0.4,
        "momentum": -0.5,
        "quality": 0.5,
        "size": -0.5,
        "low_volatility": 0.6,
    },
    # Default / unknown regime
    "neutral": {
        "value": 0.0,
        "momentum": 0.0,
        "quality": 0.0,
        "size": 0.0,
        "low_volatility": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Macro indicator processing
# ---------------------------------------------------------------------------


class MacroIndicatorProcessor:
    """Process macro indicators to inform regime classification.

    Supports: yield curve slope, credit spreads, PMI, unemployment claims,
    VIX level, and Fed funds rate.
    """

    # Thresholds for macro-to-regime mapping
    INDICATOR_THRESHOLDS = {
        "yield_curve_slope": {"contraction": -0.2, "late_cycle": 0.3, "expansion": 1.0},
        "credit_spread": {"expansion": 3.0, "late_cycle": 4.5, "contraction": 6.0},
        "pmi": {"contraction": 47.0, "recovery": 50.0, "expansion": 55.0},
        "vix": {"expansion": 15.0, "late_cycle": 22.0, "contraction": 30.0},
    }

    def classify_regime(
        self, indicators: Dict[str, float],
    ) -> Tuple[str, float]:
        """Classify macro regime from indicator values.

        Parameters
        ----------
        indicators : dict
            Mapping of indicator names to their current values.
            Supported: yield_curve_slope, credit_spread, pmi, vix.

        Returns
        -------
        Tuple[str, float]
            (regime_name, confidence).
        """
        try:
            votes: Dict[str, float] = {}

            if "yield_curve_slope" in indicators:
                slope = indicators["yield_curve_slope"]
                if slope < self.INDICATOR_THRESHOLDS["yield_curve_slope"]["contraction"]:
                    votes["contraction"] = votes.get("contraction", 0) + 1.0
                elif slope < self.INDICATOR_THRESHOLDS["yield_curve_slope"]["late_cycle"]:
                    votes["late_cycle"] = votes.get("late_cycle", 0) + 1.0
                else:
                    votes["expansion"] = votes.get("expansion", 0) + 1.0

            if "credit_spread" in indicators:
                spread = indicators["credit_spread"]
                if spread > self.INDICATOR_THRESHOLDS["credit_spread"]["contraction"]:
                    votes["contraction"] = votes.get("contraction", 0) + 1.0
                elif spread > self.INDICATOR_THRESHOLDS["credit_spread"]["late_cycle"]:
                    votes["late_cycle"] = votes.get("late_cycle", 0) + 1.0
                else:
                    votes["expansion"] = votes.get("expansion", 0) + 1.0

            if "pmi" in indicators:
                pmi = indicators["pmi"]
                if pmi < self.INDICATOR_THRESHOLDS["pmi"]["contraction"]:
                    votes["contraction"] = votes.get("contraction", 0) + 1.0
                elif pmi < self.INDICATOR_THRESHOLDS["pmi"]["recovery"]:
                    votes["recovery"] = votes.get("recovery", 0) + 1.0
                else:
                    votes["expansion"] = votes.get("expansion", 0) + 1.0

            if "vix" in indicators:
                vix = indicators["vix"]
                if vix > self.INDICATOR_THRESHOLDS["vix"]["contraction"]:
                    votes["contraction"] = votes.get("contraction", 0) + 1.0
                elif vix > self.INDICATOR_THRESHOLDS["vix"]["late_cycle"]:
                    votes["late_cycle"] = votes.get("late_cycle", 0) + 1.0
                else:
                    votes["expansion"] = votes.get("expansion", 0) + 1.0

            if not votes:
                return "neutral", 0.0

            total = sum(votes.values())
            best = max(votes, key=votes.get)
            confidence = votes[best] / total

            return best, confidence

        except Exception as e:
            logger.error("Macro regime classification failed: %s", e)
            return "neutral", 0.0


# ---------------------------------------------------------------------------
# Factor timer
# ---------------------------------------------------------------------------


class FactorTimer:
    """Adjusts factor exposures based on macro regime.

    Parameters
    ----------
    regime_factor_matrix : dict, optional
        Custom regime-to-factor weight mapping.  Defaults to the built-in
        matrix.
    transition_speed : float
        How quickly to transition between regime weights.  0.0 = instant,
        1.0 = fully smoothed (EMA-like).  Default 0.3.
    min_confidence : float
        Minimum regime confidence to act on.  Below this, return neutral.
    """

    def __init__(
        self,
        regime_factor_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        transition_speed: float = 0.3,
        min_confidence: float = 0.4,
    ) -> None:
        self._matrix = regime_factor_matrix or REGIME_FACTOR_MATRIX
        self._transition_speed = transition_speed
        self._min_confidence = min_confidence
        self._current_weights: Optional[Dict[str, float]] = None
        self._macro_processor = MacroIndicatorProcessor()
        self._history: List[Dict[str, Any]] = []

        logger.info(
            "FactorTimer: %d regimes, %d factors, transition_speed=%.2f",
            len(self._matrix), len(FACTORS), transition_speed,
        )

    def get_factor_weights(
        self,
        regime: str,
        confidence: float = 1.0,
    ) -> Dict[str, float]:
        """Get target factor weights for a given regime.

        Parameters
        ----------
        regime : str
            Current regime identifier.
        confidence : float
            Regime classification confidence [0, 1].  Weights are scaled
            by confidence.

        Returns
        -------
        dict
            ``{factor: weight}`` mapping.
        """
        try:
            if regime not in self._matrix:
                logger.warning("Unknown regime %r — using neutral.", regime)
                regime = "neutral"

            if confidence < self._min_confidence:
                logger.debug(
                    "Confidence %.2f below minimum %.2f — returning neutral.",
                    confidence, self._min_confidence,
                )
                return {f: 0.0 for f in FACTORS}

            raw = self._matrix[regime]
            scaled = {f: raw.get(f, 0.0) * confidence for f in FACTORS}

            # Apply transition smoothing
            if self._current_weights is not None and self._transition_speed > 0:
                alpha = 1.0 - self._transition_speed
                smoothed = {}
                for f in FACTORS:
                    smoothed[f] = (
                        alpha * scaled.get(f, 0.0)
                        + self._transition_speed * self._current_weights.get(f, 0.0)
                    )
                scaled = smoothed

            self._current_weights = scaled

            # Record history
            self._history.append({
                "regime": regime,
                "confidence": confidence,
                "weights": dict(scaled),
            })
            if len(self._history) > 1000:
                self._history = self._history[-500:]

            return scaled

        except Exception as e:
            logger.error("Factor weight computation failed: %s", e)
            return {f: 0.0 for f in FACTORS}

    def get_factor_weights_from_macro(
        self, indicators: Dict[str, float],
    ) -> Tuple[Dict[str, float], str, float]:
        """Determine factor weights from raw macro indicators.

        Parameters
        ----------
        indicators : dict
            Raw macro indicator values.

        Returns
        -------
        Tuple[dict, str, float]
            (factor_weights, regime_name, confidence).
        """
        regime, confidence = self._macro_processor.classify_regime(indicators)
        weights = self.get_factor_weights(regime, confidence)
        return weights, regime, confidence

    def adjust_portfolio_exposures(
        self,
        current_exposures: Dict[str, float],
        regime: str,
        confidence: float = 1.0,
        max_tilt: float = 0.3,
    ) -> Dict[str, float]:
        """Adjust existing portfolio factor exposures for the current regime.

        Parameters
        ----------
        current_exposures : dict
            ``{factor: current_exposure}`` mapping.
        regime : str
            Current regime.
        confidence : float
            Regime confidence.
        max_tilt : float
            Maximum adjustment magnitude per factor.

        Returns
        -------
        dict
            ``{factor: adjusted_exposure}`` mapping.
        """
        try:
            target_weights = self.get_factor_weights(regime, confidence)
            adjusted = {}

            for factor, current in current_exposures.items():
                tilt = target_weights.get(factor, 0.0) * max_tilt
                adjusted[factor] = current + tilt

            logger.debug(
                "Adjusted exposures for regime=%s: %s",
                regime,
                {k: f"{v:.3f}" for k, v in adjusted.items()},
            )
            return adjusted

        except Exception as e:
            logger.error("Exposure adjustment failed: %s", e)
            return dict(current_exposures)

    def get_regime_factor_summary(self) -> pd.DataFrame:
        """Return the regime-factor matrix as a DataFrame for inspection.

        Returns
        -------
        pd.DataFrame
            Rows = regimes, columns = factors.
        """
        data = {}
        for regime, weights in self._matrix.items():
            data[regime] = {f: weights.get(f, 0.0) for f in FACTORS}
        return pd.DataFrame(data).T

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the history of regime-factor decisions."""
        return list(self._history)

    def reset(self) -> None:
        """Reset smoothing state and history."""
        self._current_weights = None
        self._history.clear()
