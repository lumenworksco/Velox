"""Cross-Asset Signal Generator (ALPHA-001).

Generates macro-level signals from cross-asset relationships to inform
position sizing, regime detection, and directional bias. All signals
are normalized to [-1, +1] where positive = risk-on, negative = risk-off.

Signal sources:
    1. Bond curve slope      — 10Y-2Y spread (steepening = risk-on)
    2. Credit spreads        — HYG-IEF ratio (widening = risk-off)
    3. Dollar strength       — UUP momentum (strong dollar = risk-off for equities)
    4. Gold momentum         — GLD trend (rising gold = risk-off/inflation hedge)
    5. VIX term structure    — VIX vs VIX3M ratio (contango = risk-on, backwardation = risk-off)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cross-asset tickers
BOND_10Y = "TLT"     # 20+ Year Treasury (proxy for 10Y)
BOND_2Y = "SHY"      # 1-3 Year Treasury (proxy for 2Y)
HY_CREDIT = "HYG"    # High Yield Corporate Bond
IG_TREASURY = "IEF"  # 7-10 Year Treasury
DOLLAR = "UUP"       # Dollar Index
GOLD = "GLD"         # Gold
VIX_ETF = "VIXY"     # VIX Short-Term (proxy when VIX futures unavailable)
VIX3M_ETF = "VIXM"   # VIX Mid-Term

# Momentum lookbacks (trading days)
SHORT_LOOKBACK = 5
MEDIUM_LOOKBACK = 20
LONG_LOOKBACK = 60

# Regime thresholds
CREDIT_STRESS_THRESHOLD = -0.02    # HYG/IEF ratio drop > 2% = credit stress
FLIGHT_TO_SAFETY_SCORE = -0.7      # Composite below this = flight to safety
VIX_BACKWARDATION_THRESHOLD = 1.0  # VIX/VIX3M > 1.0 = backwardation (risk-off)

# Signal weights for composite
SIGNAL_WEIGHTS = {
    "bond_curve": 0.20,
    "credit_spread": 0.25,
    "dollar_strength": 0.15,
    "gold_momentum": 0.15,
    "vix_term_structure": 0.25,
}


@dataclass
class CrossAssetState:
    """Current state of all cross-asset signals."""
    bond_curve_slope: float = 0.0       # Normalized slope signal
    credit_spread: float = 0.0          # Normalized credit signal
    dollar_strength: float = 0.0        # Normalized dollar signal
    gold_momentum: float = 0.0          # Normalized gold signal
    vix_term_structure: float = 0.0     # Normalized VIX term signal
    composite: float = 0.0             # Weighted composite
    regime: str = "NEUTRAL"            # RISK_ON, RISK_OFF, NEUTRAL
    timestamp: Optional[datetime] = None


class CrossAssetSignalGenerator:
    """Cross-Asset signal generator for macro regime awareness.

    Monitors bond yields, credit spreads, dollar strength, gold, and VIX
    term structure to produce a composite risk-on/risk-off signal. Used
    by the main engine to adjust position sizing and filter signals.

    Usage:
        gen = CrossAssetSignalGenerator()
        signals = gen.generate_signals(market_data)
        bias = gen.get_regime_bias(market_data)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(SIGNAL_WEIGHTS)
        self._state = CrossAssetState()
        self._last_update: Optional[datetime] = None
        self._history: List[CrossAssetState] = []
        self._max_history = 100

    @property
    def state(self) -> CrossAssetState:
        """Current cross-asset state."""
        return self._state

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate cross-asset signal scores.

        Args:
            market_data: Dict mapping ticker -> DataFrame of daily OHLCV bars.
                         Should include TLT, SHY, HYG, IEF, UUP, GLD, VIXY, VIXM.

        Returns:
            Dict mapping signal name -> score (-1 to +1).
        """
        signals: Dict[str, float] = {}

        # 1. Bond curve slope (TLT vs SHY)
        signals["bond_curve"] = self._compute_bond_curve(
            market_data.get(BOND_10Y), market_data.get(BOND_2Y)
        )

        # 2. Credit spreads (HYG vs IEF)
        signals["credit_spread"] = self._compute_credit_spread(
            market_data.get(HY_CREDIT), market_data.get(IG_TREASURY)
        )

        # 3. Dollar strength (UUP momentum)
        signals["dollar_strength"] = self._compute_dollar_strength(
            market_data.get(DOLLAR)
        )

        # 4. Gold momentum (GLD trend)
        signals["gold_momentum"] = self._compute_gold_momentum(
            market_data.get(GOLD)
        )

        # 5. VIX term structure
        signals["vix_term_structure"] = self._compute_vix_term(
            market_data.get(VIX_ETF), market_data.get(VIX3M_ETF)
        )

        # Composite score
        composite = sum(
            self.weights.get(k, 0) * v for k, v in signals.items()
        )
        signals["composite"] = round(np.clip(composite, -1, 1), 4)

        # Update state
        self._state = CrossAssetState(
            bond_curve_slope=signals["bond_curve"],
            credit_spread=signals["credit_spread"],
            dollar_strength=signals["dollar_strength"],
            gold_momentum=signals["gold_momentum"],
            vix_term_structure=signals["vix_term_structure"],
            composite=signals["composite"],
            regime=self._classify_regime(signals["composite"]),
            timestamp=datetime.now(),
        )
        self._last_update = datetime.now()

        # Maintain rolling history
        self._history.append(self._state)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(
            f"Cross-asset signals: composite={signals['composite']:.3f} "
            f"regime={self._state.regime}"
        )
        return signals

    def get_regime_bias(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Get the current regime bias as a single float.

        Returns:
            Float from -1.0 (strong risk-off) to +1.0 (strong risk-on).
            0.0 indicates neutral conditions.
        """
        signals = self.generate_signals(market_data)
        return signals.get("composite", 0.0)

    def _compute_bond_curve(
        self,
        tlt_bars: Optional[pd.DataFrame],
        shy_bars: Optional[pd.DataFrame],
    ) -> float:
        """Compute bond curve slope signal from TLT/SHY ratio.

        Rising TLT/SHY ratio = curve flattening/inverting = risk-off
        Falling TLT/SHY ratio = curve steepening = risk-on
        """
        if tlt_bars is None or shy_bars is None:
            return 0.0
        if len(tlt_bars) < MEDIUM_LOOKBACK or len(shy_bars) < MEDIUM_LOOKBACK:
            return 0.0

        try:
            ratio = tlt_bars["close"].iloc[-MEDIUM_LOOKBACK:] / shy_bars["close"].iloc[-MEDIUM_LOOKBACK:]
            if len(ratio) < MEDIUM_LOOKBACK:
                return 0.0

            # Momentum of the ratio: negative momentum = steepening = risk-on
            current = ratio.iloc[-1]
            past = ratio.iloc[0]
            if past == 0:
                return 0.0

            change = (current - past) / past

            # Invert: steepening (negative change) = positive signal (risk-on)
            signal = -np.clip(change * 20, -1, 1)
            return round(float(signal), 4)
        except Exception as e:
            logger.debug(f"Bond curve computation failed: {e}")
            return 0.0

    def _compute_credit_spread(
        self,
        hyg_bars: Optional[pd.DataFrame],
        ief_bars: Optional[pd.DataFrame],
    ) -> float:
        """Compute credit spread signal from HYG/IEF ratio.

        Rising HYG/IEF = credit tightening = risk-on
        Falling HYG/IEF = credit widening = risk-off
        """
        if hyg_bars is None or ief_bars is None:
            return 0.0
        if len(hyg_bars) < MEDIUM_LOOKBACK or len(ief_bars) < MEDIUM_LOOKBACK:
            return 0.0

        try:
            ratio = hyg_bars["close"].iloc[-MEDIUM_LOOKBACK:] / ief_bars["close"].iloc[-MEDIUM_LOOKBACK:]
            if len(ratio) < MEDIUM_LOOKBACK:
                return 0.0

            current = ratio.iloc[-1]
            past = ratio.iloc[0]
            if past == 0:
                return 0.0

            change = (current - past) / past

            # Positive change = tightening = risk-on
            signal = np.clip(change * 25, -1, 1)
            return round(float(signal), 4)
        except Exception as e:
            logger.debug(f"Credit spread computation failed: {e}")
            return 0.0

    def _compute_dollar_strength(
        self, uup_bars: Optional[pd.DataFrame]
    ) -> float:
        """Compute dollar strength signal from UUP momentum.

        Rising dollar = headwind for equities/commodities = risk-off
        Falling dollar = tailwind = risk-on
        """
        if uup_bars is None or len(uup_bars) < MEDIUM_LOOKBACK:
            return 0.0

        try:
            close = uup_bars["close"]
            current = close.iloc[-1]
            past = close.iloc[-MEDIUM_LOOKBACK]
            if past == 0:
                return 0.0

            change = (current - past) / past

            # Invert: strong dollar = negative signal
            signal = -np.clip(change * 20, -1, 1)
            return round(float(signal), 4)
        except Exception as e:
            logger.debug(f"Dollar strength computation failed: {e}")
            return 0.0

    def _compute_gold_momentum(
        self, gld_bars: Optional[pd.DataFrame]
    ) -> float:
        """Compute gold momentum signal.

        Rising gold = fear/inflation = risk-off
        Falling gold = complacency/disinflation = risk-on
        """
        if gld_bars is None or len(gld_bars) < MEDIUM_LOOKBACK:
            return 0.0

        try:
            close = gld_bars["close"]
            current = close.iloc[-1]
            past = close.iloc[-MEDIUM_LOOKBACK]
            if past == 0:
                return 0.0

            change = (current - past) / past

            # Invert: rising gold = negative signal (risk-off)
            signal = -np.clip(change * 15, -1, 1)
            return round(float(signal), 4)
        except Exception as e:
            logger.debug(f"Gold momentum computation failed: {e}")
            return 0.0

    def _compute_vix_term(
        self,
        vixy_bars: Optional[pd.DataFrame],
        vixm_bars: Optional[pd.DataFrame],
    ) -> float:
        """Compute VIX term structure signal.

        VIX < VIX3M (contango)      = complacent = risk-on
        VIX > VIX3M (backwardation) = fear = risk-off

        Uses VIXY/VIXM as proxy for VIX/VIX3M when futures unavailable.
        """
        if vixy_bars is None or vixm_bars is None:
            return 0.0
        if len(vixy_bars) < SHORT_LOOKBACK or len(vixm_bars) < SHORT_LOOKBACK:
            return 0.0

        try:
            # Current ratio
            vixy_price = vixy_bars["close"].iloc[-1]
            vixm_price = vixm_bars["close"].iloc[-1]
            if vixm_price == 0:
                return 0.0

            ratio = vixy_price / vixm_price

            # 5-day average ratio for smoothing
            ratios = vixy_bars["close"].iloc[-SHORT_LOOKBACK:] / vixm_bars["close"].iloc[-SHORT_LOOKBACK:]
            avg_ratio = ratios.mean()

            # Contango (ratio < 1) = risk-on, backwardation (ratio > 1) = risk-off
            signal = -np.clip((avg_ratio - 0.95) * 10, -1, 1)
            return round(float(signal), 4)
        except Exception as e:
            logger.debug(f"VIX term structure computation failed: {e}")
            return 0.0

    @staticmethod
    def _classify_regime(composite: float) -> str:
        """Classify the regime based on composite score."""
        if composite > 0.3:
            return "RISK_ON"
        elif composite < -0.3:
            return "RISK_OFF"
        else:
            return "NEUTRAL"

    def get_state_summary(self) -> Dict[str, float]:
        """Return current state as a flat dictionary for logging/dashboard."""
        return {
            "bond_curve": self._state.bond_curve_slope,
            "credit_spread": self._state.credit_spread,
            "dollar_strength": self._state.dollar_strength,
            "gold_momentum": self._state.gold_momentum,
            "vix_term": self._state.vix_term_structure,
            "composite": self._state.composite,
            "regime": self._state.regime,
        }
