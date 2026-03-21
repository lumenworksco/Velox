"""COMP-003: FINRA short interest and squeeze detection.

Tracks short interest ratio changes over time and detects potential short
squeezes based on:
- High short interest (SI > 20% of float)
- Increasing borrowing costs (proxy via days-to-cover)
- Price momentum against short sellers
- Volume surge above average

Data source: yfinance for basic short data, with stubs for FINRA/Ortex
premium data. All methods are fail-open.

Usage::

    tracker = ShortInterestTracker()
    tracker.update("GME")
    risk = tracker.assess_squeeze_risk("GME")
    # SqueezeRisk(symbol="GME", score=0.85, factors=["high_si", "momentum"])
"""

import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & thresholds
# ---------------------------------------------------------------------------

# Short interest thresholds (as fraction of float)
SI_LOW = 0.05            # < 5% = low
SI_MODERATE = 0.10       # 5-10% = moderate
SI_HIGH = 0.20           # 10-20% = high
SI_EXTREME = 0.30        # > 30% = extreme

# Days-to-cover thresholds
DTC_MODERATE = 3.0
DTC_HIGH = 5.0
DTC_EXTREME = 10.0

# Squeeze score component weights
WEIGHT_SI_LEVEL = 0.30
WEIGHT_DTC = 0.20
WEIGHT_MOMENTUM = 0.25
WEIGHT_VOLUME_SURGE = 0.15
WEIGHT_SI_TREND = 0.10

# Cache TTL: short interest updates bi-weekly from FINRA
CACHE_TTL = 3600  # 1 hour (data rarely changes)

# Minimum data points for trend analysis
MIN_HISTORY_POINTS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShortInterestData:
    """Short interest data for a single symbol at a point in time."""
    symbol: str
    date: str                           # ISO date
    short_interest: int = 0             # Number of shares short
    shares_float: int = 0               # Total float
    si_ratio: float = 0.0               # short_interest / shares_float
    days_to_cover: float = 0.0          # short_interest / avg_daily_volume
    avg_daily_volume: int = 0
    previous_si: int = 0                # Prior reporting period
    si_change_pct: float = 0.0          # Percent change from previous


@dataclass
class SqueezeRisk:
    """Squeeze risk assessment for a symbol."""
    symbol: str
    score: float                        # 0.0 (no risk) to 1.0 (extreme)
    level: str                          # "low", "moderate", "high", "extreme"
    factors: List[str] = field(default_factory=list)
    si_ratio: float = 0.0
    days_to_cover: float = 0.0
    momentum_5d: float = 0.0
    volume_ratio: float = 0.0           # current vol / avg vol
    si_trend: str = "stable"            # "increasing", "decreasing", "stable"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


# Neutral result returned on failure
_NEUTRAL_RISK = SqueezeRisk(
    symbol="",
    score=0.0,
    level="low",
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShortInterestTracker:
    """Track short interest and detect potential short squeezes.

    Uses yfinance for basic short data. For production use, integrate
    with FINRA short interest reports or Ortex/S3 Partners for real-time
    data by overriding _fetch_short_data().
    """

    def __init__(self):
        self._data: Dict[str, ShortInterestData] = {}           # symbol -> latest
        self._history: Dict[str, List[ShortInterestData]] = {}  # symbol -> history
        self._price_cache: Dict[str, Dict] = {}                 # symbol -> price info
        self._last_update: Dict[str, float] = {}                # symbol -> epoch

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def update(self, symbol: str, force: bool = False) -> Optional[ShortInterestData]:
        """Fetch latest short interest data for a symbol.

        Uses yfinance for basic data. Returns ShortInterestData or None.
        """
        elapsed = _time.time() - self._last_update.get(symbol, 0)
        if not force and elapsed < CACHE_TTL and symbol in self._data:
            return self._data[symbol]

        si_data = self._fetch_short_data(symbol)
        if si_data:
            self._data[symbol] = si_data
            history = self._history.setdefault(symbol, [])
            history.append(si_data)
            # Keep last 26 reports (~1 year of bi-weekly data)
            self._history[symbol] = history[-26:]
            self._last_update[symbol] = _time.time()

        return si_data

    def _fetch_short_data(self, symbol: str) -> Optional[ShortInterestData]:
        """Fetch short interest data via yfinance.

        yfinance provides shortRatio (days to cover) and sharesShort
        from the info dict. Falls back gracefully if unavailable.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — short interest tracking disabled")
            return None

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            shares_short = info.get("sharesShort", 0) or 0
            shares_float = info.get("floatShares", 0) or 0
            short_ratio = info.get("shortRatio", 0.0) or 0.0  # days to cover
            avg_volume = info.get("averageDailyVolume10Day", 0) or info.get("averageVolume", 0) or 0
            prev_short = info.get("sharesShortPriorMonth", 0) or 0

            if shares_float > 0:
                si_ratio = shares_short / shares_float
            else:
                si_ratio = 0.0

            if prev_short > 0:
                si_change = (shares_short - prev_short) / prev_short
            else:
                si_change = 0.0

            # Cache price data for momentum calculation
            try:
                hist = ticker.history(period="22d", interval="1d")
                if hist is not None and not hist.empty:
                    # Flatten MultiIndex columns if present
                    if hasattr(hist.columns, 'levels') and hist.columns.nlevels > 1:
                        hist.columns = hist.columns.get_level_values(0)
                    self._price_cache[symbol] = {
                        "hist": hist,
                        "current_price": float(hist["Close"].iloc[-1]),
                        "avg_volume": float(hist["Volume"].mean()) if "Volume" in hist else avg_volume,
                        "last_volume": float(hist["Volume"].iloc[-1]) if "Volume" in hist else 0,
                    }
            except Exception:
                pass  # Price data is supplementary, not critical

            return ShortInterestData(
                symbol=symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                short_interest=shares_short,
                shares_float=shares_float,
                si_ratio=si_ratio,
                days_to_cover=short_ratio,
                avg_daily_volume=avg_volume,
                previous_si=prev_short,
                si_change_pct=si_change,
            )

        except Exception as exc:
            logger.warning("Short interest fetch failed for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Price/volume helpers
    # ------------------------------------------------------------------

    def _compute_momentum(self, symbol: str, days: int = 5) -> float:
        """Compute price momentum (percent return) over N days."""
        cached = self._price_cache.get(symbol)
        if not cached or "hist" not in cached:
            return 0.0
        try:
            hist = cached["hist"]
            close = hist["Close"]
            if len(close) < days + 1:
                return 0.0
            return float((close.iloc[-1] - close.iloc[-1 - days]) / close.iloc[-1 - days])
        except Exception:
            return 0.0

    def _compute_volume_ratio(self, symbol: str) -> float:
        """Compute current volume / average volume ratio."""
        cached = self._price_cache.get(symbol)
        if not cached:
            return 1.0
        avg_vol = cached.get("avg_volume", 0)
        last_vol = cached.get("last_volume", 0)
        if avg_vol <= 0:
            return 1.0
        return last_vol / avg_vol

    def _compute_si_trend(self, symbol: str) -> str:
        """Determine short interest trend from history."""
        history = self._history.get(symbol, [])
        if len(history) < MIN_HISTORY_POINTS:
            return "stable"

        recent = history[-MIN_HISTORY_POINTS:]
        si_values = [h.si_ratio for h in recent]

        # Simple linear trend
        if all(si_values[i] <= si_values[i + 1] for i in range(len(si_values) - 1)):
            return "increasing"
        elif all(si_values[i] >= si_values[i + 1] for i in range(len(si_values) - 1)):
            return "decreasing"

        # Check net change
        net_change = si_values[-1] - si_values[0]
        if net_change > 0.02:  # 2% absolute increase
            return "increasing"
        elif net_change < -0.02:
            return "decreasing"

        return "stable"

    # ------------------------------------------------------------------
    # Squeeze risk assessment
    # ------------------------------------------------------------------

    def assess_squeeze_risk(self, symbol: str) -> SqueezeRisk:
        """Compute a composite squeeze risk score for a symbol.

        The score is a weighted combination of:
        - SI level (as % of float)
        - Days-to-cover
        - Price momentum (against shorts)
        - Volume surge
        - SI trend direction

        Returns SqueezeRisk with score 0.0-1.0.
        """
        si_data = self._data.get(symbol)
        if not si_data:
            si_data = self.update(symbol)
        if not si_data:
            return SqueezeRisk(symbol=symbol, score=0.0, level="low")

        factors: List[str] = []
        component_scores: Dict[str, float] = {}

        # 1. SI level score
        si_score = self._score_si_level(si_data.si_ratio)
        component_scores["si_level"] = si_score
        if si_data.si_ratio >= SI_HIGH:
            factors.append("high_si")
        if si_data.si_ratio >= SI_EXTREME:
            factors.append("extreme_si")

        # 2. Days-to-cover score
        dtc_score = self._score_days_to_cover(si_data.days_to_cover)
        component_scores["days_to_cover"] = dtc_score
        if si_data.days_to_cover >= DTC_HIGH:
            factors.append("high_dtc")

        # 3. Momentum score (positive momentum = worse for shorts)
        momentum = self._compute_momentum(symbol, days=5)
        mom_score = self._score_momentum(momentum)
        component_scores["momentum"] = mom_score
        if momentum > 0.05:
            factors.append("price_momentum")

        # 4. Volume surge score
        vol_ratio = self._compute_volume_ratio(symbol)
        vol_score = self._score_volume_surge(vol_ratio)
        component_scores["volume_surge"] = vol_score
        if vol_ratio > 2.0:
            factors.append("volume_surge")

        # 5. SI trend score
        si_trend = self._compute_si_trend(symbol)
        trend_score = {"increasing": 0.8, "stable": 0.3, "decreasing": 0.0}.get(si_trend, 0.3)
        component_scores["si_trend"] = trend_score
        if si_trend == "increasing":
            factors.append("si_increasing")

        # Weighted composite
        composite = (
            si_score * WEIGHT_SI_LEVEL
            + dtc_score * WEIGHT_DTC
            + mom_score * WEIGHT_MOMENTUM
            + vol_score * WEIGHT_VOLUME_SURGE
            + trend_score * WEIGHT_SI_TREND
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # Determine level
        if composite >= 0.75:
            level = "extreme"
        elif composite >= 0.50:
            level = "high"
        elif composite >= 0.25:
            level = "moderate"
        else:
            level = "low"

        return SqueezeRisk(
            symbol=symbol,
            score=composite,
            level=level,
            factors=factors,
            si_ratio=si_data.si_ratio,
            days_to_cover=si_data.days_to_cover,
            momentum_5d=momentum,
            volume_ratio=vol_ratio,
            si_trend=si_trend,
            details=component_scores,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_si_level(si_ratio: float) -> float:
        """Score SI ratio on 0-1 scale. Higher SI = higher score."""
        if si_ratio >= SI_EXTREME:
            return 1.0
        elif si_ratio >= SI_HIGH:
            return 0.7 + 0.3 * (si_ratio - SI_HIGH) / (SI_EXTREME - SI_HIGH)
        elif si_ratio >= SI_MODERATE:
            return 0.3 + 0.4 * (si_ratio - SI_MODERATE) / (SI_HIGH - SI_MODERATE)
        elif si_ratio >= SI_LOW:
            return 0.1 + 0.2 * (si_ratio - SI_LOW) / (SI_MODERATE - SI_LOW)
        else:
            return si_ratio / SI_LOW * 0.1

    @staticmethod
    def _score_days_to_cover(dtc: float) -> float:
        """Score days-to-cover on 0-1 scale."""
        if dtc >= DTC_EXTREME:
            return 1.0
        elif dtc >= DTC_HIGH:
            return 0.6 + 0.4 * (dtc - DTC_HIGH) / (DTC_EXTREME - DTC_HIGH)
        elif dtc >= DTC_MODERATE:
            return 0.2 + 0.4 * (dtc - DTC_MODERATE) / (DTC_HIGH - DTC_MODERATE)
        else:
            return dtc / DTC_MODERATE * 0.2

    @staticmethod
    def _score_momentum(momentum_5d: float) -> float:
        """Score positive momentum on 0-1 scale (higher = more squeeze pressure)."""
        if momentum_5d <= 0:
            return 0.0
        # 10% in 5 days = score of 0.5, 30% = 1.0
        return float(np.clip(momentum_5d / 0.30, 0.0, 1.0))

    @staticmethod
    def _score_volume_surge(vol_ratio: float) -> float:
        """Score volume surge on 0-1 scale."""
        if vol_ratio <= 1.0:
            return 0.0
        # 3x average = score 0.5, 10x = 1.0
        return float(np.clip((vol_ratio - 1.0) / 9.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Batch screening
    # ------------------------------------------------------------------

    def screen_for_squeezes(
        self,
        symbols: List[str],
        min_score: float = 0.4,
    ) -> List[SqueezeRisk]:
        """Screen symbols for potential short squeezes.

        Returns list of SqueezeRisk objects for symbols above min_score,
        sorted by score descending.
        """
        results: List[SqueezeRisk] = []
        for symbol in symbols:
            try:
                self.update(symbol)
                risk = self.assess_squeeze_risk(symbol)
                if risk.score >= min_score:
                    results.append(risk)
            except Exception as exc:
                logger.debug("Squeeze screening failed for %s: %s", symbol, exc)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_most_shorted(
        self,
        symbols: List[str],
        top_n: int = 10,
    ) -> List[ShortInterestData]:
        """Return the top-N most shorted symbols by SI ratio."""
        data_list: List[ShortInterestData] = []
        for symbol in symbols:
            try:
                si = self.update(symbol)
                if si and si.si_ratio > 0:
                    data_list.append(si)
            except Exception as exc:
                logger.debug("SI fetch failed for %s: %s", symbol, exc)

        data_list.sort(key=lambda d: d.si_ratio, reverse=True)
        return data_list[:top_n]

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data."""
        if symbol:
            self._data.pop(symbol, None)
            self._history.pop(symbol, None)
            self._price_cache.pop(symbol, None)
            self._last_update.pop(symbol, None)
        else:
            self._data.clear()
            self._history.clear()
            self._price_cache.clear()
            self._last_update.clear()
