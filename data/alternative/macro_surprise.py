"""COMP-006: FRED API economic surprise index.

Fetches economic data from FRED (Federal Reserve Economic Data) using
their free API. Computes a surprise index (actual vs consensus proxied
by trend) and tracks CPI, NFP, GDP, ISM to create a macro regime
indicator.

FRED API key: set FRED_API_KEY environment variable (free from
https://fred.stlouisfed.org/docs/api/api_key.html). Falls back to
neutral if no key is set.

Fail-open: returns neutral regime if FRED is unreachable.

Usage::

    index = MacroSurpriseIndex(api_key="your_key")
    index.update()
    regime = index.get_macro_regime()
    # MacroRegime(regime="expansion", surprise_index=0.35, ...)
    cpi_surprise = index.get_indicator_surprise("CPI")
"""

import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Key economic indicators and their FRED series IDs
ECONOMIC_INDICATORS = {
    "CPI": {
        "series_id": "CPIAUCSL",         # CPI-U All Items
        "name": "Consumer Price Index",
        "frequency": "monthly",
        "higher_is": "inflationary",      # Direction interpretation
    },
    "NFP": {
        "series_id": "PAYEMS",            # Total Nonfarm Payrolls
        "name": "Nonfarm Payrolls",
        "frequency": "monthly",
        "higher_is": "expansionary",
    },
    "GDP": {
        "series_id": "GDP",               # Gross Domestic Product
        "name": "GDP",
        "frequency": "quarterly",
        "higher_is": "expansionary",
    },
    "ISM": {
        "series_id": "MANEMP",            # Manufacturing Employment (proxy)
        "name": "ISM Manufacturing",
        "frequency": "monthly",
        "higher_is": "expansionary",
    },
    "UNEMPLOYMENT": {
        "series_id": "UNRATE",            # Unemployment Rate
        "name": "Unemployment Rate",
        "frequency": "monthly",
        "higher_is": "contractionary",
    },
    "RETAIL_SALES": {
        "series_id": "RSXFS",             # Advance Retail Sales
        "name": "Retail Sales",
        "frequency": "monthly",
        "higher_is": "expansionary",
    },
    "HOUSING_STARTS": {
        "series_id": "HOUST",             # Housing Starts
        "name": "Housing Starts",
        "frequency": "monthly",
        "higher_is": "expansionary",
    },
    "FED_FUNDS": {
        "series_id": "FEDFUNDS",          # Federal Funds Rate
        "name": "Fed Funds Rate",
        "frequency": "monthly",
        "higher_is": "tightening",
    },
}

# Regime thresholds for composite surprise index
REGIME_EXPANSION = 0.3
REGIME_CONTRACTION = -0.3

# Trend window for computing "expected" values (12 periods)
TREND_WINDOW = 12

# Cache TTL (daily refresh is sufficient)
CACHE_TTL = 86400  # 24 hours

# Lookback for FRED data
LOOKBACK_YEARS = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IndicatorData:
    """Time series data for one economic indicator."""
    indicator_id: str                   # "CPI", "NFP", etc.
    series_id: str                      # FRED series ID
    name: str
    values: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    latest_value: float = 0.0
    latest_date: str = ""
    trend_value: float = 0.0           # Moving average (expected)
    surprise: float = 0.0              # Standardized surprise
    direction: str = ""                # "above_trend" or "below_trend"


@dataclass
class MacroRegime:
    """Composite macro regime assessment."""
    regime: str                        # "expansion", "contraction", "neutral"
    surprise_index: float              # -1.0 to +1.0
    confidence: float                  # 0.0 to 1.0
    indicator_surprises: Dict[str, float] = field(default_factory=dict)
    indicator_directions: Dict[str, str] = field(default_factory=dict)
    inflation_pressure: float = 0.0    # -1.0 (deflationary) to +1.0 (inflationary)
    growth_momentum: float = 0.0       # -1.0 (contracting) to +1.0 (expanding)
    labor_strength: float = 0.0        # -1.0 (weak) to +1.0 (strong)
    policy_stance: str = "neutral"     # "easing", "tightening", "neutral"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def equity_bias(self) -> float:
        """Map macro regime to equity positioning bias (-1 to +1).

        Expansion + low inflation = bullish for equities.
        Contraction + high inflation = bearish (stagflation).
        """
        growth_component = self.growth_momentum * 0.4
        inflation_penalty = -abs(self.inflation_pressure) * 0.2 if self.inflation_pressure > 0.5 else 0.0
        labor_component = self.labor_strength * 0.2
        policy_component = {"easing": 0.2, "tightening": -0.1, "neutral": 0.0}.get(self.policy_stance, 0.0)

        raw = growth_component + inflation_penalty + labor_component + policy_component
        return float(np.clip(raw, -1.0, 1.0))


# Neutral result returned on failure
_NEUTRAL_REGIME = MacroRegime(
    regime="neutral",
    surprise_index=0.0,
    confidence=0.0,
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MacroSurpriseIndex:
    """Compute economic surprise index from FRED data.

    The surprise index measures how actual economic data compares to
    its recent trend (proxy for consensus). Positive surprises indicate
    stronger-than-expected data.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("FRED_API_KEY", "")
        self._indicators: Dict[str, IndicatorData] = {}
        self._last_update: float = 0.0

    # ------------------------------------------------------------------
    # FRED API helper
    # ------------------------------------------------------------------

    def _fetch_series(
        self,
        series_id: str,
        lookback_years: int = LOOKBACK_YEARS,
    ) -> Optional[List[Tuple[str, float]]]:
        """Fetch a FRED series. Returns list of (date, value) or None."""
        if not self._api_key:
            logger.info("No FRED_API_KEY set — macro surprise index disabled")
            return None

        try:
            import urllib.request
            import json

            start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
            url = (
                f"{FRED_BASE_URL}?series_id={series_id}"
                f"&api_key={self._api_key}"
                f"&file_type=json"
                f"&observation_start={start_date}"
                f"&sort_order=asc"
            )

            req = urllib.request.Request(url, headers={"User-Agent": "VeloxBot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            observations = data.get("observations", [])
            results: List[Tuple[str, float]] = []
            for obs in observations:
                date = obs.get("date", "")
                value_str = obs.get("value", ".")
                if value_str == "." or not value_str:
                    continue
                try:
                    results.append((date, float(value_str)))
                except ValueError:
                    continue

            return results if results else None

        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return None

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------

    def update(self, force: bool = False) -> int:
        """Fetch latest data for all economic indicators.

        Returns the number of indicators successfully updated.
        """
        elapsed = _time.time() - self._last_update
        if not force and elapsed < CACHE_TTL and self._indicators:
            return len(self._indicators)

        updated = 0
        for indicator_id, meta in ECONOMIC_INDICATORS.items():
            try:
                observations = self._fetch_series(meta["series_id"])
                if not observations:
                    continue

                dates = [obs[0] for obs in observations]
                values = [obs[1] for obs in observations]

                # Compute trend (expected) using simple moving average
                if len(values) >= TREND_WINDOW:
                    trend = float(np.mean(values[-TREND_WINDOW:]))
                else:
                    trend = float(np.mean(values))

                # Compute standardized surprise
                latest = values[-1]
                std = float(np.std(values[-TREND_WINDOW:])) if len(values) >= TREND_WINDOW else 1.0
                if std < 1e-10:
                    std = 1.0
                surprise = (latest - trend) / std

                # Direction
                direction = "above_trend" if latest > trend else "below_trend"

                self._indicators[indicator_id] = IndicatorData(
                    indicator_id=indicator_id,
                    series_id=meta["series_id"],
                    name=meta["name"],
                    values=values,
                    dates=dates,
                    latest_value=latest,
                    latest_date=dates[-1] if dates else "",
                    trend_value=trend,
                    surprise=surprise,
                    direction=direction,
                )
                updated += 1

            except Exception as exc:
                logger.debug("Failed to update %s: %s", indicator_id, exc)

        self._last_update = _time.time()
        logger.info("Updated %d/%d economic indicators", updated, len(ECONOMIC_INDICATORS))
        return updated

    # ------------------------------------------------------------------
    # Indicator access
    # ------------------------------------------------------------------

    def get_indicator_surprise(self, indicator_id: str) -> Optional[float]:
        """Get the standardized surprise for a specific indicator.

        Returns None if the indicator hasn't been fetched.
        """
        ind = self._indicators.get(indicator_id)
        return ind.surprise if ind else None

    def get_indicator_data(self, indicator_id: str) -> Optional[IndicatorData]:
        """Get full data for a specific indicator."""
        return self._indicators.get(indicator_id)

    # ------------------------------------------------------------------
    # Composite regime
    # ------------------------------------------------------------------

    def get_macro_regime(self) -> MacroRegime:
        """Compute composite macro regime from all available indicators.

        Returns a MacroRegime with:
        - Overall regime classification (expansion/contraction/neutral)
        - Composite surprise index
        - Sub-components (inflation, growth, labor, policy)
        """
        if not self._indicators:
            self.update()
        if not self._indicators:
            return MacroRegime(regime="neutral", surprise_index=0.0, confidence=0.0)

        indicator_surprises: Dict[str, float] = {}
        indicator_directions: Dict[str, str] = {}

        for ind_id, ind in self._indicators.items():
            indicator_surprises[ind_id] = ind.surprise
            indicator_directions[ind_id] = ind.direction

        # Composite surprise index (weighted average)
        surprise_values = list(indicator_surprises.values())
        composite = float(np.mean(surprise_values)) if surprise_values else 0.0
        composite = float(np.clip(composite, -2.0, 2.0))

        # Normalize to -1 to +1
        normalized = float(np.tanh(composite / 2.0))

        # Sub-components
        inflation_pressure = self._compute_inflation_pressure()
        growth_momentum = self._compute_growth_momentum()
        labor_strength = self._compute_labor_strength()
        policy_stance = self._compute_policy_stance()

        # Regime classification
        if normalized > REGIME_EXPANSION:
            regime = "expansion"
        elif normalized < REGIME_CONTRACTION:
            regime = "contraction"
        else:
            regime = "neutral"

        # Confidence based on data availability
        confidence = len(self._indicators) / len(ECONOMIC_INDICATORS)

        return MacroRegime(
            regime=regime,
            surprise_index=normalized,
            confidence=confidence,
            indicator_surprises=indicator_surprises,
            indicator_directions=indicator_directions,
            inflation_pressure=inflation_pressure,
            growth_momentum=growth_momentum,
            labor_strength=labor_strength,
            policy_stance=policy_stance,
            timestamp=datetime.now(),
        )

    def _compute_inflation_pressure(self) -> float:
        """Compute inflation pressure from CPI data."""
        cpi = self._indicators.get("CPI")
        if not cpi or len(cpi.values) < 13:
            return 0.0

        # Year-over-year CPI change
        yoy_change = (cpi.values[-1] - cpi.values[-13]) / cpi.values[-13]

        # Normalize: 2% = neutral, >4% = high, <0% = deflationary
        if yoy_change > 0.04:
            return min((yoy_change - 0.02) / 0.04, 1.0)
        elif yoy_change < 0.0:
            return max(yoy_change / 0.02, -1.0)
        else:
            return (yoy_change - 0.02) / 0.02

    def _compute_growth_momentum(self) -> float:
        """Compute growth momentum from GDP and retail sales."""
        components: List[float] = []

        gdp = self._indicators.get("GDP")
        if gdp and len(gdp.values) >= 5:
            # Quarter-over-quarter GDP growth rate
            qoq = (gdp.values[-1] - gdp.values[-2]) / gdp.values[-2]
            components.append(float(np.clip(qoq * 10, -1.0, 1.0)))

        retail = self._indicators.get("RETAIL_SALES")
        if retail and len(retail.values) >= 3:
            mom = (retail.values[-1] - retail.values[-2]) / retail.values[-2]
            components.append(float(np.clip(mom * 20, -1.0, 1.0)))

        housing = self._indicators.get("HOUSING_STARTS")
        if housing and len(housing.values) >= 3:
            mom = (housing.values[-1] - housing.values[-2]) / housing.values[-2]
            components.append(float(np.clip(mom * 5, -1.0, 1.0)))

        if not components:
            return 0.0
        return float(np.clip(np.mean(components), -1.0, 1.0))

    def _compute_labor_strength(self) -> float:
        """Compute labor market strength from NFP and unemployment."""
        components: List[float] = []

        nfp = self._indicators.get("NFP")
        if nfp and len(nfp.values) >= 2:
            # Monthly change in thousands
            change = nfp.values[-1] - nfp.values[-2]
            # 200K+ = strong, 0 = neutral, negative = weak
            components.append(float(np.clip(change / 500, -1.0, 1.0)))

        unemp = self._indicators.get("UNEMPLOYMENT")
        if unemp and len(unemp.values) >= 2:
            # Lower unemployment = stronger labor market
            level = unemp.values[-1]
            # 3.5% = strong (+1), 5% = neutral (0), 7%+ = weak (-1)
            components.append(float(np.clip((5.0 - level) / 2.0, -1.0, 1.0)))

        if not components:
            return 0.0
        return float(np.clip(np.mean(components), -1.0, 1.0))

    def _compute_policy_stance(self) -> str:
        """Determine Fed policy stance from fed funds rate trend."""
        ff = self._indicators.get("FED_FUNDS")
        if not ff or len(ff.values) < 3:
            return "neutral"

        # 3-month trend
        recent = ff.values[-3:]
        if recent[-1] > recent[0] + 0.10:
            return "tightening"
        elif recent[-1] < recent[0] - 0.10:
            return "easing"
        return "neutral"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_all_surprises(self) -> Dict[str, float]:
        """Return all current indicator surprises as a dict."""
        return {
            ind_id: ind.surprise
            for ind_id, ind in self._indicators.items()
        }

    def get_data_freshness(self) -> Dict[str, str]:
        """Return latest data dates for all indicators."""
        return {
            ind_id: ind.latest_date
            for ind_id, ind in self._indicators.items()
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._indicators.clear()
        self._last_update = 0.0
