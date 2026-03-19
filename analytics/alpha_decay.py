"""V9: Alpha decay monitoring — detect when a strategy's edge is eroding.

Computes rolling Sharpe ratios over 30/60/90-day windows, estimates the decay
rate (slope of Sharpe trend), and projects a half-life for the alpha signal.
Produces a health status per strategy: healthy / warning / critical.
"""

import logging

import numpy as np
import pandas as pd

import config
from analytics.metrics import compute_sharpe

logger = logging.getLogger(__name__)

# Thresholds — pulled from config at call time so overrides work
_DEFAULT_CRITICAL_SHARPE = 0.3
_DEFAULT_WARNING_SHARPE = 0.5


class AlphaDecayMonitor:
    """Monitor alpha decay across strategies using rolling Sharpe analysis."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_decay(
        self,
        strategy: str,
        trade_history: pd.DataFrame,
    ) -> dict:
        """Compute alpha-decay metrics for a single strategy.

        Returns dict with keys:
          sharpe_30d, sharpe_60d, sharpe_90d   — rolling Sharpe values
          decay_rate   — slope of Sharpe trend (negative = decaying)
          half_life_days — estimated days until alpha halves (or None)
          status       — "healthy" | "warning" | "critical"
        """
        result = self._safe_defaults()

        if trade_history is None or trade_history.empty:
            return result

        # Filter to this strategy
        mask = trade_history["strategy"] == strategy
        strat_trades = trade_history.loc[mask].copy()

        if strat_trades.empty or "pnl_pct" not in strat_trades.columns:
            return result

        returns = strat_trades["pnl_pct"].dropna().values.astype(float)

        if len(returns) < 5:
            return result

        # Rolling Sharpe over windows (trade-count based, annualised as per-trade)
        s30 = self._window_sharpe(returns, 30)
        s60 = self._window_sharpe(returns, 60)
        s90 = self._window_sharpe(returns, 90)

        result["sharpe_30d"] = s30
        result["sharpe_60d"] = s60
        result["sharpe_90d"] = s90

        # Decay rate: linear regression slope over available Sharpe points
        sharpe_series = [v for v in [s90, s60, s30] if v is not None]
        if len(sharpe_series) >= 2:
            x = np.arange(len(sharpe_series), dtype=float)
            y = np.array(sharpe_series, dtype=float)
            if np.std(x) > 0:
                slope = float(np.polyfit(x, y, 1)[0])
                result["decay_rate"] = round(slope, 4)

                # Half-life: days until Sharpe halves (if decaying)
                if slope < -1e-6 and s30 is not None and s30 > 0:
                    # Rough estimate: how many periods until current Sharpe halves
                    half_life = abs(s30 / (2.0 * slope))
                    # Scale periods to approximate days (each window ~ 30 days apart)
                    result["half_life_days"] = round(half_life * 30.0, 1)

        # Determine status from the most recent (30d) Sharpe
        result["status"] = self._classify_status(s30)

        return result

    def get_strategy_health_report(
        self,
        trade_history: pd.DataFrame,
    ) -> dict[str, dict]:
        """Full health report for every strategy found in trade_history."""
        report: dict[str, dict] = {}

        if trade_history is None or trade_history.empty:
            return report

        if "strategy" not in trade_history.columns:
            return report

        strategies = trade_history["strategy"].unique()
        for strat in strategies:
            report[str(strat)] = self.compute_decay(str(strat), trade_history)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _window_sharpe(returns: np.ndarray, window: int) -> float | None:
        """Sharpe over the last *window* returns (per-trade basis)."""
        if len(returns) < min(window, 5):
            if len(returns) >= 5:
                # Use what we have — annualize by trade count, not 252
                return round(compute_sharpe(returns, periods_per_year=max(len(returns), 1)), 4)
            return None
        subset = returns[-window:]
        return round(compute_sharpe(subset, periods_per_year=max(len(subset), 1)), 4)

    @staticmethod
    def _classify_status(sharpe_30d: float | None) -> str:
        critical = getattr(config, "ALPHA_DECAY_CRITICAL_SHARPE", _DEFAULT_CRITICAL_SHARPE)
        warning = getattr(config, "ALPHA_DECAY_WARNING_SHARPE", _DEFAULT_WARNING_SHARPE)

        if sharpe_30d is None:
            return "healthy"  # Insufficient data, assume OK
        if sharpe_30d < critical:
            return "critical"
        if sharpe_30d < warning:
            return "warning"
        return "healthy"

    @staticmethod
    def _safe_defaults() -> dict:
        return {
            "sharpe_30d": None,
            "sharpe_60d": None,
            "sharpe_90d": None,
            "decay_rate": 0.0,
            "half_life_days": None,
            "status": "healthy",
        }
