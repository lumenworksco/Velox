"""Drawdown Risk Manager (OPS-002).

Monitors portfolio equity curve for drawdown depth and duration.
Computes drawdown-specific risk metrics (Calmar Ratio, CDaR, Ulcer Index)
and provides a dynamic exposure multiplier that scales down risk as
drawdowns deepen.

Exposure scaling schedule:
    0% drawdown  -> 1.0x exposure (full)
    5% drawdown  -> 0.7x exposure (reduce)
    8% drawdown  -> 0.0x exposure (halt trading)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Exposure scaling breakpoints: (drawdown_pct, multiplier)
# Linearly interpolated between breakpoints
DEFAULT_EXPOSURE_SCHEDULE = [
    (0.00, 1.0),   # No drawdown: full exposure
    (0.03, 0.9),   # 3% drawdown: slight reduction
    (0.05, 0.7),   # 5% drawdown: significant reduction
    (0.065, 0.4),  # 6.5% drawdown: heavy reduction
    (0.08, 0.0),   # 8% drawdown: halt all new positions
]

# CDaR uses worst N% of drawdowns
CDAR_PERCENTILE = 10  # Worst 10% of drawdowns

# Recovery tracking
MAX_RECOVERY_DAYS_WARNING = 20   # Warn if recovery takes > 20 days


@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown risk metrics."""
    current_drawdown: float = 0.0       # Current drawdown from peak (0.05 = 5%)
    max_drawdown: float = 0.0           # Maximum historical drawdown
    drawdown_duration_days: int = 0     # Days in current drawdown
    max_duration_days: int = 0          # Longest drawdown duration ever
    calmar_ratio: float = 0.0          # Annual return / max drawdown
    cdar: float = 0.0                  # Conditional Drawdown at Risk (worst 10%)
    ulcer_index: float = 0.0           # Root-mean-square of drawdowns
    recovery_factor: float = 0.0       # Net profit / max drawdown
    peak_equity: float = 0.0           # All-time peak equity
    current_equity: float = 0.0        # Current equity value
    exposure_multiplier: float = 1.0   # Recommended position sizing multiplier
    num_drawdowns: int = 0             # Count of distinct drawdown episodes
    avg_drawdown: float = 0.0          # Average drawdown depth
    avg_recovery_days: float = 0.0     # Average days to recover from drawdowns


class DrawdownRiskManager:
    """Drawdown risk monitoring and dynamic exposure management.

    Tracks the equity curve to compute drawdown metrics and provides a
    real-time exposure multiplier that automatically scales down position
    sizes as drawdowns deepen, protecting against ruin risk.

    Usage:
        mgr = DrawdownRiskManager()
        metrics = mgr.compute_metrics(equity_curve)
        multiplier = mgr.get_exposure_multiplier(current_dd=0.04)
    """

    def __init__(
        self,
        exposure_schedule: Optional[List[Tuple[float, float]]] = None,
        annualization_factor: float = 252.0,
    ):
        self.exposure_schedule = exposure_schedule or list(DEFAULT_EXPOSURE_SCHEDULE)
        self.annualization_factor = annualization_factor
        self._peak_equity: float = 0.0
        self._drawdown_start: Optional[datetime] = None
        self._all_drawdowns: List[float] = []
        self._all_durations: List[int] = []
        self._daily_index = 0

    def compute_metrics(
        self, equity_curve: np.ndarray
    ) -> DrawdownMetrics:
        """Compute comprehensive drawdown metrics from an equity curve.

        Args:
            equity_curve: 1-D array of portfolio equity values (chronological).

        Returns:
            DrawdownMetrics with all computed risk measures.
        """
        if len(equity_curve) < 2:
            return DrawdownMetrics(
                current_equity=equity_curve[-1] if len(equity_curve) else 0.0,
            )

        equity = np.asarray(equity_curve, dtype=float)

        # Running maximum (peak equity at each point)
        running_max = np.maximum.accumulate(equity)

        # Drawdown series (fractional, positive = in drawdown)
        drawdown_series = 1.0 - equity / np.where(running_max > 0, running_max, 1.0)
        drawdown_series = np.clip(drawdown_series, 0, 1)

        # Current state
        current_equity = float(equity[-1])
        peak_equity = float(running_max[-1])
        current_dd = float(drawdown_series[-1])
        max_dd = float(np.max(drawdown_series))

        # Drawdown duration
        in_drawdown = drawdown_series > 0.001
        current_duration = 0
        if in_drawdown[-1]:
            for i in range(len(in_drawdown) - 1, -1, -1):
                if in_drawdown[i]:
                    current_duration += 1
                else:
                    break

        # All drawdown episodes
        episodes = self._identify_drawdown_episodes(drawdown_series)
        episode_depths = [ep["depth"] for ep in episodes]
        episode_durations = [ep["duration"] for ep in episodes]

        max_duration = max(episode_durations) if episode_durations else 0
        avg_dd = float(np.mean(episode_depths)) if episode_depths else 0.0
        avg_recovery = float(np.mean(episode_durations)) if episode_durations else 0.0

        # Calmar Ratio: annualized return / max drawdown
        total_return = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        n_periods = len(equity)
        annual_return = (1 + total_return) ** (self.annualization_factor / max(n_periods, 1)) - 1
        calmar = annual_return / max_dd if max_dd > 0.001 else 0.0

        # CDaR: Conditional Drawdown at Risk (expected shortfall of drawdowns)
        cdar = self._compute_cdar(drawdown_series, CDAR_PERCENTILE)

        # Ulcer Index: RMS of drawdown percentage
        ulcer = float(np.sqrt(np.mean(drawdown_series ** 2)))

        # Recovery factor: net profit / max drawdown
        net_profit = equity[-1] - equity[0]
        recovery_factor = net_profit / (max_dd * equity[0]) if max_dd > 0 and equity[0] > 0 else 0.0

        # Exposure multiplier
        exposure_mult = self.get_exposure_multiplier(current_dd)

        metrics = DrawdownMetrics(
            current_drawdown=round(current_dd, 6),
            max_drawdown=round(max_dd, 6),
            drawdown_duration_days=current_duration,
            max_duration_days=max_duration,
            calmar_ratio=round(calmar, 4),
            cdar=round(cdar, 6),
            ulcer_index=round(ulcer, 6),
            recovery_factor=round(recovery_factor, 4),
            peak_equity=round(peak_equity, 2),
            current_equity=round(current_equity, 2),
            exposure_multiplier=round(exposure_mult, 4),
            num_drawdowns=len(episodes),
            avg_drawdown=round(avg_dd, 6),
            avg_recovery_days=round(avg_recovery, 1),
        )

        # Log warnings
        if current_dd > 0.05:
            logger.warning(
                f"Drawdown alert: {current_dd:.1%} drawdown, "
                f"duration={current_duration}d, exposure={exposure_mult:.0%}"
            )
        if current_duration > MAX_RECOVERY_DAYS_WARNING:
            logger.warning(
                f"Extended drawdown: {current_duration} days without recovery"
            )

        return metrics

    def get_exposure_multiplier(self, current_dd: float) -> float:
        """Get the position sizing multiplier for the current drawdown level.

        Linearly interpolates between the exposure schedule breakpoints.

        Args:
            current_dd: Current drawdown as a fraction (0.05 = 5% drawdown).

        Returns:
            Multiplier from 0.0 (halt) to 1.0 (full exposure).
        """
        if current_dd <= 0:
            return 1.0

        schedule = sorted(self.exposure_schedule, key=lambda x: x[0])

        # Below first breakpoint
        if current_dd <= schedule[0][0]:
            return schedule[0][1]

        # Above last breakpoint
        if current_dd >= schedule[-1][0]:
            return schedule[-1][1]

        # Linear interpolation between breakpoints
        for i in range(len(schedule) - 1):
            dd_low, mult_low = schedule[i]
            dd_high, mult_high = schedule[i + 1]

            if dd_low <= current_dd <= dd_high:
                if dd_high == dd_low:
                    return mult_low
                fraction = (current_dd - dd_low) / (dd_high - dd_low)
                return round(mult_low + fraction * (mult_high - mult_low), 4)

        return 0.0

    @staticmethod
    def _identify_drawdown_episodes(
        drawdown_series: np.ndarray, threshold: float = 0.001
    ) -> List[Dict]:
        """Identify distinct drawdown episodes from the drawdown series.

        An episode starts when drawdown exceeds the threshold and ends
        when equity recovers to a new high.
        """
        episodes = []
        in_episode = False
        start_idx = 0
        max_depth = 0.0

        for i, dd in enumerate(drawdown_series):
            if dd > threshold and not in_episode:
                in_episode = True
                start_idx = i
                max_depth = dd
            elif dd > threshold and in_episode:
                max_depth = max(max_depth, dd)
            elif dd <= threshold and in_episode:
                episodes.append({
                    "start": start_idx,
                    "end": i,
                    "depth": float(max_depth),
                    "duration": i - start_idx,
                })
                in_episode = False
                max_depth = 0.0

        # Handle ongoing drawdown
        if in_episode:
            episodes.append({
                "start": start_idx,
                "end": len(drawdown_series) - 1,
                "depth": float(max_depth),
                "duration": len(drawdown_series) - 1 - start_idx,
            })

        return episodes

    @staticmethod
    def _compute_cdar(
        drawdown_series: np.ndarray, percentile: int = 10
    ) -> float:
        """Compute Conditional Drawdown at Risk (CDaR).

        CDaR is the expected drawdown in the worst N% of observations.
        Analogous to CVaR but for drawdowns instead of returns.
        """
        if len(drawdown_series) == 0:
            return 0.0

        # Sort drawdowns in descending order
        sorted_dd = np.sort(drawdown_series)[::-1]

        # Take the worst percentile
        n_worst = max(1, int(len(sorted_dd) * percentile / 100))
        worst_drawdowns = sorted_dd[:n_worst]

        return float(np.mean(worst_drawdowns))

    def get_metrics_summary(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """Return metrics as a flat dictionary for logging/dashboard."""
        m = self.compute_metrics(equity_curve)
        return {
            "current_dd": m.current_drawdown,
            "max_dd": m.max_drawdown,
            "dd_duration": m.drawdown_duration_days,
            "calmar": m.calmar_ratio,
            "cdar": m.cdar,
            "ulcer_index": m.ulcer_index,
            "recovery_factor": m.recovery_factor,
            "exposure_mult": m.exposure_multiplier,
            "num_episodes": m.num_drawdowns,
        }
