"""Enhanced Seasonality Model (ALPHA-002).

Provides intraday seasonality scoring by 15-minute bucket, day-of-week,
month, and VIX regime. Also tracks calendar effects (FOMC, options
expiry, index rebalance, tax-loss selling season).

All scores are normalized to [-1, +1] where:
    +1 = historically strong bullish tendency
    -1 = historically strong bearish tendency
     0 = no significant seasonal pattern
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Intraday seasonality patterns (empirical, based on SPY/QQQ)
# ============================================================

# 15-minute bucket scores: {(hour, minute): score}
# Positive = bullish tendency, Negative = bearish
INTRADAY_SEASONALITY = {
    # Pre-open ramp (9:30-10:00): typically bullish momentum
    (9, 30): 0.3,
    (9, 45): 0.2,
    # Morning reversal zone (10:00-10:30): mean-reversion
    (10, 0): -0.1,
    (10, 15): -0.15,
    (10, 30): -0.1,
    # Mid-morning (10:30-11:30): trend continuation
    (10, 45): 0.05,
    (11, 0): 0.05,
    (11, 15): 0.0,
    # Lunch lull (11:30-13:00): low vol, choppy
    (11, 30): -0.1,
    (11, 45): -0.15,
    (12, 0): -0.2,
    (12, 15): -0.15,
    (12, 30): -0.1,
    (12, 45): -0.05,
    # Afternoon pickup (13:00-14:30): volume returns
    (13, 0): 0.05,
    (13, 15): 0.1,
    (13, 30): 0.1,
    (13, 45): 0.1,
    (14, 0): 0.1,
    (14, 15): 0.15,
    # Power hour (14:30-15:45): strong directional moves
    (14, 30): 0.2,
    (14, 45): 0.2,
    (15, 0): 0.25,
    (15, 15): 0.3,
    (15, 30): 0.35,
    # Close auction (15:45-16:00): institutional rebalancing
    (15, 45): 0.15,
}

# Day-of-week effects
# Monday: slight bearish (weekend gap risk), Friday: slight bullish (short covering)
DAY_OF_WEEK_SCORES = {
    0: -0.1,   # Monday
    1: 0.05,   # Tuesday (turn-around Tuesday)
    2: 0.0,    # Wednesday
    3: 0.05,   # Thursday
    4: 0.1,    # Friday (short covering)
}

# Month-of-year effects (January effect, sell-in-May, Santa rally)
MONTH_SCORES = {
    1: 0.3,    # January effect
    2: 0.05,
    3: 0.1,
    4: 0.15,   # Tax-refund inflows
    5: -0.1,   # Sell in May
    6: -0.05,
    7: 0.05,
    8: -0.1,   # Summer doldrums
    9: -0.2,   # September effect (worst month historically)
    10: -0.1,  # October volatility
    11: 0.2,   # Pre-holiday rally
    12: 0.25,  # Santa rally / window dressing
}

# VIX regime adjustments
VIX_REGIME_ADJUSTMENTS = {
    "low": 0.1,       # VIX < 15: low vol, slight bullish
    "normal": 0.0,    # VIX 15-25: neutral
    "elevated": -0.1, # VIX 25-35: caution
    "high": -0.3,     # VIX > 35: fear, strong bearish
}

# FOMC meeting dates for 2026 (update annually)
FOMC_DATES_2026 = {
    date(2026, 1, 28), date(2026, 1, 29),
    date(2026, 3, 17), date(2026, 3, 18),
    date(2026, 4, 28), date(2026, 4, 29),
    date(2026, 6, 16), date(2026, 6, 17),
    date(2026, 7, 28), date(2026, 7, 29),
    date(2026, 9, 15), date(2026, 9, 16),
    date(2026, 10, 27), date(2026, 10, 28),
    date(2026, 12, 15), date(2026, 12, 16),
}

# Options expiry: third Friday of each month
# Quad witching: third Friday of March, June, September, December


@dataclass
class CalendarEffect:
    """A calendar-based market effect."""
    name: str
    score: float          # -1 to +1
    description: str = ""
    active: bool = True


class EnhancedSeasonality:
    """Enhanced seasonality model for intraday and calendar effects.

    Combines multiple seasonal patterns into a single actionable score.
    The score is used to adjust signal confidence and position sizing.

    Usage:
        season = EnhancedSeasonality()
        score = season.get_seasonality_score(timestamp, vix_level=18.5)
        effects = season.get_calendar_effect(date.today())
    """

    def __init__(
        self,
        intraday_weight: float = 0.30,
        day_of_week_weight: float = 0.15,
        month_weight: float = 0.20,
        vix_regime_weight: float = 0.20,
        calendar_weight: float = 0.15,
    ):
        self.intraday_weight = intraday_weight
        self.day_of_week_weight = day_of_week_weight
        self.month_weight = month_weight
        self.vix_regime_weight = vix_regime_weight
        self.calendar_weight = calendar_weight

        # Adaptive learning: accumulate observed returns by bucket
        self._learned_scores: Dict[Tuple[int, int], List[float]] = {}
        self._learning_enabled = True

    def get_seasonality_score(
        self,
        timestamp: datetime,
        vix_level: Optional[float] = None,
    ) -> float:
        """Get composite seasonality score for a given timestamp.

        Args:
            timestamp: Current datetime (should be in ET).
            vix_level: Current VIX level (optional, uses neutral if None).

        Returns:
            Score from -1.0 (bearish seasonal tendency) to +1.0 (bullish).
        """
        components: Dict[str, float] = {}

        # 1. Intraday bucket
        components["intraday"] = self._get_intraday_score(timestamp)

        # 2. Day of week
        components["day_of_week"] = DAY_OF_WEEK_SCORES.get(
            timestamp.weekday(), 0.0
        )

        # 3. Month of year
        components["month"] = MONTH_SCORES.get(timestamp.month, 0.0)

        # 4. VIX regime
        components["vix_regime"] = self._get_vix_regime_score(vix_level)

        # 5. Calendar effects (sum of active effects)
        cal_effects = self.get_calendar_effect(timestamp.date())
        cal_score = sum(e["score"] for e in cal_effects.values()) if cal_effects else 0.0
        components["calendar"] = np.clip(cal_score, -1, 1)

        # Weighted composite
        composite = (
            self.intraday_weight * components["intraday"]
            + self.day_of_week_weight * components["day_of_week"]
            + self.month_weight * components["month"]
            + self.vix_regime_weight * components["vix_regime"]
            + self.calendar_weight * components["calendar"]
        )

        return round(float(np.clip(composite, -1, 1)), 4)

    def get_calendar_effect(self, check_date: date) -> Dict[str, float]:
        """Get active calendar effects for a given date.

        Args:
            check_date: The date to check.

        Returns:
            Dict mapping effect name -> {"score": float, "description": str}.
        """
        effects: Dict[str, Dict] = {}

        # FOMC effect
        fomc_effect = self._check_fomc(check_date)
        if fomc_effect is not None:
            effects["fomc"] = {
                "score": fomc_effect.score,
                "description": fomc_effect.description,
            }

        # Options expiry effect
        opex_effect = self._check_options_expiry(check_date)
        if opex_effect is not None:
            effects["options_expiry"] = {
                "score": opex_effect.score,
                "description": opex_effect.description,
            }

        # Index rebalance effect (end of quarter)
        rebal_effect = self._check_index_rebalance(check_date)
        if rebal_effect is not None:
            effects["index_rebalance"] = {
                "score": rebal_effect.score,
                "description": rebal_effect.description,
            }

        # Tax-loss selling season
        tax_effect = self._check_tax_loss_season(check_date)
        if tax_effect is not None:
            effects["tax_loss"] = {
                "score": tax_effect.score,
                "description": tax_effect.description,
            }

        return effects

    def record_observation(
        self, timestamp: datetime, return_pct: float
    ):
        """Record an observed return for adaptive learning.

        Over time, this adjusts intraday scores based on actual observed
        returns in each 15-minute bucket.

        Args:
            timestamp: When the return was observed.
            return_pct: The observed return (e.g., 0.002 for 0.2%).
        """
        if not self._learning_enabled:
            return

        bucket = self._get_bucket(timestamp)
        if bucket not in self._learned_scores:
            self._learned_scores[bucket] = []
        self._learned_scores[bucket].append(return_pct)

        # Cap history at 500 observations per bucket
        if len(self._learned_scores[bucket]) > 500:
            self._learned_scores[bucket] = self._learned_scores[bucket][-500:]

    def _get_intraday_score(self, timestamp: datetime) -> float:
        """Get the intraday seasonality score for the current 15-min bucket."""
        bucket = self._get_bucket(timestamp)
        base_score = INTRADAY_SEASONALITY.get(bucket, 0.0)

        # Blend with learned score if available
        if bucket in self._learned_scores and len(self._learned_scores[bucket]) >= 20:
            learned = np.mean(self._learned_scores[bucket])
            # Normalize learned return to [-1, 1] range (assume 0.5% max move per bucket)
            learned_normalized = np.clip(learned / 0.005, -1, 1)
            # 70% base, 30% learned
            return float(0.7 * base_score + 0.3 * learned_normalized)

        return base_score

    @staticmethod
    def _get_bucket(timestamp: datetime) -> Tuple[int, int]:
        """Get the 15-minute bucket for a timestamp."""
        minute_bucket = (timestamp.minute // 15) * 15
        return (timestamp.hour, minute_bucket)

    @staticmethod
    def _get_vix_regime_score(vix_level: Optional[float]) -> float:
        """Get VIX regime score."""
        if vix_level is None:
            return 0.0

        if vix_level < 15:
            return VIX_REGIME_ADJUSTMENTS["low"]
        elif vix_level < 25:
            return VIX_REGIME_ADJUSTMENTS["normal"]
        elif vix_level < 35:
            return VIX_REGIME_ADJUSTMENTS["elevated"]
        else:
            return VIX_REGIME_ADJUSTMENTS["high"]

    def _check_fomc(self, check_date: date) -> Optional[CalendarEffect]:
        """Check for FOMC meeting proximity effect.

        Pre-FOMC drift: equities tend to rise in the 24h before announcement.
        Post-FOMC: elevated volatility.
        """
        # Check if today is the day before FOMC
        next_day = check_date + timedelta(days=1)
        if next_day in FOMC_DATES_2026:
            return CalendarEffect(
                name="pre_fomc_drift",
                score=0.15,
                description="Pre-FOMC drift: historically bullish bias 24h before decision",
            )

        # On FOMC day itself: elevated vol, unpredictable
        if check_date in FOMC_DATES_2026:
            return CalendarEffect(
                name="fomc_day",
                score=-0.1,
                description="FOMC decision day: elevated volatility, reduce sizing",
            )

        # Day after FOMC: follow-through or reversal
        prev_day = check_date - timedelta(days=1)
        if prev_day in FOMC_DATES_2026:
            return CalendarEffect(
                name="post_fomc",
                score=-0.05,
                description="Post-FOMC: potential reversal of announcement move",
            )

        return None

    @staticmethod
    def _check_options_expiry(check_date: date) -> Optional[CalendarEffect]:
        """Check for monthly/quarterly options expiry (OpEx).

        Third Friday of the month = monthly OpEx.
        March, June, September, December third Friday = quad witching.
        """
        # Find third Friday of the month
        first_day = check_date.replace(day=1)
        # Find first Friday
        days_to_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)

        if check_date == third_friday:
            is_quad = check_date.month in (3, 6, 9, 12)
            if is_quad:
                return CalendarEffect(
                    name="quad_witching",
                    score=-0.15,
                    description="Quad witching: elevated vol, pinning, gamma exposure unwind",
                )
            else:
                return CalendarEffect(
                    name="monthly_opex",
                    score=-0.1,
                    description="Monthly OpEx: elevated vol from delta hedging/pin risk",
                )

        # Day before OpEx: gamma exposure tends to pin strikes
        if check_date == third_friday - timedelta(days=1):
            return CalendarEffect(
                name="pre_opex",
                score=-0.05,
                description="Pre-OpEx: gamma pinning may suppress directional moves",
            )

        return None

    @staticmethod
    def _check_index_rebalance(check_date: date) -> Optional[CalendarEffect]:
        """Check for end-of-quarter index rebalance effects.

        Large index funds rebalance on the last trading day of each quarter.
        This creates predictable volume patterns and potential price impact.
        """
        # Last 3 trading days of quarter
        quarter_end_months = {3, 6, 9, 12}
        if check_date.month not in quarter_end_months:
            return None

        # Check if we're in the last week of the quarter
        next_month = check_date.month + 1 if check_date.month < 12 else 1
        next_year = check_date.year if check_date.month < 12 else check_date.year + 1
        first_next = date(next_year, next_month, 1)
        last_day = first_next - timedelta(days=1)

        days_to_end = (last_day - check_date).days
        if 0 <= days_to_end <= 3:
            return CalendarEffect(
                name="quarter_rebalance",
                score=0.1,
                description="Quarter-end rebalance: institutional buying, window dressing",
            )

        return None

    @staticmethod
    def _check_tax_loss_season(check_date: date) -> Optional[CalendarEffect]:
        """Check for tax-loss selling season effects.

        Late October through December: institutional tax-loss harvesting
        creates selling pressure in YTD losers and buying in January.
        """
        if check_date.month == 10 and check_date.day >= 20:
            return CalendarEffect(
                name="tax_loss_early",
                score=-0.1,
                description="Early tax-loss selling season: pressure on YTD losers",
            )
        elif check_date.month == 11:
            return CalendarEffect(
                name="tax_loss_peak",
                score=-0.15,
                description="Peak tax-loss selling: heavy institutional harvesting",
            )
        elif check_date.month == 12 and check_date.day <= 20:
            return CalendarEffect(
                name="tax_loss_late",
                score=-0.1,
                description="Late tax-loss selling: tapering off, potential bounce setup",
            )

        return None

    def get_components(self, timestamp: datetime, vix_level: Optional[float] = None) -> Dict[str, float]:
        """Return all seasonality components for transparency/debugging."""
        cal_effects = self.get_calendar_effect(timestamp.date())
        cal_score = sum(e["score"] for e in cal_effects.values()) if cal_effects else 0.0

        return {
            "intraday": self._get_intraday_score(timestamp),
            "day_of_week": DAY_OF_WEEK_SCORES.get(timestamp.weekday(), 0.0),
            "month": MONTH_SCORES.get(timestamp.month, 0.0),
            "vix_regime": self._get_vix_regime_score(vix_level),
            "calendar": np.clip(cal_score, -1, 1),
            "composite": self.get_seasonality_score(timestamp, vix_level),
        }
