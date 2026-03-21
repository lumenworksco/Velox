"""COMP-009: Calendar effects — FOMC, opex, reconstitution, month-end.

Tracks calendar events that systematically affect markets and returns
event flags with expected impact direction. Covers:

- **FOMC drift**: 2-3 day pre-announcement rally tendency
- **OpEx pinning**: options expiration day gamma effects
- **Index reconstitution**: S&P 500/Russell rebalance flows
- **Month/quarter-end rebalancing**: institutional window-dressing
- **Turn-of-month effect**: historically positive first few trading days
- **Holiday effects**: pre-holiday optimism bias

All methods are fail-open: if date computation fails, events are
flagged as inactive.

Usage::

    from analytics.calendar_effects import CalendarEffectsMonitor

    monitor = CalendarEffectsMonitor()
    events = monitor.get_active_events(datetime.now())
    # [CalendarEvent(name="fomc_drift", active=True, impact="bullish", ...)]
    bias = monitor.get_net_bias(datetime.now())
    # 0.15 (slightly bullish)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 2024-2026 FOMC meeting dates (announcement day, typically Wednesday)
# Updated annually from https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES_2024 = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
]
FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
]
FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]

ALL_FOMC_DATES = FOMC_DATES_2024 + FOMC_DATES_2025 + FOMC_DATES_2026

# FOMC drift window (days before announcement)
FOMC_DRIFT_DAYS_BEFORE = 3
FOMC_DRIFT_DAYS_AFTER = 1

# Monthly options expiration (3rd Friday of each month)
# OpEx pinning window
OPEX_DAYS_BEFORE = 1
OPEX_DAYS_AFTER = 0

# Quarterly OpEx (triple/quadruple witching) months
QUAD_WITCHING_MONTHS = {3, 6, 9, 12}

# S&P 500 reconstitution: typically 3rd Friday of June and December for Russell,
# changes effective after close on 3rd Friday of September for S&P quarterly rebalance
SP500_REBALANCE_MONTHS = {3, 6, 9, 12}  # Quarterly

# Russell reconstitution: last Friday of June
RUSSELL_RECONSTITUTION_MONTH = 6

# Month-end/quarter-end window (trading days before month end)
MONTH_END_WINDOW = 3
QUARTER_END_MONTHS = {3, 6, 9, 12}

# Turn-of-month effect (first N trading days of month)
TURN_OF_MONTH_DAYS = 4

# Impact weights for composite bias
EVENT_WEIGHTS = {
    "fomc_drift": 0.20,
    "fomc_announcement": 0.10,
    "opex_pinning": 0.10,
    "quad_witching": 0.15,
    "sp500_rebalance": 0.10,
    "russell_reconstitution": 0.10,
    "month_end_rebalance": 0.10,
    "quarter_end_rebalance": 0.15,
    "turn_of_month": 0.10,
    "pre_holiday": 0.05,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class EventImpact(str, Enum):
    """Expected directional impact of a calendar event."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"       # High vol expected, direction unclear
    PINNING = "pinning"         # Reduced movement (gamma pinning)


@dataclass
class CalendarEvent:
    """A calendar-driven market event."""
    name: str                           # Event identifier
    description: str
    active: bool                        # Whether event window is currently active
    impact: EventImpact
    strength: float                     # 0.0 to 1.0 expected effect size
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    days_until: int = 0                 # Days until event (0 = today)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        status = "ACTIVE" if self.active else f"in {self.days_until}d"
        return f"{self.name} ({status}): {self.impact.value}, strength={self.strength:.2f}"


@dataclass
class CalendarSnapshot:
    """Complete calendar state for a given date."""
    date: date
    active_events: List[CalendarEvent]
    upcoming_events: List[CalendarEvent]
    net_bias: float                     # -1.0 to +1.0
    volatility_forecast: float          # 0.0 (calm) to 1.0 (volatile)
    key_event: Optional[str] = None     # Most impactful active event


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(d) -> date:
    """Coerce to date object."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise ValueError(f"Cannot parse date from {type(d)}: {d}")


def _third_friday(year: int, month: int) -> date:
    """Compute the third Friday of a given month."""
    # First day of month
    first = date(year, month, 1)
    # Day of week (0=Monday, 4=Friday)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    # Third Friday = first Friday + 14 days
    return first_friday + timedelta(days=14)


def _last_friday(year: int, month: int) -> date:
    """Compute the last Friday of a given month."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    days_after_friday = (last_day.weekday() - 4) % 7
    return last_day - timedelta(days=days_after_friday)


def _last_business_day(year: int, month: int) -> date:
    """Compute the last business day of a given month."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    # Walk backwards to find a weekday
    while last_day.weekday() >= 5:  # Saturday=5, Sunday=6
        last_day -= timedelta(days=1)
    return last_day


def _first_business_day(year: int, month: int) -> date:
    """Compute the first business day of a given month."""
    first = date(year, month, 1)
    while first.weekday() >= 5:
        first += timedelta(days=1)
    return first


def _is_business_day(d: date) -> bool:
    """Check if a date is a weekday (approximate — ignores holidays)."""
    return d.weekday() < 5


def _business_days_between(d1: date, d2: date) -> int:
    """Count approximate business days between two dates."""
    if d1 > d2:
        d1, d2 = d2, d1
    count = 0
    current = d1
    while current <= d2:
        if _is_business_day(current):
            count += 1
        current += timedelta(days=1)
    return count


# US market holidays (approximate; does not cover all years perfectly)
US_HOLIDAYS_NAMES = {
    "New Year's Day", "MLK Day", "Presidents' Day", "Good Friday",
    "Memorial Day", "Juneteenth", "Independence Day", "Labor Day",
    "Thanksgiving", "Christmas",
}


def _is_pre_holiday(d: date) -> bool:
    """Check if tomorrow or day-after is a likely market holiday.

    Uses a simple heuristic: checks for major US holidays.
    """
    year = d.year
    # Check the next 2 calendar days for known holiday patterns
    holidays = set()
    # Fixed holidays
    holidays.add(date(year, 1, 1))    # New Year's
    holidays.add(date(year, 7, 4))    # Independence Day
    holidays.add(date(year, 12, 25))  # Christmas

    # Floating holidays (approximate)
    # MLK Day: 3rd Monday of January
    jan1 = date(year, 1, 1)
    first_monday = jan1 + timedelta(days=(7 - jan1.weekday()) % 7)
    holidays.add(first_monday + timedelta(days=14))

    # Presidents' Day: 3rd Monday of February
    feb1 = date(year, 2, 1)
    first_monday = feb1 + timedelta(days=(7 - feb1.weekday()) % 7)
    holidays.add(first_monday + timedelta(days=14))

    # Memorial Day: last Monday of May
    may31 = date(year, 5, 31)
    holidays.add(may31 - timedelta(days=(may31.weekday()) % 7))

    # Labor Day: 1st Monday of September
    sep1 = date(year, 9, 1)
    holidays.add(sep1 + timedelta(days=(7 - sep1.weekday()) % 7))

    # Thanksgiving: 4th Thursday of November
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    holidays.add(first_thu + timedelta(days=21))

    # Check next 2 days
    for offset in range(1, 3):
        check_date = d + timedelta(days=offset)
        if check_date in holidays:
            return True

    return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CalendarEffectsMonitor:
    """Monitor calendar-driven market effects.

    Tracks FOMC meetings, options expiration, index reconstitution,
    month/quarter-end rebalancing, and holiday effects. Returns
    event flags and expected directional impact.
    """

    def __init__(self):
        self._fomc_dates = set()
        for d in ALL_FOMC_DATES:
            try:
                self._fomc_dates.add(_parse_date(d))
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def get_active_events(self, now=None) -> List[CalendarEvent]:
        """Get all currently active calendar events.

        Args:
            now: Current datetime (default: datetime.now()).

        Returns:
            List of CalendarEvent objects that are currently active.
        """
        if now is None:
            now = datetime.now()
        today = _parse_date(now)

        events: List[CalendarEvent] = []

        try:
            events.extend(self._check_fomc(today))
        except Exception as exc:
            logger.debug("FOMC check failed: %s", exc)

        try:
            events.extend(self._check_opex(today))
        except Exception as exc:
            logger.debug("OpEx check failed: %s", exc)

        try:
            events.extend(self._check_reconstitution(today))
        except Exception as exc:
            logger.debug("Reconstitution check failed: %s", exc)

        try:
            events.extend(self._check_month_end(today))
        except Exception as exc:
            logger.debug("Month-end check failed: %s", exc)

        try:
            events.extend(self._check_turn_of_month(today))
        except Exception as exc:
            logger.debug("Turn-of-month check failed: %s", exc)

        try:
            events.extend(self._check_pre_holiday(today))
        except Exception as exc:
            logger.debug("Pre-holiday check failed: %s", exc)

        return events

    def get_upcoming_events(self, now=None, horizon_days: int = 10) -> List[CalendarEvent]:
        """Get events coming up within the horizon.

        Returns events that are not yet active but will be soon.
        """
        if now is None:
            now = datetime.now()
        today = _parse_date(now)

        upcoming: List[CalendarEvent] = []
        for offset in range(1, horizon_days + 1):
            future = today + timedelta(days=offset)
            if not _is_business_day(future):
                continue
            events = self.get_active_events(future)
            for evt in events:
                if evt.active:
                    evt.days_until = offset
                    evt.active = False  # Not yet active
                    upcoming.append(evt)

        # Deduplicate by event name (keep earliest)
        seen = set()
        unique: List[CalendarEvent] = []
        for evt in upcoming:
            if evt.name not in seen:
                seen.add(evt.name)
                unique.append(evt)

        return unique

    # ------------------------------------------------------------------
    # Individual event checks
    # ------------------------------------------------------------------

    def _check_fomc(self, today: date) -> List[CalendarEvent]:
        """Check FOMC drift and announcement day effects."""
        events = []

        for fomc_date in sorted(self._fomc_dates):
            days_until = (fomc_date - today).days

            # FOMC drift: 1-3 days before announcement
            if 0 < days_until <= FOMC_DRIFT_DAYS_BEFORE:
                events.append(CalendarEvent(
                    name="fomc_drift",
                    description=f"Pre-FOMC drift ({days_until}d before announcement)",
                    active=True,
                    impact=EventImpact.BULLISH,
                    strength=0.6 - (days_until - 1) * 0.15,  # Stronger closer to event
                    start_date=fomc_date - timedelta(days=FOMC_DRIFT_DAYS_BEFORE),
                    end_date=fomc_date,
                    days_until=days_until,
                    metadata={"fomc_date": fomc_date.isoformat()},
                ))
                break  # Only report nearest FOMC

            # FOMC announcement day
            if days_until == 0:
                events.append(CalendarEvent(
                    name="fomc_announcement",
                    description="FOMC announcement day — expect volatility at 2pm ET",
                    active=True,
                    impact=EventImpact.VOLATILE,
                    strength=0.8,
                    start_date=fomc_date,
                    end_date=fomc_date,
                    days_until=0,
                    metadata={"fomc_date": fomc_date.isoformat()},
                ))
                break

            # Day after FOMC
            if days_until == -1:
                events.append(CalendarEvent(
                    name="fomc_follow_through",
                    description="Day after FOMC — follow-through or reversal",
                    active=True,
                    impact=EventImpact.VOLATILE,
                    strength=0.4,
                    start_date=fomc_date + timedelta(days=1),
                    end_date=fomc_date + timedelta(days=1),
                    days_until=0,
                ))
                break

        return events

    def _check_opex(self, today: date) -> List[CalendarEvent]:
        """Check monthly/quarterly options expiration effects."""
        events = []
        opex_date = _third_friday(today.year, today.month)
        days_until = (opex_date - today).days

        if 0 <= days_until <= OPEX_DAYS_BEFORE:
            is_quad = today.month in QUAD_WITCHING_MONTHS

            if is_quad:
                events.append(CalendarEvent(
                    name="quad_witching",
                    description="Quarterly options expiration (quad witching) — elevated volume and pinning",
                    active=True,
                    impact=EventImpact.PINNING,
                    strength=0.7,
                    start_date=opex_date - timedelta(days=OPEX_DAYS_BEFORE),
                    end_date=opex_date,
                    days_until=days_until,
                    metadata={"opex_date": opex_date.isoformat(), "quad_witching": True},
                ))
            else:
                events.append(CalendarEvent(
                    name="opex_pinning",
                    description="Monthly options expiration — gamma pinning near strikes",
                    active=True,
                    impact=EventImpact.PINNING,
                    strength=0.4,
                    start_date=opex_date - timedelta(days=OPEX_DAYS_BEFORE),
                    end_date=opex_date,
                    days_until=days_until,
                    metadata={"opex_date": opex_date.isoformat(), "quad_witching": False},
                ))

        return events

    def _check_reconstitution(self, today: date) -> List[CalendarEvent]:
        """Check index reconstitution/rebalance events."""
        events = []

        # S&P 500 quarterly rebalance: effective after close on 3rd Friday of Mar/Jun/Sep/Dec
        if today.month in SP500_REBALANCE_MONTHS:
            rebalance_date = _third_friday(today.year, today.month)
            days_until = (rebalance_date - today).days
            if 0 <= days_until <= 5:
                events.append(CalendarEvent(
                    name="sp500_rebalance",
                    description=f"S&P 500 quarterly rebalance ({days_until}d away)",
                    active=days_until <= 2,
                    impact=EventImpact.VOLATILE,
                    strength=0.5,
                    start_date=rebalance_date - timedelta(days=5),
                    end_date=rebalance_date,
                    days_until=days_until,
                ))

        # Russell reconstitution: last Friday of June
        if today.month == RUSSELL_RECONSTITUTION_MONTH:
            russell_date = _last_friday(today.year, 6)
            days_until = (russell_date - today).days
            if 0 <= days_until <= 7:
                events.append(CalendarEvent(
                    name="russell_reconstitution",
                    description=f"Russell reconstitution ({days_until}d away) — heavy small-cap flows",
                    active=days_until <= 3,
                    impact=EventImpact.VOLATILE,
                    strength=0.6,
                    start_date=russell_date - timedelta(days=7),
                    end_date=russell_date,
                    days_until=days_until,
                    metadata={"affects": "small_cap"},
                ))

        return events

    def _check_month_end(self, today: date) -> List[CalendarEvent]:
        """Check month-end and quarter-end rebalancing effects."""
        events = []
        last_bd = _last_business_day(today.year, today.month)
        bdays_to_end = _business_days_between(today, last_bd)

        if bdays_to_end <= MONTH_END_WINDOW:
            is_quarter_end = today.month in QUARTER_END_MONTHS

            if is_quarter_end:
                events.append(CalendarEvent(
                    name="quarter_end_rebalance",
                    description=f"Quarter-end rebalancing ({bdays_to_end} business days to end)",
                    active=True,
                    impact=EventImpact.VOLATILE,
                    strength=0.5,
                    start_date=last_bd - timedelta(days=MONTH_END_WINDOW + 2),
                    end_date=last_bd,
                    days_until=bdays_to_end,
                    metadata={"quarter": (today.month - 1) // 3 + 1},
                ))
            else:
                events.append(CalendarEvent(
                    name="month_end_rebalance",
                    description=f"Month-end rebalancing ({bdays_to_end} business days to end)",
                    active=True,
                    impact=EventImpact.NEUTRAL,
                    strength=0.3,
                    start_date=last_bd - timedelta(days=MONTH_END_WINDOW + 2),
                    end_date=last_bd,
                    days_until=bdays_to_end,
                ))

        return events

    def _check_turn_of_month(self, today: date) -> List[CalendarEvent]:
        """Check turn-of-month effect (first few trading days tend bullish)."""
        events = []
        first_bd = _first_business_day(today.year, today.month)
        bdays_from_start = _business_days_between(first_bd, today)

        if bdays_from_start <= TURN_OF_MONTH_DAYS:
            events.append(CalendarEvent(
                name="turn_of_month",
                description=f"Turn-of-month effect (day {bdays_from_start} of month)",
                active=True,
                impact=EventImpact.BULLISH,
                strength=0.3 - (bdays_from_start - 1) * 0.05,
                start_date=first_bd,
                end_date=first_bd + timedelta(days=TURN_OF_MONTH_DAYS + 2),
                days_until=0,
            ))

        return events

    def _check_pre_holiday(self, today: date) -> List[CalendarEvent]:
        """Check pre-holiday bullish bias."""
        events = []
        if _is_pre_holiday(today):
            events.append(CalendarEvent(
                name="pre_holiday",
                description="Pre-holiday session — historically bullish bias",
                active=True,
                impact=EventImpact.BULLISH,
                strength=0.25,
                start_date=today,
                end_date=today,
                days_until=0,
            ))
        return events

    # ------------------------------------------------------------------
    # Composite bias
    # ------------------------------------------------------------------

    def get_net_bias(self, now=None) -> float:
        """Compute net directional bias from all active calendar events.

        Returns a float from -1.0 (strong bearish) to +1.0 (strong bullish).
        Pinning and volatile events contribute 0 directional bias but
        may dampen the overall signal.
        """
        events = self.get_active_events(now)
        if not events:
            return 0.0

        total_bias = 0.0
        dampening = 1.0

        for event in events:
            weight = EVENT_WEIGHTS.get(event.name, 0.05)
            if event.impact == EventImpact.BULLISH:
                total_bias += weight * event.strength
            elif event.impact == EventImpact.BEARISH:
                total_bias -= weight * event.strength
            elif event.impact in (EventImpact.PINNING, EventImpact.VOLATILE):
                dampening *= (1.0 - event.strength * 0.3)

        return max(-1.0, min(1.0, total_bias * dampening))

    def get_volatility_forecast(self, now=None) -> float:
        """Estimate expected volatility from calendar events.

        Returns 0.0 (no calendar-driven vol) to 1.0 (high expected vol).
        """
        events = self.get_active_events(now)
        if not events:
            return 0.0

        vol_scores: List[float] = []
        for event in events:
            if event.impact in (EventImpact.VOLATILE, EventImpact.PINNING):
                vol_scores.append(event.strength)
            elif event.impact in (EventImpact.BULLISH, EventImpact.BEARISH):
                vol_scores.append(event.strength * 0.3)

        if not vol_scores:
            return 0.0

        # Use max rather than sum to avoid over-counting
        return min(max(vol_scores) * 1.2, 1.0)

    def get_snapshot(self, now=None) -> CalendarSnapshot:
        """Get a complete calendar state snapshot.

        Returns active events, upcoming events, net bias, and
        volatility forecast in a single call.
        """
        if now is None:
            now = datetime.now()
        today = _parse_date(now)

        active = self.get_active_events(now)
        upcoming = self.get_upcoming_events(now, horizon_days=7)

        # Identify the most impactful active event
        key_event = None
        if active:
            strongest = max(active, key=lambda e: e.strength)
            key_event = strongest.name

        return CalendarSnapshot(
            date=today,
            active_events=active,
            upcoming_events=upcoming,
            net_bias=self.get_net_bias(now),
            volatility_forecast=self.get_volatility_forecast(now),
            key_event=key_event,
        )
