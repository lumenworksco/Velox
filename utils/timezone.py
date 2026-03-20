"""Timezone utilities — canonical Eastern time helpers.

All trading logic should use these helpers instead of datetime.now()
to ensure consistent timezone-aware timestamps in US/Eastern.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """Return the current time as a timezone-aware datetime in US/Eastern.

    Always use this instead of ``datetime.now()`` in trading code.
    """
    return datetime.now(ET)


def ensure_et(dt: datetime) -> datetime:
    """Convert any datetime to US/Eastern.

    - Naive datetimes are assumed to already represent Eastern time and are
      tagged with the ET tzinfo (no conversion).
    - Aware datetimes in other timezones are converted to Eastern.
    - Aware datetimes already in ET are returned unchanged.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def to_iso_et(dt: datetime) -> str:
    """Return an ISO 8601 string in US/Eastern.

    Ensures the datetime is converted to Eastern before formatting.
    """
    return ensure_et(dt).isoformat()
