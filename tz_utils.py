"""Timezone utility functions — ensures consistent ET-aware datetimes throughout the codebase.

BUG-009: Centralizes timezone handling to prevent naive/aware datetime mismatches.
CRIT-003: Fallback import if config is unavailable.
"""

from datetime import datetime

try:
    import config
    _ET = config.ET
except Exception:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """Return the current time as a timezone-aware datetime in US/Eastern."""
    return datetime.now(_ET)


def ensure_et(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in US/Eastern.

    - If naive (no tzinfo), assumes it was intended as ET and attaches the timezone.
    - If already aware, converts to ET.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ET)
    return dt.astimezone(_ET)
