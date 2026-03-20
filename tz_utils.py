"""Timezone utility functions — ensures consistent ET-aware datetimes throughout the codebase.

BUG-009: Centralizes timezone handling to prevent naive/aware datetime mismatches.
"""

from datetime import datetime

import config


def now_et() -> datetime:
    """Return the current time as a timezone-aware datetime in US/Eastern."""
    return datetime.now(config.ET)


def ensure_et(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware in US/Eastern.

    - If naive (no tzinfo), assumes it was intended as ET and attaches the timezone.
    - If already aware, converts to ET.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=config.ET)
    return dt.astimezone(config.ET)
