"""Shared utilities — re-exports from submodules for convenient imports.

Usage::

    from utils import safe_divide, now_et, ensure_et
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from utils.timezone import ET, now_et, ensure_et, to_iso_et
from utils.math_utils import safe_divide, clamp


def generate_idempotency_key(
    symbol: str,
    side: str,
    qty: int | float,
    strategy: str,
) -> str:
    """Generate a deterministic idempotency key with 5-second bucketing.

    The key is derived from the inputs plus a time bucket so that identical
    orders submitted within the same 5-second window get the same key,
    preventing duplicate submissions.

    Returns a hex string suitable for use as a broker idempotency token.
    """
    now = now_et()
    bucket = int(now.timestamp()) // 5  # 5-second buckets
    raw = f"{symbol}|{side}|{qty}|{strategy}|{bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


__all__ = [
    # timezone
    "ET",
    "now_et",
    "ensure_et",
    "to_iso_et",
    # math
    "safe_divide",
    "clamp",
    # idempotency
    "generate_idempotency_key",
]
