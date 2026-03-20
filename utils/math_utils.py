"""Math utilities — safe arithmetic helpers used across the trading bot."""

from __future__ import annotations

import math


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* on zero / NaN / Inf.

    >>> safe_divide(10, 2)
    5.0
    >>> safe_divide(1, 0)
    0.0
    >>> safe_divide(1, 0, default=-1.0)
    -1.0
    """
    try:
        if denominator == 0:
            return default
        result = numerator / denominator
        if math.isnan(result) or math.isinf(result):
            return default
        return float(result)
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* between *min_val* and *max_val* (inclusive).

    >>> clamp(5, 0, 10)
    5
    >>> clamp(-3, 0, 10)
    0
    >>> clamp(15, 0, 10)
    10
    """
    return max(min_val, min(value, max_val))
