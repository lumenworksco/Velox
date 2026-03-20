"""Shared utility functions for the trading bot."""


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide numerator by denominator, returning *default* when denominator is
    zero or near-zero.

    This avoids ZeroDivisionError in hot paths like position sizing, range
    calculations, and Kelly criterion computations.

    Parameters
    ----------
    numerator : float
        The value to divide.
    denominator : float
        The divisor.  If ``abs(denominator) < 1e-12`` the *default* is returned.
    default : float, optional
        Value returned when division is unsafe.  Defaults to ``0.0``.
    """
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator
