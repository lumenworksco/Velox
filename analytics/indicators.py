"""Shared technical indicator calculations used across strategies."""

import numpy as np
import pandas as pd


def compute_vwap(bars: pd.DataFrame) -> float | None:
    """Compute VWAP from intraday bars.

    Args:
        bars: DataFrame with 'high', 'low', 'close', 'volume' columns.

    Returns:
        VWAP value as float, or None if data is insufficient.
    """
    if bars.empty or "volume" not in bars.columns:
        return None
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vol = bars["volume"].cumsum()
    cum_vp = (typical * bars["volume"]).cumsum()
    if cum_vol.iloc[-1] == 0:
        return None
    return cum_vp.iloc[-1] / cum_vol.iloc[-1]


def compute_vwap_bands(
    bars: pd.DataFrame, std_mult: float = 2.0
) -> tuple[float, float, float] | None:
    """Compute VWAP with upper and lower bands.

    Args:
        bars: DataFrame with 'high', 'low', 'close', 'volume' columns.
        std_mult: Standard deviation multiplier for bands.

    Returns:
        Tuple of (vwap, upper_band, lower_band), or None if data is insufficient.
    """
    if bars.empty or "volume" not in bars.columns:
        return None

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vol = bars["volume"].cumsum()
    cum_vp = (typical_price * bars["volume"]).cumsum()

    if cum_vol.iloc[-1] == 0:
        return None

    vwap = cum_vp.iloc[-1] / cum_vol.iloc[-1]

    cum_vp2 = (typical_price**2 * bars["volume"]).cumsum()
    variance = cum_vp2.iloc[-1] / cum_vol.iloc[-1] - vwap**2
    std_dev = np.sqrt(max(variance, 0))

    upper = vwap + std_mult * std_dev
    lower = vwap - std_mult * std_dev

    return vwap, upper, lower
