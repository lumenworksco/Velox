"""Shared dataclasses and types for all strategies."""

from dataclasses import dataclass


@dataclass
class Signal:
    symbol: str
    strategy: str          # "ORB", "VWAP", "MOMENTUM"
    side: str              # "buy" or "sell"
    entry_price: float
    take_profit: float
    stop_loss: float
    reason: str = ""
    hold_type: str = "day"  # "day" or "swing" (multi-day)
    pair_id: str = ""       # Links two legs of a pairs trade


@dataclass
class ORBRange:
    high: float
    low: float
    volume: float          # total volume during ORB period
    prev_close: float      # yesterday's close for gap check


@dataclass
class VWAPState:
    vwap: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    cumulative_volume: float = 0.0
    cumulative_vp: float = 0.0    # volume * price
    cumulative_vp2: float = 0.0   # volume * price^2
