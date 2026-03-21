"""Transaction cost and slippage models for realistic backtesting and pre-trade analysis.

Models:
- FixedSlippage: flat bps cost per trade
- VolumeSlippage: cost scales with order size relative to average volume
- MarketImpactSlippage: square-root market impact model (Almgren-Chriss)
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class SlippageCost:
    slippage_bps: float       # Estimated slippage in basis points
    commission_usd: float     # Broker commission in USD
    total_cost_usd: float     # Total execution cost
    adjusted_price: float     # Price after slippage adjustment


class SlippageModel(ABC):
    @abstractmethod
    def estimate(self, price: float, qty: int, side: str,
                 avg_daily_volume: float = 0, spread_bps: float = 0) -> SlippageCost:
        pass


class FixedSlippage(SlippageModel):
    """Fixed basis-point slippage model. Good for liquid large-caps."""

    def __init__(self, slippage_bps: float = 1.0, commission_per_share: float = 0.0):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share

    def estimate(self, price: float, qty: int, side: str,
                 avg_daily_volume: float = 0, spread_bps: float = 0) -> SlippageCost:
        slip = price * (self.slippage_bps / 10000)
        adjusted = price + slip if side == "buy" else price - slip
        commission = qty * self.commission_per_share
        total = abs(adjusted - price) * qty + commission
        return SlippageCost(self.slippage_bps, commission, round(total, 2), round(adjusted, 4))


class VolumeSlippage(SlippageModel):
    """Volume-dependent slippage. Cost increases as order is larger fraction of ADV."""

    def __init__(self, base_bps: float = 0.5, volume_impact_factor: float = 0.1,
                 commission_per_share: float = 0.0):
        self.base_bps = base_bps
        self.volume_impact = volume_impact_factor
        self.commission_per_share = commission_per_share

    def estimate(self, price: float, qty: int, side: str,
                 avg_daily_volume: float = 1_000_000, spread_bps: float = 0) -> SlippageCost:
        if avg_daily_volume <= 0:
            avg_daily_volume = 1_000_000
        participation = qty / avg_daily_volume
        slip_bps = self.base_bps + (self.volume_impact * participation * 10000)
        slip_bps = min(slip_bps, 50.0)  # Cap at 50 bps
        slip = price * (slip_bps / 10000)
        adjusted = price + slip if side == "buy" else price - slip
        commission = qty * self.commission_per_share
        total = abs(adjusted - price) * qty + commission
        return SlippageCost(round(slip_bps, 2), commission, round(total, 2), round(adjusted, 4))


class MarketImpactSlippage(SlippageModel):
    """Almgren-Chriss square-root market impact model.

    Impact ≈ sigma * sqrt(qty / ADV) * lambda
    Where sigma is daily volatility and lambda is a market-dependent constant.
    """

    def __init__(self, volatility: float = 0.02, impact_lambda: float = 0.1,
                 commission_per_share: float = 0.0):
        self.volatility = volatility
        self.impact_lambda = impact_lambda
        self.commission_per_share = commission_per_share

    def estimate(self, price: float, qty: int, side: str,
                 avg_daily_volume: float = 1_000_000, spread_bps: float = 0) -> SlippageCost:
        if avg_daily_volume <= 0:
            avg_daily_volume = 1_000_000
        participation = qty / avg_daily_volume
        impact = self.volatility * math.sqrt(participation) * self.impact_lambda
        slip_bps = impact * 10000
        # MED-013: Cap maximum slippage to configurable limit
        max_slippage_bps = getattr(config, 'MAX_SLIPPAGE_BPS', 50)
        slip_bps = min(slip_bps, max_slippage_bps)
        half_spread = spread_bps / 2 if spread_bps > 0 else 0.5
        total_slip_bps = slip_bps + half_spread
        slip = price * (total_slip_bps / 10000)
        adjusted = price + slip if side == "buy" else price - slip
        commission = qty * self.commission_per_share
        total = abs(adjusted - price) * qty + commission
        return SlippageCost(round(total_slip_bps, 2), commission, round(total, 2), round(adjusted, 4))


def get_default_model() -> SlippageModel:
    """Return the configured default slippage model."""
    model_name = getattr(config, 'SLIPPAGE_MODEL', 'volume')
    if model_name == 'fixed':
        return FixedSlippage(
            slippage_bps=getattr(config, 'SLIPPAGE_FIXED_BPS', 1.0)
        )
    elif model_name == 'market_impact':
        return MarketImpactSlippage()
    else:
        return VolumeSlippage(
            base_bps=getattr(config, 'SLIPPAGE_BASE_BPS', 0.5),
            volume_impact_factor=getattr(config, 'SLIPPAGE_VOLUME_FACTOR', 0.1),
        )
