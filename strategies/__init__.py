"""Strategy package — re-exports all strategy classes for backward compatibility."""

from strategies.base import Signal, ORBRange, VWAPState
from strategies.regime import MarketRegime
from strategies.orb import ORBStrategy
from strategies.vwap import VWAPStrategy
from strategies.momentum import MomentumStrategy
from strategies.gap_go import GapGoStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.pairs_trading import PairsTradingStrategy

__all__ = [
    "Signal",
    "ORBRange",
    "VWAPState",
    "MarketRegime",
    "ORBStrategy",
    "VWAPStrategy",
    "MomentumStrategy",
    "GapGoStrategy",
    "SectorRotationStrategy",
    "PairsTradingStrategy",
]
