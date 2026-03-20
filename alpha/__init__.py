"""Alpha generation package.

Provides cross-asset signal generation and enhanced seasonality models
for regime-aware alpha overlays on top of core trading strategies.

Modules:
    cross_asset   — Cross-asset signals (bonds, credit, dollar, gold, VIX)
    seasonality   — Enhanced intraday/calendar seasonality scoring
"""

from alpha.cross_asset import CrossAssetSignalGenerator
from alpha.seasonality import EnhancedSeasonality

__all__ = [
    "CrossAssetSignalGenerator",
    "EnhancedSeasonality",
]
