"""data package — Alpaca data fetching, feature store, dynamic universe, data quality.

Backward-compatible: all functions previously in data.py (get_daily_bars, etc.)
are re-exported here alongside V11 additions.
"""

# --- Original data.py functions (moved to data/fetcher.py) ---
from data.fetcher import (
    get_trading_client,
    get_data_client,
    get_account,
    get_clock,
    get_positions,
    get_open_orders,
    get_bars,
    get_daily_bars,
    get_intraday_bars,
    get_filled_exit_info,
    get_filled_exit_price,
    get_snapshot,
    get_snapshots,
    verify_connectivity,
    verify_data_feed,
)

# --- V11 additions ---
from data.feature_store import FeatureStore, get_feature_store
from data.universe import DynamicUniverse, get_dynamic_universe
from data.quality import (
    DataQualityFramework,
    DataQualityScore,
    QualityIssue,
    check_bar_quality,
    get_quality_framework,
)

__all__ = [
    # Original data functions
    "get_trading_client",
    "get_data_client",
    "get_account",
    "get_clock",
    "get_positions",
    "get_open_orders",
    "get_bars",
    "get_daily_bars",
    "get_intraday_bars",
    "get_filled_exit_info",
    "get_filled_exit_price",
    "get_snapshot",
    "get_snapshots",
    "verify_connectivity",
    "verify_data_feed",
    # Feature store
    "FeatureStore",
    "get_feature_store",
    # Dynamic universe
    "DynamicUniverse",
    "get_dynamic_universe",
    # Data quality
    "DataQualityFramework",
    "DataQualityScore",
    "QualityIssue",
    "check_bar_quality",
    "get_quality_framework",
]
