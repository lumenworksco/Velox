"""Risk package — V6 risk engine + backward-compatible V1-V5 risk manager."""

# V6 risk modules
from risk.vol_targeting import VolatilityTargetingRiskEngine
from risk.daily_pnl_lock import DailyPnLLock, LockState
from risk.beta_neutralizer import BetaNeutralizer

# V8 risk modules
from risk.kelly import KellyEngine
from risk.portfolio_heat import PortfolioHeatTracker

# V10 advanced risk modules
from risk.factor_model import FactorRiskModel
from risk.hrp import HierarchicalRiskParity
from risk.stress_testing import StressTestFramework
from risk.intraday_controls import IntradayRiskControls
from risk.gap_risk import GapRiskManager
from risk.correlation import DynamicCorrelation

# Next-gen risk modules (RISK-001 through RISK-008)
from risk.dynamic_hedging import DynamicHedgingEngine
from risk.margin_monitor import MarginMonitor
from risk.corporate_actions import CorporateActionDetector
from risk.conformal_stops import ConformalStopEngine

# Backward compatibility: re-export RiskManager, TradeRecord, VIX functions
from risk.risk_manager import (
    RiskManager,
    TradeRecord,
    get_vix_level,
    get_vix_risk_scalar,
)

__all__ = [
    # V6
    "VolatilityTargetingRiskEngine",
    "DailyPnLLock",
    "LockState",
    "BetaNeutralizer",
    # V8
    "KellyEngine",
    "PortfolioHeatTracker",
    # V10 advanced risk
    "FactorRiskModel",
    "HierarchicalRiskParity",
    "StressTestFramework",
    "IntradayRiskControls",
    "GapRiskManager",
    "DynamicCorrelation",
    # Next-gen risk (RISK-001 through RISK-008)
    "DynamicHedgingEngine",
    "MarginMonitor",
    "CorporateActionDetector",
    "ConformalStopEngine",
    # V1-V5 backward compat
    "RiskManager",
    "TradeRecord",
    "get_vix_level",
    "get_vix_risk_scalar",
]
