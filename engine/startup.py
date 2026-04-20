"""V10 Engine — Startup initialization: strategies, risk engines, optional modules.

Consolidates the ~200 lines of module initialization from main() into a single
function that returns a dict of all initialized components.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import config
import database
from data import verify_connectivity, verify_data_feed, get_account
from strategies.stat_mean_reversion import StatMeanReversion
from strategies.kalman_pairs import KalmanPairsTrader
from strategies.micro_momentum import IntradayMicroMomentum
from strategies.vwap import VWAPStrategy
from strategies.regime import MarketRegime
from risk import RiskManager, VolatilityTargetingRiskEngine, DailyPnLLock, BetaNeutralizer
from engine.broker_sync import sync_positions_with_broker
from engine.exit_processor import handle_ws_close

logger = logging.getLogger(__name__)

# Optional imports (all fail-open)
try:
    from strategies.orb_v2 import ORBStrategyV2
except ImportError:
    ORBStrategyV2 = None

try:
    from strategies.pead import PEADStrategy
except ImportError:
    PEADStrategy = None

try:
    from strategies.overnight import OvernightManager
except ImportError:
    OvernightManager = None

try:
    from position_monitor import PositionMonitor
except ImportError:
    PositionMonitor = None

try:
    from news_sentiment import AlpacaNewsSentiment
except ImportError:
    AlpacaNewsSentiment = None

try:
    from llm_signal_scorer import LLMSignalScorer
except ImportError:
    LLMSignalScorer = None

try:
    from adaptive_exit_manager import AdaptiveExitManager
except ImportError:
    AdaptiveExitManager = None

try:
    from analytics.walk_forward import WalkForwardValidator
except ImportError:
    WalkForwardValidator = None

try:
    from analytics.cross_asset import CrossAssetMonitor
except ImportError:
    CrossAssetMonitor = None

try:
    from analytics.signal_ranker import SignalRanker
except ImportError:
    SignalRanker = None

try:
    from analytics.alpha_decay import AlphaDecayMonitor
except ImportError:
    AlphaDecayMonitor = None

try:
    from risk.adaptive_allocation import AdaptiveAllocator
except ImportError:
    AdaptiveAllocator = None

try:
    from analytics.param_optimizer import BayesianOptimizer
except ImportError:
    BayesianOptimizer = None

try:
    from watchdog import Watchdog, PositionReconciler, AuditTrail
except ImportError:
    Watchdog = PositionReconciler = AuditTrail = None


def _init_optional(name: str, cls, enabled_flag: str, *args, **kwargs):
    """Initialize an optional module with fail-open behavior."""
    if not getattr(config, enabled_flag, False) or cls is None:
        return None
    try:
        instance = cls(*args, **kwargs)
        logger.info(f"{name} enabled")
        return instance
    except Exception as e:
        logger.warning(f"{name} init failed: {e}")
        return None


def initialize_strategies():
    """Create all strategy instances."""
    # 2026-04-17: log disabled strategies at startup so ops can confirm the
    # demotion is in effect (see DISABLED_STRATEGIES in config/settings.py).
    disabled = getattr(config, 'DISABLED_STRATEGIES', set()) or set()
    reasons = getattr(config, 'DISABLED_STRATEGIES_REASON', {}) or {}
    for name in sorted(disabled):
        logger.warning(
            "Strategy %s is DISABLED (entry scans skipped) — reason: %s",
            name, reasons.get(name, "unspecified"),
        )
    return {
        "stat_mr": StatMeanReversion(),
        "kalman_pairs": KalmanPairsTrader(),
        "micro_mom": IntradayMicroMomentum(),
        "vwap_strategy": VWAPStrategy(),
        "orb_strategy": ORBStrategyV2() if ORBStrategyV2 and config.ORB_ENABLED else None,
        "pead_strategy": _init_optional("PEAD strategy", PEADStrategy, "PEAD_ENABLED"),
    }


def initialize_risk_engines():
    """Create risk management components."""
    return {
        "vol_engine": VolatilityTargetingRiskEngine(),
        "pnl_lock": DailyPnLLock(),
        "beta_neutral": BetaNeutralizer(),
        "regime_detector": MarketRegime(),
    }


def initialize_optional_modules():
    """Create all optional V7/V9 modules (fail-open)."""
    modules = {
        "news_sentiment": _init_optional("News sentiment", AlpacaNewsSentiment, "NEWS_SENTIMENT_ENABLED"),
        "llm_scorer": _init_optional("LLM scorer", LLMSignalScorer, "LLM_SCORING_ENABLED"),
        "adaptive_exits": AdaptiveExitManager() if AdaptiveExitManager and config.ADAPTIVE_EXITS_ENABLED else None,
        "walk_forward": WalkForwardValidator() if WalkForwardValidator and config.WALK_FORWARD_ENABLED else None,
        "overnight_manager": _init_optional("Overnight manager", OvernightManager, "OVERNIGHT_HOLD_ENABLED"),
        "cross_asset_monitor": _init_optional("Cross-asset monitor", CrossAssetMonitor, "CROSS_ASSET_ENABLED"),
        "signal_ranker": _init_optional("Signal ranker", SignalRanker, "SIGNAL_RANKING_ENABLED"),
        "alpha_decay_monitor": _init_optional("Alpha decay monitor", AlphaDecayMonitor, "ALPHA_DECAY_ENABLED"),
        "param_optimizer": _init_optional("Parameter optimizer", BayesianOptimizer, "PARAM_OPTIMIZER_ENABLED"),
        "watchdog": _init_optional("Watchdog", Watchdog, "WATCHDOG_ENABLED"),
        "reconciler": _init_optional("Position reconciler", PositionReconciler, "RECONCILIATION_ENABLED"),
        "audit_trail": _init_optional("Audit trail", AuditTrail, "STRUCTURED_LOGGING_ENABLED"),
    }

    # Adaptive allocator needs extra init args
    if config.ADAPTIVE_ALLOCATION_ENABLED and AdaptiveAllocator:
        try:
            strategies = list(config.STRATEGY_ALLOCATIONS.keys())
            modules["adaptive_allocator"] = AdaptiveAllocator(strategies, config.STRATEGY_ALLOCATIONS)
            logger.info("Adaptive allocator enabled")
        except Exception as e:
            logger.warning(f"Adaptive allocator init failed: {e}")
            modules["adaptive_allocator"] = None
    else:
        modules["adaptive_allocator"] = None

    return modules


def initialize_v10_components():
    """Create V10-specific components: OMS, circuit breaker, VaR, correlation limiter."""
    components = {
        "order_manager": None,
        "tiered_cb": None,
        "kill_switch": None,
        "var_monitor": None,
        "corr_limiter": None,
    }
    try:
        from oms import OrderManager, KillSwitch
        from risk.circuit_breaker import TieredCircuitBreaker
        from risk.var_monitor import VaRMonitor
        from risk.correlation_limiter import CorrelationLimiter
        from engine.signal_processor import set_order_manager

        components["order_manager"] = OrderManager()
        set_order_manager(components["order_manager"])
        components["tiered_cb"] = TieredCircuitBreaker()
        components["kill_switch"] = KillSwitch()
        components["var_monitor"] = VaRMonitor()
        components["corr_limiter"] = CorrelationLimiter()
        logger.info("OMS, circuit breaker, VaR monitor, and correlation limiter initialized")
    except Exception as e:
        logger.warning(f"Component init failed (non-fatal): {e}")

    return components


def initialize_risk_manager(equity: float, cash: float):
    """Create and configure the RiskManager with startup reconciliation."""
    risk = RiskManager()
    risk.reset_daily(equity, cash)
    risk.load_from_db()

    # V10 BUG-024: Validate DB state against broker on startup
    try:
        logger.info("Running startup broker reconciliation...")
        sync_positions_with_broker(risk, datetime.now(config.ET), None)
        logger.info(f"Startup reconciliation complete: {len(risk.open_trades)} positions tracked")
    except Exception as e:
        logger.error(f"Startup reconciliation failed: {e}")

    return risk


def initialize_websocket(risk, ws_monitor_cls=None):
    """Set up WebSocket position monitor."""
    if not config.WEBSOCKET_MONITORING or not (ws_monitor_cls or PositionMonitor):
        return None

    cls = ws_monitor_cls or PositionMonitor
    ws_monitor = cls(risk)
    ws_monitor.set_close_callback(
        lambda symbol, reason: handle_ws_close(symbol, reason, risk, ws_monitor)
    )
    for symbol in risk.open_trades:
        ws_monitor.subscribe(symbol)
    ws_monitor.start()
    logger.info("WebSocket position monitor started")
    return ws_monitor


def initialize_dashboard(order_manager=None, kill_switch=None, tiered_cb=None):
    """Start the web dashboard and register components."""
    if not config.WEB_DASHBOARD_ENABLED:
        return False

    try:
        from web_dashboard import start_web_dashboard, set_v10_components
        start_web_dashboard()
        if order_manager or kill_switch or tiered_cb:
            set_v10_components(order_manager, kill_switch, tiered_cb)
        logger.info(f"Web dashboard started at http://localhost:{config.WEB_DASHBOARD_PORT}")
        return True
    except Exception as e:
        logger.warning(f"Web dashboard failed to start: {e}")
        return False
