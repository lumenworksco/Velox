"""Velox V10 — Autonomous Algorithmic Trading System.

Six-strategy portfolio with HMM regime detection, adaptive allocation,
cross-asset signals, signal ranking, intraday seasonality, volatility-targeted
sizing, Kelly criterion, daily P&L locks, beta neutralization, smart execution,
overnight holds, parameter optimization, and operational hardening.

Strategies: StatMR (40%), VWAP (20%), KalmanPairs (20%), PEAD (10%), ORB (5%), MicroMom (5%).
"""

import argparse
import asyncio
import logging
import signal
import sys
import time as time_mod
from datetime import datetime, time, timedelta

from rich.live import Live

import config
import database
import analytics as analytics_mod
from data import (
    verify_connectivity, verify_data_feed, get_account, get_clock,
    get_positions, get_snapshot, get_snapshots,
    get_filled_exit_price, get_filled_exit_info,
)
# V10: Decomposed engine modules
from engine.broker_sync import sync_positions_with_broker, check_shadow_exits
from engine.signal_processor import process_signals
from engine.exit_processor import (
    handle_strategy_exits, handle_ws_close, get_current_prices,
)
from engine.scanner import scan_all_strategies, check_all_exits, run_beta_neutralization
from engine.daily_tasks import daily_reset, weekly_tasks, eod_close
from strategies.base import Signal
from strategies.regime import MarketRegime
from strategies.stat_mean_reversion import StatMeanReversion
from strategies.kalman_pairs import KalmanPairsTrader
from strategies.micro_momentum import IntradayMicroMomentum
from strategies.vwap import VWAPStrategy

try:
    from strategies.orb_v2 import ORBStrategyV2
except ImportError:
    ORBStrategyV2 = None
from risk import (
    RiskManager, TradeRecord,
    VolatilityTargetingRiskEngine, DailyPnLLock, BetaNeutralizer,
)
from execution import (
    submit_bracket_order,
    close_position,
    close_partial_position,
    can_short,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon
from correlation import load_correlation_cache, is_too_correlated
from analytics.consistency_score import compute_consistency_score

try:
    import numpy as np
except ImportError:
    np = None

try:
    from position_monitor import PositionMonitor
except ImportError:
    PositionMonitor = None

try:
    import notifications
except ImportError:
    notifications = None

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
    from walk_forward import WalkForwardValidator
except ImportError:
    WalkForwardValidator = None

# --- V10 module imports (all fail-open) ---
try:
    from strategies.pead import PEADStrategy
except ImportError:
    PEADStrategy = None

try:
    from strategies.overnight import OvernightManager
except ImportError:
    OvernightManager = None

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
    from analytics.intraday_seasonality import IntradaySeasonality
except ImportError:
    IntradaySeasonality = None

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
    Watchdog = None
    PositionReconciler = None
    AuditTrail = None

try:
    from smart_routing import SmartOrderRouter, FillMonitor
except ImportError:
    SmartOrderRouter = None
    FillMonitor = None

# --- Logging setup ---
_file_handler = logging.FileHandler(config.LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_et() -> datetime:
    return datetime.now(config.ET)


_market_open_cache: tuple[float, bool] | None = None  # (timestamp, is_open)

def is_market_hours(t: time) -> bool:
    """Check if market is open, using Alpaca clock API for holiday detection."""
    global _market_open_cache
    import time as _time

    # V10 BUG-025: Check actual market status (holidays, early close) via API
    # Cache for 60 seconds to avoid API spam
    now_ts = _time.time()
    if _market_open_cache and (now_ts - _market_open_cache[0]) < 60:
        return _market_open_cache[1]

    # Wall-clock check first (fast path for obviously closed times)
    if not (config.MARKET_OPEN <= t <= config.MARKET_CLOSE):
        _market_open_cache = (now_ts, False)
        return False

    # API check for holidays and early closes
    try:
        clock = get_clock()
        is_open = clock.is_open
        _market_open_cache = (now_ts, is_open)
        return is_open
    except Exception:
        # Fallback to wall-clock if API fails
        _market_open_cache = (now_ts, True)
        return True


def is_trading_hours(t: time) -> bool:
    return config.TRADING_START <= t <= config.ORB_EXIT_TIME


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup_checks() -> dict:
    """Run all startup checks. Exit on failure."""
    console.print("\n[bold]Running startup checks...[/bold]\n")

    # 1. Verify API connectivity
    try:
        info = verify_connectivity()
        print_startup_info(info)
    except Exception as e:
        console.print(f"[bold red]FATAL: Cannot connect to Alpaca API: {e}[/bold red]")
        console.print("Check ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        sys.exit(1)

    # 2. Check market status
    if not info["market_open"]:
        next_open = info.get("next_open", "unknown")
        console.print(f"[yellow]Market is closed. Next open: {next_open}[/yellow]")
        console.print("[yellow]Bot will wait for market open...[/yellow]")

    # 3. Verify data feed
    console.print("Verifying data feed...")
    try:
        if not verify_data_feed("SPY"):
            console.print("[bold red]FATAL: Cannot fetch market data. Check API permissions.[/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]FATAL: Data feed error: {e}[/bold red]")
        sys.exit(1)
    console.print("[green]Data feed verified.[/green]")

    # 4. Print symbol list
    console.print(
        f"\n[bold]Symbol universe:[/bold] {len(config.SYMBOLS)} symbols "
        f"({len(config.CORE_SYMBOLS)} core + {len(config.SYMBOLS) - len(config.CORE_SYMBOLS)} extended)"
    )
    console.print(", ".join(config.SYMBOLS[:10]) + f"... and {len(config.SYMBOLS) - 10} more\n")

    return info


# ---------------------------------------------------------------------------
# V10: Functions extracted to engine/ package:
#   sync_positions_with_broker, check_shadow_exits -> engine.broker_sync
#   process_signals, _process_single_signal -> engine.signal_processor
#   handle_strategy_exits, handle_ws_close, get_current_prices -> engine.exit_processor
# ---------------------------------------------------------------------------

# ===========================================================================
# MAIN (synchronous mode)
# ===========================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Velox V10 Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--live", action="store_true", help="Alias for ALPACA_LIVE=true")
    args = parser.parse_args()

    if args.live:
        import os
        os.environ["ALPACA_LIVE"] = "true"

    # Backtest mode
    if args.backtest:
        from backtester import run_backtest
        database.init_db()
        run_backtest()
        return

    # Walk-forward mode
    if args.walkforward:
        from backtester import walk_forward_test
        database.init_db()
        walk_forward_test()
        return

    # Register SIGTERM handler for graceful Docker shutdown
    _shutdown_requested = False

    def _sigterm_handler(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        logger.info("SIGTERM received — initiating graceful shutdown")
        raise KeyboardInterrupt()  # Reuse existing shutdown path

    signal.signal(signal.SIGTERM, _sigterm_handler)

    console.print("[bold cyan]Starting Velox V10 Trading Bot...[/bold cyan]\n")

    # Validate configuration
    config.validate()

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # V10: Consolidated initialization via engine/startup.py
    from engine.startup import (
        initialize_strategies, initialize_risk_engines,
        initialize_optional_modules, initialize_v10_components,
        initialize_risk_manager, initialize_websocket, initialize_dashboard,
    )

    strats = initialize_strategies()
    stat_mr = strats["stat_mr"]
    kalman_pairs = strats["kalman_pairs"]
    micro_mom = strats["micro_mom"]
    vwap_strategy = strats["vwap_strategy"]
    orb_strategy = strats["orb_strategy"]
    pead_strategy = strats.get("pead_strategy")

    engines = initialize_risk_engines()
    vol_engine = engines["vol_engine"]
    pnl_lock = engines["pnl_lock"]
    beta_neutral = engines["beta_neutral"]
    regime_detector = engines["regime_detector"]

    # V10: PDT rule enforcement
    from risk.pdt_tracker import PDTTracker
    pdt = PDTTracker()

    mods = initialize_optional_modules()
    news_sentiment = mods["news_sentiment"]
    llm_scorer = mods["llm_scorer"]
    walk_forward = mods.get("walk_forward")
    overnight_manager = mods.get("overnight_manager")
    cross_asset_monitor = mods.get("cross_asset_monitor")
    signal_ranker = mods.get("signal_ranker")
    alpha_decay_monitor = mods.get("alpha_decay_monitor")
    adaptive_allocator = mods.get("adaptive_allocator")
    param_optimizer = mods.get("param_optimizer")
    watchdog = mods.get("watchdog")
    reconciler = mods.get("reconciler")

    last_cross_asset_update = now_et()
    last_watchdog_check = now_et()
    last_reconciliation = now_et()

    risk = initialize_risk_manager(info["equity"], info["cash"])

    v10 = initialize_v10_components()
    order_manager = v10["order_manager"]
    tiered_cb = v10["tiered_cb"]
    kill_switch = v10["kill_switch"]
    var_monitor = v10["var_monitor"]
    corr_limiter = v10["corr_limiter"]

    ws_monitor = initialize_websocket(risk)
    initialize_dashboard(order_manager, kill_switch, tiered_cb)

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        logger.warning(f"Filter cache load failed: {e}")

    # --- Loop state ---
    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_analytics_update = now_et()
    last_day = now_et().date()
    eod_summary_printed = False
    current_analytics = None
    universe_prepared_today = False
    last_sunday_task = None
    latest_consistency_score = 0.0

    # Feature flags
    features = ["MR40%", "VWAP20%", "PAIRS20%", "PEAD10%", "ORB5%", "MICRO5%"]
    if config.ALLOW_SHORT:
        features.append("Short")
    if config.TELEGRAM_ENABLED:
        features.append("Notify")
    if config.WEB_DASHBOARD_ENABLED:
        features.append("Web")
    if config.WEBSOCKET_MONITORING:
        features.append("WS")

    features_str = ", ".join(features)
    console.print(f"\n[bold green]Velox V10 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: STAT_MR + VWAP + KALMAN_PAIRS + ORB + MICRO_MOM + PEAD[/dim]")
    console.print(f"[dim]Features: {features_str}[/dim]\n")

    # -----------------------------------------------------------------------
    # Initialize strategies at startup (don't wait for scheduled times)
    # -----------------------------------------------------------------------
    startup_now = now_et()
    if is_market_hours(startup_now.time()) or startup_now.time() >= config.MR_UNIVERSE_PREP_TIME:
        console.print("[cyan]Preparing StatMR universe at startup...[/cyan]")
        try:
            stat_mr.prepare_universe(config.STANDARD_SYMBOLS, startup_now)
            universe_prepared_today = True
            console.print(f"[green]StatMR universe ready: {len(stat_mr.universe)} symbols[/green]")
            # Save OU params to database
            for sym, ou in stat_mr.ou_params.items():
                try:
                    database.save_ou_parameters(sym, ou)
                except Exception:
                    pass
        except Exception as e:
            console.print(f"[yellow]StatMR universe prep failed: {e}[/yellow]")

    # Initialize Kalman pairs if empty or stale
    try:
        active_pairs_count = len(database.get_active_kalman_pairs())
    except Exception:
        active_pairs_count = 0

    if active_pairs_count == 0:
        console.print("[cyan]Initializing Kalman pairs at startup (none active)...[/cyan]")
        try:
            kalman_pairs.select_pairs_weekly(startup_now)
            console.print(f"[green]Kalman pairs ready: {len(kalman_pairs.active_pairs)} pairs[/green]")
        except Exception as e:
            console.print(f"[yellow]Kalman pairs init failed: {e}[/yellow]")
    else:
        # Load existing pairs from DB into memory
        try:
            db_pairs = database.get_active_kalman_pairs()
            kalman_pairs.active_pairs = [
                (p['symbol1'], p['symbol2']) for p in db_pairs
            ]
            console.print(f"[green]Kalman pairs loaded from DB: {len(kalman_pairs.active_pairs)} pairs[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to load Kalman pairs from DB: {e}[/yellow]")

    # Stop logging to terminal -- dashboard takes over
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(
                risk, regime_detector.regime, start_time, now_et(), last_scan,
                len(config.SYMBOLS), current_analytics,
                pnl_lock_state=pnl_lock.state.value,
                vol_scalar=vol_engine.last_scalar,
                portfolio_beta=beta_neutral.portfolio_beta,
                consistency_score=latest_consistency_score,
            ),
            console=console,
            refresh_per_second=0.2,
            transient=False,
        ) as live:
            while True:
                current = now_et()
                current_time = current.time()

                # -------------------------------------------------------
                # Daily reset
                # -------------------------------------------------------
                if current.date() != last_day:
                    daily_reset(
                        risk, stat_mr, kalman_pairs, micro_mom, vwap_strategy,
                        pnl_lock, beta_neutral,
                        orb_strategy=orb_strategy, pead_strategy=pead_strategy,
                        overnight_manager=overnight_manager, news_sentiment=news_sentiment,
                        llm_scorer=llm_scorer, tiered_cb=tiered_cb,
                    )
                    universe_prepared_today = False
                    last_day = current.date()
                    eod_summary_printed = False

                # -------------------------------------------------------
                # Sunday tasks — weekly pair selection, optimization, validation
                # -------------------------------------------------------
                if (current.weekday() == 6 and current_time >= time(0, 0)
                        and last_sunday_task != current.date()):
                    last_sunday_task = current.date()
                    weekly_tasks(current, kalman_pairs, param_optimizer, walk_forward)

                # -------------------------------------------------------
                # Update regime
                # -------------------------------------------------------
                regime = regime_detector.update(current)

                # -------------------------------------------------------
                # Market hours logic
                # -------------------------------------------------------
                if is_market_hours(current_time):

                    # 9:00 AM tasks: universe prep + Monday pair selection
                    if not universe_prepared_today and current_time >= config.MR_UNIVERSE_PREP_TIME:
                        # V10: Dynamic universe selection based on regime
                        scan_symbols = config.STANDARD_SYMBOLS
                        try:
                            from strategies.dynamic_universe import DynamicUniverse
                            _dyn_universe = DynamicUniverse()
                            selection = _dyn_universe.select(regime=regime)
                            if selection.symbols:
                                scan_symbols = selection.symbols
                                logger.info(
                                    f"Dynamic universe: {len(scan_symbols)} symbols "
                                    f"(+{len(selection.added)} -{len(selection.removed)}) regime={regime}"
                                )
                        except Exception as e:
                            logger.debug(f"Dynamic universe failed (using static): {e}")

                        try:
                            stat_mr.prepare_universe(scan_symbols, current)
                            universe_prepared_today = True
                            logger.info(f"MR universe prepared: {len(stat_mr.universe)} symbols")
                            for sym, ou in stat_mr.ou_params.items():
                                try:
                                    database.save_ou_parameters(sym, ou)
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"MR universe prep failed: {e}")

                        # Monday: refresh pairs
                        if current.weekday() == 0:
                            try:
                                kalman_pairs.select_pairs_weekly(current)
                                logger.info(f"Monday pair selection: {len(kalman_pairs.active_pairs)} pairs")
                            except Exception as e:
                                logger.error(f"Monday pair selection failed: {e}")

                    # V7: ORB opening range recording at 10:00 AM
                    if orb_strategy and current_time >= time(10, 0) and not getattr(orb_strategy, '_ranges_recorded_today', False):
                        try:
                            from data import get_intraday_bars
                            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                            market_open = datetime(current.year, current.month, current.day, 9, 30, tzinfo=config.ET)
                            orb_10am = datetime(current.year, current.month, current.day, 10, 0, tzinfo=config.ET)
                            for symbol in config.STANDARD_SYMBOLS[:config.ORB_SCAN_SYMBOLS]:
                                try:
                                    bars_930_1000 = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=market_open, end=orb_10am)
                                    if bars_930_1000 is not None and not bars_930_1000.empty:
                                        orb_strategy.record_opening_range(symbol, bars_930_1000)
                                except Exception:
                                    pass
                            orb_strategy._ranges_recorded_today = True
                            logger.info(f"ORB opening ranges recorded: {len(orb_strategy.opening_ranges)} symbols")
                        except Exception as e:
                            logger.error(f"ORB opening range recording failed: {e}")

                    # Trading hours scan
                    if is_trading_hours(current_time):
                        # 1. Update PnL lock state
                        pnl_lock.update(risk.day_pnl)

                        # V10: Tiered circuit breaker (replaces single-threshold)
                        skip_scan = False
                        if tiered_cb:
                            day_pnl_pct = risk.day_pnl / max(risk.current_equity, 1)
                            tier = tiered_cb.update(day_pnl_pct, current)
                            if tiered_cb.should_close_all and kill_switch:
                                kill_switch.activate("tiered_cb_black", risk_manager=risk, order_manager=order_manager)
                                skip_scan = True
                            elif tiered_cb.should_close_day_trades:
                                # Red tier: close day-hold positions
                                for sym in list(risk.open_trades.keys()):
                                    t = risk.open_trades[sym]
                                    if t.hold_type == "day":
                                        try:
                                            from execution import close_position as _close_pos
                                            _close_pos(sym, reason="circuit_breaker_red")
                                            risk.close_trade(sym, t.entry_price, current, exit_reason="circuit_breaker_red")
                                        except Exception:
                                            pass
                                skip_scan = True
                            elif not tiered_cb.allow_new_entries:
                                skip_scan = True
                        elif risk.check_circuit_breaker():
                            # Fallback to legacy single-threshold
                            skip_scan = True

                        if skip_scan:
                            if notifications and config.TELEGRAM_ENABLED:
                                try:
                                    tier_name = tiered_cb.current_tier.name if tiered_cb else "ACTIVE"
                                    notifications.notify_circuit_breaker(risk.day_pnl)
                                except Exception:
                                    pass
                            logger.warning(f"Circuit breaker {tier_name if tiered_cb else 'ACTIVE'} — skipping scan cycle (day P&L: {risk.day_pnl:.2f})")
                            last_scan = current
                            continue

                        # 2-3. Scan all strategies + detect events
                        signals = scan_all_strategies(
                            current, regime, stat_mr, kalman_pairs, micro_mom,
                            vwap_strategy, orb_strategy, pead_strategy, signal_ranker,
                            day_pnl_pct=day_pnl_pct,
                        )

                        # 4. Process signals
                        if signals:
                            process_signals(
                                signals, risk, regime, current,
                                vol_engine, pnl_lock, ws_monitor,
                                news_sentiment, llm_scorer,
                                regime_detector, cross_asset_monitor,
                                var_monitor, corr_limiter,
                            )

                        # 5. Check strategy exits
                        check_all_exits(
                            current, risk, stat_mr, kalman_pairs, micro_mom,
                            orb_strategy, pead_strategy, ws_monitor,
                        )

                        # 6. Beta neutralization
                        run_beta_neutralization(
                            current, risk, beta_neutral, vol_engine, pnl_lock,
                            ws_monitor, regime,
                        )

                        # V9: Cross-asset monitor update (every 15 min)
                        if cross_asset_monitor:
                            if (current - last_cross_asset_update).total_seconds() >= config.CROSS_ASSET_UPDATE_INTERVAL:
                                try:
                                    cross_asset_monitor.update(current)
                                    last_cross_asset_update = current
                                except Exception as e:
                                    logger.error(f"Cross-asset update failed: {e}")

                        # V9: Watchdog health check
                        if watchdog:
                            if (current - last_watchdog_check).total_seconds() >= config.WATCHDOG_CHECK_INTERVAL:
                                try:
                                    health = watchdog.check_health()
                                    if not health.overall_healthy:
                                        for issue in health.issues:
                                            watchdog.recover(issue)
                                    last_watchdog_check = current
                                except Exception as e:
                                    logger.error(f"Watchdog check failed: {e}")

                        # V9: Position reconciliation (every 30 min)
                        if reconciler:
                            if (current - last_reconciliation).total_seconds() >= config.RECONCILIATION_INTERVAL:
                                try:
                                    reconciler.reconcile()
                                    last_reconciliation = current
                                except Exception as e:
                                    logger.error(f"Reconciliation failed: {e}")

                        # Check shadow exits
                        check_shadow_exits(current)

                    # EOD: close day-hold positions + respect overnight holds (BUG-034, BUG-039)
                    if current_time >= config.ORB_EXIT_TIME:
                        eod_close(current, risk, ws_monitor, overnight_manager, regime)

                    # Sync with broker
                    # V10 BUG-026: Always sync with broker regardless of WebSocket state
                    # WebSocket handles real-time updates; sync handles drift correction
                    sync_positions_with_broker(risk, current, ws_monitor)
                    try:
                        account = get_account()
                        risk.update_equity(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to update account: {e}")

                    last_scan = current

                # -------------------------------------------------------
                # EOD summary at EOD_SUMMARY_TIME
                # -------------------------------------------------------
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary, consistency_score=latest_consistency_score)
                    logger.info(f"Day summary: {summary}")

                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_daily_summary(summary, risk.current_equity)
                        except Exception as e:
                            logger.warning(f"Daily summary notification failed: {e}")

                    # Save daily snapshot
                    try:
                        wr = summary.get("win_rate", 0) if summary.get("trades", 0) > 0 else 0
                        database.save_daily_snapshot(
                            date=current.strftime("%Y-%m-%d"),
                            portfolio_value=risk.current_equity,
                            cash=risk.current_cash,
                            day_pnl=risk.day_pnl * risk.starting_equity,
                            day_pnl_pct=risk.day_pnl,
                            total_trades=summary.get("trades", 0),
                            win_rate=wr,
                            sharpe_rolling=current_analytics.get("sharpe_7d", 0) if current_analytics else 0,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save daily snapshot: {e}")

                    # Compute and save consistency score
                    try:
                        daily_pnls = database.get_daily_pnl_series(days=30)
                        sharpe = current_analytics.get("sharpe_7d", 0) if current_analytics else 0
                        max_dd = current_analytics.get("max_drawdown", 0) if current_analytics else 0
                        latest_consistency_score = compute_consistency_score(daily_pnls, sharpe, max_dd)
                        pct_positive = sum(1 for p in daily_pnls if p > 0) / max(len(daily_pnls), 1)
                        database.save_consistency_log(
                            date=current.strftime("%Y-%m-%d"),
                            consistency_score=latest_consistency_score,
                            pct_positive=pct_positive,
                            sharpe=sharpe,
                            max_drawdown=max_dd,
                            vol_scalar_avg=vol_engine.last_scalar,
                            beta_avg=beta_neutral.portfolio_beta,
                        )
                        logger.info(f"Consistency score: {latest_consistency_score:.1f}")
                    except Exception as e:
                        logger.error(f"Consistency score computation failed: {e}")

                    # V9: Alpha decay check at EOD
                    if alpha_decay_monitor:
                        try:
                            report = alpha_decay_monitor.get_strategy_health_report()
                            for strat, health in report.items():
                                status = health.get("status", "unknown")
                                if status == "critical":
                                    logger.warning(f"Alpha decay CRITICAL: {strat} — consider demoting")
                                elif status == "warning":
                                    logger.info(f"Alpha decay warning: {strat} — Sharpe declining")
                        except Exception as e:
                            logger.error(f"Alpha decay check failed: {e}")

                    # V9: Adaptive allocation rebalance at EOD
                    if adaptive_allocator:
                        try:
                            new_weights = adaptive_allocator.compute_weights(
                                regime=regime,
                                regime_detector=regime_detector,
                            )
                            if new_weights:
                                vol_engine.set_adaptive_weights(new_weights)
                                logger.info(f"Adaptive weights updated: {new_weights}")
                        except Exception as e:
                            logger.error(f"Adaptive allocation failed: {e}")

                    eod_summary_printed = True

                # -------------------------------------------------------
                # Save state periodically
                # -------------------------------------------------------
                if (current - last_state_save).total_seconds() >= config.STATE_SAVE_INTERVAL_SEC:
                    try:
                        database.save_open_positions(risk.open_trades)
                    except Exception as e:
                        logger.error(f"Failed to save state: {e}")
                    last_state_save = current

                # -------------------------------------------------------
                # Update analytics every 5 minutes
                # -------------------------------------------------------
                if (current - last_analytics_update).total_seconds() >= 300:
                    try:
                        current_analytics = analytics_mod.compute_analytics()
                        if current_analytics and notifications and config.TELEGRAM_ENABLED:
                            dd = current_analytics.get("max_drawdown", 0)
                            if dd > 0.05:
                                try:
                                    notifications.notify_drawdown_warning(dd)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Failed to compute analytics: {e}")

                    # V10: Update VaR monitor
                    if var_monitor:
                        try:
                            pnl_series_var = database.get_daily_pnl_series(days=60)
                            if pnl_series_var and len(pnl_series_var) >= 10:
                                var_result = var_monitor.update(pnl_series_var, risk.current_equity)
                                # Log if VaR budget is running low
                                if var_monitor.risk_budget_remaining < 0.3:
                                    logger.warning(
                                        f"VaR budget low: {var_monitor.risk_budget_remaining:.0%} remaining "
                                        f"(VaR95=${var_result.var_95:.0f}, {var_result.var_95_pct:.2%})"
                                    )
                        except Exception as e:
                            logger.debug(f"VaR update failed: {e}")

                    last_analytics_update = current

                # -------------------------------------------------------
                # Update vol scalar periodically (every analytics cycle)
                # -------------------------------------------------------
                try:
                    from risk import get_vix_level
                    vix = get_vix_level()
                    pnl_series = database.get_daily_pnl_series(days=20)
                    if np is not None and len(pnl_series) > 2:
                        rolling_std = float(np.std(pnl_series))
                    else:
                        rolling_std = 0.01
                    vol_engine.compute_vol_scalar(
                        vix=vix,
                        portfolio_atr_vol=0.01,  # Simplified; could compute from positions
                        rolling_pnl_std=rolling_std,
                    )
                except Exception:
                    pass  # Vol scalar stays at last value

                # -------------------------------------------------------
                # Update dashboard
                # -------------------------------------------------------
                # V9: Update web dashboard shared state
                if config.WEB_DASHBOARD_ENABLED:
                    try:
                        from web_dashboard import update_v9_state
                        v9_update = {}
                        if regime_detector and hasattr(regime_detector, 'hmm_regime'):
                            v9_update['hmm_regime'] = str(regime_detector.hmm_regime or "UNKNOWN")
                            v9_update['hmm_probabilities'] = getattr(regime_detector, 'hmm_probabilities', {})
                        if cross_asset_monitor:
                            v9_update['cross_asset_bias'] = cross_asset_monitor.get_equity_bias()
                            try:
                                v9_update['cross_asset_signals'] = cross_asset_monitor.compute_signals()
                            except Exception:
                                pass
                        if overnight_manager:
                            v9_update['overnight_count'] = len(overnight_manager.get_overnight_positions())
                        if adaptive_allocator and hasattr(vol_engine, '_adaptive_weights'):
                            v9_update['adaptive_weights'] = getattr(vol_engine, '_adaptive_weights', {})
                        update_v9_state(**v9_update)
                    except Exception:
                        pass  # Web dashboard update is non-critical

                # Build V9 dashboard kwargs
                _v9_dash = {}
                try:
                    if regime_detector and hasattr(regime_detector, 'hmm_regime'):
                        _v9_dash['hmm_regime_state'] = str(regime_detector.hmm_regime or "")
                        _v9_dash['hmm_probabilities'] = getattr(regime_detector, 'hmm_probabilities', None)
                    if cross_asset_monitor:
                        _v9_dash['cross_asset_bias'] = cross_asset_monitor.get_equity_bias()
                    if overnight_manager:
                        _v9_dash['overnight_count'] = len(overnight_manager.get_overnight_positions())
                    if alpha_decay_monitor:
                        try:
                            report = alpha_decay_monitor.get_strategy_health_report()
                            warnings = [f"{s}: {d.get('status','')}" for s, d in report.items()
                                       if d.get('status') in ('warning', 'critical')]
                            _v9_dash['alpha_warnings'] = warnings
                        except Exception:
                            pass
                except Exception:
                    pass

                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics,
                        pnl_lock_state=pnl_lock.state.value,
                        vol_scalar=vol_engine.last_scalar,
                        portfolio_beta=beta_neutral.portfolio_beta,
                        consistency_score=latest_consistency_score,
                        **_v9_dash,
                    )
                )

                # Sleep until next scan — interruptible 1s ticks
                for _ in range(config.SCAN_INTERVAL_SEC):
                    time_mod.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)



# ===========================================================================
# (Legacy async mode removed in V10 — synchronous main loop has all features)
# ===========================================================================


# ===========================================================================
# DIAGNOSTIC MODE
# ===========================================================================

def run_diagnostic():
    """Run one diagnostic scan cycle and print what's blocking trades."""
    print("\n=== VELOX V10 SIGNAL DIAGNOSTIC MODE ===\n")
    print("Initializing strategies and filters...\n")

    from earnings import has_earnings_soon, load_earnings_cache
    from correlation import is_too_correlated, load_correlation_cache
    from data import get_clock, get_account

    stat_mr = StatMeanReversion()
    kalman_pairs = KalmanPairsTrader()
    micro_mom = IntradayMicroMomentum()

    clock = get_clock()
    now = clock.timestamp

    account = get_account()
    print(f"Account equity: ${float(account.equity):,.2f}")
    print(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
    print(f"Time: {now}\n")

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception as e:
        print(f"  Earnings cache: {e}")
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        print(f"  Correlation cache: {e}")

    # Detect regime
    regime_detector = MarketRegime()
    regime = regime_detector.update(now)
    print(f"Market regime: {regime}\n")

    # Track rejections
    rejection_counts: dict[str, int] = {}
    signals_generated: list[str] = []

    def count(reason: str):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    mr_signals = []
    pair_signals = []
    micro_signals = []

    if clock.is_open:
        # Prepare MR universe
        print("Preparing MR universe...")
        stat_mr.prepare_universe(config.STANDARD_SYMBOLS, now)
        print(f"  MR universe: {len(stat_mr.universe)} symbols")

        # Scan strategies
        try:
            mr_signals = stat_mr.scan(now, regime)
            for sig in mr_signals:
                signals_generated.append(f"{sig.symbol} STAT_MR {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  StatMR scan error: {e}")

        try:
            pair_signals = kalman_pairs.scan(now, regime)
            for sig in pair_signals:
                signals_generated.append(f"{sig.symbol} KALMAN_PAIRS {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  KalmanPairs scan error: {e}")

        try:
            micro_mom.detect_event(now)
            micro_signals = micro_mom.scan(now, day_pnl_pct=0.0, regime=regime)
            for sig in micro_signals:
                signals_generated.append(f"{sig.symbol} MICRO_MOM {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  MicroMom scan error: {e}")

    # Filter diagnostic
    all_signals = mr_signals + pair_signals + micro_signals
    blocked_signals = []
    passed_signals = []

    for sig in all_signals:
        try:
            if has_earnings_soon(sig.symbol):
                count("filter_earnings")
                blocked_signals.append(f"{sig.symbol} {sig.strategy}: BLOCKED by earnings")
                continue
        except Exception:
            pass

        if sig.strategy != "KALMAN_PAIRS":
            try:
                if is_too_correlated(sig.symbol, []):
                    count("filter_correlation")
                    blocked_signals.append(f"{sig.symbol} {sig.strategy}: BLOCKED by correlation")
                    continue
            except Exception:
                pass

        passed_signals.append(f"{sig.symbol} {sig.strategy} {sig.side} @ ${sig.entry_price:.2f}")

    # Print results
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SCAN RESULTS")
    print("=" * 60)
    print(f"\n  Total symbols scanned:    {len(config.SYMBOLS)}")
    print(f"  MR universe size:         {len(stat_mr.universe)}")
    print(f"  Active pairs:             {len(kalman_pairs.active_pairs)}")
    print(f"  Signals generated:        {len(all_signals)}")
    print(f"  Signals blocked:          {len(blocked_signals)}")
    print(f"  Signals passed:           {len(passed_signals)}")

    if rejection_counts:
        print(f"\n  REJECTION BREAKDOWN:")
        for reason, cnt in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30} {cnt:>4}")

    if blocked_signals:
        print(f"\n  BLOCKED SIGNALS:")
        for b in blocked_signals:
            print(f"    {b}")

    if passed_signals:
        print(f"\n  PASSED SIGNALS (would trade):")
        for p in passed_signals:
            print(f"    {p}")
    else:
        print(f"\n  NO SIGNALS PASSED -- all blocked or no setups found")

    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="Velox V10 Trading Bot")
    _parser.add_argument("--diagnose", action="store_true", help="Run one diagnostic scan cycle (read-only)")
    _args = _parser.parse_args()

    if _args.diagnose:
        run_diagnostic()
    else:
        main()
