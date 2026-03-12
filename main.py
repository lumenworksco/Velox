"""Entry point V4 — MTF confirmation, VIX scaling, news filter, sector rotation,
pairs trading, advanced exits, plus all V3 features."""

import argparse
import asyncio
import logging
import sys
import time as time_mod
from datetime import datetime, time, timedelta
from pathlib import Path

from rich.console import Console
from rich.live import Live

import config
import database
import analytics as analytics_mod
from data import verify_connectivity, verify_data_feed, get_account, get_clock, get_positions
from strategies import MarketRegime, ORBStrategy, VWAPStrategy, MomentumStrategy, GapGoStrategy, Signal
from risk import RiskManager, TradeRecord, get_vix_risk_scalar
from execution import (
    submit_bracket_order,
    close_position,
    close_orb_positions,
    check_vwap_time_stops,
    check_momentum_max_hold,
    cancel_all_open_orders,
    can_short,
    close_gap_go_positions,
    check_sector_max_hold,
    check_pairs_max_hold,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon, get_excluded_count
from correlation import load_correlation_cache, is_too_correlated

# V3 imports (conditional)
try:
    from ml_filter import ml_filter, extract_live_features
except ImportError:
    ml_filter = None

try:
    from relative_strength import RelativeStrengthTracker
except ImportError:
    RelativeStrengthTracker = None

try:
    from position_monitor import PositionMonitor
except ImportError:
    PositionMonitor = None

try:
    import notifications
except ImportError:
    notifications = None

# V4 imports (conditional)
try:
    from strategies.mtf_confirmation import mtf_confirmer
except ImportError:
    mtf_confirmer = None

try:
    from news_filter import news_filter
except ImportError:
    news_filter = None

try:
    from strategies.sector_rotation import SectorRotationStrategy
except ImportError:
    SectorRotationStrategy = None

try:
    from strategies.pairs_trading import PairsTradingStrategy
except ImportError:
    PairsTradingStrategy = None

try:
    from exit_manager import ExitManager
except ImportError:
    ExitManager = None

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


def now_et() -> datetime:
    return datetime.now(config.ET)


def is_market_hours(t: time) -> bool:
    return config.MARKET_OPEN <= t <= config.MARKET_CLOSE


def is_trading_hours(t: time) -> bool:
    return config.TRADING_START <= t <= config.ORB_EXIT_TIME


def is_orb_recording_period(t: time) -> bool:
    return config.MARKET_OPEN <= t < config.ORB_END


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
    console.print(f"\n[bold]Symbol universe:[/bold] {len(config.SYMBOLS)} symbols ({len(config.CORE_SYMBOLS)} core + {len(config.SYMBOLS) - len(config.CORE_SYMBOLS)} extended)")
    console.print(", ".join(config.SYMBOLS[:10]) + f"... and {len(config.SYMBOLS) - 10} more")
    console.print(f"[dim]Leveraged ETFs (VWAP only): {', '.join(sorted(config.LEVERAGED_ETFS))}[/dim]\n")

    return info


def process_signals(
    signals: list[Signal],
    risk: RiskManager,
    regime: str,
    now: datetime,
    rs_tracker=None,
    ws_monitor=None,
    market_data: dict | None = None,
):
    """Process signals: check filters, risk, size, and submit orders."""

    # V4: VIX halt — skip all signals if VIX >= halt threshold
    if config.VIX_RISK_SCALING_ENABLED and get_vix_risk_scalar() == 0.0:
        logger.info("VIX halt active — skipping all signals")
        for signal in signals:
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, "vix_halt")
        return

    # V4: Group pairs signals by pair_id for atomic processing
    pair_groups: dict[str, list[Signal]] = {}
    non_pair_signals: list[Signal] = []
    for signal in signals:
        if signal.pair_id:
            pair_groups.setdefault(signal.pair_id, []).append(signal)
        else:
            non_pair_signals.append(signal)

    # Process non-pair signals normally
    for signal in non_pair_signals:
        _process_single_signal(signal, risk, regime, now, rs_tracker, ws_monitor, market_data)

    # Process pairs atomically (both legs or neither)
    for pair_id, pair_signals in pair_groups.items():
        if len(pair_signals) != 2:
            logger.warning(f"Pair {pair_id} has {len(pair_signals)} signals, skipping")
            continue

        # Check if both legs can be opened
        all_ok = True
        for sig in pair_signals:
            if sig.symbol in risk.open_trades:
                all_ok = False
                break
            allowed, reason = risk.can_open_trade(strategy=sig.strategy)
            if not allowed:
                all_ok = False
                break

        if all_ok:
            for sig in pair_signals:
                _process_single_signal(sig, risk, regime, now, rs_tracker, ws_monitor, market_data)
        else:
            for sig in pair_signals:
                database.log_signal(now, sig.symbol, sig.strategy, sig.side, False, "pair_blocked")


def _process_single_signal(
    signal: Signal,
    risk: RiskManager,
    regime: str,
    now: datetime,
    rs_tracker=None,
    ws_monitor=None,
    market_data: dict | None = None,
):
    """Process a single signal through all filters and submit if valid."""
    skip_reason = ""

    # Skip if already in this symbol
    if signal.symbol in risk.open_trades:
        skip_reason = "already_in_position"
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # Earnings filter
    if has_earnings_soon(signal.symbol):
        skip_reason = "earnings_soon"
        logger.info(f"Signal skipped for {signal.symbol}: earnings soon")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # Correlation filter (skip for pairs — they're inherently correlated)
    if signal.strategy != "PAIRS":
        open_symbols = list(risk.open_trades.keys())
        if open_symbols and is_too_correlated(signal.symbol, open_symbols):
            skip_reason = "high_correlation"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return

    # V4: Multi-timeframe confirmation
    if mtf_confirmer and config.MTF_CONFIRMATION_ENABLED:
        try:
            confirmed = mtf_confirmer.confirm(
                {"symbol": signal.symbol, "strategy": signal.strategy, "side": signal.side},
                now,
            )
            if not confirmed:
                skip_reason = "mtf_rejected"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
        except Exception as e:
            logger.warning(f"MTF confirmation error for {signal.symbol}: {e}")

    # V4: News sentiment filter (skip for sector ETFs and pairs)
    if news_filter and config.NEWS_FILTER_ENABLED and signal.strategy not in ("SECTOR_ROTATION", "PAIRS"):
        try:
            blocked, reason = news_filter.should_block(signal.symbol, signal.side)
            if blocked:
                skip_reason = f"news_{news_filter.get_sentiment(signal.symbol).lower()}"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
        except Exception as e:
            logger.warning(f"News filter error for {signal.symbol}: {e}")

    # V3: ML signal filter
    if config.USE_ML_FILTER and ml_filter and ml_filter._active.get(signal.strategy):
        try:
            features = extract_live_features(signal, regime, market_data or {})
            prob = ml_filter.should_trade(signal.strategy, features)
            if prob < config.ML_PROBABILITY_THRESHOLD:
                skip_reason = f"ml_filter_{prob:.2f}"
                logger.info(f"ML filter rejected {signal.symbol} ({signal.strategy}): prob={prob:.2f}")
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
        except Exception as e:
            logger.warning(f"ML filter error for {signal.symbol}: {e}")

    # V3: Relative strength filter
    if config.USE_RS_FILTER and rs_tracker:
        try:
            rs_score = rs_tracker.score(signal.symbol)
            if signal.side == "buy" and rs_score < config.RS_LONG_THRESHOLD:
                skip_reason = f"rs_weak_{rs_score:.2f}"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
            elif signal.side == "sell" and rs_score > config.RS_SHORT_THRESHOLD:
                skip_reason = f"rs_strong_{rs_score:.2f}"
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
        except Exception as e:
            logger.warning(f"RS filter error for {signal.symbol}: {e}")

    # V3: Short selling pre-check
    if signal.side == "sell":
        if not config.ALLOW_SHORT:
            skip_reason = "shorting_disabled"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return
        shortable, short_reason = can_short(signal.symbol, 1, signal.entry_price)
        if not shortable:
            skip_reason = f"short_blocked_{short_reason}"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return

    # Check risk limits
    allowed, reason = risk.can_open_trade(strategy=signal.strategy)
    if not allowed:
        skip_reason = reason
        logger.info(f"Trade blocked for {signal.symbol}: {reason}")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # Calculate position size (V3: strategy-weighted + short multiplier)
    qty = risk.calculate_position_size(
        signal.entry_price, signal.stop_loss, regime,
        strategy=signal.strategy, side=signal.side,
    )
    if qty <= 0:
        skip_reason = "position_size_zero"
        logger.info(f"Position size 0 for {signal.symbol}, skipping")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # Submit bracket order
    order_id = submit_bracket_order(signal, qty)
    if order_id is None:
        skip_reason = "order_failed"
        logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # Register the trade
    time_stop = None
    max_hold_date = None
    hold_type = getattr(signal, 'hold_type', 'day')

    if signal.strategy == "VWAP":
        time_stop = now + timedelta(minutes=config.VWAP_TIME_STOP_MINUTES)
    elif signal.strategy == "MOMENTUM":
        max_hold_date = now + timedelta(days=config.MOMENTUM_MAX_HOLD_DAYS)
    elif signal.strategy == "GAP_GO":
        time_stop = datetime.combine(now.date(), config.GAP_EXIT_TIME, tzinfo=config.ET)
    elif signal.strategy == "SECTOR_ROTATION":
        max_hold_date = now + timedelta(days=config.SECTOR_MAX_HOLD_DAYS)
    elif signal.strategy == "PAIRS":
        max_hold_date = now + timedelta(days=config.PAIRS_MAX_HOLD_DAYS)

    trade = TradeRecord(
        symbol=signal.symbol,
        strategy=signal.strategy,
        side=signal.side,
        entry_price=signal.entry_price,
        entry_time=now,
        qty=qty,
        take_profit=signal.take_profit,
        stop_loss=signal.stop_loss,
        order_id=order_id,
        time_stop=time_stop,
        hold_type=hold_type,
        max_hold_date=max_hold_date,
        pair_id=getattr(signal, 'pair_id', ''),
        highest_price_seen=signal.entry_price,
    )
    risk.register_trade(trade)

    # V3: Subscribe to WebSocket monitoring
    if ws_monitor:
        ws_monitor.subscribe(signal.symbol)

    # V3: Telegram notification
    if notifications and config.TELEGRAM_ENABLED:
        try:
            notifications.notify_trade_opened(trade)
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")

    # Log signal as acted on
    database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")


def sync_positions_with_broker(risk: RiskManager, now: datetime, ws_monitor=None):
    """Sync open trades with actual broker positions to detect fills/stops."""
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            trade = risk.open_trades[symbol]
            risk.close_trade(symbol, trade.entry_price, now, exit_reason="broker_sync")
            logger.info(f"Position {symbol} no longer at broker — marking closed")

            # V3: Unsubscribe from WS and notify
            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notifications and config.TELEGRAM_ENABLED:
                try:
                    notifications.notify_trade_closed(trade)
                except Exception:
                    pass

    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Algo Trading Bot V4")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--train-ml", action="store_true", help="Train ML signal filter")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
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

    # Train ML mode
    if args.train_ml:
        database.init_db()
        if ml_filter:
            ml_filter.retrain_all()
        else:
            console.print("[red]ML filter module not available[/red]")
        return

    # Optimize mode
    if args.optimize:
        from optimizer import weekly_optimization
        database.init_db()
        weekly_optimization()
        return

    console.print("[bold cyan]Starting Algo Trading Bot V4...[/bold cyan]\n")

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # Initialize strategies
    regime_detector = MarketRegime()
    orb = ORBStrategy()
    vwap = VWAPStrategy()
    momentum = MomentumStrategy()
    gap_go = GapGoStrategy() if config.GAP_GO_ENABLED else None
    risk = RiskManager()

    # V4: Initialize sector rotation strategy
    sector_rotation = None
    if config.SECTOR_ROTATION_ENABLED and SectorRotationStrategy:
        sector_rotation = SectorRotationStrategy()
        console.print("[green]Sector Rotation strategy initialized.[/green]")

    # V4: Initialize pairs trading strategy
    pairs_trading = None
    if config.PAIRS_TRADING_ENABLED and PairsTradingStrategy:
        pairs_trading = PairsTradingStrategy()
        console.print("[green]Pairs Trading strategy initialized.[/green]")

    # V4: Initialize exit manager
    exit_mgr = None
    if config.ADVANCED_EXITS_ENABLED and ExitManager:
        exit_mgr = ExitManager()
        console.print("[green]Advanced exit manager initialized.[/green]")

    # Initialize risk with account info
    risk.reset_daily(info["equity"], info["cash"])

    # Load persisted state from DB
    risk.load_from_db()

    # V3: Initialize relative strength tracker
    rs_tracker = None
    if config.USE_RS_FILTER and RelativeStrengthTracker:
        rs_tracker = RelativeStrengthTracker()
        console.print("[green]Relative strength tracker initialized.[/green]")

    # V3: Load ML models
    if config.USE_ML_FILTER and ml_filter:
        try:
            ml_filter.load_models()
            active = [s for s, a in ml_filter._active.items() if a]
            if active:
                console.print(f"[green]ML filter loaded for: {', '.join(active)}[/green]")
            else:
                console.print("[yellow]ML filter: no trained models yet (need 100+ trades)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]ML filter load failed: {e}[/yellow]")

    # V3: Calculate initial strategy weights
    if config.DYNAMIC_ALLOCATION:
        try:
            risk.update_strategy_weights()
            weights = risk.get_strategy_weights()
            if weights:
                weight_str = ", ".join(f"{s}: {w:.0%}" for s, w in weights.items())
                console.print(f"[green]Capital allocation: {weight_str}[/green]")
        except Exception as e:
            console.print(f"[yellow]Dynamic allocation init failed: {e}[/yellow]")

    # V3: Start WebSocket position monitor
    ws_monitor = None
    if config.WEBSOCKET_MONITORING and PositionMonitor:
        ws_monitor = PositionMonitor(risk)
        ws_monitor.set_close_callback(
            lambda symbol, reason: _handle_ws_close(symbol, reason, risk, ws_monitor)
        )
        # Subscribe to existing open positions
        for symbol in risk.open_trades:
            ws_monitor.subscribe(symbol)
        ws_monitor.start()
        console.print("[green]WebSocket position monitor started.[/green]")

    # V3: Start web dashboard
    if config.WEB_DASHBOARD_ENABLED:
        try:
            from web_dashboard import start_web_dashboard
            start_web_dashboard()
            console.print(f"[green]Web dashboard: http://localhost:{config.WEB_DASHBOARD_PORT}[/green]")
        except Exception as e:
            console.print(f"[yellow]Web dashboard failed to start: {e}[/yellow]")

    # V4: Print VIX status
    if config.VIX_RISK_SCALING_ENABLED:
        try:
            from risk import get_vix_level, get_vix_risk_scalar as _scalar
            vix = get_vix_level()
            scalar = _scalar()
            console.print(f"[green]VIX: {vix:.1f} (risk scalar: {scalar:.0%})[/green]")
        except Exception as e:
            console.print(f"[yellow]VIX fetch failed: {e}[/yellow]")

    # V4: Print MTF/News status
    if config.MTF_CONFIRMATION_ENABLED and mtf_confirmer:
        console.print("[green]Multi-timeframe confirmation enabled.[/green]")
    if config.NEWS_FILTER_ENABLED and news_filter:
        console.print("[green]News sentiment filter enabled.[/green]")

    # Load filters
    console.print("Loading earnings calendar...")
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Earnings filter load failed: {e} (continuing without)[/yellow]")

    console.print("Loading correlation data...")
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Correlation filter load failed: {e} (continuing without)[/yellow]")

    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_analytics_update = now_et()
    last_day = now_et().date()
    eod_summary_printed = False
    current_analytics = None
    gap_candidates_found = False
    gap_first_candle_recorded = False
    allocation_updated_today = False
    last_sunday_task = None  # Track Sunday midnight ML/optimize/pairs runs
    news_cache_cleared_today = False  # V4: track news cache clear
    last_vix_alert_level = None  # V4: track VIX alert to avoid spam

    # Feature flags summary
    features = []
    if config.USE_ML_FILTER:
        features.append("ML")
    if config.DYNAMIC_ALLOCATION:
        features.append("Alloc")
    if config.WEBSOCKET_MONITORING:
        features.append("WS")
    if config.ALLOW_SHORT:
        features.append("Short")
    if config.GAP_GO_ENABLED:
        features.append("Gap")
    if config.USE_RS_FILTER:
        features.append("RS")
    if config.TELEGRAM_ENABLED:
        features.append("TG")
    if config.WEB_DASHBOARD_ENABLED:
        features.append("Web")
    if config.AUTO_OPTIMIZE:
        features.append("Opt")
    if config.MTF_CONFIRMATION_ENABLED:
        features.append("MTF")
    if config.VIX_RISK_SCALING_ENABLED:
        features.append("VIX")
    if config.NEWS_FILTER_ENABLED:
        features.append("News")
    if config.SECTOR_ROTATION_ENABLED:
        features.append("Sector")
    if config.PAIRS_TRADING_ENABLED:
        features.append("Pairs")
    if config.ADVANCED_EXITS_ENABLED:
        features.append("AdvExit")

    strategies_list = ["ORB", "VWAP"]
    if config.ALLOW_MOMENTUM:
        strategies_list.append("MOMENTUM")
    if config.GAP_GO_ENABLED:
        strategies_list.append("GAP_GO")
    if config.SECTOR_ROTATION_ENABLED:
        strategies_list.append("SECTOR_ROT")
    if config.PAIRS_TRADING_ENABLED:
        strategies_list.append("PAIRS")

    features_str = ", ".join(features) if features else "none"
    console.print(f"\n[bold green]Bot V4 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: {' + '.join(strategies_list)}[/dim]")
    console.print(f"[dim]Features: {features_str}[/dim]\n")

    # Stop logging to terminal — dashboard takes over
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(risk, regime_detector.regime, start_time, now_et(), last_scan,
                          len(config.SYMBOLS), current_analytics, get_excluded_count(),
                          risk.get_strategy_weights()),
            console=console,
            refresh_per_second=0.2,
            transient=False,
        ) as live:
            while True:
                current = now_et()
                current_time = current.time()

                # Daily reset
                if current.date() != last_day:
                    logger.info("New trading day — resetting state")
                    orb.reset_daily()
                    vwap.reset_daily()
                    momentum.reset_daily()
                    if gap_go:
                        gap_go.reset_daily()
                    if sector_rotation:
                        sector_rotation.reset_daily()
                    gap_candidates_found = False
                    gap_first_candle_recorded = False
                    allocation_updated_today = False
                    news_cache_cleared_today = False

                    if rs_tracker:
                        rs_tracker.clear_cache()

                    try:
                        account = get_account()
                        risk.reset_daily(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to reset daily: {e}")
                    last_day = current.date()
                    eod_summary_printed = False

                    # Refresh daily caches
                    try:
                        load_earnings_cache(config.SYMBOLS)
                        load_correlation_cache(config.SYMBOLS)
                    except Exception as e:
                        logger.error(f"Failed to refresh daily caches: {e}")

                # V4: News cache clear at 9:25 AM
                if (not news_cache_cleared_today
                        and news_filter
                        and current_time >= config.NEWS_CACHE_CLEAR_TIME):
                    news_filter.clear_cache()
                    news_cache_cleared_today = True

                # Sunday midnight tasks (ML retrain + optimization + pairs discovery)
                if (current.weekday() == 6 and current_time >= time(0, 0)
                        and last_sunday_task != current.date()):
                    last_sunday_task = current.date()

                    if config.USE_ML_FILTER and ml_filter:
                        try:
                            logger.info("Sunday: retraining ML models...")
                            ml_filter.retrain_all()
                            if notifications and config.TELEGRAM_ENABLED:
                                notifications.notify_ml_retrain(ml_filter._active)
                        except Exception as e:
                            logger.error(f"ML retrain failed: {e}")

                    if config.AUTO_OPTIMIZE:
                        try:
                            logger.info("Sunday: running parameter optimization...")
                            from optimizer import weekly_optimization
                            weekly_optimization()
                        except Exception as e:
                            logger.error(f"Parameter optimization failed: {e}")

                    # V4: Pairs trading weekly discovery + revalidation
                    if pairs_trading and config.PAIRS_TRADING_ENABLED:
                        try:
                            logger.info("Sunday: discovering cointegrated pairs...")
                            pairs_trading.discover_pairs(config.STANDARD_SYMBOLS, current)
                            pairs_trading.revalidate_pairs()
                        except Exception as e:
                            logger.error(f"Pairs discovery failed: {e}")

                # Update regime
                regime = regime_detector.update(current)

                # V3: Update RS tracker periodically
                if rs_tracker and is_market_hours(current_time):
                    try:
                        rs_tracker.update(current)
                    except Exception as e:
                        logger.warning(f"RS tracker update failed: {e}")

                # V3: Daily capital allocation update at 9:00 AM
                if (config.DYNAMIC_ALLOCATION
                        and not allocation_updated_today
                        and current_time >= config.ALLOCATION_RECALC_TIME):
                    try:
                        risk.update_strategy_weights()
                        allocation_updated_today = True
                    except Exception as e:
                        logger.error(f"Failed to update strategy weights: {e}")

                # V4: VIX alert notifications
                if config.VIX_RISK_SCALING_ENABLED and notifications and config.TELEGRAM_ENABLED:
                    try:
                        from risk import get_vix_level as _gvl
                        vix = _gvl()
                        scalar = get_vix_risk_scalar()
                        alert_level = None
                        if vix >= 40:
                            alert_level = "extreme"
                        elif vix >= 30:
                            alert_level = "high"
                        elif vix >= 25:
                            alert_level = "elevated"
                        if alert_level and alert_level != last_vix_alert_level:
                            notifications.notify_vix_alert(vix, scalar)
                            last_vix_alert_level = alert_level
                        elif vix < 25:
                            last_vix_alert_level = None
                    except Exception:
                        pass

                # Market hours logic
                if is_market_hours(current_time):

                    # V3: Gap & Go pre-market scan at 9:00 AM
                    if (gap_go and not gap_candidates_found
                            and current_time >= config.GAP_PREMARKET_SCAN_TIME):
                        try:
                            gap_go.find_gap_candidates(config.SYMBOLS, current)
                            gap_candidates_found = True
                            logger.info(f"Gap & Go: found {len(gap_go.candidates)} candidates")
                        except Exception as e:
                            logger.error(f"Gap & Go pre-market scan failed: {e}")

                    # ORB recording period (9:30-10:00)
                    if is_orb_recording_period(current_time):
                        if not orb.ranges_recorded and current_time >= time(9, 55):
                            orb.record_opening_ranges(config.STANDARD_SYMBOLS, current)

                        # V3: Gap & Go first candle recording at 9:45
                        if (gap_go and not gap_first_candle_recorded
                                and current_time >= time(9, 45)):
                            try:
                                gap_go.record_first_candle(current)
                                gap_first_candle_recorded = True
                            except Exception as e:
                                logger.error(f"Gap & Go first candle record failed: {e}")

                    # Trading hours (10:00-15:45)
                    if is_trading_hours(current_time):
                        signals = []

                        # Run strategies based on regime
                        if regime in ("BULLISH", "UNKNOWN"):
                            orb_signals = orb.scan(config.STANDARD_SYMBOLS, current)
                            signals.extend(orb_signals)

                        # V3: ORB short signals in bearish regime
                        if config.ALLOW_SHORT and regime in ("BEARISH", "UNKNOWN"):
                            orb_short_signals = orb.scan(config.STANDARD_SYMBOLS, current, regime=regime)
                            for sig in orb_short_signals:
                                if sig.side == "sell" and sig.symbol not in [s.symbol for s in signals]:
                                    signals.append(sig)

                        # VWAP runs on all symbols in all regimes
                        vwap_signals = vwap.scan(config.SYMBOLS, current, regime)
                        signals.extend(vwap_signals)

                        # Momentum: once daily at 10:30 AM
                        if (config.ALLOW_MOMENTUM
                            and current_time >= config.MOMENTUM_SCAN_TIME
                            and not momentum.scanned_today):
                            mom_signals = momentum.scan(
                                config.STANDARD_SYMBOLS, current, regime_detector
                            )
                            signals.extend(mom_signals)

                        # V3: Gap & Go scan (9:45-11:30)
                        if (gap_go and gap_first_candle_recorded
                                and current_time >= config.GAP_ENTRY_TIME
                                and current_time < config.GAP_EXIT_TIME):
                            try:
                                gap_signals = gap_go.scan(current)
                                signals.extend(gap_signals)
                            except Exception as e:
                                logger.error(f"Gap & Go scan failed: {e}")

                        # V4: Sector Rotation scan at 10:30 AM
                        if (sector_rotation and config.SECTOR_ROTATION_ENABLED
                                and current_time >= config.SECTOR_SCAN_TIME
                                and not sector_rotation.scanned_today):
                            try:
                                sector_signals = sector_rotation.scan(
                                    config.SECTOR_ROTATION_ETFS, current, regime_detector
                                )
                                signals.extend(sector_signals)
                            except Exception as e:
                                logger.error(f"Sector Rotation scan failed: {e}")

                        # V4: Pairs Trading intraday scan
                        if pairs_trading and config.PAIRS_TRADING_ENABLED and pairs_trading.pairs:
                            try:
                                pair_signals = pairs_trading.scan(
                                    config.STANDARD_SYMBOLS, current
                                )
                                signals.extend(pair_signals)
                            except Exception as e:
                                logger.error(f"Pairs Trading scan failed: {e}")

                        # Process signals
                        if signals:
                            process_signals(
                                signals, risk, regime, current,
                                rs_tracker=rs_tracker,
                                ws_monitor=ws_monitor,
                            )

                        # V4: Advanced exit checks
                        if exit_mgr:
                            try:
                                exit_actions = exit_mgr.check_exits(risk, current)
                                for action in exit_actions:
                                    if ws_monitor and action["symbol"] not in risk.open_trades:
                                        ws_monitor.unsubscribe(action["symbol"])
                            except Exception as e:
                                logger.error(f"Exit manager check failed: {e}")

                        # Check VWAP time stops
                        expired = check_vwap_time_stops(risk.open_trades, current)
                        for symbol in expired:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="time_stop")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                        # Check momentum max hold
                        expired_mom = check_momentum_max_hold(risk.open_trades, current)
                        for symbol in expired_mom:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="max_hold")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                        # V3: Gap & Go time stop at 11:30
                        if gap_go and current_time >= config.GAP_EXIT_TIME:
                            closed_gaps = close_gap_go_positions(risk.open_trades, current)
                            for symbol in closed_gaps:
                                if symbol in risk.open_trades:
                                    risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="gap_time_stop")
                                    if ws_monitor:
                                        ws_monitor.unsubscribe(symbol)

                        # V4: Sector rotation max hold
                        if sector_rotation and config.SECTOR_ROTATION_ENABLED:
                            expired_sector = check_sector_max_hold(risk.open_trades, current)
                            for symbol in expired_sector:
                                if symbol in risk.open_trades:
                                    risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="sector_max_hold")
                                    if ws_monitor:
                                        ws_monitor.unsubscribe(symbol)
                                    if symbol in sector_rotation.held_sectors:
                                        del sector_rotation.held_sectors[symbol]

                        # V4: Pairs max hold + z-score exits
                        if pairs_trading and config.PAIRS_TRADING_ENABLED:
                            expired_pairs = check_pairs_max_hold(risk.open_trades, current)
                            for symbol in expired_pairs:
                                if symbol in risk.open_trades:
                                    risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="pairs_max_hold")
                                    if ws_monitor:
                                        ws_monitor.unsubscribe(symbol)

                            # Z-score convergence/divergence exits
                            try:
                                zscore_exits = pairs_trading.check_pair_exits(risk.open_trades, current)
                                for symbol in zscore_exits:
                                    if symbol in risk.open_trades:
                                        from data import get_snapshot
                                        try:
                                            snap = get_snapshot(symbol)
                                            exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else risk.open_trades[symbol].entry_price
                                        except Exception:
                                            exit_price = risk.open_trades[symbol].entry_price
                                        close_position(symbol, reason="pairs_zscore_exit")
                                        risk.close_trade(symbol, exit_price, current, exit_reason="pairs_zscore_exit")
                                        if ws_monitor:
                                            ws_monitor.unsubscribe(symbol)
                            except Exception as e:
                                logger.error(f"Pairs z-score exit check failed: {e}")

                    # ORB exit time (15:45)
                    if current_time >= config.ORB_EXIT_TIME:
                        closed = close_orb_positions(risk.open_trades, current)
                        for symbol in closed:
                            if symbol in risk.open_trades:
                                risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="eod_close")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)

                    # Sync with broker (use polling if WS not connected)
                    if not (ws_monitor and ws_monitor.is_connected):
                        sync_positions_with_broker(risk, current, ws_monitor)
                    else:
                        # Still update equity from account
                        try:
                            account = get_account()
                            risk.update_equity(float(account.equity), float(account.cash))
                        except Exception as e:
                            logger.error(f"Failed to update account: {e}")

                    # Check circuit breaker
                    if risk.check_circuit_breaker():
                        if notifications and config.TELEGRAM_ENABLED:
                            try:
                                notifications.notify_circuit_breaker(risk.day_pnl)
                            except Exception:
                                pass

                    last_scan = current

                # EOD summary + daily snapshot at 16:15
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary)
                    logger.info(f"Day summary: {summary}")

                    # V3: Telegram daily summary
                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_daily_summary(summary, risk.current_equity)
                        except Exception as e:
                            logger.warning(f"Telegram daily summary failed: {e}")

                    # Save daily snapshot to DB
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

                    eod_summary_printed = True

                # Save state to DB periodically
                if (current - last_state_save).total_seconds() >= config.STATE_SAVE_INTERVAL_SEC:
                    try:
                        database.save_open_positions(risk.open_trades)
                    except Exception as e:
                        logger.error(f"Failed to save state: {e}")
                    last_state_save = current

                # Update analytics every 5 minutes
                if (current - last_analytics_update).total_seconds() >= 300:
                    try:
                        current_analytics = analytics_mod.compute_analytics()

                        # V3: Drawdown warning
                        if current_analytics and notifications and config.TELEGRAM_ENABLED:
                            dd = current_analytics.get("max_drawdown", 0)
                            if dd > 0.05:
                                try:
                                    notifications.notify_drawdown_warning(dd)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Failed to compute analytics: {e}")
                    last_analytics_update = current

                # Update dashboard
                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics, get_excluded_count(),
                        risk.get_strategy_weights(),
                    )
                )

                # Sleep until next scan
                time_mod.sleep(config.SCAN_INTERVAL_SEC)

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


def _handle_ws_close(symbol: str, reason: str, risk: RiskManager, ws_monitor):
    """Callback for WebSocket-triggered position closes."""
    if symbol in risk.open_trades:
        trade = risk.open_trades[symbol]
        try:
            close_position(symbol)
        except Exception as e:
            logger.error(f"WS close failed for {symbol}: {e}")
            return
        risk.close_trade(symbol, trade.entry_price, now_et(), exit_reason=reason)
        ws_monitor.unsubscribe(symbol)

        if notifications and config.TELEGRAM_ENABLED:
            try:
                notifications.notify_trade_closed(trade)
            except Exception:
                pass


# ===================================================================
# Async Mode — behind ASYNC_MODE flag
# ===================================================================

async def _async_scanner(
    risk, regime_detector, orb, vwap, momentum, gap_go,
    sector_rotation, pairs_trading, rs_tracker, ws_monitor,
):
    """Async task: scan for signals and process them every SCAN_INTERVAL_SEC."""
    gap_candidates_found = False
    gap_first_candle_recorded = False

    while True:
        try:
            current = now_et()
            ct = current.time()
            regime = regime_detector.regime

            if is_market_hours(ct):
                # Gap & Go pre-market scan
                if gap_go and not gap_candidates_found and ct >= config.GAP_PREMARKET_SCAN_TIME:
                    try:
                        gap_go.find_gap_candidates(config.SYMBOLS, current)
                        gap_candidates_found = True
                    except Exception as e:
                        logger.error(f"[async] Gap scan failed: {e}")

                # ORB recording
                if is_orb_recording_period(ct):
                    if not orb.ranges_recorded and ct >= time(9, 55):
                        orb.record_opening_ranges(config.STANDARD_SYMBOLS, current)
                    if gap_go and not gap_first_candle_recorded and ct >= time(9, 45):
                        try:
                            gap_go.record_first_candle(current)
                            gap_first_candle_recorded = True
                        except Exception as e:
                            logger.error(f"[async] Gap first candle failed: {e}")

                # Trading hours
                if is_trading_hours(ct):
                    signals = []
                    if regime in ("BULLISH", "UNKNOWN"):
                        signals.extend(orb.scan(config.STANDARD_SYMBOLS, current))
                    if config.ALLOW_SHORT and regime in ("BEARISH", "UNKNOWN"):
                        for sig in orb.scan(config.STANDARD_SYMBOLS, current, regime=regime):
                            if sig.side == "sell" and sig.symbol not in [s.symbol for s in signals]:
                                signals.append(sig)
                    signals.extend(vwap.scan(config.SYMBOLS, current, regime))
                    if (config.ALLOW_MOMENTUM and ct >= config.MOMENTUM_SCAN_TIME
                            and not momentum.scanned_today):
                        signals.extend(momentum.scan(config.STANDARD_SYMBOLS, current, regime_detector))
                    if gap_go and gap_first_candle_recorded and ct >= config.GAP_ENTRY_TIME and ct < config.GAP_EXIT_TIME:
                        try:
                            signals.extend(gap_go.scan(current))
                        except Exception as e:
                            logger.error(f"[async] Gap scan failed: {e}")
                    if sector_rotation and config.SECTOR_ROTATION_ENABLED and ct >= config.SECTOR_SCAN_TIME and not sector_rotation.scanned_today:
                        try:
                            signals.extend(sector_rotation.scan(config.SECTOR_ROTATION_ETFS, current, regime_detector))
                        except Exception as e:
                            logger.error(f"[async] Sector scan failed: {e}")
                    if pairs_trading and config.PAIRS_TRADING_ENABLED and pairs_trading.pairs:
                        try:
                            signals.extend(pairs_trading.scan(config.STANDARD_SYMBOLS, current))
                        except Exception as e:
                            logger.error(f"[async] Pairs scan failed: {e}")

                    if signals:
                        process_signals(signals, risk, regime, current,
                                        rs_tracker=rs_tracker, ws_monitor=ws_monitor)

            # Daily reset for gap tracking
            if ct < config.MARKET_OPEN:
                gap_candidates_found = False
                gap_first_candle_recorded = False

        except Exception as e:
            logger.error(f"[async] Scanner error: {e}", exc_info=True)

        await asyncio.sleep(config.SCAN_INTERVAL_SEC)


async def _async_exit_checker(risk, exit_mgr, ws_monitor, sector_rotation, pairs_trading):
    """Async task: check exit conditions every 30 seconds."""
    while True:
        try:
            current = now_et()
            ct = current.time()

            if is_trading_hours(ct):
                # Advanced exits
                if exit_mgr:
                    try:
                        exit_actions = exit_mgr.check_exits(risk, current)
                        for action in exit_actions:
                            if ws_monitor and action["symbol"] not in risk.open_trades:
                                ws_monitor.unsubscribe(action["symbol"])
                    except Exception as e:
                        logger.error(f"[async] Exit check failed: {e}")

                # VWAP time stops
                expired = check_vwap_time_stops(risk.open_trades, current)
                for symbol in expired:
                    if symbol in risk.open_trades:
                        risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="time_stop")
                        if ws_monitor:
                            ws_monitor.unsubscribe(symbol)

                # Momentum max hold
                for symbol in check_momentum_max_hold(risk.open_trades, current):
                    if symbol in risk.open_trades:
                        risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="max_hold")
                        if ws_monitor:
                            ws_monitor.unsubscribe(symbol)

                # Gap & Go time stop
                if ct >= config.GAP_EXIT_TIME:
                    for symbol in close_gap_go_positions(risk.open_trades, current):
                        if symbol in risk.open_trades:
                            risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="gap_time_stop")
                            if ws_monitor:
                                ws_monitor.unsubscribe(symbol)

                # Sector max hold
                if sector_rotation and config.SECTOR_ROTATION_ENABLED:
                    for symbol in check_sector_max_hold(risk.open_trades, current):
                        if symbol in risk.open_trades:
                            risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="sector_max_hold")
                            if ws_monitor:
                                ws_monitor.unsubscribe(symbol)
                            if symbol in sector_rotation.held_sectors:
                                del sector_rotation.held_sectors[symbol]

                # Pairs max hold + z-score exits
                if pairs_trading and config.PAIRS_TRADING_ENABLED:
                    for symbol in check_pairs_max_hold(risk.open_trades, current):
                        if symbol in risk.open_trades:
                            risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="pairs_max_hold")
                            if ws_monitor:
                                ws_monitor.unsubscribe(symbol)
                    try:
                        for symbol in pairs_trading.check_pair_exits(risk.open_trades, current):
                            if symbol in risk.open_trades:
                                from data import get_snapshot
                                try:
                                    snap = get_snapshot(symbol)
                                    exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else risk.open_trades[symbol].entry_price
                                except Exception:
                                    exit_price = risk.open_trades[symbol].entry_price
                                close_position(symbol, reason="pairs_zscore_exit")
                                risk.close_trade(symbol, exit_price, current, exit_reason="pairs_zscore_exit")
                                if ws_monitor:
                                    ws_monitor.unsubscribe(symbol)
                    except Exception as e:
                        logger.error(f"[async] Pairs z-score exit failed: {e}")

                # ORB EOD close
                if ct >= config.ORB_EXIT_TIME:
                    for symbol in close_orb_positions(risk.open_trades, current):
                        if symbol in risk.open_trades:
                            risk.close_trade(symbol, risk.open_trades[symbol].entry_price, current, exit_reason="eod_close")
                            if ws_monitor:
                                ws_monitor.unsubscribe(symbol)

        except Exception as e:
            logger.error(f"[async] Exit checker error: {e}", exc_info=True)

        await asyncio.sleep(30)


async def _async_broker_sync(risk, ws_monitor):
    """Async task: sync positions with broker every 60 seconds."""
    while True:
        try:
            current = now_et()
            if is_market_hours(current.time()):
                if not (ws_monitor and ws_monitor.is_connected):
                    sync_positions_with_broker(risk, current, ws_monitor)
                else:
                    try:
                        account = get_account()
                        risk.update_equity(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"[async] Account update failed: {e}")

                if risk.check_circuit_breaker():
                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_circuit_breaker(risk.day_pnl)
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"[async] Broker sync error: {e}", exc_info=True)

        await asyncio.sleep(60)


async def _async_state_saver(risk):
    """Async task: save open positions to DB every STATE_SAVE_INTERVAL_SEC."""
    while True:
        await asyncio.sleep(config.STATE_SAVE_INTERVAL_SEC)
        try:
            database.save_open_positions(risk.open_trades)
        except Exception as e:
            logger.error(f"[async] State save failed: {e}")


async def _async_daily_tasks(
    risk, regime_detector, orb, vwap, momentum, gap_go,
    sector_rotation, pairs_trading, rs_tracker,
):
    """Async task: handle daily resets, VIX alerts, regime updates, analytics."""
    last_day = now_et().date()
    allocation_updated_today = False
    news_cache_cleared_today = False
    last_sunday_task = None
    last_vix_alert_level = None
    eod_summary_printed = False
    current_analytics = None

    while True:
        try:
            current = now_et()
            ct = current.time()

            # Daily reset
            if current.date() != last_day:
                logger.info("[async] New trading day — resetting state")
                orb.reset_daily()
                vwap.reset_daily()
                momentum.reset_daily()
                if gap_go:
                    gap_go.reset_daily()
                if sector_rotation:
                    sector_rotation.reset_daily()
                allocation_updated_today = False
                news_cache_cleared_today = False
                eod_summary_printed = False

                if rs_tracker:
                    rs_tracker.clear_cache()

                try:
                    account = get_account()
                    risk.reset_daily(float(account.equity), float(account.cash))
                except Exception as e:
                    logger.error(f"[async] Daily reset failed: {e}")
                last_day = current.date()

                try:
                    load_earnings_cache(config.SYMBOLS)
                    load_correlation_cache(config.SYMBOLS)
                except Exception as e:
                    logger.error(f"[async] Cache refresh failed: {e}")

            # News cache clear
            if (not news_cache_cleared_today and news_filter
                    and ct >= config.NEWS_CACHE_CLEAR_TIME):
                news_filter.clear_cache()
                news_cache_cleared_today = True

            # Sunday tasks
            if (current.weekday() == 6 and ct >= time(0, 0)
                    and last_sunday_task != current.date()):
                last_sunday_task = current.date()
                if config.USE_ML_FILTER and ml_filter:
                    try:
                        ml_filter.retrain_all()
                    except Exception as e:
                        logger.error(f"[async] ML retrain failed: {e}")
                if config.AUTO_OPTIMIZE:
                    try:
                        from optimizer import weekly_optimization
                        weekly_optimization()
                    except Exception as e:
                        logger.error(f"[async] Optimization failed: {e}")
                if pairs_trading and config.PAIRS_TRADING_ENABLED:
                    try:
                        pairs_trading.discover_pairs(config.STANDARD_SYMBOLS, current)
                        pairs_trading.revalidate_pairs()
                    except Exception as e:
                        logger.error(f"[async] Pairs discovery failed: {e}")

            # Regime update
            regime_detector.update(current)

            # RS tracker
            if rs_tracker and is_market_hours(ct):
                try:
                    rs_tracker.update(current)
                except Exception as e:
                    logger.warning(f"[async] RS update failed: {e}")

            # Capital allocation
            if (config.DYNAMIC_ALLOCATION and not allocation_updated_today
                    and ct >= config.ALLOCATION_RECALC_TIME):
                try:
                    risk.update_strategy_weights()
                    allocation_updated_today = True
                except Exception as e:
                    logger.error(f"[async] Allocation update failed: {e}")

            # VIX alerts
            if config.VIX_RISK_SCALING_ENABLED and notifications and config.TELEGRAM_ENABLED:
                try:
                    from risk import get_vix_level as _gvl
                    vix = _gvl()
                    alert_level = None
                    if vix >= 40:
                        alert_level = "extreme"
                    elif vix >= 30:
                        alert_level = "high"
                    elif vix >= 25:
                        alert_level = "elevated"
                    if alert_level and alert_level != last_vix_alert_level:
                        notifications.notify_vix_alert(vix, get_vix_risk_scalar())
                        last_vix_alert_level = alert_level
                    elif vix < 25:
                        last_vix_alert_level = None
                except Exception:
                    pass

            # EOD summary
            if ct >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                summary = risk.get_day_summary()
                print_day_summary(summary)
                logger.info(f"Day summary: {summary}")
                if notifications and config.TELEGRAM_ENABLED:
                    try:
                        notifications.notify_daily_summary(summary, risk.current_equity)
                    except Exception:
                        pass
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
                    logger.error(f"[async] Daily snapshot failed: {e}")
                eod_summary_printed = True

            # Analytics
            try:
                current_analytics = analytics_mod.compute_analytics()
                if current_analytics and notifications and config.TELEGRAM_ENABLED:
                    dd = current_analytics.get("max_drawdown", 0)
                    if dd > 0.05:
                        try:
                            notifications.notify_drawdown_warning(dd)
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[async] Daily tasks error: {e}", exc_info=True)

        await asyncio.sleep(60)


async def _async_dashboard_updater(risk, regime_detector, start_time, live):
    """Async task: refresh the Rich dashboard."""
    while True:
        try:
            current = now_et()
            live.update(
                build_dashboard(
                    risk, regime_detector.regime, start_time, current, current,
                    len(config.SYMBOLS), None, get_excluded_count(),
                    risk.get_strategy_weights(),
                )
            )
        except Exception as e:
            logger.error(f"[async] Dashboard update failed: {e}")
        await asyncio.sleep(5)


async def async_main():
    """Async entry point — runs all tasks concurrently under supervision."""
    import asyncio as _asyncio
    from supervisor import TaskSupervisor

    console.print("[bold cyan]Starting Algo Trading Bot V4 (ASYNC MODE)...[/bold cyan]\n")

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # Initialize strategies (same as sync)
    regime_detector = MarketRegime()
    orb = ORBStrategy()
    vwap = VWAPStrategy()
    momentum = MomentumStrategy()
    gap_go = GapGoStrategy() if config.GAP_GO_ENABLED else None
    risk = RiskManager()

    sector_rotation = None
    if config.SECTOR_ROTATION_ENABLED and SectorRotationStrategy:
        sector_rotation = SectorRotationStrategy()
    pairs_trading = None
    if config.PAIRS_TRADING_ENABLED and PairsTradingStrategy:
        pairs_trading = PairsTradingStrategy()
    exit_mgr = None
    if config.ADVANCED_EXITS_ENABLED and ExitManager:
        exit_mgr = ExitManager()

    risk.reset_daily(info["equity"], info["cash"])
    risk.load_from_db()

    rs_tracker = None
    if config.USE_RS_FILTER and RelativeStrengthTracker:
        rs_tracker = RelativeStrengthTracker()

    if config.USE_ML_FILTER and ml_filter:
        try:
            ml_filter.load_models()
        except Exception:
            pass

    if config.DYNAMIC_ALLOCATION:
        try:
            risk.update_strategy_weights()
        except Exception:
            pass

    # WS monitor (still threaded — it uses its own asyncio loop)
    ws_monitor = None
    if config.WEBSOCKET_MONITORING and PositionMonitor:
        ws_monitor = PositionMonitor(risk)
        ws_monitor.set_close_callback(
            lambda symbol, reason: _handle_ws_close(symbol, reason, risk, ws_monitor)
        )
        for symbol in risk.open_trades:
            ws_monitor.subscribe(symbol)
        ws_monitor.start()

    # Web dashboard
    if config.WEB_DASHBOARD_ENABLED:
        try:
            from web_dashboard import start_web_dashboard
            start_web_dashboard()
        except Exception:
            pass

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception:
        pass
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception:
        pass

    start_time = now_et()
    logging.getLogger().removeHandler(_stream_handler)

    console.print(f"\n[bold green]Bot V4 (ASYNC) is running. Press Ctrl+C to stop.[/bold green]\n")

    supervisor = TaskSupervisor()

    # Optional crash notification via Telegram
    if notifications and config.TELEGRAM_ENABLED:
        async def _crash_notify(name, exc, count):
            try:
                notifications.send_message(f"Task '{name}' crashed (#{count}): {exc}")
            except Exception:
                pass
        supervisor.set_notify(_crash_notify)

    try:
        with Live(
            build_dashboard(risk, regime_detector.regime, start_time, now_et(), None,
                            len(config.SYMBOLS), None, get_excluded_count(),
                            risk.get_strategy_weights()),
            console=console, refresh_per_second=0.2, transient=False,
        ) as live:
            # Launch all supervised tasks
            await supervisor.launch(
                "scanner", _async_scanner,
                risk, regime_detector, orb, vwap, momentum, gap_go,
                sector_rotation, pairs_trading, rs_tracker, ws_monitor,
            )
            await supervisor.launch(
                "exit_checker", _async_exit_checker,
                risk, exit_mgr, ws_monitor, sector_rotation, pairs_trading,
            )
            await supervisor.launch(
                "broker_sync", _async_broker_sync,
                risk, ws_monitor,
            )
            await supervisor.launch(
                "state_saver", _async_state_saver,
                risk,
            )
            await supervisor.launch(
                "daily_tasks", _async_daily_tasks,
                risk, regime_detector, orb, vwap, momentum, gap_go,
                sector_rotation, pairs_trading, rs_tracker,
            )
            await supervisor.launch(
                "dashboard", _async_dashboard_updater,
                risk, regime_detector, start_time, live,
            )

            # Run forever until cancelled
            try:
                await _asyncio.Event().wait()
            except _asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down async tasks...[/yellow]")
    finally:
        await supervisor.stop_all()
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")


if __name__ == "__main__":
    if config.ASYNC_MODE:
        import asyncio
        asyncio.run(async_main())
    else:
        main()
