"""V10 Engine — Daily reset, weekly tasks, and EOD close logic.

Extracts the daily/weekly/EOD operations from main.py into reusable functions.
"""

import logging
from datetime import datetime, time

import config
import database
from data import get_account, get_snapshot
from execution import close_position
from earnings import load_earnings_cache
from correlation import load_correlation_cache

logger = logging.getLogger(__name__)


def daily_reset(
    risk,
    stat_mr, kalman_pairs, micro_mom, vwap_strategy,
    pnl_lock, beta_neutral,
    orb_strategy=None, pead_strategy=None,
    overnight_manager=None, news_sentiment=None, llm_scorer=None,
    tiered_cb=None,
):
    """Reset all strategies and risk engines for a new trading day."""
    logger.info("New trading day -- resetting state")

    stat_mr.reset_daily()
    micro_mom.reset_daily()
    vwap_strategy.reset_daily()
    pnl_lock.reset_daily()
    beta_neutral.reset_daily()

    if tiered_cb:
        tiered_cb.reset_daily()

    if orb_strategy:
        orb_strategy.reset_daily()
        orb_strategy._ranges_recorded_today = False

    if news_sentiment:
        try:
            news_sentiment.clear_daily_cache()
        except Exception:
            pass

    if llm_scorer:
        try:
            llm_scorer.reset_daily()
        except Exception:
            pass

    if pead_strategy:
        try:
            pead_strategy.reset_daily()
        except Exception:
            pass

    if overnight_manager:
        try:
            overnight_manager.reset_daily()
        except Exception:
            pass

    # Reset risk manager with fresh account data
    try:
        account = get_account()
        risk.reset_daily(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to reset daily account: {e}")

    # Refresh daily caches
    try:
        load_earnings_cache(config.SYMBOLS)
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        logger.error(f"Failed to refresh daily caches: {e}")


def weekly_tasks(current: datetime, kalman_pairs, param_optimizer=None, walk_forward=None):
    """Run weekly tasks (Sunday): pair selection, parameter optimization, walk-forward."""
    try:
        logger.info("Sunday: selecting cointegrated pairs...")
        kalman_pairs.select_pairs_weekly(current)
    except Exception as e:
        logger.error(f"Weekly pair selection failed: {e}")

    if param_optimizer:
        try:
            results = param_optimizer.optimize_all()
            if results:
                logger.info(f"Parameter optimization: {len(results)} strategies updated")
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")

    if walk_forward:
        try:
            results = walk_forward.run_weekly_validation()
            if results:
                logger.info(f"Walk-forward validation: {len(results)} strategies tested")
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")


def eod_close(
    current: datetime,
    risk,
    ws_monitor=None,
    overnight_manager=None,
    regime: str = "UNKNOWN",
):
    """Close all day-hold positions at EOD. Respects overnight holds (BUG-034)."""
    # V10 BUG-034: Track overnight hold symbols to skip in EOD close
    overnight_hold_symbols: set[str] = set()
    if overnight_manager:
        try:
            holds = overnight_manager.select_overnight_holds(risk.open_trades, regime)
            if holds:
                overnight_hold_symbols = {h.symbol for h in holds}
                logger.info(f"Overnight holds selected: {list(overnight_hold_symbols)}")
        except Exception as e:
            logger.error(f"Overnight hold selection failed: {e}")

    day_strategies = ("MICRO_MOM", "BETA_HEDGE", "ORB", "VWAP", "STAT_MR", "KALMAN_PAIRS")
    closed_count = 0

    for symbol in list(risk.open_trades.keys()):
        trade = risk.open_trades[symbol]

        if symbol in overnight_hold_symbols:
            continue

        if trade.hold_type == "day" and trade.strategy in day_strategies:
            try:
                close_position(symbol, reason="eod_close")
                try:
                    snap = get_snapshot(symbol)
                    ep = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    ep = trade.entry_price
                risk.close_trade(symbol, ep, current, exit_reason="eod_close")
                if ws_monitor:
                    ws_monitor.unsubscribe(symbol)
                closed_count += 1
            except Exception as e:
                logger.error(f"EOD close failed for {symbol}: {e}")

    if closed_count:
        logger.info(f"EOD: closed {closed_count} day-hold positions")

    return closed_count


def eod_summary(
    current: datetime,
    risk,
    vol_engine,
    consistency_score: float = 0.0,
    alpha_decay_monitor=None,
    adaptive_allocator=None,
    regime_detector=None,
):
    """Generate and log the end-of-day performance summary."""
    try:
        from dashboard import print_day_summary
        summary = risk.get_day_summary()
        print_day_summary(summary, consistency_score)
    except Exception as e:
        logger.error(f"EOD summary failed: {e}")
        summary = {}

    # Notifications
    try:
        import notifications
        if config.TELEGRAM_ENABLED:
            notifications.notify_daily_summary(summary, risk.current_equity)
    except Exception:
        pass

    # Save daily snapshot
    try:
        database.save_daily_snapshot(
            current.strftime("%Y-%m-%d"),
            risk.current_equity,
            risk.current_cash,
            summary.get("day_pnl", 0),
            summary.get("day_pnl_pct", 0),
            summary.get("total_trades", 0),
            summary.get("win_rate", 0),
            summary.get("sharpe_rolling", 0),
        )
    except Exception as e:
        logger.error(f"Failed to save daily snapshot: {e}")

    # Compute and save consistency score
    try:
        from analytics.consistency_score import compute_consistency_score
        pnl_series = database.get_daily_pnl_series(days=30)
        if pnl_series and len(pnl_series) >= 5:
            cs = compute_consistency_score(
                pnl_series,
                summary.get("sharpe_rolling", 0),
                summary.get("max_drawdown", 0),
            )
            database.save_consistency_log(
                current.strftime("%Y-%m-%d"), cs,
                sum(1 for p in pnl_series if p > 0) / max(len(pnl_series), 1),
                summary.get("sharpe_rolling", 0),
                summary.get("max_drawdown", 0),
                vol_engine.last_scalar,
                0.0,
            )
            return cs
    except Exception as e:
        logger.error(f"Consistency score computation failed: {e}")

    return consistency_score
