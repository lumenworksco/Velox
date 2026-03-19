"""V10 Engine — Signal processing pipeline: filtering, sizing, and order submission.

Integrates:
- OMS order tracking (order lifecycle, idempotency, audit trail)
- Event bus (signal.generated, order.submitted, position.opened events)
- Transaction cost model (reject negative-EV trades — PROFIT-GAP-001)
"""

import logging
from datetime import datetime, timedelta

import config
import database
from strategies.base import Signal
from risk import RiskManager, TradeRecord, VolatilityTargetingRiskEngine, DailyPnLLock
from execution import submit_bracket_order, close_position, can_short
from earnings import has_earnings_soon
from correlation import is_too_correlated

logger = logging.getLogger(__name__)

# Lazy-loaded optional modules
try:
    from analytics.intraday_seasonality import IntradaySeasonality
except ImportError:
    IntradaySeasonality = None

# V10 BUG-042: Singleton IntradaySeasonality (created once, not per-signal)
_seasonality_instance = None

def _get_seasonality():
    global _seasonality_instance
    if _seasonality_instance is None and IntradaySeasonality:
        _seasonality_instance = IntradaySeasonality()
    return _seasonality_instance

# OMS integration (fail-open: if OMS not available, orders still go through)
try:
    from oms.order import Order, OrderState
    from oms.order_manager import OrderManager
    from oms.transaction_cost import is_trade_profitable_after_costs
    _OMS_AVAILABLE = True
except ImportError:
    _OMS_AVAILABLE = False

# Event bus integration (fail-open)
try:
    from engine.events import get_event_bus, Event, EventTypes
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False

_notifications = None
_order_manager = None


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


def set_order_manager(mgr):
    """Set the OMS order manager for order tracking."""
    global _order_manager
    _order_manager = mgr


def _emit_event(event_type: str, data: dict, source: str = "signal_processor"):
    """Emit an event on the bus (fail-open)."""
    if _EVENTS_AVAILABLE:
        try:
            bus = get_event_bus()
            bus.publish(Event(event_type, data, source=source))
        except Exception:
            pass


def process_signals(
    signals: list[Signal],
    risk: RiskManager,
    regime: str,
    now: datetime,
    vol_engine: VolatilityTargetingRiskEngine,
    pnl_lock: DailyPnLLock,
    ws_monitor=None,
    news_sentiment=None,
    llm_scorer=None,
    regime_detector=None,
    cross_asset_monitor=None,
    var_monitor=None,
    corr_limiter=None,
):
    """Process signals: check filters, risk, size, and submit orders.

    Filters: position conflict, earnings, correlation, short pre-check,
    news sentiment (soft), LLM scoring (optional), VaR budget, concentration.
    """
    if not pnl_lock.is_trading_allowed():
        logger.info("PnL lock LOSS_HALT active -- skipping all signals")
        for signal in signals:
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, "pnl_halt")
        return

    # Group pairs signals by pair_id for atomic processing
    pair_groups: dict[str, list[Signal]] = {}
    non_pair_signals: list[Signal] = []
    for signal in signals:
        if signal.pair_id:
            pair_groups.setdefault(signal.pair_id, []).append(signal)
        else:
            non_pair_signals.append(signal)

    # Process non-pair signals
    for signal in non_pair_signals:
        _process_single_signal(signal, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                               news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                               var_monitor, corr_limiter)

    # Process pairs atomically (both legs or neither)
    for pair_id, pair_signals in pair_groups.items():
        if len(pair_signals) != 2:
            logger.warning(f"Pair {pair_id} has {len(pair_signals)} signals, skipping")
            continue

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
            # V10 BUG-030: Submit both legs with rollback if second fails
            first_sig, second_sig = pair_signals[0], pair_signals[1]
            _process_single_signal(first_sig, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                                   news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                               var_monitor, corr_limiter)

            if first_sig.symbol in risk.open_trades:
                _process_single_signal(second_sig, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                                       news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                               var_monitor, corr_limiter)

                if second_sig.symbol not in risk.open_trades:
                    logger.warning(f"Pair {pair_id}: second leg {second_sig.symbol} failed, closing first leg {first_sig.symbol}")
                    rollback_ok = False
                    for attempt in range(3):
                        try:
                            close_position(first_sig.symbol, reason="pair_rollback")
                            trade = risk.open_trades.get(first_sig.symbol)
                            if trade:
                                risk.close_trade(first_sig.symbol, trade.entry_price, now, exit_reason="pair_rollback")
                            rollback_ok = True
                            break
                        except Exception as e:
                            logger.error(f"Pair rollback attempt {attempt+1}/3 failed for {first_sig.symbol}: {e}")
                    if not rollback_ok:
                        logger.critical(f"PAIR ROLLBACK FAILED after 3 attempts for {first_sig.symbol} — manual intervention required")
            else:
                database.log_signal(now, second_sig.symbol, second_sig.strategy, second_sig.side, False, "pair_first_leg_failed")
        else:
            for sig in pair_signals:
                database.log_signal(now, sig.symbol, sig.strategy, sig.side, False, "pair_blocked")


def _process_single_signal(
    signal: Signal,
    risk: RiskManager,
    regime: str,
    now: datetime,
    vol_engine: VolatilityTargetingRiskEngine,
    pnl_lock: DailyPnLLock,
    ws_monitor=None,
    news_sentiment=None,
    llm_scorer=None,
    regime_detector=None,
    cross_asset_monitor=None,
    var_monitor=None,
    corr_limiter=None,
):
    """Process a single signal through filters and submit if valid."""
    skip_reason = ""

    # 1. Position conflict
    if signal.symbol in risk.open_trades:
        skip_reason = "already_in_position"
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 2. Earnings filter
    if has_earnings_soon(signal.symbol):
        skip_reason = "earnings_soon"
        logger.info(f"Signal skipped for {signal.symbol}: earnings soon")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 3. Correlation filter (skip for pairs — inherently correlated)
    if signal.strategy != "KALMAN_PAIRS":
        open_symbols = list(risk.open_trades.keys())
        if open_symbols and is_too_correlated(signal.symbol, open_symbols):
            skip_reason = "high_correlation"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return

    # 4. Short selling pre-check
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

    # 5. Risk limits
    allowed, reason = risk.can_open_trade(strategy=signal.strategy)
    if not allowed:
        skip_reason = reason
        logger.info(f"Trade blocked for {signal.symbol}: {reason}")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 5a. V10: Correlation-based concentration check (skip for pairs — inherently correlated)
    if corr_limiter and signal.strategy != "KALMAN_PAIRS":
        try:
            open_symbols = list(risk.open_trades.keys())
            if open_symbols:
                conc = corr_limiter.check_new_position(signal.symbol, open_symbols)
                if conc.too_concentrated:
                    skip_reason = f"concentration_{conc.reason}"
                    logger.info(f"Trade blocked for {signal.symbol}: {skip_reason}")
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    return
        except Exception:
            pass  # Fail-open

    # 5b. News sentiment size adjustment (soft filter)
    news_mult = 1.0
    if news_sentiment:
        try:
            news_mult, news_reason = news_sentiment.get_sentiment_size_mult(signal.symbol)
            if news_mult == 0.0:
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, news_reason)
                return
        except Exception:
            news_mult = 1.0

    # 5c. LLM signal scoring (optional, fail-open)
    llm_mult = 1.0
    if llm_scorer and config.LLM_SCORING_ENABLED:
        try:
            context = {
                'spy_day_return': risk.day_pnl,
                'vix_level': getattr(vol_engine, '_last_vix', 20.0),
            }
            scored = llm_scorer.score_signal(signal, context)
            if scored.score < config.LLM_SCORE_THRESHOLD:
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False,
                                   f'llm_low_score_{scored.score:.2f}')
                return
            if config.LLM_SCORE_SIZE_MULT:
                llm_mult = scored.size_mult
        except Exception:
            llm_mult = 1.0

    # 6. Position sizing via vol-targeting engine
    vol_scalar = vol_engine.last_scalar
    lock_mult = pnl_lock.get_size_multiplier()
    qty = vol_engine.calculate_position_size(
        equity=risk.current_equity,
        entry_price=signal.entry_price,
        stop_price=signal.stop_loss,
        vol_scalar=vol_scalar,
        strategy=signal.strategy,
        side=signal.side,
        pnl_lock_mult=lock_mult,
    )

    # Regime affinity multiplier (fail-open)
    regime_mult = 1.0
    if regime_detector and getattr(config, "HMM_REGIME_ENABLED", False):
        try:
            regime_mult = regime_detector.get_regime_affinity(signal.strategy)
        except Exception:
            regime_mult = 1.0

    # V10 BUG-042: Intraday seasonality multiplier (singleton, fail-open)
    seasonality_mult = 1.0
    if getattr(config, "INTRADAY_SEASONALITY_ENABLED", False) and IntradaySeasonality:
        try:
            seasonality_mult = _get_seasonality().get_window_score(signal.strategy, now)
        except Exception:
            seasonality_mult = 1.0

    # Cross-asset bias multiplier (fail-open)
    cross_asset_mult = 1.0
    if getattr(config, "CROSS_ASSET_ENABLED", False) and cross_asset_monitor:
        try:
            bias = cross_asset_monitor.get_equity_bias()
            if bias < -0.5:
                cross_asset_mult = getattr(config, "CROSS_ASSET_FLIGHT_REDUCTION", 0.30)
        except Exception:
            cross_asset_mult = 1.0

    # V10: VaR risk budget multiplier (fail-open: 1.0)
    var_mult = 1.0
    if var_monitor:
        try:
            var_mult = var_monitor.size_multiplier
        except Exception:
            var_mult = 1.0

    # Apply all multipliers
    qty = int(qty * news_mult * llm_mult * regime_mult * seasonality_mult * cross_asset_mult * var_mult)

    if qty <= 0:
        skip_reason = "position_size_zero"
        logger.info(f"Position size 0 for {signal.symbol}, skipping")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        _emit_event(EventTypes.SIGNAL_FILTERED if _EVENTS_AVAILABLE else "signal.filtered",
                    {"symbol": signal.symbol, "reason": skip_reason})
        return

    # 6b. V10 PROFIT-GAP-001: Transaction cost filter (reject negative-EV trades)
    if _OMS_AVAILABLE and getattr(config, "COST_FILTER_ENABLED", True):
        try:
            # Use strategy-specific win rate if available, else 55%
            strategy_win_rates = getattr(config, "STRATEGY_WIN_RATES", {})
            win_rate = strategy_win_rates.get(signal.strategy, 0.55)

            profitable, cost_details = is_trade_profitable_after_costs(
                entry_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                qty=qty,
                side=signal.side,
                win_rate=win_rate,
            )
            if not profitable:
                skip_reason = f"negative_ev_after_costs_{cost_details['cost_bps']:.0f}bps"
                logger.info(
                    f"Trade {signal.symbol} rejected: EV=${cost_details['expected_value']:.2f} "
                    f"after costs (${cost_details['total_cost']:.2f}, {cost_details['cost_bps']:.1f}bps)"
                )
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                _emit_event(EventTypes.SIGNAL_FILTERED if _EVENTS_AVAILABLE else "signal.filtered",
                            {"symbol": signal.symbol, "reason": skip_reason, **cost_details})
                return
        except Exception as e:
            logger.debug(f"Cost filter failed (proceeding): {e}")

    # 7. Submit bracket order (with OMS tracking if available)
    oms_order = None
    if _OMS_AVAILABLE and _order_manager:
        try:
            oms_order = _order_manager.create_order(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                order_type="bracket",
                qty=qty,
                limit_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                pair_id=getattr(signal, "pair_id", ""),
                idempotency_key=f"{signal.symbol}_{signal.strategy}_{now.strftime('%Y%m%d%H%M%S')}",
            )
        except Exception as e:
            logger.debug(f"OMS order creation failed (proceeding): {e}")

    order_id = submit_bracket_order(signal, qty)

    if order_id is None:
        skip_reason = "order_failed"
        logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        if oms_order and _order_manager:
            _order_manager.update_state(oms_order.oms_id, OrderState.FAILED)
        _emit_event(EventTypes.ORDER_FAILED if _EVENTS_AVAILABLE else "order.failed",
                    {"symbol": signal.symbol, "strategy": signal.strategy})
        return

    # Update OMS with broker order ID
    if oms_order and _order_manager:
        broker_id = order_id if isinstance(order_id, str) else str(order_id[0]) if order_id else ""
        _order_manager.update_state(oms_order.oms_id, OrderState.SUBMITTED, broker_order_id=broker_id)
    _emit_event(EventTypes.ORDER_SUBMITTED if _EVENTS_AVAILABLE else "order.submitted",
                {"symbol": signal.symbol, "strategy": signal.strategy, "qty": qty, "order_id": str(order_id)})

    # 8. Register trade with time stops / max hold
    time_stop = None
    max_hold_date = None
    hold_type = getattr(signal, "hold_type", "day")

    if signal.strategy == "STAT_MR":
        pass  # z-score exits handle it
    elif signal.strategy == "KALMAN_PAIRS":
        max_hold_date = now + timedelta(days=config.PAIRS_MAX_HOLD_DAYS)
    elif signal.strategy == "MICRO_MOM":
        time_stop = now + timedelta(minutes=config.MICRO_MAX_HOLD_MINUTES)
    elif signal.strategy == "BETA_HEDGE":
        hold_type = "day"

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
        pair_id=getattr(signal, "pair_id", ""),
        highest_price_seen=signal.entry_price,
    )
    risk.register_trade(trade)

    if ws_monitor:
        ws_monitor.subscribe(signal.symbol)

    notif = _get_notifications()
    if notif and config.TELEGRAM_ENABLED:
        try:
            notif.notify_trade_opened(trade)
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")

    # Emit position opened event
    _emit_event(EventTypes.POSITION_OPENED if _EVENTS_AVAILABLE else "position.opened", {
        "symbol": signal.symbol,
        "strategy": signal.strategy,
        "side": signal.side,
        "qty": qty,
        "entry_price": signal.entry_price,
        "take_profit": signal.take_profit,
        "stop_loss": signal.stop_loss,
    })
