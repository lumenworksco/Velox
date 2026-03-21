"""V10 Engine — Signal processing pipeline: filtering, sizing, and order submission.

Integrates:
- OMS order tracking (order lifecycle, idempotency, audit trail)
- Event bus (signal.generated, order.submitted, position.opened events)
- Transaction cost model (reject negative-EV trades — PROFIT-GAP-001)
"""

import logging
import threading
import time as _time
from datetime import datetime, timedelta

import config
import database
from strategies.base import Signal
from risk import RiskManager, TradeRecord, VolatilityTargetingRiskEngine, DailyPnLLock
from execution import submit_bracket_order, close_position, can_short
from earnings import has_earnings_soon
from correlation import is_too_correlated

from engine.event_log import log_event, EventType

logger = logging.getLogger(__name__)

# Lazy-loaded optional modules
try:
    from analytics.intraday_seasonality import IntradaySeasonality
except ImportError:
    IntradaySeasonality = None

# WIRE-001: Feature store integration (fail-open)
try:
    from data.feature_store import get_feature_store as _get_feature_store
    _FEATURE_STORE_AVAILABLE = True
except ImportError:
    _FEATURE_STORE_AVAILABLE = False

# WIRE-002: ML model integration (fail-open)
_ml_model = None
_ml_model_load_attempted = False
_ml_model_lock = threading.Lock()

def _get_ml_model():
    """Lazy-load the most recent trained ML model (singleton, fail-open)."""
    global _ml_model, _ml_model_load_attempted
    if _ml_model_load_attempted:
        return _ml_model
    with _ml_model_lock:
        if _ml_model_load_attempted:
            return _ml_model
        _ml_model_load_attempted = True
        try:
            import glob
            import os
            from ml.training import ModelTrainer
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            model_files = sorted(glob.glob(os.path.join(model_dir, "model_*.pkl")))
            if model_files:
                _ml_model = ModelTrainer.load_model(model_files[-1])
                logger.info("WIRE-002: ML model loaded: %s", model_files[-1])
            else:
                logger.info("WIRE-002: No trained ML model found in %s", model_dir)
        except Exception as e:
            logger.info("WIRE-002: ML model load skipped (fail-open): %s", e)
    return _ml_model

# WIRE-003: VPIN integration (fail-open)
try:
    from microstructure.vpin import VPIN as _VPIN_Class
    _VPIN_AVAILABLE = True
except ImportError:
    _VPIN_AVAILABLE = False

_vpin_instances: dict[str, object] = {}
_vpin_lock = threading.Lock()

# V10 BUG-042: Singleton IntradaySeasonality (created once, not per-signal)
# BUG-008: Thread-safe singleton creation via double-checked locking
_seasonality_instance = None
_seasonality_lock = threading.Lock()

def _get_seasonality():
    global _seasonality_instance
    if _seasonality_instance is None and IntradaySeasonality:
        with _seasonality_lock:
            if _seasonality_instance is None:
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

# BUG-024 + RISK-005: Halted/delisted symbol blocklist with Alpaca asset status
_halted_symbols: dict[str, float] = {}  # symbol -> expiry timestamp
_HALT_BLOCKLIST_TTL_SEC = 300  # Block for 5 minutes after detection
_pair_lock = threading.Lock()  # CRIT-009: Lock for atomic pair trade submission

# RISK-005: Alpaca asset status blocklist — refreshed every 5 minutes
_asset_status_blocklist: dict[str, float] = {}  # symbol -> expiry timestamp
_asset_status_lock = threading.Lock()
_ASSET_STATUS_REFRESH_SEC = 300  # Refresh every 5 minutes
_last_asset_status_refresh: float = 0.0


def _refresh_asset_status_blocklist() -> None:
    """RISK-005: Refresh the halted/delisted blocklist from Alpaca asset status API.

    Called automatically when the blocklist is stale (> 5 minutes old).
    Fail-open: if the API call fails, the stale blocklist is kept.
    """
    global _last_asset_status_refresh

    current_ts = _time.time()
    with _asset_status_lock:
        if (current_ts - _last_asset_status_refresh) < _ASSET_STATUS_REFRESH_SEC:
            return  # Still fresh

    try:
        from broker.alpaca_client import get_trading_client

        client = get_trading_client()
        # Fetch the full set of open-position symbols + recently seen symbols
        # to check their tradability status
        symbols_to_check = set(_halted_symbols.keys()) | set(_asset_status_blocklist.keys())

        new_blocklist: dict[str, float] = {}
        expiry = current_ts + _ASSET_STATUS_REFRESH_SEC

        for symbol in symbols_to_check:
            try:
                asset = client.get_asset(symbol)
                if not asset.tradable or asset.status == "inactive":
                    new_blocklist[symbol] = expiry
                    logger.info(f"RISK-005: {symbol} is not tradable (status={asset.status}), blocking")
            except Exception:
                pass  # If we can't check a specific asset, skip it

        with _asset_status_lock:
            _asset_status_blocklist.update(new_blocklist)
            _last_asset_status_refresh = current_ts

        logger.debug(f"RISK-005: Asset status blocklist refreshed, {len(new_blocklist)} blocked symbols")

    except Exception as e:
        logger.warning(f"RISK-005: Asset status refresh failed (fail-open): {e}")
        with _asset_status_lock:
            _last_asset_status_refresh = current_ts  # Don't retry immediately


def _is_asset_blocked(symbol: str) -> bool:
    """RISK-005: Check if a symbol is blocked due to Alpaca asset status (halted/delisted).

    Also checks the symbol on-demand if not in the blocklist, to catch newly halted symbols.
    Fail-open: returns False if the check fails.
    """
    current_ts = _time.time()

    # Trigger periodic refresh
    _refresh_asset_status_blocklist()

    # Check blocklist
    with _asset_status_lock:
        if symbol in _asset_status_blocklist:
            if current_ts < _asset_status_blocklist[symbol]:
                return True
            else:
                del _asset_status_blocklist[symbol]

    # On-demand check for symbols not in blocklist
    try:
        from broker.alpaca_client import get_trading_client

        client = get_trading_client()
        asset = client.get_asset(symbol)

        if not asset.tradable or asset.status == "inactive":
            with _asset_status_lock:
                _asset_status_blocklist[symbol] = current_ts + _ASSET_STATUS_REFRESH_SEC
            logger.warning(f"RISK-005: {symbol} is not tradable (status={asset.status}), blocking")
            return True

    except Exception as e:
        logger.debug(f"RISK-005: Asset status check failed for {symbol} (fail-open): {e}")

    return False


def _is_symbol_halted(symbol: str, now: datetime) -> bool:
    """BUG-024 + RISK-005: Check if a symbol is halted or blocked.

    Combines volume-based halt detection (BUG-024) with Alpaca asset
    status checks (RISK-005). Fail-open on all checks.
    """
    # RISK-005: Check Alpaca asset status blocklist first (fast path)
    if _is_asset_blocked(symbol):
        return True

    # BUG-024: Check volume-based blocklist
    current_ts = _time.time()
    if symbol in _halted_symbols:
        if current_ts < _halted_symbols[symbol]:
            return True
        else:
            del _halted_symbols[symbol]

    try:
        from data import get_intraday_bars
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        lookback = now - timedelta(minutes=15)
        bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=lookback, end=now)
        if bars is None or bars.empty or len(bars) < 5:
            return False

        # Check last 5 bars for zero volume
        last_5_vol = bars["volume"].iloc[-5:]
        if (last_5_vol == 0).all():
            _halted_symbols[symbol] = current_ts + _HALT_BLOCKLIST_TTL_SEC
            logger.warning(f"BUG-024: {symbol} appears halted (zero volume for 5+ bars), blocking for {_HALT_BLOCKLIST_TTL_SEC}s")
            return True
    except Exception as e:
        logger.debug(f"Halt detection check failed for {symbol}: {e}")

    return False


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
        except Exception as e:
            logger.debug(f"Event bus publish failed: {e}")


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
    # CRIT-009: Lock around pair submission to prevent unhedged exposure
    with _pair_lock:
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
                                # CRIT-009: Verify fill after rollback
                                _time.sleep(0.5)  # Brief wait for fill
                                try:
                                    from data import get_positions
                                    broker_pos = {p.symbol: p for p in get_positions()}
                                    if first_sig.symbol in broker_pos:
                                        logger.warning(f"CRIT-009: Rollback position still exists at broker for {first_sig.symbol}")
                                        rollback_ok = False
                                        continue  # Retry
                                except Exception:
                                    pass  # If we can't verify, assume OK
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

    # 1a. BUG-024: Halted symbol detection
    if _is_symbol_halted(signal.symbol, now):
        skip_reason = "symbol_halted"
        logger.info(f"Signal skipped for {signal.symbol}: appears halted (zero volume)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 1b. PDT rule check for day trades
    if signal.hold_type == "day":
        try:
            from risk.pdt_tracker import PDTTracker
            # Access global PDT tracker via risk manager or create check
            if not getattr(risk, '_pdt', None):
                risk._pdt = PDTTracker()
            if not risk._pdt.can_day_trade(risk.current_equity):
                skip_reason = "pdt_limit"
                logger.warning(f"PDT: Blocked day trade for {signal.symbol} — at limit")
                log_event(EventType.PDT_BLOCKED, "signal_processor",
                          symbol=signal.symbol, strategy=signal.strategy,
                          details=f"Day trade blocked — PDT limit reached")
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return
        except Exception as e:
            logger.warning(f"PDT check failed for {signal.symbol} (fail-open): {e}")

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
        except Exception as e:
            logger.warning(f"Concentration check failed for {signal.symbol} (fail-open): {e}")

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

    # WIRE-001: Feature store — fetch features for ML/sizing enrichment (fail-open)
    _signal_features = None
    if _FEATURE_STORE_AVAILABLE:
        try:
            fs = _get_feature_store()
            _signal_features = fs.get_all_features(signal.symbol)
            if not _signal_features:
                _signal_features = None
        except Exception as e:
            logger.debug("WIRE-001: Feature store lookup failed for %s (fail-open): %s", signal.symbol, e)

    # WIRE-002: ML model confidence multiplier (fail-open)
    ml_conf_mult = 1.0
    if _signal_features:
        try:
            model = _get_ml_model()
            if model is not None:
                import pandas as _pd
                feat_df = _pd.DataFrame([_signal_features])
                prediction = model.predict(feat_df)
                # prediction is array; use first element as confidence (0-1 for classification)
                confidence = float(prediction[0]) if len(prediction) > 0 else 0.5
                # Scale: confidence 0.5 = neutral (1.0x), 1.0 = high (1.3x), 0.0 = low (0.7x)
                ml_conf_mult = 0.7 + 0.6 * confidence
                ml_conf_mult = max(0.5, min(ml_conf_mult, 1.5))
        except Exception as e:
            logger.debug("WIRE-002: ML prediction failed for %s (fail-open): %s", signal.symbol, e)
            ml_conf_mult = 1.0

    # WIRE-003: VPIN toxicity check — reduce size by 50% when VPIN > 0.7 (fail-open)
    vpin_mult = 1.0
    if _VPIN_AVAILABLE:
        try:
            with _vpin_lock:
                vpin_inst = _vpin_instances.get(signal.symbol)
            if vpin_inst is not None:
                vpin_value = vpin_inst.compute_vpin()
                if vpin_value > 0.7:
                    vpin_mult = 0.5
                    logger.info("WIRE-003: VPIN=%.2f for %s — reducing size by 50%%", vpin_value, signal.symbol)
        except Exception as e:
            logger.debug("WIRE-003: VPIN check failed for %s (fail-open): %s", signal.symbol, e)

    # Apply all multipliers
    # CRIT-008: Cap combined multipliers to prevent position size overflow (3x+ leverage)
    combined_mult = news_mult * llm_mult * regime_mult * seasonality_mult * cross_asset_mult * var_mult * ml_conf_mult * vpin_mult
    combined_mult = min(combined_mult, 2.0)  # Hard cap at 2x base size
    qty = int(qty * combined_mult)

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
                strategy=signal.strategy,  # BUG-026: pass strategy for per-strategy win rate
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
    log_event(EventType.ORDER_SUBMITTED, "signal_processor",
              symbol=signal.symbol, strategy=signal.strategy,
              details=f"qty={qty} side={signal.side} order_id={order_id}")

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
