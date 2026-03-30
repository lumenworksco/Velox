"""V10 Engine — Exit processing for strategy-driven and WebSocket-triggered closes.

Emits position.closed and position.partial_close events on the event bus.
"""

import logging
from datetime import datetime

import config
from data import get_snapshot, get_snapshots, get_filled_exit_price
from execution import close_position, close_partial_position
from risk import RiskManager

logger = logging.getLogger(__name__)

# Event bus integration (fail-open)
try:
    from engine.events import get_event_bus, Event, EventTypes
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False


def _emit_event(event_type: str, data: dict, source: str = "exit_processor"):
    if _EVENTS_AVAILABLE:
        try:
            bus = get_event_bus()
            bus.publish(Event(event_type, data, source=source))
        except Exception:
            pass

# Lazy-loaded optional modules
_notifications = None
_intraday_controls = None


def set_intraday_controls(controls) -> None:
    """Set the shared IntradayRiskControls instance (called from main.py)."""
    global _intraday_controls
    _intraday_controls = controls


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


def now_et():
    return datetime.now(config.ET)


def handle_strategy_exits(exit_actions: list[dict], risk: RiskManager, now: datetime, ws_monitor=None):
    """Process exit actions returned by strategy check_exits() methods.

    Each action dict: {symbol, action, reason, ...}
    action = "full" -> close_position, "partial" -> close_partial_position
    """
    notif = _get_notifications()

    for action in exit_actions:
        symbol = action["symbol"]
        if symbol not in risk.open_trades:
            continue
        trade = risk.open_trades[symbol]
        reason = action.get("reason", "strategy_exit")

        if action.get("action") == "partial":
            partial_qty = action.get("qty", max(1, trade.qty // 2))
            try:
                close_partial_position(symbol, partial_qty)
                logger.info(f"Partial exit {symbol}: {partial_qty} shares, reason={reason}")
                _emit_event(EventTypes.POSITION_PARTIAL_CLOSE if _EVENTS_AVAILABLE else "position.partial_close", {
                    "symbol": symbol, "strategy": trade.strategy, "qty": partial_qty, "reason": reason,
                })
            except Exception as e:
                logger.error(f"Partial close failed for {symbol}: {e}")
        else:
            try:
                close_position(symbol, reason=reason)
            except Exception as e:
                logger.error(f"Close failed for {symbol}: {e}")
                continue

            exit_price = get_filled_exit_price(symbol, side=trade.side)
            if exit_price is None:
                try:
                    snap = get_snapshot(symbol)
                    exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    exit_price = trade.entry_price

            pnl = (exit_price - trade.entry_price) * trade.qty * (1 if trade.side == "buy" else -1)
            risk.close_trade(symbol, exit_price, now, exit_reason=reason)

            # V11.3 T2: Feed intraday risk controls with P&L data
            if _intraday_controls is not None:
                try:
                    pnl_pct = pnl / max(risk.current_equity, 1)
                    is_stop = "stop" in reason.lower()
                    _intraday_controls.record_pnl(pnl_pct, is_stop_loss=is_stop,
                                                   is_loss=(pnl < 0), is_win=(pnl > 0), now=now)
                except Exception:
                    pass

            # Register cooldown if this was a stop-loss exit
            if "stop" in reason.lower():
                try:
                    from engine.signal_processor import register_stopout
                    register_stopout(symbol)
                except Exception:
                    pass
            _emit_event(EventTypes.POSITION_CLOSED if _EVENTS_AVAILABLE else "position.closed", {
                "symbol": symbol, "strategy": trade.strategy, "side": trade.side,
                "entry_price": trade.entry_price, "exit_price": exit_price,
                "qty": trade.qty, "pnl": round(pnl, 2), "reason": reason,
            })
            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notif and config.TELEGRAM_ENABLED:
                try:
                    notif.notify_trade_closed(trade)
                except Exception as e:
                    logger.error(f"Failed to send close notification for {symbol}: {e}")


def handle_ws_close(symbol: str, reason: str, risk: RiskManager, ws_monitor):
    """Callback for WebSocket-triggered position closes.

    Don't close for SL/TP hits (broker bracket order handles those).
    Only close for reasons the broker doesn't know about (time stops, hard stops).
    """
    if symbol not in risk.open_trades:
        return

    trade = risk.open_trades[symbol]

    # Bracket order SL/TP handled by broker — defer to broker_sync
    if reason in ("stop_loss_ws", "take_profit_ws"):
        logger.info(f"WS: {symbol} {reason} detected — deferring to broker bracket order")
        return

    try:
        close_position(symbol, reason=reason)
    except Exception as e:
        logger.error(f"WS close failed for {symbol}: {e}")
        return

    exit_price = get_filled_exit_price(symbol, side=trade.side)
    if exit_price is None:
        try:
            snap = get_snapshot(symbol)
            exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
        except Exception:
            exit_price = trade.entry_price

    risk.close_trade(symbol, exit_price, now_et(), exit_reason=reason)
    ws_monitor.unsubscribe(symbol)

    notif = _get_notifications()
    if notif and config.TELEGRAM_ENABLED:
        try:
            notif.notify_trade_closed(trade)
        except Exception as e:
            logger.error(f"Failed to send close notification for {symbol}: {e}")


def get_current_prices(open_trades: dict) -> dict[str, float]:
    """Fetch current prices for open trades (for beta calculation etc.)."""
    symbols = list(open_trades.keys())
    if not symbols:
        return {}
    prices: dict[str, float] = {}
    try:
        snapshots = get_snapshots(symbols)
        for sym, snap in snapshots.items():
            if snap and snap.latest_trade:
                prices[sym] = float(snap.latest_trade.price)
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
    for sym, trade in open_trades.items():
        if sym not in prices:
            prices[sym] = trade.entry_price
    return prices
