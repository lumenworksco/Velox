"""Order execution — bracket orders, retries, EOD exits, momentum orders."""

import logging
import time
from datetime import datetime

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

import config
from data import get_trading_client
from strategies.base import Signal

logger = logging.getLogger(__name__)


def _submit_order(signal: Signal, qty: int, client=None):
    """Internal: submit a single order. Returns order object."""
    if client is None:
        client = get_trading_client()

    side = OrderSide.BUY if signal.side == "buy" else OrderSide.SELL

    if signal.strategy == "MOMENTUM":
        # Momentum uses GTC limit order with bracket (holds overnight)
        return client.submit_order(
            LimitOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=round(signal.entry_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )
    elif signal.strategy == "ORB":
        # ORB uses limit order at breakout price (day only)
        return client.submit_order(
            LimitOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(signal.entry_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )
    else:
        # VWAP uses market order (speed matters, day only)
        return client.submit_order(
            MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )


def can_short(symbol: str, qty: int, entry_price: float) -> tuple[bool, str]:
    """V3: Pre-trade checks before shorting. Returns (allowed, reason)."""
    if not config.ALLOW_SHORT:
        return False, "shorting_disabled"

    if symbol in config.NO_SHORT_SYMBOLS:
        return False, "no_short_symbol"

    try:
        client = get_trading_client()

        # Check if asset is shortable
        asset = client.get_asset(symbol)
        if not asset.shortable:
            return False, "not_shortable"

        # Check buying power (shorts require ~150% margin)
        account = client.get_account()
        required = qty * entry_price * 1.5
        if float(account.buying_power) < required:
            return False, "insufficient_buying_power"

        return True, "ok"

    except Exception as e:
        logger.error(f"Short pre-check failed for {symbol}: {e}")
        return False, f"check_error: {e}"


def submit_bracket_order(signal: Signal, qty: int) -> str | None:
    """Submit a bracket order (entry + TP + SL). Returns order ID or None on failure."""
    client = get_trading_client()

    try:
        order = _submit_order(signal, qty, client)
        logger.info(
            f"Order submitted: {signal.side.upper()} {qty} {signal.symbol} "
            f"({signal.strategy}) order_id={order.id}"
        )
        return str(order.id)

    except Exception as e:
        logger.error(f"Bracket order failed for {signal.symbol}: {e}")

        # Retry once after 2 seconds
        time.sleep(2)
        try:
            order = _submit_order(signal, qty, client)
            logger.info(f"Order retry succeeded: {order.id}")
            return str(order.id)
        except Exception as e2:
            logger.error(f"Order retry also failed for {signal.symbol}: {e2}")
            return None


def close_position(symbol: str, reason: str = "") -> bool:
    """Close an open position by symbol. Returns True on success."""
    client = get_trading_client()
    try:
        client.close_position(symbol)
        logger.info(f"Position closed: {symbol} ({reason})")
        return True
    except Exception as e:
        logger.error(f"Failed to close {symbol}: {e}")
        return False


def close_all_positions(reason: str = "EOD") -> int:
    """Close all open positions. Returns count of positions closed."""
    client = get_trading_client()
    try:
        client.close_all_positions(cancel_orders=True)
        logger.info(f"All positions closed ({reason})")
        return 0
    except Exception as e:
        logger.error(f"Failed to close all positions: {e}")
        return -1


def cancel_all_open_orders() -> bool:
    """Cancel all pending orders."""
    client = get_trading_client()
    try:
        client.cancel_orders()
        logger.info("All open orders cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel orders: {e}")
        return False


def close_orb_positions(open_trades: dict, now: datetime) -> list[str]:
    """Close all ORB positions (for 3:45 PM exit). Returns list of closed symbols."""
    closed = []
    for symbol, trade in list(open_trades.items()):
        if trade.strategy == "ORB":
            if close_position(symbol, reason="ORB EOD exit"):
                closed.append(symbol)
    return closed


def check_vwap_time_stops(open_trades: dict, now: datetime) -> list[str]:
    """Check VWAP trades for time stop (45 min). Returns list of symbols to close."""
    expired = []
    for symbol, trade in open_trades.items():
        if trade.strategy == "VWAP" and trade.time_stop:
            if now >= trade.time_stop:
                if close_position(symbol, reason="VWAP time stop"):
                    expired.append(symbol)
    return expired


def check_momentum_max_hold(open_trades: dict, now: datetime) -> list[str]:
    """Check momentum trades for max hold period. Returns list of symbols to close."""
    expired = []
    for symbol, trade in open_trades.items():
        if trade.strategy == "MOMENTUM" and trade.max_hold_date:
            if now >= trade.max_hold_date:
                if close_position(symbol, reason="momentum max hold"):
                    expired.append(symbol)
    return expired


def close_gap_go_positions(open_trades: dict, now: datetime) -> list[str]:
    """V3: Close all Gap & Go positions at 11:30 AM time stop."""
    closed = []
    for symbol, trade in list(open_trades.items()):
        if trade.strategy == "GAP_GO":
            if close_position(symbol, reason="gap_go time stop"):
                closed.append(symbol)
    return closed


def check_sector_max_hold(open_trades: dict, now: datetime) -> list[str]:
    """V4: Check sector rotation trades for max hold period."""
    expired = []
    for symbol, trade in open_trades.items():
        if trade.strategy == "SECTOR_ROTATION" and trade.max_hold_date:
            if now >= trade.max_hold_date:
                if close_position(symbol, reason="sector max hold"):
                    expired.append(symbol)
    return expired


def check_pairs_max_hold(open_trades: dict, now: datetime) -> list[str]:
    """V4: Check pairs trades for max hold period."""
    expired = []
    for symbol, trade in open_trades.items():
        if trade.strategy == "PAIRS" and trade.max_hold_date:
            if now >= trade.max_hold_date:
                if close_position(symbol, reason="pairs max hold"):
                    expired.append(symbol)
    return expired


def close_partial_position(symbol: str, qty: int) -> bool:
    """V4: Close a partial position (for scaled exits)."""
    client = get_trading_client()
    try:
        client.close_position(symbol, qty=str(qty))
        logger.info(f"Partial close: {symbol} qty={qty}")
        return True
    except Exception as e:
        logger.error(f"Failed partial close {symbol} qty={qty}: {e}")
        return False
