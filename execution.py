"""Order execution — bracket orders, TWAP splitting, retries, EOD exits."""

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
from engine.failure_modes import FailureMode, handle_failure

logger = logging.getLogger(__name__)


def _validate_bracket_params(signal: Signal) -> tuple[bool, str]:
    """MED-032: Validate bracket order parameters before submission.

    Checks that stop_loss and take_profit are on the correct side of entry_price
    for the given trade direction.

    Returns:
        (valid, reason) tuple.
    """
    entry = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit

    if entry <= 0:
        return False, f"Invalid entry price: {entry}"
    if sl <= 0 or tp <= 0:
        return False, f"Invalid SL={sl} or TP={tp} (must be > 0)"

    if signal.side == "buy":
        # Long: stop_loss must be below entry, take_profit must be above entry
        if sl >= entry:
            return False, f"Long SL={sl} >= entry={entry}"
        if tp <= entry:
            return False, f"Long TP={tp} <= entry={entry}"
    elif signal.side == "sell":
        # Short: stop_loss must be above entry, take_profit must be below entry
        if sl <= entry:
            return False, f"Short SL={sl} <= entry={entry}"
        if tp >= entry:
            return False, f"Short TP={tp} >= entry={entry}"

    return True, ""


def _submit_order(signal: Signal, qty: int, client=None):
    """Internal: submit a single order. Returns order object.

    V6 strategy routing:
      - STAT_MR / KALMAN_PAIRS  -> LIMIT (mean-reversion, not time-sensitive)
      - MICRO_MOM / BETA_HEDGE  -> MARKET (speed matters)
    Legacy strategies (MOMENTUM, ORB, VWAP, etc.) are kept as fallback.
    """
    if client is None:
        client = get_trading_client()

    # MED-032: Validate bracket parameters before submission
    valid, reason = _validate_bracket_params(signal)
    if not valid:
        raise ValueError(f"Bracket order validation failed for {signal.symbol}: {reason}")

    side = OrderSide.BUY if signal.side == "buy" else OrderSide.SELL

    # --- V6 strategies ---------------------------------------------------
    if signal.strategy in ("STAT_MR", "KALMAN_PAIRS"):
        # Mean-reversion: limit order, not time-sensitive
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

    elif signal.strategy in ("MICRO_MOM", "BETA_HEDGE"):
        # Momentum / hedge: market order, speed matters
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

    elif signal.strategy == "PEAD":
        # CRIT-014: PEAD: limit order (swing trade, not time-sensitive)
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
        # VWAP and others: market order (speed matters, day only)
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


def submit_twap_order(
    signal: Signal, total_qty: int, slices: int = 5, interval_sec: int = 60
) -> list[str]:
    """Split large orders into time-weighted slices (TWAP).

    For orders > $2000, split into `slices` smaller orders spaced
    `interval_sec` apart.  Each slice is a bracket order with the same TP/SL.

    Returns list of order IDs.
    """
    client = get_trading_client()
    slice_qty = total_qty // slices
    remainder = total_qty % slices
    order_ids: list[str] = []

    for i in range(slices):
        # V10 BUG-004: Distribute remainder across first N slices (1 extra each)
        qty = slice_qty + (1 if i < remainder else 0)
        if qty <= 0:
            continue

        try:
            order = _submit_order(signal, qty, client)
            oid = str(order.id)
            order_ids.append(oid)
            logger.info(
                f"TWAP slice {i+1}/{slices}: {signal.side.upper()} {qty} "
                f"{signal.symbol} ({signal.strategy}) order_id={oid}"
            )
        except Exception as e:
            logger.error(
                f"TWAP slice {i+1}/{slices} failed for {signal.symbol}: {e}"
            )

        # Sleep between slices (not after the last one)
        if i < slices - 1:
            time.sleep(interval_sec)

    return order_ids


def submit_bracket_order(signal: Signal, qty: int) -> str | list[str] | None:
    """Submit bracket order, auto-routing to TWAP for large orders.

    Returns a single order ID (str), a list of order IDs (TWAP), or None on
    failure.
    """
    # Auto-route large mean-reversion orders to TWAP
    order_value = qty * signal.entry_price
    if order_value > 2000 and signal.strategy in ("STAT_MR", "KALMAN_PAIRS"):
        return submit_twap_order(signal, qty)

    client = get_trading_client()

    # HIGH-009: 3 retries with exponential backoff (1s, 2s, 4s)
    backoff_delays = [1, 2, 4]
    last_order_id = None

    for attempt in range(1 + len(backoff_delays)):  # 1 initial + 3 retries
        try:
            # Before retrying, check if a previous attempt actually went through
            if attempt > 0 and last_order_id:
                try:
                    existing = client.get_order_by_id(last_order_id)
                    status = str(existing.status).lower()
                    if status in ("new", "accepted", "partially_filled", "filled"):
                        logger.info(
                            f"Order {last_order_id} already {status} — skipping retry"
                        )
                        return str(last_order_id)
                except Exception as exc:
                    handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                                   "execution.order_status_check", exc,
                                   symbol=signal.symbol, strategy=signal.strategy)

            order = _submit_order(signal, qty, client)
            logger.info(
                f"Order submitted: {signal.side.upper()} {qty} {signal.symbol} "
                f"({signal.strategy}) order_id={order.id}"
                + (f" (attempt {attempt + 1})" if attempt > 0 else "")
            )
            return str(order.id)

        except Exception as e:
            last_order_id_str = f" (last_order_id={last_order_id})" if last_order_id else ""
            if attempt < len(backoff_delays):
                delay = backoff_delays[attempt]
                logger.warning(
                    f"Order attempt {attempt + 1} failed for {signal.symbol}: {e}"
                    f"{last_order_id_str} — retrying in {delay}s"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Order failed after {attempt + 1} attempts for "
                    f"{signal.symbol}: {e}{last_order_id_str}"
                )
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


def close_all_positions(reason: str = "EOD") -> tuple[bool, int, list[str]]:
    """Close all open positions.

    HIGH-031: Returns (success, count, failed_symbols) tuple instead of just int.

    Args:
        reason: Reason string for logging.

    Returns:
        Tuple of:
          - success: True if all positions closed without error
          - count: Number of positions successfully closed
          - failed_symbols: List of symbols that failed to close
    """
    client = get_trading_client()
    failed_symbols: list[str] = []
    closed_count = 0

    try:
        # Get current positions before closing
        positions = client.get_all_positions()
        total = len(positions)

        if total == 0:
            logger.info(f"No positions to close ({reason})")
            return True, 0, []

        # Attempt to close all
        client.close_all_positions(cancel_orders=True)

        # Verify which positions actually closed
        time.sleep(1)  # Brief pause for settlement
        remaining = client.get_all_positions()
        remaining_symbols = {p.symbol for p in remaining}

        for pos in positions:
            if pos.symbol in remaining_symbols:
                failed_symbols.append(pos.symbol)
            else:
                closed_count += 1

        success = len(failed_symbols) == 0
        if success:
            logger.info(f"All {closed_count} positions closed ({reason})")
        else:
            logger.warning(
                f"Closed {closed_count}/{total} positions ({reason}), "
                f"failed: {failed_symbols}"
            )
        return success, closed_count, failed_symbols

    except Exception as e:
        logger.error(f"Failed to close all positions: {e}")
        return False, closed_count, failed_symbols


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


## V10: Removed dead strategy functions:
# check_momentum_max_hold, close_gap_go_positions,
# check_sector_max_hold, check_pairs_max_hold
# (MOMENTUM, GAP_GO, SECTOR_ROTATION, PAIRS strategies no longer exist)


def close_partial_position(symbol: str, qty: int) -> bool:
    """Close a partial position (for scaled exits)."""
    client = get_trading_client()
    try:
        # V10 BUG-016: Validate qty is int before string conversion (Alpaca API requires str)
        client.close_position(symbol, qty=str(int(qty)))
        logger.info(f"Partial close: {symbol} qty={qty}")
        return True
    except Exception as e:
        logger.error(f"Failed partial close {symbol} qty={qty}: {e}")
        return False
