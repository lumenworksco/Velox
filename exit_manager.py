"""Advanced exit mechanics — scaled TP, trailing stops, ATR trailing, volatility exits."""

import copy
import logging
import threading
from datetime import datetime

import pandas_ta as ta

import config
from data import get_intraday_bars
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)


class ExitManager:
    """Manages advanced exit logic for open positions.

    Exit types:
    1. Scaled take profit: 33% at 1/3 target, 50% of remaining at 2/3, rest at full
    2. Breakeven stop: move SL to entry + 0.1% after first partial
    3. RSI momentum exit: close if RSI > 80 and trade is profitable
    4. Volatility expansion: close if ATR > 2x entry ATR and losing
    5. Trailing stop: for swing positions, trail at highest_price * (1 - 1.5%)
    """

    def check_exits(self, risk_manager, now: datetime) -> list[dict]:
        """Check all open trades for exit conditions.

        Returns list of actions taken: [{"symbol": ..., "action": ..., "qty": ...}]
        """
        if not config.ADVANCED_EXITS_ENABLED:
            return []

        actions = []

        for symbol in list(risk_manager.open_trades.keys()):
            trade = risk_manager.open_trades.get(symbol)
            if not trade:
                continue

            try:
                action = self._evaluate_trade(trade, risk_manager, now)
                if action:
                    actions.append(action)
            except Exception as e:
                logger.warning(f"Exit check error for {symbol}: {e}")

        return actions

    def _evaluate_trade(self, trade, risk_manager, now: datetime) -> dict | None:
        """Evaluate a single trade for exit conditions.

        Thread-safety: reads trade state under the risk_manager lock, then
        evaluates on a local snapshot so concurrent WebSocket updates to
        highest_price_seen / lowest_price_seen cannot cause torn reads.
        """
        symbol = trade.symbol

        # Get current price data (I/O — done outside lock)
        try:
            from data import get_snapshot
            snap = get_snapshot(symbol)
            if not snap or not snap.latest_trade:
                return None
            current_price = float(snap.latest_trade.price)
        except Exception as e:
            logger.debug(f"Snapshot fetch failed for {symbol}: {e}")
            return None

        if current_price <= 0:
            return None

        # V10 BUG-006: Snapshot mutable trade fields under lock, then update
        # price extremes atomically so concurrent WebSocket updates cannot
        # interleave with this evaluation.
        lock = getattr(risk_manager, '_lock', None)
        if lock is not None:
            lock.acquire()
        try:
            # Update price extremes for trailing stops
            if trade.side == "buy" and current_price > trade.highest_price_seen:
                trade.highest_price_seen = current_price
            elif trade.side == "sell":
                if trade.lowest_price_seen == 0 or current_price < trade.lowest_price_seen:
                    trade.lowest_price_seen = current_price

            # Snapshot mutable fields for evaluation outside lock
            snap_highest = trade.highest_price_seen
            snap_lowest = trade.lowest_price_seen
            snap_stop_loss = trade.stop_loss
            snap_qty = trade.qty
            snap_partial_exits = trade.partial_exits
            snap_partial_closed_qty = trade.partial_closed_qty
        finally:
            if lock is not None:
                lock.release()

        # Guard against invalid entry price
        if trade.entry_price <= 0:
            logger.warning(f"Exit manager: invalid entry_price for {trade.symbol}")
            return None

        # Calculate current P&L
        if trade.side == "buy":
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price

        # --- 1. SCALED TAKE PROFIT ---
        if config.SCALED_TP_ENABLED and trade.qty > 1:
            action = self._check_scaled_tp(trade, current_price, risk_manager, now)
            if action:
                return action

        # --- 2. TRAILING STOP ---
        # V8: ATR-based trailing for all strategies
        if config.ATR_TRAILING_ENABLED and trade.entry_atr > 0:
            action = self._check_atr_trailing_stop(trade, current_price, risk_manager, now)
            if action:
                return action

        # --- 3. RSI MOMENTUM EXIT ---
        if pnl_pct > 0.005:  # Only if profitable
            action = self._check_rsi_exit(trade, current_price, risk_manager, now)
            if action:
                return action

        # --- 4. VOLATILITY EXPANSION EXIT ---
        if pnl_pct < 0 and trade.entry_atr > 0:
            action = self._check_volatility_exit(trade, current_price, risk_manager, now)
            if action:
                return action

        return None

    # Maximum number of partial take-profit levels
    MAX_PARTIAL_LEVELS = 2

    def _check_scaled_tp(self, trade, current_price: float, risk_manager, now: datetime) -> dict | None:
        """Check scaled take profit levels."""
        # Guard: skip if all partial levels already taken
        if trade.partial_exits >= self.MAX_PARTIAL_LEVELS:
            return None

        entry = trade.entry_price
        target = trade.take_profit

        if trade.side == "buy":
            target_range = target - entry
            if target_range <= 0:
                return None

            # Level 1: 33% at 1/3 of target
            if current_price >= entry + target_range * 0.33 and trade.partial_exits == 0:
                qty_to_close = max(1, int(trade.qty * 0.33))
                self._execute_partial(risk_manager, trade.symbol, qty_to_close,
                                     current_price, now, "partial_tp_1")
                trade.partial_closed_qty = trade.partial_closed_qty + qty_to_close
                trade.partial_exits += 1
                return {"symbol": trade.symbol, "action": "partial_tp_1", "qty": qty_to_close}

            # Level 2: 50% of remaining at 2/3 of target
            if current_price >= entry + target_range * 0.66 and trade.partial_exits == 1:
                remaining_qty = trade.qty - trade.partial_closed_qty
                qty_to_close = max(1, int(remaining_qty * 0.50))
                # Move stop to breakeven
                if config.BREAKEVEN_STOP_ENABLED:
                    trade.stop_loss = entry * 1.001
                self._execute_partial(risk_manager, trade.symbol, qty_to_close,
                                     current_price, now, "partial_tp_2")
                trade.partial_closed_qty = trade.partial_closed_qty + qty_to_close
                trade.partial_exits += 1
                return {"symbol": trade.symbol, "action": "partial_tp_2", "qty": qty_to_close}

        elif trade.side == "sell":
            target_range = entry - target  # Positive for shorts
            if target_range <= 0:
                return None

            if current_price <= entry - target_range * 0.33 and trade.partial_exits == 0:
                qty_to_close = max(1, int(trade.qty * 0.33))
                self._execute_partial(risk_manager, trade.symbol, qty_to_close,
                                     current_price, now, "partial_tp_1")
                trade.partial_closed_qty = trade.partial_closed_qty + qty_to_close
                trade.partial_exits += 1
                return {"symbol": trade.symbol, "action": "partial_tp_1", "qty": qty_to_close}

            if current_price <= entry - target_range * 0.66 and trade.partial_exits == 1:
                remaining_qty = trade.qty - trade.partial_closed_qty
                qty_to_close = max(1, int(remaining_qty * 0.50))
                if config.BREAKEVEN_STOP_ENABLED:
                    trade.stop_loss = entry * 0.999
                self._execute_partial(risk_manager, trade.symbol, qty_to_close,
                                     current_price, now, "partial_tp_2")
                trade.partial_closed_qty = trade.partial_closed_qty + qty_to_close
                trade.partial_exits += 1
                return {"symbol": trade.symbol, "action": "partial_tp_2", "qty": qty_to_close}

        return None

    def _check_trailing_stop(self, trade, current_price: float, risk_manager, now: datetime) -> dict | None:
        """Update and check trailing stop for swing positions."""
        if trade.side == "buy":
            if trade.highest_price_seen > trade.entry_price:
                trailing_stop = trade.highest_price_seen * (1 - config.TRAILING_STOP_PCT)
                if trailing_stop > trade.stop_loss:
                    trade.stop_loss = trailing_stop

                if current_price <= trade.stop_loss:
                    risk_manager.close_trade(trade.symbol, current_price, now, "trailing_stop")
                    return {"symbol": trade.symbol, "action": "trailing_stop", "qty": trade.qty}

        elif trade.side == "sell":
            if trade.lowest_price_seen > 0 and trade.lowest_price_seen < trade.entry_price:
                trailing_stop = trade.lowest_price_seen * (1 + config.TRAILING_STOP_PCT)
                if trailing_stop < trade.stop_loss:
                    trade.stop_loss = trailing_stop

                if current_price >= trade.stop_loss:
                    risk_manager.close_trade(trade.symbol, current_price, now, "trailing_stop")
                    return {"symbol": trade.symbol, "action": "trailing_stop", "qty": trade.qty}

        return None

    def _check_atr_trailing_stop(self, trade, current_price: float, risk_manager, now: datetime) -> dict | None:
        """V8: ATR-based trailing stop for all strategies.

        - Trailing stop = highest_price - (entry_atr x ATR_TRAIL_MULT) for longs
        - Trailing stop = lowest_price + (entry_atr x ATR_TRAIL_MULT) for shorts
        - Only activates after position is in profit by 0.5x ATR
        - Trail only ratchets in profitable direction (never widens)
        """
        atr_mult = config.ATR_TRAIL_MULT.get(trade.strategy)
        if atr_mult is None:
            return None

        trail_distance = trade.entry_atr * atr_mult
        activation_distance = trade.entry_atr * config.ATR_TRAIL_ACTIVATION

        if trade.side == "buy":
            # Check activation: must be in profit by at least 0.5x ATR
            if current_price < trade.entry_price + activation_distance:
                return None

            # Compute trailing stop from highest price
            atr_trail_stop = trade.highest_price_seen - trail_distance

            # Only ratchet up, never down — and must be better than current SL
            if atr_trail_stop > trade.stop_loss:
                trade.stop_loss = atr_trail_stop

            if current_price <= trade.stop_loss:
                risk_manager.close_trade(trade.symbol, current_price, now, "atr_trailing_stop")
                return {"symbol": trade.symbol, "action": "atr_trailing_stop", "qty": trade.qty}

        elif trade.side == "sell":
            # Check activation: must be in profit by at least 0.5x ATR
            if current_price > trade.entry_price - activation_distance:
                return None

            # For shorts, use lowest_price_seen as reference
            if trade.lowest_price_seen <= 0:
                trade.lowest_price_seen = current_price
            atr_trail_stop = trade.lowest_price_seen + trail_distance

            # Only ratchet down (for shorts), must be better (lower) than current SL
            if atr_trail_stop < trade.stop_loss:
                trade.stop_loss = atr_trail_stop

            if current_price >= trade.stop_loss:
                risk_manager.close_trade(trade.symbol, current_price, now, "atr_trailing_stop")
                return {"symbol": trade.symbol, "action": "atr_trailing_stop", "qty": trade.qty}

        return None

    def _check_rsi_exit(self, trade, current_price: float, risk_manager, now: datetime) -> dict | None:
        """Exit if RSI is extremely overbought/oversold and trade is profitable."""
        try:
            market_open = datetime(now.year, now.month, now.day, 9, 30, tzinfo=config.ET)
            bars = get_intraday_bars(trade.symbol, TimeFrame(1, TimeFrameUnit.Minute), start=market_open, end=now)
            if bars.empty or len(bars) < 20:
                return None

            rsi_series = ta.rsi(bars["close"], length=14)
            if rsi_series is None or rsi_series.empty:
                return None
            rsi = rsi_series.iloc[-1]

            if trade.side == "buy" and rsi > config.RSI_EXIT_THRESHOLD:
                risk_manager.close_trade(trade.symbol, current_price, now, "rsi_overbought")
                return {"symbol": trade.symbol, "action": "rsi_overbought", "qty": trade.qty}

            if trade.side == "sell" and rsi < (100 - config.RSI_EXIT_THRESHOLD):
                risk_manager.close_trade(trade.symbol, current_price, now, "rsi_oversold")
                return {"symbol": trade.symbol, "action": "rsi_oversold", "qty": trade.qty}

        except Exception as e:
            logger.debug(f"RSI exit check failed for {trade.symbol}: {e}")

        return None

    def _check_volatility_exit(self, trade, current_price: float, risk_manager, now: datetime) -> dict | None:
        """Exit if ATR has expanded significantly since entry (market conditions changed)."""
        if trade.entry_atr <= 0:
            return None

        try:
            market_open = datetime(now.year, now.month, now.day, 9, 30, tzinfo=config.ET)
            bars = get_intraday_bars(trade.symbol, TimeFrame(1, TimeFrameUnit.Minute), start=market_open, end=now)
            if bars.empty or len(bars) < 20:
                return None

            atr_series = ta.atr(bars["high"], bars["low"], bars["close"], length=14)
            if atr_series is None or atr_series.empty:
                return None
            current_atr = atr_series.iloc[-1]

            if current_atr > trade.entry_atr * config.ATR_EXPANSION_MULT:
                risk_manager.close_trade(trade.symbol, current_price, now, "volatility_expansion")
                return {"symbol": trade.symbol, "action": "volatility_expansion", "qty": trade.qty}

        except Exception as e:
            logger.debug(f"Volatility exit check failed for {trade.symbol}: {e}")

        return None

    def _execute_partial(self, risk_manager, symbol: str, qty: int,
                         price: float, now: datetime, reason: str):
        """Execute a partial position close via the broker."""
        try:
            from execution import close_partial_position
            close_partial_position(symbol, qty)
        except Exception as e:
            logger.error(f"Partial close execution failed for {symbol}: {e}")
            return

        risk_manager.partial_close(symbol, qty, price, now, reason)

        # CRIT-028: Register partial close in OMS order manager if available
        try:
            from engine.signal_processor import _order_manager, _OMS_AVAILABLE
            if _OMS_AVAILABLE and _order_manager:
                _order_manager.create_order(
                    symbol=symbol,
                    strategy=risk_manager.open_trades[symbol].strategy if symbol in risk_manager.open_trades else "",
                    side="sell" if risk_manager.open_trades.get(symbol, None) and risk_manager.open_trades[symbol].side == "buy" else "buy",
                    order_type="market",
                    qty=qty,
                    limit_price=price,
                    idempotency_key=f"partial_{symbol}_{reason}_{now.strftime('%Y%m%d%H%M%S')}",
                )
        except Exception as e:
            logger.debug(f"OMS registration for partial close failed (non-critical): {e}")

    def update_highest_prices(self, risk_manager, quotes: dict):
        """Update price extremes from live quote data (called by WebSocket handler).

        V10 BUG-006: Acquire risk_manager lock to prevent race with
        _evaluate_trade reading the same fields concurrently.
        """
        lock = getattr(risk_manager, '_lock', None)
        if lock is not None:
            lock.acquire()
        try:
            for symbol, price in quotes.items():
                if symbol in risk_manager.open_trades:
                    trade = risk_manager.open_trades[symbol]
                    if trade.side == "buy" and price > trade.highest_price_seen:
                        trade.highest_price_seen = price
                    elif trade.side == "sell":
                        if trade.lowest_price_seen == 0 or price < trade.lowest_price_seen:
                            trade.lowest_price_seen = price
        finally:
            if lock is not None:
                lock.release()
