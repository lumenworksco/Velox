"""V3: WebSocket position monitor — real-time SL/TP/trailing stop via Alpaca stream."""

import asyncio
import logging
import threading
from datetime import datetime

import config

logger = logging.getLogger(__name__)


class PositionMonitor:
    """Real-time position monitoring via Alpaca WebSocket quotes.

    Runs in a background thread. Falls back to polling if disconnected.
    """

    def __init__(self, risk_manager):
        self._risk = risk_manager
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None
        self._connected = False
        self._running = False
        self._subscribed_symbols: set[str] = set()
        self._close_callback = None  # Set by main.py to handle position closes
        self.monitoring_disabled = False  # Set True after repeated WS failures

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_close_callback(self, callback):
        """Set callback for closing positions: callback(symbol, reason)."""
        self._close_callback = callback

    def start(self):
        """Start the WebSocket monitor in a background thread."""
        if not config.WEBSOCKET_MONITORING:
            logger.info("WebSocket monitoring disabled")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-monitor")
        self._thread.start()
        logger.info("WebSocket position monitor started")

    def stop(self):
        """Stop the monitor."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._connected = False
        logger.info("WebSocket position monitor stopped")

    def subscribe(self, symbol: str):
        """Subscribe to quotes for a new symbol (thread-safe)."""
        if symbol in self._subscribed_symbols:
            return
        self._subscribed_symbols.add(symbol)
        if self._loop and self._stream and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._add_subscription(symbol))
            )

    def unsubscribe(self, symbol: str):
        """Unsubscribe from quotes for a symbol."""
        self._subscribed_symbols.discard(symbol)
        if self._loop and self._stream and self._connected:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._remove_subscription(symbol))
            )

    def _run_loop(self):
        """Background thread event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        consecutive_failures = 0

        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_stream())
                consecutive_failures = 0  # Reset on successful connection
            except Exception as e:
                # BUG-FIX (2026-04-14): `stop()` calls loop.stop(), which causes
                # any pending Future inside _connect_and_stream to raise
                # "Event loop stopped before Future completed". That's expected
                # on shutdown — log it as INFO, not ERROR, so it doesn't pollute
                # the log and trigger false alerts.
                if not self._running:
                    logger.info(f"WebSocket shutting down ({e})")
                    break
                logger.error(f"WebSocket error: {e}")
                self._connected = False
                consecutive_failures += 1

                # Alert after 3 consecutive failures
                if consecutive_failures == 3:
                    self.monitoring_disabled = True
                    logger.critical(
                        f"WebSocket failed {consecutive_failures} consecutive times — "
                        "real-time monitoring disabled! SL/TP checks rely on polling."
                    )
                    try:
                        from notifications import send_telegram
                        send_telegram(
                            f"⚠️ VELOX ALERT: WebSocket disconnected {consecutive_failures}x. "
                            "Real-time SL/TP monitoring disabled."
                        )
                    except Exception:
                        pass

            if self._running:
                # V12 HOTFIX: Rate-limit reconnect log — this fires every 5s
                # when we have no open positions (~7000 log lines per session
                # in paper trading when no positions are held). Log only on
                # first disconnect + every 60s.
                if not hasattr(self, "_last_disconnect_log"):
                    self._last_disconnect_log = 0.0
                import time as _time_mod
                _now_ts = _time_mod.time()
                if _now_ts - self._last_disconnect_log > 60:
                    logger.info(
                        f"WebSocket disconnected, reconnecting in "
                        f"{config.WEBSOCKET_RECONNECT_SEC}s..."
                    )
                    self._last_disconnect_log = _now_ts
                _time_mod.sleep(config.WEBSOCKET_RECONNECT_SEC)

    async def _connect_and_stream(self):
        """Connect to Alpaca WebSocket and stream quotes."""
        try:
            from alpaca.data.live import StockDataStream
            from alpaca.data.live.stock import DataFeed

            api_key = config.API_KEY or None
            api_secret = config.API_SECRET or None

            self._stream = StockDataStream(
                api_key=api_key,
                secret_key=api_secret,
                feed=DataFeed.IEX,
            )

            # Subscribe handler for quotes
            symbols = list(self._subscribed_symbols)
            if not symbols:
                # Nothing to monitor yet — wait and retry
                # V12 HOTFIX: Rate-limit log to once per 60s (was flooding
                # with thousands of lines per session).
                self._connected = False
                if not hasattr(self, "_last_no_symbols_log"):
                    self._last_no_symbols_log = 0.0
                import time as _time_mod
                _now_ts = _time_mod.time()
                if _now_ts - self._last_no_symbols_log > 60:
                    logger.info("WebSocket: no symbols to monitor, waiting...")
                    self._last_no_symbols_log = _now_ts
                import asyncio
                await asyncio.sleep(10)
                return

            self._stream.subscribe_quotes(self._on_quote, *symbols)

            self._connected = True
            logger.info(f"WebSocket connected, monitoring {len(symbols)} symbols")

            await self._stream._run_forever()

        except Exception as e:
            self._connected = False
            raise

    async def _on_quote(self, quote):
        """Handle incoming quote — check SL/TP/trailing stop."""
        try:
            symbol = quote.symbol
            if symbol not in self._risk.open_trades:
                return

            trade = self._risk.open_trades[symbol]
            # Use ask for longs (worst execution), bid for shorts
            if trade.side == "buy":
                current_price = float(quote.ask_price) if quote.ask_price else float(quote.bid_price)
            else:
                current_price = float(quote.bid_price) if quote.bid_price else float(quote.ask_price)

            if current_price <= 0:
                return

            # Check stop loss
            if trade.side == "buy" and current_price <= trade.stop_loss:
                logger.info(f"WS: {symbol} hit stop loss at {current_price:.2f} (SL={trade.stop_loss:.2f})")
                self._trigger_close(symbol, "stop_loss_ws")
                return
            elif trade.side == "sell" and current_price >= trade.stop_loss:
                logger.info(f"WS: {symbol} short hit stop at {current_price:.2f} (SL={trade.stop_loss:.2f})")
                self._trigger_close(symbol, "stop_loss_ws")
                return

            # Check take profit
            if trade.side == "buy" and current_price >= trade.take_profit:
                logger.info(f"WS: {symbol} hit take profit at {current_price:.2f} (TP={trade.take_profit:.2f})")
                self._trigger_close(symbol, "take_profit_ws")
                return
            elif trade.side == "sell" and current_price <= trade.take_profit:
                logger.info(f"WS: {symbol} short hit TP at {current_price:.2f} (TP={trade.take_profit:.2f})")
                self._trigger_close(symbol, "take_profit_ws")
                return

            # Update trailing stop via ATR-based trailing (handled by exit_manager)

            # Short hard stop check (4% max loss)
            if trade.side == "sell" and config.ALLOW_SHORT:
                if trade.entry_price <= 0:
                    return
                loss_pct = (current_price - trade.entry_price) / trade.entry_price
                if loss_pct > config.SHORT_HARD_STOP_PCT:
                    logger.info(f"WS: {symbol} short hard stop at {loss_pct:.2%} loss")
                    self._trigger_close(symbol, "short_hard_stop")
                    return

        except Exception as e:
            logger.error(f"WS quote handler error for {quote.symbol}: {e}")

    def _trigger_close(self, symbol: str, reason: str):
        """Trigger a position close via the callback."""
        if self._close_callback:
            try:
                self._close_callback(symbol, reason)
            except Exception as e:
                logger.error(f"Close callback failed for {symbol}: {e}")

    async def _add_subscription(self, symbol: str):
        """Add a symbol subscription to the running stream."""
        try:
            if self._stream:
                self._stream.subscribe_quotes(self._on_quote, symbol)
                logger.info(f"WS: Subscribed to {symbol}")
        except Exception as e:
            logger.error(f"WS: Failed to subscribe to {symbol}: {e}")

    async def _remove_subscription(self, symbol: str):
        """Remove a symbol subscription."""
        try:
            if self._stream:
                self._stream.unsubscribe_quotes(symbol)
                logger.info(f"WS: Unsubscribed from {symbol}")
        except Exception as e:
            logger.error(f"WS: Failed to unsubscribe from {symbol}: {e}")
