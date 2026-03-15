"""Intraday Micro-Momentum — 15% of capital allocation.

Capitalizes on quick momentum moves following economic data releases
(detected via SPY volume spikes). Trades the highest-beta stocks
in the direction of the SPY move. Very short hold time (max 8 min).

Target: 0.3-0.6% per trade, high frequency on event days.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)


# Beta table — sourced from config for tunability
STOCK_BETAS = config.MICRO_BETA_TABLE


class IntradayMicroMomentum:
    """Trade high-beta stocks during economic data release momentum.

    Workflow:
    1. detect_event() — check if SPY has a volume spike + price move
    2. scan() — on event detection, select top-beta stocks in SPY direction
    3. check_exits() — 8-minute hard time stop, or target/stop hit
    4. Disabled if daily P&L > MICRO_MAX_DAILY_GAIN_DISABLE (+1.5%)
    """

    def __init__(self):
        self._event_active = False
        self._event_direction: str = ""  # "up" or "down"
        self._event_time: datetime | None = None
        self._daily_trade_count = 0
        self._trades_this_event = 0
        self._triggered_symbols: set = set()

    def reset_daily(self):
        self._event_active = False
        self._event_direction = ""
        self._event_time = None
        self._daily_trade_count = 0
        self._trades_this_event = 0
        self._triggered_symbols = set()

    def detect_event(self, now: datetime) -> bool:
        """Detect a potential economic data release via SPY volume spike.

        Checks if:
        1. SPY 1-min volume > MICRO_SPY_VOL_SPIKE_MULT (3.0) × 20-bar avg volume
        2. SPY |price move| > MICRO_SPY_MIN_MOVE_PCT (0.15%) in last bar

        Returns True if event detected.
        """
        # Cooldown: don't detect new events within cooldown period
        if self._event_time and (now - self._event_time).total_seconds() < config.MICRO_EVENT_COOLDOWN_SEC:
            return self._event_active

        try:
            lookback = now - timedelta(minutes=30)
            bars = get_intraday_bars("SPY", TimeFrame.Minute, start=lookback, end=now)

            if bars is None or bars.empty or len(bars) < 20:
                return False

            volume = bars["volume"]
            close = bars["close"]

            # Volume spike check
            avg_vol = volume.rolling(20).mean().iloc[-1]
            if avg_vol <= 0:
                return False

            vol_ratio = volume.iloc[-1] / avg_vol

            # Price move check
            if len(close) < 2:
                return False
            price_move_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]

            if (vol_ratio >= config.MICRO_SPY_VOL_SPIKE_MULT
                    and abs(price_move_pct) >= config.MICRO_SPY_MIN_MOVE_PCT):

                self._event_active = True
                self._event_direction = "up" if price_move_pct > 0 else "down"
                self._event_time = now
                self._trades_this_event = 0

                logger.info(
                    f"MICRO event detected: SPY {self._event_direction} "
                    f"vol_ratio={vol_ratio:.1f}x move={price_move_pct:.3%}"
                )
                return True

            return False

        except Exception as e:
            logger.debug(f"Micro event detection error: {e}")
            return False

    def scan(self, now: datetime, day_pnl_pct: float = 0.0, regime: str = "UNKNOWN") -> list[Signal]:
        """Generate signals for highest-beta stocks in SPY direction.

        Called every scan cycle. Only generates signals during active events.

        Constraints:
        - Max MICRO_MAX_TRADES_PER_EVENT (3) per event
        - Disabled if day P&L > MICRO_MAX_DAILY_GAIN_DISABLE (+1.5%)
        - Top MICRO_TOP_BETA_STOCKS (5) highest-beta from standard symbols
        """
        signals = []

        if not self._event_active:
            return signals

        # Disable if day P&L is already good enough
        if day_pnl_pct >= config.MICRO_MAX_DAILY_GAIN_DISABLE:
            return signals

        # Max trades per event
        if self._trades_this_event >= config.MICRO_MAX_TRADES_PER_EVENT:
            return signals

        # Event window: only trade within event window
        if self._event_time and (now - self._event_time).total_seconds() > config.MICRO_EVENT_WINDOW_SEC:
            self._event_active = False
            return signals

        # Select top-beta stocks not already triggered
        candidates = [
            (sym, beta) for sym, beta in STOCK_BETAS.items()
            if sym not in self._triggered_symbols
            and sym not in config.LEVERAGED_ETFS
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:config.MICRO_TOP_BETA_STOCKS]

        for symbol, beta in top:
            if self._trades_this_event >= config.MICRO_MAX_TRADES_PER_EVENT:
                break

            try:
                # Get current price
                lookback = now - timedelta(minutes=5)
                bars = get_intraday_bars(symbol, TimeFrame.Minute, start=lookback, end=now)
                if bars is None or bars.empty:
                    continue

                price = bars["close"].iloc[-1]

                if self._event_direction == "up":
                    # Buy high-beta stocks
                    stop_loss = price * (1 - config.MICRO_STOP_PCT)
                    take_profit = price * (1 + config.MICRO_TARGET_PCT)
                    side = "buy"
                else:
                    # Short high-beta stocks
                    if not config.ALLOW_SHORT or symbol in config.NO_SHORT_SYMBOLS:
                        continue
                    stop_loss = price * (1 + config.MICRO_STOP_PCT)
                    take_profit = price * (1 - config.MICRO_TARGET_PCT)
                    side = "sell"

                signals.append(Signal(
                    symbol=symbol,
                    strategy="MICRO_MOM",
                    side=side,
                    entry_price=round(price, 2),
                    take_profit=round(take_profit, 2),
                    stop_loss=round(stop_loss, 2),
                    reason=f"Micro {self._event_direction} beta={beta:.1f}",
                    hold_type="day",
                ))

                self._triggered_symbols.add(symbol)
                self._trades_this_event += 1

            except Exception as e:
                logger.debug(f"Micro scan error for {symbol}: {e}")

        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check micro-momentum positions for time stop.

        8-minute hard time stop — these are ultra-short-hold trades.

        Returns list of exit actions.
        """
        exits = []

        for symbol, trade in open_trades.items():
            if trade.strategy != "MICRO_MOM":
                continue

            # 8-minute time stop
            if hasattr(trade, 'entry_time') and trade.entry_time:
                elapsed_min = (now - trade.entry_time).total_seconds() / 60
                if elapsed_min >= config.MICRO_MAX_HOLD_MINUTES:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Micro time stop ({elapsed_min:.0f}min)",
                    })

        return exits
