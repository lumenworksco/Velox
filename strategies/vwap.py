"""VWAP Mean Reversion strategy."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars
from strategies.base import Signal, VWAPState

logger = logging.getLogger(__name__)


class VWAPStrategy:
    """VWAP Mean Reversion strategy."""

    def __init__(self):
        self.states: dict[str, VWAPState] = {}
        self.triggered: dict[str, datetime] = {}  # symbol -> last trigger time (cooldown)
        self.daily_moves: dict[str, float] = {}    # symbol -> day's move % for trend filter

    def reset_daily(self):
        self.states.clear()
        self.triggered.clear()
        self.daily_moves.clear()

    def _compute_vwap(self, bars: pd.DataFrame) -> tuple[float, float, float] | None:
        """Compute VWAP and bands from intraday bars."""
        if bars.empty:
            return None

        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_vol = bars["volume"].cumsum()
        cum_vp = (typical_price * bars["volume"]).cumsum()

        if cum_vol.iloc[-1] == 0:
            return None

        vwap = cum_vp.iloc[-1] / cum_vol.iloc[-1]

        # Standard deviation of typical price weighted by volume
        cum_vp2 = (typical_price**2 * bars["volume"]).cumsum()
        variance = cum_vp2.iloc[-1] / cum_vol.iloc[-1] - vwap**2
        std_dev = np.sqrt(max(variance, 0))

        upper = vwap + config.VWAP_BAND_STD * std_dev
        lower = vwap - config.VWAP_BAND_STD * std_dev

        return vwap, upper, lower

    def scan(self, symbols: list[str], now: datetime, regime: str) -> list[Signal]:
        """Scan for VWAP mean reversion signals."""
        signals = []
        today = now.date()
        market_open = datetime(today.year, today.month, today.day, 9, 30, tzinfo=config.ET)

        for symbol in symbols:
            # Cooldown: don't re-trigger within 5 minutes
            if symbol in self.triggered:
                if (now - self.triggered[symbol]).total_seconds() < 300:
                    continue

            try:
                # Get all 1-min bars from open until now
                bars = get_intraday_bars(symbol, TimeFrame.Minute, start=market_open, end=now)
                if bars.empty or len(bars) < 15:
                    continue

                # Trend filter: skip if stock moved > 3% today
                first_price = bars["open"].iloc[0]
                last_price = bars["close"].iloc[-1]
                day_move = abs(last_price - first_price) / first_price
                self.daily_moves[symbol] = day_move

                if day_move > config.MAX_INTRADAY_MOVE_PCT:
                    continue

                # Compute VWAP + bands
                result = self._compute_vwap(bars)
                if result is None:
                    continue
                vwap, upper, lower = result

                # Compute RSI(14)
                rsi_series = ta.rsi(bars["close"], length=14)
                if rsi_series is None or rsi_series.empty:
                    continue
                rsi = rsi_series.iloc[-1]

                prev_bar = bars.iloc[-2]
                curr_bar = bars.iloc[-1]

                # BUY signal: price touched lower band and bounced back above
                if prev_bar["low"] <= lower and curr_bar["close"] > lower and rsi < config.VWAP_RSI_OVERSOLD:
                    # V5: Confirmation bar check
                    if config.VWAP_CONFIRMATION_BARS >= 2 and len(bars) >= 3:
                        bar_2ago = bars.iloc[-3]
                        # Require: 2 bars ago touched band, prev bar confirmed bounce
                        if not (bar_2ago["low"] <= lower and prev_bar["close"] > prev_bar["open"]):
                            continue

                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = lower - config.VWAP_STOP_EXTENSION * std_dev

                    # V5: Enforce minimum stop/TP distances
                    entry = curr_bar["close"]
                    min_stop = entry * config.VWAP_MIN_STOP_PCT
                    min_tp = entry * config.VWAP_MIN_TP_PCT
                    if abs(entry - stop_loss) < min_stop:
                        stop_loss = entry - min_stop
                    if abs(vwap - entry) < min_tp:
                        pass  # VWAP TP is the target, don't override upward

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="buy",
                        entry_price=round(curr_bar["close"], 2),
                        take_profit=round(vwap, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP bounce at {lower:.2f}, RSI={rsi:.1f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

                # V3 SELL/SHORT signal: upper band rejection (bearish/choppy regime only)
                elif (
                    config.ALLOW_SHORT
                    and regime in ("BEARISH", "UNKNOWN")
                    and symbol not in config.NO_SHORT_SYMBOLS
                    and prev_bar["high"] >= upper
                    and curr_bar["close"] < upper
                    and rsi > config.VWAP_RSI_OVERBOUGHT
                    and day_move > 0.01  # Stock must be up > 1% already today
                ):
                    # V5: Confirmation bar check
                    if config.VWAP_CONFIRMATION_BARS >= 2 and len(bars) >= 3:
                        bar_2ago = bars.iloc[-3]
                        # Require: 2 bars ago touched band, prev bar confirmed rejection
                        if not (bar_2ago["high"] >= upper and prev_bar["close"] < prev_bar["open"]):
                            continue

                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = upper + config.VWAP_STOP_EXTENSION * std_dev

                    # V5: Enforce minimum stop/TP distances
                    entry = curr_bar["close"]
                    min_stop = entry * config.VWAP_MIN_STOP_PCT
                    min_tp = entry * config.VWAP_MIN_TP_PCT
                    if abs(stop_loss - entry) < min_stop:
                        stop_loss = entry + min_stop
                    if abs(entry - vwap) < min_tp:
                        pass  # VWAP TP is the target, don't override downward

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="sell",
                        entry_price=round(curr_bar["close"], 2),
                        take_profit=round(vwap, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP rejection at {upper:.2f}, RSI={rsi:.1f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

            except Exception as e:
                logger.warning(f"VWAP scan error for {symbol}: {e}")

        return signals
