"""EMA Ribbon Scalper — high-frequency momentum strategy using EMA alignment."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)


class EMAScalper:
    """Trade strong intraday momentum using EMA ribbon alignment.

    Uses 5/8/13/21 EMA ribbon. Only enters when all 4 EMAs are perfectly
    aligned and spreading apart, indicating strong directional momentum.

    Hold time: 5-30 minutes
    Timeframe: 2-minute bars
    Best on: High-ATR stocks from FOCUS_LIST only
    """

    def __init__(self):
        self.triggered: dict[str, datetime] = {}  # cooldown tracking

    def reset_daily(self):
        self.triggered.clear()

    def _compute_vwap(self, bars: pd.DataFrame) -> float | None:
        """Compute VWAP from intraday bars."""
        if bars.empty:
            return None
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_vol = bars["volume"].cumsum()
        cum_vp = (typical * bars["volume"]).cumsum()
        if cum_vol.iloc[-1] == 0:
            return None
        return cum_vp.iloc[-1] / cum_vol.iloc[-1]

    def scan(self, symbols: list[str], now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan FOCUS_LIST for EMA ribbon momentum signals."""
        signals = []

        if not config.EMA_SCALP_ENABLED:
            return signals

        # Only during allowed hours
        current_time = now.time()
        if not (config.EMA_SCALP_START_TIME <= current_time <= config.EMA_SCALP_END_TIME):
            return signals

        # Only FOCUS_LIST symbols
        scan_symbols = [s for s in symbols if s in config.FOCUS_LIST]

        for symbol in scan_symbols:
            # Cooldown check
            if symbol in self.triggered:
                if (now - self.triggered[symbol]).total_seconds() < config.EMA_SCALP_COOLDOWN_SEC:
                    continue

            try:
                # Get 2-min bars (need ~60 bars = 2 hours of data)
                lookback = now - timedelta(hours=2)
                bars = get_intraday_bars(
                    symbol, TimeFrame(2, "Min"), start=lookback, end=now
                )
                if bars is None or bars.empty or len(bars) < 30:
                    continue

                close = bars["close"]
                volume = bars["volume"]

                # Compute EMA ribbon
                emas = {}
                for period in config.EMA_SCALP_PERIODS:
                    emas[period] = close.ewm(span=period, adjust=False).mean()

                # Check ribbon alignment
                periods = config.EMA_SCALP_PERIODS  # [5, 8, 13, 21]
                latest = {p: emas[p].iloc[-1] for p in periods}
                prev = {p: emas[p].iloc[-3] for p in periods}  # 3 bars ago

                # Bullish: 5 > 8 > 13 > 21
                bullish_ribbon = all(
                    latest[periods[i]] > latest[periods[i + 1]]
                    for i in range(len(periods) - 1)
                )

                # Bearish: 5 < 8 < 13 < 21
                bearish_ribbon = all(
                    latest[periods[i]] < latest[periods[i + 1]]
                    for i in range(len(periods) - 1)
                )

                if not bullish_ribbon and not bearish_ribbon:
                    continue

                # Ribbon must be expanding (momentum accelerating)
                ribbon_width_now = abs(latest[periods[0]] - latest[periods[-1]])
                ribbon_width_prev = abs(prev[periods[0]] - prev[periods[-1]])
                if ribbon_width_now <= ribbon_width_prev * 1.1:
                    continue  # Not expanding

                # Volume confirmation
                avg_vol = volume.rolling(20).mean().iloc[-1]
                if avg_vol == 0 or volume.iloc[-1] < config.EMA_SCALP_VOLUME_MULT * avg_vol:
                    continue

                # VWAP filter
                vwap = self._compute_vwap(bars)
                if vwap is None:
                    continue

                price = close.iloc[-1]
                slowest_ema = latest[periods[-1]]  # EMA(21)

                # --- LONG Signal ---
                if (bullish_ribbon
                        and price > vwap
                        and regime != "BEARISH"):

                    stop_loss = slowest_ema * 0.999  # Just below slowest EMA
                    stop_dist = price - stop_loss
                    if stop_dist <= 0:
                        continue
                    take_profit = price + stop_dist * config.EMA_SCALP_TP_MULT

                    # Enforce min stop distance
                    if stop_dist / price < config.EMA_SCALP_STOP_PCT:
                        stop_loss = price * (1 - config.EMA_SCALP_STOP_PCT)
                        take_profit = price * (1 + config.EMA_SCALP_STOP_PCT * config.EMA_SCALP_TP_MULT)

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="EMA_SCALP",
                        side="buy",
                        entry_price=round(price, 2),
                        take_profit=round(take_profit, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"EMA ribbon bullish+expanding, vol={volume.iloc[-1]:.0f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

                # --- SHORT Signal ---
                elif (bearish_ribbon
                      and price < vwap
                      and regime != "BULLISH"
                      and config.ALLOW_SHORT
                      and symbol not in config.NO_SHORT_SYMBOLS):

                    stop_loss = slowest_ema * 1.001  # Just above slowest EMA
                    stop_dist = stop_loss - price
                    if stop_dist <= 0:
                        continue
                    take_profit = price - stop_dist * config.EMA_SCALP_TP_MULT

                    if stop_dist / price < config.EMA_SCALP_STOP_PCT:
                        stop_loss = price * (1 + config.EMA_SCALP_STOP_PCT)
                        take_profit = price * (1 - config.EMA_SCALP_STOP_PCT * config.EMA_SCALP_TP_MULT)

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="EMA_SCALP",
                        side="sell",
                        entry_price=round(price, 2),
                        take_profit=round(take_profit, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"EMA ribbon bearish+expanding, vol={volume.iloc[-1]:.0f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

            except Exception as e:
                logger.warning(f"EMA scalp error for {symbol}: {e}")

        return signals
