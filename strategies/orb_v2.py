"""Opening Range Breakout v2 (ORB) strategy for Velox V7.

Cleaner rewrite with:
- Pre-market volume universe filtering at 9:35 AM
- Gap and range validity checks at 10:00 AM
- Volume-confirmed breakout signals 10:00–11:30 AM
- 2-hour time stop on open trades
"""

import logging
from datetime import datetime, time, timedelta

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_intraday_bars, get_snapshot
from strategies.base import Signal

logger = logging.getLogger(__name__)


class ORBStrategyV2:
    """Opening Range Breakout v2 — scans for volume-confirmed breakouts."""

    def __init__(self):
        self.opening_ranges: dict[str, dict] = {}
        # Each entry: {high, low, range_pct, gap_pct, valid, established}
        self._universe: list[str] = []
        self._trades_today: int = 0

    # ------------------------------------------------------------------
    # Morning prep
    # ------------------------------------------------------------------

    def prepare_morning_universe(self, symbols: list[str], now: datetime):
        """At 9:35 AM, rank symbols by pre-market volume and keep top N."""
        ranked = []
        for symbol in symbols:
            try:
                snap = get_snapshot(symbol)
                if snap is None:
                    continue
                vol = getattr(snap, "latest_trade", None)
                # Snapshots expose minute_bar with volume; use latest bar volume
                minute_bar = getattr(snap, "minute_bar", None)
                pre_vol = minute_bar.volume if minute_bar else 0
                ranked.append((symbol, pre_vol))
            except Exception as e:
                logger.debug(f"Snapshot failed for {symbol}: {e}")
                continue

        ranked.sort(key=lambda x: x[1], reverse=True)
        self._universe = [s for s, _ in ranked[:config.ORB_SCAN_SYMBOLS]]
        logger.info(
            f"ORBv2 morning universe: {len(self._universe)} symbols "
            f"(from {len(symbols)} candidates)"
        )

    # ------------------------------------------------------------------
    # Record opening range (called at 10:00 AM)
    # ------------------------------------------------------------------

    def record_opening_range(self, symbol: str, bars_930_1000):
        """Record ORB high/low from 9:30–10:00 bars for a single symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        bars_930_1000 : pd.DataFrame
            DataFrame with columns: high, low, close, open, volume.
            Covers the 9:30–10:00 window.
        """
        if bars_930_1000.empty or len(bars_930_1000) < 2:
            logger.warning(f"ORBv2: insufficient bars for {symbol}, skipping")
            return

        orb_high = bars_930_1000["high"].max()
        orb_low = bars_930_1000["low"].min()
        orb_mid = (orb_high + orb_low) / 2
        range_pct = (orb_high - orb_low) / orb_mid if orb_mid > 0 else 0

        # Gap % — use first bar open vs previous close
        first_open = bars_930_1000["open"].iloc[0]
        prev_close = bars_930_1000["close"].iloc[0]  # caller may embed prev close
        # More robust: try to get previous close from snapshot or daily bars
        # For now, use the open of the first bar compared to the close of
        # the *last bar of yesterday*.  The caller is responsible for passing
        # bars that include prev-close info, or we approximate with the first
        # bar's open vs low.
        # In practice the caller passes 30-min intraday bars; use the bar
        # *before* the opening bar if available, else approximate gap = 0.
        gap_pct = 0.0
        try:
            snap = get_snapshot(symbol)
            if snap and hasattr(snap, "prev_daily_bar") and snap.prev_daily_bar:
                prev_close = snap.prev_daily_bar.close
                gap_pct = abs(first_open - prev_close) / prev_close
        except Exception as e:
            logger.debug(f"Prev close lookup failed for {symbol}: {e}")

        valid = (
            gap_pct < config.ORB_MAX_GAP_PCT
            and range_pct < config.ORB_MAX_RANGE_PCT
        )

        self.opening_ranges[symbol] = {
            "high": orb_high,
            "low": orb_low,
            "range_pct": range_pct,
            "gap_pct": gap_pct,
            "valid": valid,
            "established": True,
        }
        logger.info(
            f"ORBv2 range {symbol}: high={orb_high:.2f} low={orb_low:.2f} "
            f"range={range_pct:.3f} gap={gap_pct:.3f} valid={valid}"
        )

    # ------------------------------------------------------------------
    # Scan for breakout signals
    # ------------------------------------------------------------------

    def scan(self, now: datetime, regime: str) -> list[Signal]:
        """Scan universe for ORB breakout signals.

        Active window: 10:00 AM – config.ORB_ACTIVE_UNTIL (11:30 AM).
        Skips BEARISH regime (breakouts fail in down-trending markets).
        """
        signals: list[Signal] = []

        now_time = now.time()
        if now_time < time(10, 0) or now_time > config.ORB_ACTIVE_UNTIL:
            return signals

        if regime == "BEARISH":
            return signals

        for symbol, orb in self.opening_ranges.items():
            if not orb.get("valid"):
                continue

            orb_high = orb["high"]
            orb_low = orb["low"]
            orb_range = orb_high - orb_low
            if orb_range <= 0:
                continue

            try:
                # Fetch 5-min bars for last hour
                start = now - timedelta(hours=1)
                bars = get_intraday_bars(
                    symbol,
                    TimeFrame(5, TimeFrameUnit.Minute),
                    start=start,
                    end=now,
                )
                if bars.empty:
                    continue

                latest = bars.iloc[-1]
                price = latest["close"]
                current_vol = latest["volume"]

                # Volume ratio vs average bar volume
                avg_vol = bars["volume"].mean()
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

                # --- LONG breakout ---
                breakout_buf = config.ORB_BREAKOUT_BUFFER
                if price > orb_high * (1 + breakout_buf) and vol_ratio > config.ORB_VOLUME_RATIO:
                    entry = price
                    tp = entry + config.ORB_TP_MULT * orb_range
                    sl = entry - config.ORB_SL_MULT * orb_range

                    # Enforce minimum stop distance
                    min_stop_dist = entry * config.ORB_MIN_STOP_PCT
                    if (entry - sl) < min_stop_dist:
                        sl = entry - min_stop_dist

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="ORB",
                        side="buy",
                        entry_price=round(entry, 2),
                        take_profit=round(tp, 2),
                        stop_loss=round(sl, 2),
                        reason=f"ORBv2 long breakout above {orb_high:.2f}, vol_ratio={vol_ratio:.1f}",
                        hold_type="day",
                    ))

                # --- SHORT breakdown ---
                elif (
                    config.ALLOW_SHORT
                    and price < orb_low * (1 - breakout_buf)
                    and vol_ratio > config.ORB_VOLUME_RATIO
                ):
                    entry = price
                    tp = entry - config.ORB_TP_MULT * orb_range
                    sl = entry + config.ORB_SL_MULT * orb_range

                    # Enforce minimum stop distance
                    min_stop_dist = entry * config.ORB_MIN_STOP_PCT
                    if (sl - entry) < min_stop_dist:
                        sl = entry + min_stop_dist

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="ORB",
                        side="sell",
                        entry_price=round(entry, 2),
                        take_profit=round(tp, 2),
                        stop_loss=round(sl, 2),
                        reason=f"ORBv2 short breakdown below {orb_low:.2f}, vol_ratio={vol_ratio:.1f}",
                        hold_type="day",
                    ))

            except Exception as e:
                logger.warning(f"ORBv2 scan error for {symbol}: {e}")

        return signals

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    def check_exits(self, open_trades: list, now: datetime) -> list[dict]:
        """Check open ORB trades for time-based exits.

        Returns a list of exit actions:
            [{symbol, action: 'full', reason: 'orb_time_stop'}, ...]
        """
        exits: list[dict] = []

        for trade in open_trades:
            if getattr(trade, "strategy", "") != "ORB":
                continue

            entry_time = getattr(trade, "entry_time", None)
            if entry_time is None:
                continue

            elapsed = now - entry_time
            if elapsed >= timedelta(hours=config.ORB_TIME_STOP_HOURS):
                exits.append({
                    "symbol": trade.symbol,
                    "action": "full",
                    "reason": "orb_time_stop",
                })

        return exits

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self):
        """Clear all state for a fresh trading day."""
        self.opening_ranges.clear()
        self._universe.clear()
        self._trades_today = 0
        logger.info("ORBv2 daily state reset")
