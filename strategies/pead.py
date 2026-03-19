"""Post-Earnings Announcement Drift (PEAD) — event-driven swing strategy.

Trades the well-documented drift after earnings surprises:
- LONG: surprise > +5%, volume > 2x, gap up
- SHORT: surprise < -5%, volume > 2x, gap down
- Entry: Day AFTER earnings at market open
- Exit: 10-20 day hold with 5% TP / 3% SL / ATR trailing after +2%
- Max 5 concurrent PEAD positions, 2% of portfolio per position
"""

import logging
from datetime import datetime, timedelta

import config
from strategies.base import Signal

logger = logging.getLogger(__name__)


class PEADStrategy:
    """Post-Earnings Announcement Drift strategy."""

    def __init__(self):
        self.triggered: dict[str, datetime] = {}  # symbol -> last trigger time
        self._candidates: list[dict] = []  # Pending earnings candidates
        self._scanned_today = False

    def scan(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Daily scan -- check for recent earnings surprises.

        Uses yfinance to get earnings surprise data for symbols that reported
        in the last 24 hours. Filters by surprise %, volume, and gap direction.

        Since this is swing, only scan once per day (at or after 9:00 AM).
        Must be fail-open: if yfinance fails, return empty list.
        """
        if not config.PEAD_ENABLED:
            return []

        # Only scan once per day
        if self._scanned_today:
            return []

        # Only scan at or after 9:00 AM ET
        scan_time = now.time()
        from datetime import time as dt_time
        if scan_time < dt_time(9, 0):
            return []

        self._scanned_today = True

        # Check how many PEAD positions we already have (via triggered)
        active_count = len(self.triggered)
        if active_count >= config.PEAD_MAX_POSITIONS:
            logger.debug(f"PEAD: max positions ({config.PEAD_MAX_POSITIONS}) reached")
            return []

        # Reduce in HIGH_VOL_BEAR regime
        if regime == "HIGH_VOL_BEAR":
            logger.info("PEAD: skipping scan in HIGH_VOL_BEAR regime")
            return []

        # Get earnings surprises
        symbols = config.STANDARD_SYMBOLS
        surprises = self._get_earnings_surprises(symbols)
        if not surprises:
            return []

        signals = []
        slots_remaining = config.PEAD_MAX_POSITIONS - active_count

        for data in surprises:
            if slots_remaining <= 0:
                break

            symbol = data["symbol"]
            surprise_pct = data["surprise_pct"]
            volume_ratio = data["volume_ratio"]
            gap_pct = data["gap_pct"]

            # Skip if already triggered recently (within 30 days)
            if symbol in self.triggered:
                last = self.triggered[symbol]
                if (now - last).days < 30:
                    continue

            # Filter: minimum surprise magnitude
            if abs(surprise_pct) < config.PEAD_MIN_SURPRISE_PCT:
                continue

            # Filter: minimum volume ratio
            if volume_ratio < config.PEAD_MIN_VOLUME_RATIO:
                continue

            # Determine direction
            if surprise_pct > 0 and gap_pct > 0:
                side = "buy"
                entry_price = data.get("current_price", 0)
                if entry_price <= 0:
                    continue
                take_profit = round(entry_price * (1 + config.PEAD_TAKE_PROFIT), 2)
                stop_loss = round(entry_price * (1 - config.PEAD_STOP_LOSS), 2)
                reason = (f"PEAD long: surprise={surprise_pct:+.1f}% "
                          f"vol={volume_ratio:.1f}x gap={gap_pct:+.1f}%")
            elif surprise_pct < 0 and gap_pct < 0:
                if not config.ALLOW_SHORT:
                    continue
                if symbol in config.NO_SHORT_SYMBOLS:
                    continue
                side = "sell"
                entry_price = data.get("current_price", 0)
                if entry_price <= 0:
                    continue
                take_profit = round(entry_price * (1 - config.PEAD_TAKE_PROFIT), 2)
                stop_loss = round(entry_price * (1 + config.PEAD_STOP_LOSS), 2)
                reason = (f"PEAD short: surprise={surprise_pct:+.1f}% "
                          f"vol={volume_ratio:.1f}x gap={gap_pct:+.1f}%")
            else:
                # Surprise and gap direction mismatch -- skip
                continue

            signals.append(Signal(
                symbol=symbol,
                strategy="PEAD",
                side=side,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reason=reason,
                hold_type="swing",
            ))
            self.triggered[symbol] = now
            slots_remaining -= 1

        logger.info(f"PEAD scan: {len(signals)} signals from {len(surprises)} surprises")
        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check time stop (max 20 days), profit target (5%), stop loss (3%).

        Returns list of exit action dicts: {symbol, action: "full", reason}.
        """
        exits = []

        for symbol, trade in open_trades.items():
            if trade.strategy != "PEAD":
                continue

            try:
                # Time stop: max hold days
                if hasattr(trade, "entry_time") and trade.entry_time:
                    hold_days = (now - trade.entry_time).days
                    if hold_days >= config.PEAD_HOLD_DAYS_MAX:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD time stop ({hold_days}d >= {config.PEAD_HOLD_DAYS_MAX}d)",
                        })
                        continue

                # P&L based exits using current market price
                try:
                    from data import get_snapshot
                    snap = get_snapshot(symbol)
                    current_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    current_price = trade.entry_price
                if trade.side == "buy":
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                    if pnl_pct >= config.PEAD_TAKE_PROFIT:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD take profit ({pnl_pct:.1%})",
                        })
                    elif pnl_pct <= -config.PEAD_STOP_LOSS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD stop loss ({pnl_pct:.1%})",
                        })
                elif trade.side == "sell":
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price
                    if pnl_pct >= config.PEAD_TAKE_PROFIT:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD take profit ({pnl_pct:.1%})",
                        })
                    elif pnl_pct <= -config.PEAD_STOP_LOSS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD stop loss ({pnl_pct:.1%})",
                        })

            except Exception as e:
                logger.debug(f"PEAD exit check error for {symbol}: {e}")

        return exits

    def reset_daily(self):
        """Clear daily state (allow re-scan next day)."""
        self._scanned_today = False
        self._candidates = []

    def _get_earnings_surprises(self, symbols: list[str]) -> list[dict]:
        """Fetch recent earnings surprises via yfinance.

        Returns list of {symbol, surprise_pct, volume_ratio, gap_pct, current_price}.
        Fail-open: returns empty list on error.
        """
        results = []
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("PEAD: yfinance not installed, skipping")
            return []

        # Suppress yfinance logging noise
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)

                    # Get earnings history for surprise data
                    earnings = getattr(ticker, "earnings_dates", None)
                    if earnings is None or (hasattr(earnings, "empty") and earnings.empty):
                        continue

                    # Find most recent past earnings date
                    import pandas as pd
                    now = pd.Timestamp.now(tz="America/New_York")
                    past_earnings = earnings[earnings.index <= now]
                    if past_earnings.empty:
                        continue

                    last_earnings = past_earnings.index[0]
                    days_since = (now - last_earnings).days
                    if days_since > 1:
                        continue  # Only interested in earnings from last 24h

                    # Get surprise percentage
                    row = past_earnings.iloc[0]
                    surprise_pct = 0.0
                    if "Surprise(%)" in past_earnings.columns:
                        val = row.get("Surprise(%)")
                        if pd.notna(val):
                            surprise_pct = float(val)
                    if surprise_pct == 0.0:
                        continue

                    # Get volume ratio and gap from recent price data
                    hist = ticker.history(period="5d")
                    if hist is None or len(hist) < 2:
                        continue

                    # Volume ratio: last day vs average of prior days
                    last_vol = hist["Volume"].iloc[-1]
                    avg_vol = hist["Volume"].iloc[:-1].mean()
                    volume_ratio = last_vol / avg_vol if avg_vol > 0 else 0

                    # Gap: open vs prior close
                    gap_pct = ((hist["Open"].iloc[-1] - hist["Close"].iloc[-2])
                               / hist["Close"].iloc[-2]) * 100

                    current_price = float(hist["Close"].iloc[-1])

                    results.append({
                        "symbol": symbol,
                        "surprise_pct": surprise_pct,
                        "volume_ratio": volume_ratio,
                        "gap_pct": gap_pct,
                        "current_price": current_price,
                    })

                except Exception as e:
                    logger.debug(f"PEAD earnings check failed for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"PEAD earnings scan failed: {e}")
        finally:
            yf_logger.setLevel(prev_level)

        return results
