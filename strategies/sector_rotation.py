"""Sector Rotation strategy — rank sector ETFs by momentum, long top / short bottom."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data import get_daily_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)


class SectorRotationStrategy:
    """Sector Rotation: daily scan ranks 11 sector ETFs by composite momentum.

    Long top 2 sectors, short bottom 2 (if allowed).
    Holds up to SECTOR_MAX_HOLD_DAYS trading days (swing).
    Scanned once daily at 10:30 AM ET.
    """

    def __init__(self):
        self.scanned_today: bool = False
        self.held_sectors: dict[str, str] = {}  # symbol -> "long"/"short"

    def reset_daily(self):
        self.scanned_today = False

    def scan(self, symbols: list[str], now: datetime, regime_detector=None) -> list[Signal]:
        """Daily scan at 10:30 AM. Rank 11 sectors, long top 2, short bottom 2."""
        if not config.SECTOR_ROTATION_ENABLED:
            return []
        if self.scanned_today:
            return []

        self.scanned_today = True

        regime = regime_detector.regime if regime_detector else "UNKNOWN"
        logger.info(f"Sector Rotation: scanning {len(config.SECTOR_ROTATION_ETFS)} ETFs (regime={regime})")

        # ------------------------------------------------------------------
        # 1. Fetch 30 daily bars for each sector ETF and compute momentum
        # ------------------------------------------------------------------
        scored: list[dict] = []

        for etf in config.SECTOR_ROTATION_ETFS:
            try:
                bars = get_daily_bars(etf, days=30)
                if bars is None or bars.empty:
                    logger.debug(f"Sector Rotation: no data for {etf}")
                    continue

                close = bars["close"]
                if len(close) < 21:
                    logger.debug(f"Sector Rotation: not enough bars for {etf} ({len(close)})")
                    continue

                # Return windows
                ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
                ret_10d = (close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else 0
                ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0

                # Volume ratio: recent vs historical
                vol_ratio = bars["volume"].iloc[-5:].mean() / bars["volume"].iloc[-20:].mean()
                vol_ratio = min(vol_ratio, 2.0)  # Cap at 2x

                # Trend: above/below 20-EMA
                ema20 = close.ewm(span=20).mean().iloc[-1]
                trend = 1.0 if close.iloc[-1] > ema20 else -1.0

                # Composite score
                score = (ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2) * vol_ratio * trend

                scored.append({
                    "symbol": etf,
                    "score": score,
                    "price": close.iloc[-1],
                    "ret_5d": ret_5d,
                    "ret_10d": ret_10d,
                    "ret_20d": ret_20d,
                    "vol_ratio": vol_ratio,
                    "trend": trend,
                })
                logger.debug(
                    f"Sector Rotation: {etf} score={score:.4f} "
                    f"(5d={ret_5d:.3f} 10d={ret_10d:.3f} 20d={ret_20d:.3f} "
                    f"vol={vol_ratio:.2f} trend={trend:+.0f})"
                )

            except Exception as e:
                logger.warning(f"Sector Rotation: error scoring {etf}: {e}")

        if not scored:
            logger.info("Sector Rotation: no ETFs scored, skipping")
            return []

        # ------------------------------------------------------------------
        # 2. Rank by score
        # ------------------------------------------------------------------
        scored.sort(key=lambda x: x["score"], reverse=True)

        logger.info("Sector Rotation ranking:")
        for rank, s in enumerate(scored, 1):
            logger.info(
                f"  #{rank} {s['symbol']:5s}  score={s['score']:+.4f}  "
                f"5d={s['ret_5d']:+.3f}  10d={s['ret_10d']:+.3f}  20d={s['ret_20d']:+.3f}"
            )

        # ------------------------------------------------------------------
        # 3. Generate signals
        # ------------------------------------------------------------------
        signals: list[Signal] = []

        # --- Long: top 2 ---
        if regime != "BEARISH":
            for rank, entry in enumerate(scored[:2], 1):
                if entry["score"] <= config.SECTOR_MIN_SCORE:
                    continue
                if entry["symbol"] in self.held_sectors:
                    continue

                price = entry["price"]
                stop_loss = round(price * (1 - config.SECTOR_STOP_PCT), 2)
                take_profit = round(price * (1 + config.SECTOR_STOP_PCT * 2), 2)

                signals.append(Signal(
                    symbol=entry["symbol"],
                    strategy="SECTOR_ROTATION",
                    side="buy",
                    entry_price=round(price, 2),
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    reason=f"Sector rotation: score={entry['score']:.4f}, rank={rank}",
                    hold_type="swing",
                ))
                self.held_sectors[entry["symbol"]] = "long"

        # --- Short: bottom 2 ---
        if regime != "BULLISH" and config.ALLOW_SHORT:
            for rank_offset, entry in enumerate(reversed(scored[-2:]), 1):
                rank = len(scored) - rank_offset + 1
                if entry["score"] >= -config.SECTOR_MIN_SCORE:
                    continue
                if entry["symbol"] in self.held_sectors:
                    continue

                price = entry["price"]
                stop_loss = round(price * (1 + config.SECTOR_STOP_PCT), 2)
                take_profit = round(price * (1 - config.SECTOR_STOP_PCT * 2), 2)

                signals.append(Signal(
                    symbol=entry["symbol"],
                    strategy="SECTOR_ROTATION",
                    side="sell",
                    entry_price=round(price, 2),
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    reason=f"Sector rotation: score={entry['score']:.4f}, rank={rank}",
                    hold_type="swing",
                ))
                self.held_sectors[entry["symbol"]] = "short"

        if signals:
            logger.info(f"Sector Rotation: generated {len(signals)} signals")
        else:
            logger.info("Sector Rotation: no actionable signals today")

        return signals
