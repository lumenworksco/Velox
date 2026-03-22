"""Sector Momentum Rotation strategy (ALPHA-008 / STRAT-003).

Ranks sectors by multi-horizon momentum (1-month, 3-month, 6-month) and
takes long positions in the top 3 sectors, short positions in the bottom 3
using sector ETFs. Rebalances weekly.

Regime filter: only active in LOW_VOL_BULL and MEAN_REVERTING HMM states.

Sector ETFs: XLK, XLF, XLV, XLE, XLI, XLP, XLU, XLY, XLC, XLB, XLRE
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data import get_daily_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)

# Sector ETFs used for rotation
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLY", "XLC", "XLB", "XLRE"]

# Momentum horizons in trading days
MOMENTUM_WINDOWS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

# Weights for composite momentum score
MOMENTUM_WEIGHTS = {
    "1m": 0.40,
    "3m": 0.35,
    "6m": 0.25,
}

# Allowed HMM regimes for this strategy (V11.2: BULLISH only per regime guard)
ALLOWED_REGIMES = {"LOW_VOL_BULL", "MEAN_REVERTING", "BULLISH"}

# Relative-strength threshold: only enter if symbol RS vs sector ETF > this
RS_THRESHOLD = 0.15

# Number of sectors to go long/short
TOP_N = 3
BOTTOM_N = 3

# Weekly rebalance day (0=Monday, 4=Friday)
REBALANCE_DAY = 0  # Monday


@dataclass
class SectorScore:
    """Momentum score for a single sector ETF."""
    symbol: str
    momentum_1m: float = 0.0
    momentum_3m: float = 0.0
    momentum_6m: float = 0.0
    composite_score: float = 0.0
    volume_20d_avg: float = 0.0


class SectorMomentumStrategy:
    """Sector Momentum Rotation — long top sectors, short bottom sectors.

    Workflow:
    1. prepare_universe(date) — verify sector ETFs are tradeable
    2. generate_signals(bars) — rank sectors, generate long/short signals
    3. Rebalance weekly (Monday after market open)
    4. Regime filter: skip in high-vol or bearish HMM states

    Position sizing: equal-weight across selected sectors within
    the strategy's capital allocation.
    """

    def __init__(self):
        self.universe: List[str] = list(SECTOR_ETFS)
        self.sector_scores: Dict[str, SectorScore] = {}
        self._last_rebalance_date: Optional[datetime] = None
        self._current_longs: List[str] = []
        self._current_shorts: List[str] = []
        self._universe_ready = False

    def reset_daily(self):
        """Clear per-day state. Preserve weekly rebalance state."""
        # Do NOT clear _last_rebalance_date or current positions — weekly strategy
        # Clear per-sector scores so they are recomputed on the next scan
        self.sector_scores.clear()
        self._universe_ready = False

    def prepare_universe(self, date: datetime) -> List[str]:
        """Verify sector ETFs are available and liquid.

        Called at startup and daily at universe prep time.
        Returns the list of sector ETFs in the tradeable universe.
        """
        verified = []
        for symbol in SECTOR_ETFS:
            try:
                bars = get_daily_bars(symbol, days=10)
                if bars is not None and not bars.empty and len(bars) >= 5:
                    avg_vol = bars["volume"].tail(5).mean()
                    if avg_vol > 100_000:  # Minimum liquidity threshold
                        verified.append(symbol)
            except Exception as e:
                logger.debug(f"Sector ETF verification failed for {symbol}: {e}")
                continue

        self.universe = verified if verified else list(SECTOR_ETFS)
        self._universe_ready = True
        logger.info(f"Sector momentum universe: {len(self.universe)} ETFs verified")
        return self.universe

    def generate_signals(self, bars: Dict[str, pd.DataFrame],
                         regime: str = "UNKNOWN",
                         hmm_regime: str = "UNKNOWN",
                         now: Optional[datetime] = None) -> List[Signal]:
        """Rank sectors by momentum and generate long/short signals.

        Only generates signals on weekly rebalance day (Monday) and when
        the HMM regime is in ALLOWED_REGIMES.

        Args:
            bars: Dict mapping symbol -> DataFrame of daily OHLCV bars.
                  If empty, will fetch bars internally.
            regime: Legacy regime string (not used for filtering).
            hmm_regime: HMM regime state name for regime filtering.
            now: Current datetime (defaults to datetime.now()).

        Returns:
            List of Signal objects — longs for top sectors, shorts for bottom.
        """
        if now is None:
            now = datetime.now(config.ET)

        signals: List[Signal] = []

        # Regime filter: only trade in favorable regimes
        if hmm_regime not in ALLOWED_REGIMES:
            logger.debug(f"Sector momentum: skipping, regime={hmm_regime} not in {ALLOWED_REGIMES}")
            return signals

        # Weekly rebalance check
        if not self._should_rebalance(now):
            return signals

        # Compute momentum scores for all sector ETFs
        self.sector_scores = {}
        for symbol in self.universe:
            try:
                score = self._compute_sector_score(symbol, bars.get(symbol), now)
                if score is not None:
                    self.sector_scores[symbol] = score
            except Exception as e:
                logger.debug(f"Sector score computation failed for {symbol}: {e}")
                continue

        if len(self.sector_scores) < TOP_N + BOTTOM_N:
            logger.warning(
                f"Sector momentum: insufficient scores ({len(self.sector_scores)}), "
                f"need at least {TOP_N + BOTTOM_N}"
            )
            return signals

        # Rank by composite score
        ranked = sorted(
            self.sector_scores.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )

        # Top N for longs
        top_sectors = [s.symbol for s in ranked[:TOP_N]]
        # Bottom N for shorts
        bottom_sectors = [s.symbol for s in ranked[-BOTTOM_N:]]

        # Generate long signals (with RS filter)
        for symbol in top_sectors:
            score = self.sector_scores[symbol]

            # V11.2 RS filter: sector must outperform SPY by RS_THRESHOLD
            rs_vs_spy = self._compute_rs_vs_spy(symbol, bars.get(symbol), bars.get("SPY"))
            if rs_vs_spy is not None and rs_vs_spy < RS_THRESHOLD:
                logger.debug(
                    f"Sector momentum: skipping {symbol}, RS vs SPY={rs_vs_spy:.3f} < {RS_THRESHOLD}"
                )
                continue

            try:
                price = self._get_current_price(symbol, bars.get(symbol))
                if price is None or price <= 0:
                    continue

                # Target: hold for 1 week, TP at 2% above, SL at 3% below
                take_profit = round(price * 1.02, 2)
                stop_loss = round(price * 0.97, 2)

                signals.append(Signal(
                    symbol=symbol,
                    strategy="SECTOR_MOM",
                    side="buy",
                    entry_price=round(price, 2),
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    reason=(
                        f"Sector momentum LONG rank={top_sectors.index(symbol)+1} "
                        f"score={score.composite_score:.3f} "
                        f"m1={score.momentum_1m:.3f} m3={score.momentum_3m:.3f}"
                    ),
                    hold_type="swing",
                    confidence=min(0.5 + abs(score.composite_score) * 2, 0.95),
                    metadata={
                        "momentum_1m": score.momentum_1m,
                        "momentum_3m": score.momentum_3m,
                        "momentum_6m": score.momentum_6m,
                        "composite": score.composite_score,
                    },
                ))
            except Exception as e:
                logger.debug(f"Sector momentum signal gen failed for {symbol}: {e}")

        # Generate short signals
        if config.ALLOW_SHORT:
            for symbol in bottom_sectors:
                if symbol in config.NO_SHORT_SYMBOLS:
                    continue
                score = self.sector_scores[symbol]
                try:
                    price = self._get_current_price(symbol, bars.get(symbol))
                    if price is None or price <= 0:
                        continue

                    take_profit = round(price * 0.98, 2)
                    stop_loss = round(price * 1.03, 2)

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="SECTOR_MOM",
                        side="sell",
                        entry_price=round(price, 2),
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        reason=(
                            f"Sector momentum SHORT rank={len(ranked) - bottom_sectors.index(symbol)} "
                            f"score={score.composite_score:.3f} "
                            f"m1={score.momentum_1m:.3f} m3={score.momentum_3m:.3f}"
                        ),
                        hold_type="swing",
                        confidence=min(0.5 + abs(score.composite_score) * 2, 0.95),
                        metadata={
                            "momentum_1m": score.momentum_1m,
                            "momentum_3m": score.momentum_3m,
                            "momentum_6m": score.momentum_6m,
                            "composite": score.composite_score,
                        },
                    ))
                except Exception as e:
                    logger.debug(f"Sector momentum short signal gen failed for {symbol}: {e}")

        self._last_rebalance_date = now.date()
        self._current_longs = top_sectors
        self._current_shorts = bottom_sectors

        logger.info(
            f"Sector momentum rebalance: {len(signals)} signals, "
            f"longs={top_sectors}, shorts={bottom_sectors}"
        )
        return signals

    def _should_rebalance(self, now: datetime) -> bool:
        """Check if it's time for weekly rebalance.

        Rebalances on Monday (REBALANCE_DAY) if we haven't already this week.
        """
        if now.weekday() != REBALANCE_DAY:
            return False

        if self._last_rebalance_date is not None:
            days_since = (now.date() - self._last_rebalance_date).days
            if days_since < 5:  # Already rebalanced this week
                return False

        return True

    def _compute_sector_score(self, symbol: str,
                              bars_df: Optional[pd.DataFrame],
                              now: datetime) -> Optional[SectorScore]:
        """Compute multi-horizon momentum score for a sector ETF.

        Momentum = total return over the lookback window.
        Composite = weighted average of 1m, 3m, 6m momentum.
        """
        # Fetch bars if not provided
        if bars_df is None or bars_df.empty:
            bars_df = get_daily_bars(symbol, days=140)  # Need ~126 trading days + buffer
            if bars_df is None or bars_df.empty:
                return None

        close = bars_df["close"]
        if len(close) < MOMENTUM_WINDOWS["1m"]:
            return None

        score = SectorScore(symbol=symbol)

        # Compute momentum for each horizon
        for horizon, window in MOMENTUM_WINDOWS.items():
            if len(close) >= window:
                ret = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]
            else:
                ret = 0.0

            if horizon == "1m":
                score.momentum_1m = ret
            elif horizon == "3m":
                score.momentum_3m = ret
            elif horizon == "6m":
                score.momentum_6m = ret

        # Composite score: weighted average
        score.composite_score = (
            MOMENTUM_WEIGHTS["1m"] * score.momentum_1m
            + MOMENTUM_WEIGHTS["3m"] * score.momentum_3m
            + MOMENTUM_WEIGHTS["6m"] * score.momentum_6m
        )

        # 20-day average volume
        if len(bars_df) >= 20 and "volume" in bars_df.columns:
            score.volume_20d_avg = bars_df["volume"].tail(20).mean()

        return score

    def _get_current_price(self, symbol: str,
                           bars_df: Optional[pd.DataFrame]) -> Optional[float]:
        """Get the most recent price for a symbol."""
        if bars_df is not None and not bars_df.empty:
            return float(bars_df["close"].iloc[-1])

        try:
            bars = get_daily_bars(symbol, days=3)
            if bars is not None and not bars.empty:
                return float(bars["close"].iloc[-1])
        except Exception:
            pass

        return None

    def _compute_rs_vs_spy(self, symbol: str,
                           bars_df: Optional[pd.DataFrame],
                           spy_bars: Optional[pd.DataFrame]) -> Optional[float]:
        """Compute 20-day relative strength of symbol vs SPY.

        Returns the difference in 20-day returns (symbol - SPY), or None
        if data is insufficient.
        """
        lookback = MOMENTUM_WINDOWS["1m"]  # 21 trading days

        if bars_df is None or bars_df.empty:
            bars_df = get_daily_bars(symbol, days=lookback + 5)
        if bars_df is None or len(bars_df) < lookback:
            return None

        if spy_bars is None or spy_bars.empty:
            spy_bars = get_daily_bars("SPY", days=lookback + 5)
        if spy_bars is None or len(spy_bars) < lookback:
            return None

        sym_ret = (bars_df["close"].iloc[-1] - bars_df["close"].iloc[-lookback]) / bars_df["close"].iloc[-lookback]
        spy_ret = (spy_bars["close"].iloc[-1] - spy_bars["close"].iloc[-lookback]) / spy_bars["close"].iloc[-lookback]
        return float(sym_ret - spy_ret)

    def get_rankings(self) -> List[Dict]:
        """Return current sector rankings for dashboard/logging."""
        if not self.sector_scores:
            return []

        ranked = sorted(
            self.sector_scores.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )
        return [
            {
                "rank": i + 1,
                "symbol": s.symbol,
                "score": round(s.composite_score, 4),
                "mom_1m": round(s.momentum_1m, 4),
                "mom_3m": round(s.momentum_3m, 4),
                "mom_6m": round(s.momentum_6m, 4),
            }
            for i, s in enumerate(ranked)
        ]
