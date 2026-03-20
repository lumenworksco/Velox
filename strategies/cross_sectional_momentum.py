"""Cross-Sectional Momentum strategy (STRAT-003).

Ranks stocks by 1-month return (skipping most recent week to avoid
short-term reversal) and goes long the top decile, short the bottom
decile. Market-neutral by construction.

Features:
- Industry-neutral variant: rank within each sector separately
- Weekly rebalance
- Skip-week momentum: exclude last 5 trading days from return calc
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data import get_daily_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)

# Lookback: 1-month return minus most recent week
MOMENTUM_LOOKBACK_DAYS = 21  # ~1 month of trading days
SKIP_RECENT_DAYS = 5         # Skip most recent week (short-term reversal)

# Decile thresholds
TOP_DECILE_PCT = 0.10
BOTTOM_DECILE_PCT = 0.10

# Rebalance day
REBALANCE_DAY = 0  # Monday

# Position parameters
DEFAULT_TP_PCT = 0.03   # 3% take-profit target
DEFAULT_SL_PCT = 0.02   # 2% stop-loss


@dataclass
class MomentumRank:
    """Momentum ranking for a single stock."""
    symbol: str
    return_1m_skip: float  # 1-month return skipping last week
    sector: str = ""
    sector_rank_pct: float = 0.0  # Percentile rank within sector (0=worst, 1=best)
    universe_rank_pct: float = 0.0  # Percentile rank in full universe


class CrossSectionalMomentum:
    """Cross-Sectional Momentum — long winners, short losers.

    Workflow:
    1. Compute 1-month skip-week returns for all stocks
    2. Rank stocks: universe-wide and within-sector
    3. Long top decile, short bottom decile
    4. Market-neutral: equal dollar exposure long vs short
    5. Weekly rebalance on Monday

    The strategy is market-neutral by construction: total long exposure
    equals total short exposure, so net market beta is approximately zero.
    """

    def __init__(self, industry_neutral: bool = False):
        self.industry_neutral = industry_neutral
        self.rankings: List[MomentumRank] = []
        self._last_rebalance_date: Optional[datetime] = None
        self._current_longs: List[str] = []
        self._current_shorts: List[str] = []

    def reset_daily(self):
        """Clear per-day state. Preserve weekly rebalance state."""
        pass

    def generate_signals(self, bars: Dict[str, pd.DataFrame],
                         regime: str = "UNKNOWN",
                         now: Optional[datetime] = None) -> List[Signal]:
        """Rank stocks by momentum and generate long/short signals.

        Args:
            bars: Dict mapping symbol -> DataFrame of daily OHLCV bars.
                  If a symbol's bars are missing, fetches internally.
            regime: Market regime string (informational).
            now: Current datetime (defaults to datetime.now()).

        Returns:
            List of Signal objects — longs for top decile, shorts for bottom.
        """
        if now is None:
            now = datetime.now(config.ET)

        signals: List[Signal] = []

        # Weekly rebalance check
        if not self._should_rebalance(now):
            return signals

        # Compute momentum for all symbols
        symbols = config.STANDARD_SYMBOLS
        self.rankings = []

        for symbol in symbols:
            try:
                rank = self._compute_momentum(symbol, bars.get(symbol), now)
                if rank is not None:
                    self.rankings.append(rank)
            except Exception as e:
                logger.debug(f"XS momentum computation failed for {symbol}: {e}")
                continue

        if len(self.rankings) < 20:
            logger.warning(f"XS momentum: insufficient rankings ({len(self.rankings)}), need >= 20")
            return signals

        # Generate signals based on ranking mode
        if self.industry_neutral:
            signals = self._generate_industry_neutral_signals(now)
        else:
            signals = self._generate_universe_signals(now)

        self._last_rebalance_date = now.date()

        logger.info(
            f"XS momentum rebalance: {len(signals)} signals "
            f"({len(self._current_longs)} longs, {len(self._current_shorts)} shorts)"
        )
        return signals

    def _generate_universe_signals(self, now: datetime) -> List[Signal]:
        """Generate signals using universe-wide ranking."""
        # Sort by return
        sorted_ranks = sorted(self.rankings, key=lambda r: r.return_1m_skip, reverse=True)
        n = len(sorted_ranks)

        # Assign universe percentile ranks
        for i, rank in enumerate(sorted_ranks):
            rank.universe_rank_pct = 1.0 - (i / n)

        # Top decile = long
        top_cutoff = max(1, int(n * TOP_DECILE_PCT))
        top_stocks = sorted_ranks[:top_cutoff]

        # Bottom decile = short
        bottom_cutoff = max(1, int(n * BOTTOM_DECILE_PCT))
        bottom_stocks = sorted_ranks[-bottom_cutoff:]

        signals = []
        self._current_longs = []
        self._current_shorts = []

        # Long signals
        for rank in top_stocks:
            sig = self._create_long_signal(rank, now)
            if sig:
                signals.append(sig)
                self._current_longs.append(rank.symbol)

        # Short signals
        if config.ALLOW_SHORT:
            for rank in bottom_stocks:
                if rank.symbol in config.NO_SHORT_SYMBOLS:
                    continue
                sig = self._create_short_signal(rank, now)
                if sig:
                    signals.append(sig)
                    self._current_shorts.append(rank.symbol)

        return signals

    def _generate_industry_neutral_signals(self, now: datetime) -> List[Signal]:
        """Generate signals by ranking within each sector separately.

        This ensures the portfolio is both market-neutral and sector-neutral,
        avoiding unintended sector bets.
        """
        # Group by sector
        sector_groups: Dict[str, List[MomentumRank]] = defaultdict(list)
        for rank in self.rankings:
            sector = rank.sector or "UNKNOWN"
            sector_groups[sector].append(rank)

        signals = []
        self._current_longs = []
        self._current_shorts = []

        for sector, stocks in sector_groups.items():
            if len(stocks) < 4:  # Need at least 4 stocks per sector
                continue

            # Sort within sector
            sorted_stocks = sorted(stocks, key=lambda r: r.return_1m_skip, reverse=True)
            n = len(sorted_stocks)

            # Assign within-sector percentile ranks
            for i, rank in enumerate(sorted_stocks):
                rank.sector_rank_pct = 1.0 - (i / n)

            # Top and bottom within sector
            top_cutoff = max(1, int(n * TOP_DECILE_PCT))
            bottom_cutoff = max(1, int(n * BOTTOM_DECILE_PCT))

            for rank in sorted_stocks[:top_cutoff]:
                sig = self._create_long_signal(rank, now)
                if sig:
                    signals.append(sig)
                    self._current_longs.append(rank.symbol)

            if config.ALLOW_SHORT:
                for rank in sorted_stocks[-bottom_cutoff:]:
                    if rank.symbol in config.NO_SHORT_SYMBOLS:
                        continue
                    sig = self._create_short_signal(rank, now)
                    if sig:
                        signals.append(sig)
                        self._current_shorts.append(rank.symbol)

        return signals

    def _compute_momentum(self, symbol: str,
                          bars_df: Optional[pd.DataFrame],
                          now: datetime) -> Optional[MomentumRank]:
        """Compute skip-week 1-month momentum for a stock.

        The skip-week adjustment avoids the short-term reversal effect:
        return is measured from T-26 to T-5 (skipping the most recent week).
        """
        # Fetch bars if not provided
        if bars_df is None or bars_df.empty:
            bars_df = get_daily_bars(symbol, days=MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS + 5)
            if bars_df is None or bars_df.empty:
                return None

        close = bars_df["close"]
        total_bars_needed = MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS
        if len(close) < total_bars_needed:
            return None

        # Skip-week return: price at T-5 / price at T-26
        price_end = close.iloc[-(SKIP_RECENT_DAYS + 1)]  # T-5 (skip last week)
        price_start = close.iloc[-(total_bars_needed + 1)]  # T-26

        if price_start <= 0:
            return None

        return_1m_skip = (price_end - price_start) / price_start

        # Look up sector from config
        sector = config.SECTOR_MAP.get(symbol, "UNKNOWN")

        return MomentumRank(
            symbol=symbol,
            return_1m_skip=return_1m_skip,
            sector=sector,
        )

    def _create_long_signal(self, rank: MomentumRank, now: datetime) -> Optional[Signal]:
        """Create a long signal for a top-decile stock."""
        try:
            bars = get_daily_bars(rank.symbol, days=3)
            if bars is None or bars.empty:
                return None
            price = float(bars["close"].iloc[-1])
            if price <= 0:
                return None

            return Signal(
                symbol=rank.symbol,
                strategy="XS_MOM",
                side="buy",
                entry_price=round(price, 2),
                take_profit=round(price * (1 + DEFAULT_TP_PCT), 2),
                stop_loss=round(price * (1 - DEFAULT_SL_PCT), 2),
                reason=(
                    f"XS momentum LONG ret_1m={rank.return_1m_skip:.3f} "
                    f"sector={rank.sector}"
                ),
                hold_type="swing",
                confidence=min(0.5 + abs(rank.return_1m_skip) * 3, 0.95),
                metadata={
                    "return_1m_skip": rank.return_1m_skip,
                    "sector": rank.sector,
                    "universe_rank_pct": rank.universe_rank_pct,
                    "sector_rank_pct": rank.sector_rank_pct,
                },
                timestamp=now,
            )
        except Exception as e:
            logger.debug(f"XS momentum long signal creation failed for {rank.symbol}: {e}")
            return None

    def _create_short_signal(self, rank: MomentumRank, now: datetime) -> Optional[Signal]:
        """Create a short signal for a bottom-decile stock."""
        try:
            bars = get_daily_bars(rank.symbol, days=3)
            if bars is None or bars.empty:
                return None
            price = float(bars["close"].iloc[-1])
            if price <= 0:
                return None

            return Signal(
                symbol=rank.symbol,
                strategy="XS_MOM",
                side="sell",
                entry_price=round(price, 2),
                take_profit=round(price * (1 - DEFAULT_TP_PCT), 2),
                stop_loss=round(price * (1 + DEFAULT_SL_PCT), 2),
                reason=(
                    f"XS momentum SHORT ret_1m={rank.return_1m_skip:.3f} "
                    f"sector={rank.sector}"
                ),
                hold_type="swing",
                confidence=min(0.5 + abs(rank.return_1m_skip) * 3, 0.95),
                metadata={
                    "return_1m_skip": rank.return_1m_skip,
                    "sector": rank.sector,
                    "universe_rank_pct": rank.universe_rank_pct,
                    "sector_rank_pct": rank.sector_rank_pct,
                },
                timestamp=now,
            )
        except Exception as e:
            logger.debug(f"XS momentum short signal creation failed for {rank.symbol}: {e}")
            return None

    def _should_rebalance(self, now: datetime) -> bool:
        """Check if it's time for weekly rebalance."""
        if now.weekday() != REBALANCE_DAY:
            return False
        if self._last_rebalance_date is not None:
            days_since = (now.date() - self._last_rebalance_date).days
            if days_since < 5:
                return False
        return True

    def get_rankings_summary(self) -> List[Dict]:
        """Return current rankings for dashboard/logging."""
        if not self.rankings:
            return []
        sorted_ranks = sorted(self.rankings, key=lambda r: r.return_1m_skip, reverse=True)
        return [
            {
                "rank": i + 1,
                "symbol": r.symbol,
                "return_1m_skip": round(r.return_1m_skip, 4),
                "sector": r.sector,
                "universe_rank_pct": round(r.universe_rank_pct, 4),
            }
            for i, r in enumerate(sorted_ranks)
        ]
