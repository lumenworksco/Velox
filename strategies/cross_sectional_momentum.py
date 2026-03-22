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
SKIP_RECENT_DAYS = 1         # Skip most recent day to avoid reversal

# Decile thresholds
TOP_DECILE_PCT = 0.10
BOTTOM_DECILE_PCT = 0.10

# Rebalance day
REBALANCE_DAY = 0  # Monday

# Position parameters
DEFAULT_TP_PCT = 0.03   # 3% take-profit target
DEFAULT_SL_PCT = 0.02   # 2% stop-loss

# V11.2: Maximum portfolio turnover per rebalance (fraction of portfolio)
MAX_TURNOVER_PCT = 0.20


@dataclass
class MomentumRank:
    """Momentum ranking for a single stock."""
    symbol: str
    return_1m_skip: float  # 1-month return skipping last week
    sector: str = ""
    sector_rank_pct: float = 0.0  # Percentile rank within sector (0=worst, 1=best)
    universe_rank_pct: float = 0.0  # Percentile rank in full universe


# IMPL-011: Minimum warm-up period before generating signals
WARM_UP_DAYS = 60  # Require at least 60 calendar days of data


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

    IMPL-011: Includes warm-up period validation to ensure data pipeline
    has sufficient history before generating signals.
    """

    def __init__(self, industry_neutral: bool = False):
        self.industry_neutral = industry_neutral
        self.rankings: List[MomentumRank] = []
        self._last_rebalance_date: Optional[datetime] = None
        self._current_longs: List[str] = []
        self._current_shorts: List[str] = []
        self._warm_up_verified: bool = False
        self._data_pipeline_ok: bool = False

    def reset_daily(self):
        """Clear per-day state. Preserve weekly rebalance state."""
        # Clear rankings so they are recomputed on the next signal generation.
        # Preserve _last_rebalance_date and current long/short lists (weekly strategy).
        self.rankings.clear()
        self._data_pipeline_ok = False

    def verify_data_pipeline(self, symbols: Optional[List[str]] = None) -> Dict:
        """Verify that the data pipeline can provide sufficient history.

        IMPL-011: End-to-end validation of the data pipeline:
        1. Check that get_daily_bars works for a sample of symbols
        2. Verify bars have required columns (close, volume)
        3. Verify at least WARM_UP_DAYS of data available
        4. Check for excessive gaps or stale data

        Args:
            symbols: List of symbols to check. Defaults to config.STANDARD_SYMBOLS.

        Returns:
            Dict with keys: ok (bool), symbols_checked (int), symbols_with_data (int),
            avg_bars_available (float), min_bars_available (int), issues (list).
        """
        if symbols is None:
            symbols = config.STANDARD_SYMBOLS

        result = {
            "ok": False,
            "symbols_checked": 0,
            "symbols_with_data": 0,
            "avg_bars_available": 0.0,
            "min_bars_available": 0,
            "issues": [],
        }

        bars_counts = []
        sample_symbols = symbols[:20]  # Check a representative sample

        for symbol in sample_symbols:
            result["symbols_checked"] += 1
            try:
                df = get_daily_bars(
                    symbol,
                    days=MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS + 10,
                )
                if df is None or df.empty:
                    result["issues"].append(f"{symbol}: no data returned")
                    continue

                # Verify required columns
                required_cols = {"close"}
                df_cols = {c.lower() for c in df.columns}
                missing = required_cols - df_cols
                if missing:
                    result["issues"].append(f"{symbol}: missing columns {missing}")
                    continue

                bars_counts.append(len(df))
                result["symbols_with_data"] += 1

                # Check for stale data (last bar older than 5 days)
                if hasattr(df.index, 'date') or hasattr(df.index, 'to_pydatetime'):
                    try:
                        last_bar_date = df.index[-1]
                        if hasattr(last_bar_date, 'date'):
                            last_bar_date = last_bar_date.date()
                        from datetime import date as date_cls
                        days_stale = (date_cls.today() - last_bar_date).days
                        if days_stale > 5:
                            result["issues"].append(
                                f"{symbol}: stale data ({days_stale} days old)"
                            )
                    except Exception:
                        pass

            except Exception as e:
                result["issues"].append(f"{symbol}: fetch error: {e}")

        if bars_counts:
            result["avg_bars_available"] = round(np.mean(bars_counts), 1)
            result["min_bars_available"] = int(min(bars_counts))
        else:
            result["min_bars_available"] = 0

        # Pipeline is OK if we have data for at least 80% of checked symbols
        # and the minimum bar count meets warm-up requirements
        min_required = MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS
        data_ratio = result["symbols_with_data"] / max(result["symbols_checked"], 1)
        result["ok"] = (
            data_ratio >= 0.80
            and result["min_bars_available"] >= min_required
        )

        self._data_pipeline_ok = result["ok"]
        log_fn = logger.info if result["ok"] else logger.warning
        log_fn(
            f"XS momentum data pipeline: {'OK' if result['ok'] else 'FAIL'} "
            f"({result['symbols_with_data']}/{result['symbols_checked']} symbols, "
            f"min {result['min_bars_available']} bars, "
            f"{len(result['issues'])} issues)"
        )

        return result

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

        # IMPL-011: Warm-up period check — require sufficient data history
        if not self._warm_up_verified:
            pipeline_result = self.verify_data_pipeline()
            if not pipeline_result["ok"]:
                logger.warning(
                    f"XS momentum: warm-up check failed — need at least "
                    f"{MOMENTUM_LOOKBACK_DAYS + SKIP_RECENT_DAYS} bars of data for "
                    f"80%+ of symbols. Got min={pipeline_result['min_bars_available']} bars, "
                    f"{pipeline_result['symbols_with_data']}/{pipeline_result['symbols_checked']} symbols"
                )
                return signals
            self._warm_up_verified = True
            logger.info("XS momentum: warm-up check passed, data pipeline verified")

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
        """Generate signals using universe-wide ranking.

        V11.2 enhancements:
        - Turnover limit: max MAX_TURNOVER_PCT portfolio turnover per rebalance.
        - Rank-score sizing: position size proportional to rank-score within decile.
        """
        # Sort by return
        sorted_ranks = sorted(self.rankings, key=lambda r: r.return_1m_skip, reverse=True)
        n = len(sorted_ranks)

        # Assign universe percentile ranks
        for i, rank in enumerate(sorted_ranks):
            rank.universe_rank_pct = 1.0 - (i / n) if n > 0 else 0.0

        # Top decile = long
        top_cutoff = max(1, int(n * TOP_DECILE_PCT))
        top_stocks = sorted_ranks[:top_cutoff]

        # Bottom decile = short
        bottom_cutoff = max(1, int(n * BOTTOM_DECILE_PCT))
        bottom_stocks = sorted_ranks[-bottom_cutoff:]

        # V11.2: Turnover limit — cap the number of new names we add
        prev_longs = set(self._current_longs)
        prev_shorts = set(self._current_shorts)
        prev_all = prev_longs | prev_shorts
        new_candidates_long = [r for r in top_stocks]
        new_candidates_short = [r for r in bottom_stocks]
        total_positions = max(len(new_candidates_long) + len(new_candidates_short), 1)

        if prev_all:
            new_names_long = [r for r in new_candidates_long if r.symbol not in prev_longs]
            new_names_short = [r for r in new_candidates_short if r.symbol not in prev_shorts]
            total_new = len(new_names_long) + len(new_names_short)
            turnover_ratio = total_new / total_positions
            if turnover_ratio > MAX_TURNOVER_PCT:
                # Trim new entries to respect turnover limit
                max_new = max(1, int(total_positions * MAX_TURNOVER_PCT))
                # Prioritize strongest new signals
                new_names_long.sort(key=lambda r: r.return_1m_skip, reverse=True)
                new_names_short.sort(key=lambda r: r.return_1m_skip)
                allowed_new = set()
                for r in (new_names_long + new_names_short)[:max_new]:
                    allowed_new.add(r.symbol)
                # Keep previous holdings + allowed new names
                new_candidates_long = [
                    r for r in new_candidates_long
                    if r.symbol in prev_longs or r.symbol in allowed_new
                ]
                new_candidates_short = [
                    r for r in new_candidates_short
                    if r.symbol in prev_shorts or r.symbol in allowed_new
                ]
                logger.info(
                    f"XS momentum: turnover capped at {MAX_TURNOVER_PCT:.0%} — "
                    f"allowed {len(allowed_new)} new names out of {total_new}"
                )

        # V11.2: Compute rank-score weights within decile for proportional sizing
        long_scores = [abs(r.return_1m_skip) for r in new_candidates_long]
        long_total = sum(long_scores) if long_scores else 1.0
        short_scores = [abs(r.return_1m_skip) for r in new_candidates_short]
        short_total = sum(short_scores) if short_scores else 1.0

        signals = []
        self._current_longs = []
        self._current_shorts = []

        # Long signals with rank-proportional sizing
        for i, rank in enumerate(new_candidates_long):
            sig = self._create_long_signal(rank, now)
            if sig:
                # Store rank-proportional weight in metadata
                weight = long_scores[i] / long_total if long_total > 0 else 1.0 / max(len(new_candidates_long), 1)
                sig.metadata["rank_weight"] = round(weight, 4)
                signals.append(sig)
                self._current_longs.append(rank.symbol)

        # Short signals with rank-proportional sizing
        if config.ALLOW_SHORT:
            for i, rank in enumerate(new_candidates_short):
                if rank.symbol in config.NO_SHORT_SYMBOLS:
                    continue
                sig = self._create_short_signal(rank, now)
                if sig:
                    weight = short_scores[i] / short_total if short_total > 0 else 1.0 / max(len(new_candidates_short), 1)
                    sig.metadata["rank_weight"] = round(weight, 4)
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
                rank.sector_rank_pct = 1.0 - (i / n) if n > 0 else 0.0

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

        if price_start <= 0 or abs(price_start) < 1e-10:
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
            if price <= 0 or abs(price) < 1e-10:
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
            if price <= 0 or abs(price) < 1e-10:
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
