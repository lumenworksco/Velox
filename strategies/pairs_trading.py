"""V4: Statistical Pairs Trading — market-neutral cointegration strategy."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from data import get_daily_bars, get_snapshot
from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class PairRelationship:
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    mean_spread: float
    std_spread: float
    coint_pvalue: float
    correlation: float


class PairsTradingStrategy:
    """Statistical pairs trading using cointegration.

    Pair discovery runs weekly (Sunday midnight).
    Signal generation runs intraday during market hours.
    """

    def __init__(self):
        self.pairs: list[PairRelationship] = []
        self.active_pair_ids: set[str] = set()  # Currently open pair_ids
        self.last_discovery: datetime | None = None

    def reset_daily(self):
        pass  # Pairs span multiple days — no daily reset needed

    def discover_pairs(self, symbols: list[str], now: datetime):
        """Run cointegration tests on top liquid symbols.

        Called weekly on Sunday midnight. Computationally expensive —
        limited to top PAIRS_DISCOVERY_SYMBOLS symbols.
        """
        if not config.PAIRS_TRADING_ENABLED:
            return

        logger.info(f"Pairs discovery: testing {len(symbols[:config.PAIRS_DISCOVERY_SYMBOLS])} symbols...")

        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            logger.error("statsmodels not installed — pairs trading disabled")
            return

        # Fetch 90 days of daily closes for each symbol
        prices = {}
        test_symbols = symbols[:config.PAIRS_DISCOVERY_SYMBOLS]

        for symbol in test_symbols:
            try:
                bars = get_daily_bars(symbol, days=100)
                if len(bars) >= 60:
                    prices[symbol] = bars["close"]
            except Exception:
                continue

        if len(prices) < 10:
            logger.warning(f"Pairs discovery: only {len(prices)} symbols with enough data")
            return

        # Test all pairs for cointegration
        syms = list(prices.keys())
        candidates = []

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s1, s2 = syms[i], syms[j]
                try:
                    # Align price series
                    p1 = prices[s1]
                    p2 = prices[s2]
                    aligned = pd.concat([p1, p2], axis=1, join="inner").dropna()
                    if len(aligned) < 60:
                        continue

                    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

                    # Cointegration test
                    score, pvalue, _ = coint(a.values, b.values)
                    if pvalue >= config.PAIRS_COINT_PVALUE:
                        continue

                    # Correlation check
                    corr = a.corr(b)
                    if corr < config.PAIRS_MIN_CORRELATION:
                        continue

                    # Hedge ratio via simple OLS: s1 = hedge_ratio * s2 + residual
                    hedge_ratio = float(np.polyfit(b.values, a.values, 1)[0])

                    # Spread statistics
                    spread = a.values - hedge_ratio * b.values
                    mean_spread = float(np.mean(spread))
                    std_spread = float(np.std(spread))

                    if std_spread == 0:
                        continue

                    candidates.append(PairRelationship(
                        symbol_a=s1,
                        symbol_b=s2,
                        hedge_ratio=hedge_ratio,
                        mean_spread=mean_spread,
                        std_spread=std_spread,
                        coint_pvalue=pvalue,
                        correlation=corr,
                    ))
                except Exception:
                    continue

        # Keep top 20 most cointegrated pairs
        candidates.sort(key=lambda p: p.coint_pvalue)
        self.pairs = candidates[:20]
        self.last_discovery = now

        logger.info(
            f"Pairs discovery complete: {len(self.pairs)} cointegrated pairs found "
            f"(from {len(syms)} symbols)"
        )
        for p in self.pairs[:5]:
            logger.info(
                f"  {p.symbol_a}/{p.symbol_b}: p={p.coint_pvalue:.4f}, "
                f"corr={p.correlation:.3f}, hedge={p.hedge_ratio:.3f}"
            )

    def revalidate_pairs(self):
        """Remove pairs whose correlation has dropped below threshold."""
        if not self.pairs:
            return

        valid = []
        for pair in self.pairs:
            try:
                bars_a = get_daily_bars(pair.symbol_a, days=35)
                bars_b = get_daily_bars(pair.symbol_b, days=35)
                aligned = pd.concat(
                    [bars_a["close"], bars_b["close"]], axis=1, join="inner"
                ).dropna()
                if len(aligned) < 20:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if corr >= config.PAIRS_REVALIDATION_THRESHOLD:
                    pair.correlation = corr
                    valid.append(pair)
                else:
                    logger.info(f"Pair {pair.symbol_a}/{pair.symbol_b} dropped: corr={corr:.3f}")
            except Exception:
                valid.append(pair)  # Keep on error (fail-open)

        self.pairs = valid

    def scan(self, symbols: list[str], now: datetime) -> list[Signal]:
        """Check z-scores for all discovered pairs and generate signals."""
        if not config.PAIRS_TRADING_ENABLED or not self.pairs:
            return []

        signals = []

        for pair in self.pairs:
            try:
                # Skip if already in this pair
                pair_key = f"{pair.symbol_a}_{pair.symbol_b}"
                if pair_key in self.active_pair_ids:
                    continue

                # Get current prices
                snap_a = get_snapshot(pair.symbol_a)
                snap_b = get_snapshot(pair.symbol_b)
                if not snap_a or not snap_b:
                    continue

                price_a = float(snap_a.latest_trade.price) if snap_a.latest_trade else None
                price_b = float(snap_b.latest_trade.price) if snap_b.latest_trade else None
                if not price_a or not price_b:
                    continue

                # Calculate current spread and z-score
                current_spread = price_a - pair.hedge_ratio * price_b
                z_score = (current_spread - pair.mean_spread) / pair.std_spread

                if abs(z_score) < config.PAIRS_ZSCORE_ENTRY:
                    continue

                # Generate linked pair signals
                pair_id = f"pair_{pair.symbol_a}_{pair.symbol_b}_{uuid.uuid4().hex[:8]}"

                if z_score > config.PAIRS_ZSCORE_ENTRY:
                    # Spread too wide: short A, long B
                    stop_spread = pair.mean_spread + config.PAIRS_ZSCORE_STOP * pair.std_spread
                    stop_pct_a = abs(stop_spread - current_spread) / price_a

                    signals.append(Signal(
                        symbol=pair.symbol_a,
                        strategy="PAIRS",
                        side="sell",
                        entry_price=round(price_a, 2),
                        take_profit=round(price_a * (1 - 0.02), 2),
                        stop_loss=round(price_a * (1 + stop_pct_a), 2),
                        reason=f"Pairs short: z={z_score:.2f} vs {pair.symbol_b}",
                        hold_type="swing",
                        pair_id=pair_id,
                    ))
                    signals.append(Signal(
                        symbol=pair.symbol_b,
                        strategy="PAIRS",
                        side="buy",
                        entry_price=round(price_b, 2),
                        take_profit=round(price_b * (1 + 0.02), 2),
                        stop_loss=round(price_b * (1 - stop_pct_a), 2),
                        reason=f"Pairs long: z={z_score:.2f} vs {pair.symbol_a}",
                        hold_type="swing",
                        pair_id=pair_id,
                    ))

                elif z_score < -config.PAIRS_ZSCORE_ENTRY:
                    # Spread too narrow: long A, short B
                    stop_spread = pair.mean_spread - config.PAIRS_ZSCORE_STOP * pair.std_spread
                    stop_pct_a = abs(current_spread - stop_spread) / price_a

                    signals.append(Signal(
                        symbol=pair.symbol_a,
                        strategy="PAIRS",
                        side="buy",
                        entry_price=round(price_a, 2),
                        take_profit=round(price_a * (1 + 0.02), 2),
                        stop_loss=round(price_a * (1 - stop_pct_a), 2),
                        reason=f"Pairs long: z={z_score:.2f} vs {pair.symbol_b}",
                        hold_type="swing",
                        pair_id=pair_id,
                    ))
                    signals.append(Signal(
                        symbol=pair.symbol_b,
                        strategy="PAIRS",
                        side="sell",
                        entry_price=round(price_b, 2),
                        take_profit=round(price_b * (1 - 0.02), 2),
                        stop_loss=round(price_b * (1 + stop_pct_a), 2),
                        reason=f"Pairs short: z={z_score:.2f} vs {pair.symbol_a}",
                        hold_type="swing",
                        pair_id=pair_id,
                    ))

                if signals:
                    self.active_pair_ids.add(pair_key)

            except Exception as e:
                logger.warning(f"Pairs scan error for {pair.symbol_a}/{pair.symbol_b}: {e}")

        return signals

    def check_pair_exits(self, open_trades: dict, now: datetime) -> list[str]:
        """Check if any pairs should be exited based on z-score convergence or max hold.

        Returns list of symbols to close.
        """
        symbols_to_close = []

        # Group open trades by pair_id
        pair_trades: dict[str, list] = {}
        for symbol, trade in open_trades.items():
            if trade.strategy == "PAIRS" and trade.pair_id:
                pair_trades.setdefault(trade.pair_id, []).append(trade)

        for pair_id, trades in pair_trades.items():
            if len(trades) != 2:
                continue

            # Check max hold
            for trade in trades:
                hold_days = (now - trade.entry_time).days
                if hold_days >= config.PAIRS_MAX_HOLD_DAYS:
                    for t in trades:
                        symbols_to_close.append(t.symbol)
                    # Remove from active pairs
                    pair_key = "_".join(pair_id.split("_")[1:3])
                    self.active_pair_ids.discard(pair_key)
                    break

            # Check z-score convergence
            if trades[0].symbol not in [s for s in symbols_to_close]:
                try:
                    # Find the pair relationship
                    t1, t2 = trades[0], trades[1]
                    pair = None
                    for p in self.pairs:
                        if {p.symbol_a, p.symbol_b} == {t1.symbol, t2.symbol}:
                            pair = p
                            break

                    if pair:
                        snap_a = get_snapshot(pair.symbol_a)
                        snap_b = get_snapshot(pair.symbol_b)
                        if snap_a and snap_b and snap_a.latest_trade and snap_b.latest_trade:
                            price_a = float(snap_a.latest_trade.price)
                            price_b = float(snap_b.latest_trade.price)
                            spread = price_a - pair.hedge_ratio * price_b
                            z_score = (spread - pair.mean_spread) / pair.std_spread

                            # Exit if converged or diverged beyond stop
                            if abs(z_score) < config.PAIRS_ZSCORE_EXIT:
                                for t in trades:
                                    symbols_to_close.append(t.symbol)
                                pair_key = f"{pair.symbol_a}_{pair.symbol_b}"
                                self.active_pair_ids.discard(pair_key)
                            elif abs(z_score) > config.PAIRS_ZSCORE_STOP:
                                for t in trades:
                                    symbols_to_close.append(t.symbol)
                                pair_key = f"{pair.symbol_a}_{pair.symbol_b}"
                                self.active_pair_ids.discard(pair_key)
                except Exception as e:
                    logger.warning(f"Pairs exit check error: {e}")

        return symbols_to_close
