"""COMP-008: Industry-neutral momentum strategy (residual momentum).

Computes residual returns after removing industry/sector factor exposure,
then ranks stocks by residual momentum. Generates long signals on strong
positive residuals and short signals on strong negative residuals.

This isolates stock-specific momentum from industry momentum, which
research shows provides more persistent alpha.

Follows the existing strategy pattern: generate_signals(), get_exit_params(),
reset_daily(), get_metadata().

Usage::

    strategy = ResidualMomentumStrategy()
    strategy.prepare_universe(date)
    signals = strategy.generate_signals(bars)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data import get_daily_bars
from strategies.base import ExitParams, Signal, Strategy, StrategyMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_NAME = "RESIDUAL_MOM"

# Sector ETFs used as industry factors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

# Stock-to-sector mapping (common large-caps; extend as needed)
STOCK_SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOG": "XLC", "GOOGL": "XLC",
    "AMZN": "XLY", "META": "XLC", "TSLA": "XLY", "AVGO": "XLK", "AMD": "XLK",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "CAT": "XLI", "HON": "XLI", "UNP": "XLI", "GE": "XLI", "RTX": "XLI",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    "DIS": "XLC", "NFLX": "XLC", "T": "XLC", "VZ": "XLC", "CMCSA": "XLC",
    "LIN": "XLB", "APD": "XLB", "ECL": "XLB", "SHW": "XLB", "NEM": "XLB",
    "PLD": "XLRE", "AMT": "XLRE", "CCI": "XLRE", "EQIX": "XLRE", "SPG": "XLRE",
}

# Momentum lookback windows (trading days)
RESIDUAL_LOOKBACK = 63         # ~3 months for residual computation
MOMENTUM_WINDOW = 21           # ~1 month for momentum ranking
SKIP_RECENT = 5                # Skip most recent 5 days (reversal buffer)

# Signal thresholds
LONG_PERCENTILE = 80           # Top 20% residual momentum -> long
SHORT_PERCENTILE = 20          # Bottom 20% -> short

# Position sizing
DEFAULT_ALLOCATION = 0.10       # 10% of portfolio allocation
MAX_POSITIONS = 6               # Max concurrent positions

# Stop/target parameters
STOP_LOSS_PCT = 0.04            # 4% stop loss
TAKE_PROFIT_PCT = 0.08          # 8% take profit

# Minimum R-squared for valid sector regression
MIN_R_SQUARED = 0.05

# Minimum bars required
MIN_BARS = RESIDUAL_LOOKBACK + SKIP_RECENT + 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ResidualScore:
    """Residual momentum score for a single stock."""
    symbol: str
    sector_etf: str
    residual_momentum: float       # Cumulative residual return
    raw_momentum: float            # Raw (unadjusted) momentum
    sector_beta: float             # Regression beta to sector
    r_squared: float               # Regression R-squared
    residual_vol: float            # Volatility of residuals
    z_score: float                 # Residual momentum z-score


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class ResidualMomentumStrategy(Strategy):
    """Industry-neutral residual momentum strategy.

    Lifecycle:
        1. prepare_universe() — select stocks with sector mappings
        2. generate_signals() — compute residual momentum, rank, signal extremes
        3. get_exit_params() — check stops and targets
        4. reset_daily() — clear per-day state
    """

    def __init__(self):
        self._universe: List[str] = []
        self._sector_bars: Dict[str, pd.DataFrame] = {}
        self._scores: Dict[str, ResidualScore] = {}
        self._triggered_today: set = set()
        self._last_universe_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def prepare_universe(self, date) -> List[str]:
        """Select stocks that have a known sector mapping."""
        date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

        if self._last_universe_date == date_str and self._universe:
            return self._universe

        # Use stocks from the sector map that are also in the dynamic universe
        self._universe = list(STOCK_SECTOR_MAP.keys())
        self._last_universe_date = date_str

        # Pre-fetch sector ETF bars
        self._fetch_sector_bars()

        logger.info("ResidualMomentum universe: %d stocks", len(self._universe))
        return self._universe

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Compute residual momentum and generate signals on extremes.

        Steps:
            1. For each stock, regress returns on sector ETF returns
            2. Extract residuals (stock-specific returns)
            3. Compute cumulative residual over momentum window
            4. Rank and signal top/bottom percentiles
        """
        if not self._universe:
            return []

        scores: List[ResidualScore] = []

        for symbol in self._universe:
            if symbol in self._triggered_today:
                continue

            stock_bars = bars.get(symbol)
            if stock_bars is None or len(stock_bars) < MIN_BARS:
                continue

            try:
                score = self._compute_residual_score(symbol, stock_bars)
                if score:
                    scores.append(score)
                    self._scores[symbol] = score
            except Exception as exc:
                logger.debug("Residual computation failed for %s: %s", symbol, exc)

        if len(scores) < 5:
            logger.debug("Too few scored stocks (%d) for residual momentum", len(scores))
            return []

        # Rank by residual momentum
        signals = self._generate_signals_from_scores(scores, bars)
        return signals

    def get_exit_params(
        self, trade, current_price: float, bars: Optional[pd.DataFrame] = None
    ) -> ExitParams:
        """Check stops and targets for open positions."""
        entry = trade.entry_price
        side = trade.side

        if side == "buy":
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry

        # Stop loss
        if pnl_pct <= -STOP_LOSS_PCT:
            return ExitParams(
                should_exit=True,
                exit_reason=f"Stop loss hit ({pnl_pct:.1%})",
            )

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            return ExitParams(
                should_exit=True,
                exit_reason=f"Take profit hit ({pnl_pct:.1%})",
            )

        # Trailing stop: if > 50% of target, move stop to breakeven
        if pnl_pct >= TAKE_PROFIT_PCT * 0.5:
            new_stop = entry * (1.001 if side == "buy" else 0.999)
            return ExitParams(
                should_exit=False,
                new_stop_loss=new_stop,
            )

        return ExitParams(should_exit=False)

    def reset_daily(self):
        """Clear per-day state."""
        self._triggered_today.clear()
        self._scores.clear()
        logger.debug("ResidualMomentum daily state reset")

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name=STRATEGY_NAME,
            version="1.0",
            description="Industry-neutral residual momentum — longs strong residuals, shorts weak",
            default_allocation=DEFAULT_ALLOCATION,
            min_bars_required=MIN_BARS,
            supported_sides=["long", "short"],
            timeframe="1d",
            max_positions=MAX_POSITIONS,
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _fetch_sector_bars(self) -> None:
        """Fetch daily bars for all sector ETFs."""
        for etf in SECTOR_ETFS:
            if etf in self._sector_bars:
                continue
            try:
                bars = get_daily_bars(etf, limit=RESIDUAL_LOOKBACK + 20)
                if bars is not None and len(bars) >= MIN_BARS:
                    self._sector_bars[etf] = bars
            except Exception as exc:
                logger.debug("Failed to fetch sector ETF %s: %s", etf, exc)

    def _compute_residual_score(
        self,
        symbol: str,
        stock_bars: pd.DataFrame,
    ) -> Optional[ResidualScore]:
        """Compute residual momentum score for a single stock.

        1. Get the stock's sector ETF
        2. Compute daily returns for both
        3. Regress stock returns on sector returns
        4. Extract residuals
        5. Compute cumulative residual return over momentum window
        """
        sector_etf = STOCK_SECTOR_MAP.get(symbol)
        if not sector_etf or sector_etf not in self._sector_bars:
            return None

        sector_bars = self._sector_bars[sector_etf]

        # Compute returns
        try:
            stock_close = stock_bars["close"] if "close" in stock_bars.columns else stock_bars["Close"]
            sector_close = sector_bars["close"] if "close" in sector_bars.columns else sector_bars["Close"]
        except (KeyError, AttributeError):
            return None

        stock_returns = stock_close.pct_change().dropna()
        sector_returns = sector_close.pct_change().dropna()

        # Align
        aligned = pd.concat(
            [stock_returns.rename("stock"), sector_returns.rename("sector")],
            axis=1,
        ).dropna()

        if len(aligned) < MIN_BARS - 10:
            return None

        # Use last RESIDUAL_LOOKBACK days
        aligned = aligned.iloc[-RESIDUAL_LOOKBACK:]
        x = aligned["sector"].values
        y = aligned["stock"].values

        # OLS regression: stock_return = alpha + beta * sector_return + residual
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        alpha, beta = beta_hat[0], beta_hat[1]
        predicted = X @ beta_hat
        residuals = y - predicted

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        if r_squared < MIN_R_SQUARED:
            # Very low R-squared means the sector doesn't explain the stock well
            # Still compute residuals, they're just mostly raw returns
            pass

        # Residual momentum: cumulative residual over momentum window, skipping recent
        if len(residuals) < MOMENTUM_WINDOW + SKIP_RECENT:
            return None

        momentum_residuals = residuals[-(MOMENTUM_WINDOW + SKIP_RECENT):-SKIP_RECENT]
        residual_momentum = float(np.sum(momentum_residuals))

        # Raw momentum for comparison
        raw_returns = y[-(MOMENTUM_WINDOW + SKIP_RECENT):-SKIP_RECENT]
        raw_momentum = float(np.sum(raw_returns))

        # Residual volatility
        residual_vol = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        # Z-score of residual momentum
        z_score = residual_momentum / residual_vol if residual_vol > 1e-10 else 0.0

        return ResidualScore(
            symbol=symbol,
            sector_etf=sector_etf,
            residual_momentum=residual_momentum,
            raw_momentum=raw_momentum,
            sector_beta=float(beta),
            r_squared=float(r_squared),
            residual_vol=residual_vol,
            z_score=z_score,
        )

    def _generate_signals_from_scores(
        self,
        scores: List[ResidualScore],
        bars: Dict[str, pd.DataFrame],
    ) -> List[Signal]:
        """Rank scores and generate signals for extreme percentiles."""
        # Sort by residual momentum
        scores.sort(key=lambda s: s.residual_momentum)

        n = len(scores)
        long_cutoff = int(n * LONG_PERCENTILE / 100)
        short_cutoff = int(n * SHORT_PERCENTILE / 100)

        signals: List[Signal] = []

        # Long signals (top residual momentum)
        for score in scores[long_cutoff:]:
            signal = self._create_signal(score, "buy", bars)
            if signal:
                signals.append(signal)

        # Short signals (bottom residual momentum) — if shorting is allowed
        if getattr(config, "ALLOW_SHORT", False):
            for score in scores[:short_cutoff]:
                signal = self._create_signal(score, "sell", bars)
                if signal:
                    signals.append(signal)

        return signals

    def _create_signal(
        self,
        score: ResidualScore,
        side: str,
        bars: Dict[str, pd.DataFrame],
    ) -> Optional[Signal]:
        """Create a Signal from a ResidualScore."""
        stock_bars = bars.get(score.symbol)
        if stock_bars is None or len(stock_bars) < 1:
            return None

        try:
            close_col = "close" if "close" in stock_bars.columns else "Close"
            entry_price = float(stock_bars[close_col].iloc[-1])
        except (KeyError, IndexError):
            return None

        if entry_price <= 0:
            return None

        if side == "buy":
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)
            take_profit = entry_price * (1 - TAKE_PROFIT_PCT)

        # Confidence from z-score magnitude (capped at 1.0)
        confidence = min(abs(score.z_score) / 3.0, 1.0)

        self._triggered_today.add(score.symbol)

        return Signal(
            symbol=score.symbol,
            strategy=STRATEGY_NAME,
            side=side,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            reason=(
                f"Residual momentum z={score.z_score:.2f} "
                f"(sector={score.sector_etf}, beta={score.sector_beta:.2f}, "
                f"R2={score.r_squared:.2f})"
            ),
            hold_type="swing",
            confidence=confidence,
            metadata={
                "residual_momentum": score.residual_momentum,
                "raw_momentum": score.raw_momentum,
                "sector_etf": score.sector_etf,
                "sector_beta": score.sector_beta,
                "r_squared": score.r_squared,
                "residual_vol": score.residual_vol,
                "z_score": score.z_score,
            },
            timestamp=datetime.now(),
        )
