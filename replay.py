"""Market Replay & Config Comparison — replay historical trading days through the signal pipeline."""

import logging
import math
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import config

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# Subset of liquid symbols for replay (fetching all 50+ is slow)
_REPLAY_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "AMD", "NFLX", "COIN", "PLTR", "SOFI", "JPM", "BAC",
]


# ============================================================
# Result dataclasses
# ============================================================

@dataclass
class ReplayResult:
    """Result of replaying a single trading day."""
    date: str
    signals: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    config_used: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two replay runs."""
    date: str
    result_a: ReplayResult = field(default_factory=lambda: ReplayResult(date=""))
    result_b: ReplayResult = field(default_factory=lambda: ReplayResult(date=""))
    delta_pnl: float = 0.0
    delta_sharpe: float = 0.0
    delta_win_rate: float = 0.0


# ============================================================
# Simulated trade tracking
# ============================================================

@dataclass
class _SimTrade:
    """Internal: tracks a simulated position during replay."""
    symbol: str
    strategy: str
    side: str
    entry_price: float
    take_profit: float
    stop_loss: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(ET))
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""
    pnl: float = 0.0


# ============================================================
# MarketReplay
# ============================================================

class MarketReplay:
    """
    Replays a historical trading day through the full signal pipeline.
    Fetches real minute bars from Alpaca and simulates ORB / VWAP signals.
    """

    def __init__(self, symbols: list[str] | None = None):
        self._scan_interval_minutes = max(config.SCAN_INTERVAL_SEC // 60, 1)
        self._symbols = symbols or _REPLAY_SYMBOLS

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def replay_day(self, date: str, config_overrides: dict | None = None) -> ReplayResult:
        """
        Replays the given trading day with optional config overrides.
        Returns: all signals generated, risk decisions, simulated fills, P&L.
        """
        if not getattr(config, "REPLAY_ENABLED", True):
            logger.info("Replay disabled via config")
            return ReplayResult(date=date)

        overrides = config_overrides or {}
        result = ReplayResult(date=date, config_used=overrides)

        originals = {}
        try:
            # Apply config overrides
            originals = self._apply_overrides(overrides)

            # Load real minute bar data from Alpaca
            bars_by_symbol = self._load_bars(date)
            if not bars_by_symbol:
                logger.warning(f"No bar data found for {date}")
                return result

            # Simulate time progression
            open_trades: dict[str, _SimTrade] = {}
            all_signals: list[dict] = []
            all_trades: list[dict] = []

            scan_times = self._generate_scan_times(date)

            # Pre-compute ORB ranges (9:30-10:00) for each symbol
            orb_ranges = self._compute_orb_ranges(bars_by_symbol)

            for sim_time in scan_times:
                try:
                    # Generate signals at this scan time
                    signals = self._generate_signals(
                        sim_time, bars_by_symbol, orb_ranges, overrides
                    )
                    for sig in signals:
                        sig_dict = {
                            "time": sim_time.isoformat(),
                            "symbol": sig.get("symbol", ""),
                            "strategy": sig.get("strategy", ""),
                            "side": sig.get("side", ""),
                            "entry_price": sig.get("entry_price", 0.0),
                            "take_profit": sig.get("take_profit", 0.0),
                            "stop_loss": sig.get("stop_loss", 0.0),
                        }
                        all_signals.append(sig_dict)

                        # Open simulated position if not already in one for this symbol
                        symbol = sig.get("symbol", "")
                        if symbol and symbol not in open_trades:
                            trade = _SimTrade(
                                symbol=symbol,
                                strategy=sig.get("strategy", ""),
                                side=sig.get("side", "buy"),
                                entry_price=sig.get("entry_price", 0.0),
                                take_profit=sig.get("take_profit", 0.0),
                                stop_loss=sig.get("stop_loss", 0.0),
                                entry_time=sim_time,
                            )
                            open_trades[symbol] = trade

                    # Check open trades against bar high/low for TP/SL
                    closed = self._check_exits(sim_time, open_trades, bars_by_symbol)
                    for t in closed:
                        all_trades.append(self._trade_to_dict(t))
                        open_trades.pop(t.symbol, None)

                except Exception as e:
                    logger.warning(f"Replay scan error at {sim_time}: {e}")
                    continue

            # Force close remaining positions at market close
            close_time = datetime.strptime(date, "%Y-%m-%d").replace(
                hour=16, minute=0, tzinfo=ET
            )
            for symbol, trade in list(open_trades.items()):
                try:
                    close_price = self._get_price_at_time(
                        symbol, close_time, bars_by_symbol
                    )
                    if close_price:
                        trade.exit_price = close_price
                        trade.exit_time = close_time
                        trade.exit_reason = "market_close"
                        trade.pnl = self._calc_pnl(trade)
                        all_trades.append(self._trade_to_dict(trade))
                except Exception as e:
                    logger.warning(f"Error closing {symbol} at EOD: {e}")

            result.signals = all_signals
            result.trades = all_trades
            result.total_pnl = sum(t.get("pnl", 0.0) for t in all_trades)
            result.win_rate = self._calc_win_rate(all_trades)
            result.sharpe = self._calc_sharpe(all_trades)

        except Exception as e:
            logger.error(f"Replay failed for {date}: {e}")
        finally:
            self._restore_overrides(originals)

        return result

    def compare_configs(self, date: str, config_a: dict, config_b: dict) -> ComparisonResult:
        """
        Runs the same day through two different configs.
        Returns side-by-side comparison of signals, trades, and P&L.
        """
        result = ComparisonResult(date=date)
        try:
            result.result_a = self.replay_day(date, config_a)
            result.result_b = self.replay_day(date, config_b)
            result.delta_pnl = result.result_a.total_pnl - result.result_b.total_pnl
            result.delta_sharpe = result.result_a.sharpe - result.result_b.sharpe
            result.delta_win_rate = result.result_a.win_rate - result.result_b.win_rate
        except Exception as e:
            logger.error(f"Config comparison failed for {date}: {e}")
        return result

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------

    def _load_bars(self, date: str) -> dict[str, pd.DataFrame]:
        """Fetch real 1-minute bars from Alpaca for all replay symbols."""
        try:
            from data import get_intraday_bars
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as e:
            logger.error(f"Cannot import data module for replay: {e}")
            return {}

        dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ET)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

        bars_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in self._symbols:
            try:
                df = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), market_open, market_close)
                if df is not None and not df.empty:
                    bars_by_symbol[symbol] = df
            except Exception as e:
                logger.debug(f"No bars for {symbol} on {date}: {e}")
                continue

        logger.info(f"Loaded bars for {len(bars_by_symbol)} symbols on {date}")
        return bars_by_symbol

    # ----------------------------------------------------------
    # ORB range computation
    # ----------------------------------------------------------

    def _compute_orb_ranges(self, bars_by_symbol: dict[str, pd.DataFrame]) -> dict:
        """Compute the 9:30-10:00 opening range for each symbol."""
        orb = {}
        for symbol, df in bars_by_symbol.items():
            try:
                # Convert index to ET for time filtering
                idx = df.index
                if hasattr(idx, 'tz') and idx.tz is not None:
                    et_times = idx.tz_convert(ET)
                else:
                    et_times = idx.tz_localize(ET)

                # Filter to 9:30-9:59 ET
                orb_mask = (et_times.hour == 9) & (et_times.minute >= 30)
                orb_bars = df[orb_mask]
                if orb_bars.empty or len(orb_bars) < 5:
                    continue

                orb[symbol] = {
                    "high": float(orb_bars["high"].max()),
                    "low": float(orb_bars["low"].min()),
                    "volume": float(orb_bars["volume"].sum()),
                }
            except Exception as e:
                logger.debug(f"ORB range error for {symbol}: {e}")
        return orb

    # ----------------------------------------------------------
    # Signal generation
    # ----------------------------------------------------------

    def _generate_signals(self, sim_time: datetime, bars: dict[str, pd.DataFrame],
                          orb_ranges: dict, overrides: dict) -> list[dict]:
        """
        Generate signals for the current scan time using ORB and VWAP logic.

        - ORB (10:00-11:30): breakout above/below opening range
        - VWAP (10:30-15:30): reversion when price deviates >1 std from VWAP
        """
        signals = []
        sim_hour = sim_time.hour
        sim_minute = sim_time.minute

        try:
            for symbol, df in bars.items():
                # Get bars up to current sim_time
                bars_so_far = self._bars_up_to(df, sim_time)
                if bars_so_far.empty:
                    continue

                current_close = float(bars_so_far["close"].iloc[-1])
                current_high = float(bars_so_far["high"].iloc[-1])
                current_low = float(bars_so_far["low"].iloc[-1])

                # --- ORB signals (10:00 - 11:30) ---
                orb_active_until = getattr(config, "ORB_ACTIVE_UNTIL", time(11, 30))
                if (sim_hour == 10 or (sim_hour == 11 and sim_minute <= 30)):
                    if sim_time <= sim_time.replace(
                            hour=orb_active_until.hour,
                            minute=orb_active_until.minute):
                        orb_range = orb_ranges.get(symbol)
                        if orb_range:
                            orb_high = orb_range["high"]
                            orb_low = orb_range["low"]
                            orb_width = orb_high - orb_low
                            if orb_width > 0:
                                tp_mult = getattr(config, "ORB_TP_MULT", 1.5)
                                # Long breakout
                                if current_close > orb_high * 1.001:  # 0.1% buffer
                                    signals.append({
                                        "symbol": symbol,
                                        "strategy": "ORB",
                                        "side": "buy",
                                        "entry_price": current_close,
                                        "take_profit": current_close + orb_width * tp_mult,
                                        "stop_loss": orb_low,
                                    })
                                # Short breakdown
                                elif current_close < orb_low * 0.999:
                                    signals.append({
                                        "symbol": symbol,
                                        "strategy": "ORB",
                                        "side": "sell",
                                        "entry_price": current_close,
                                        "take_profit": current_close - orb_width * tp_mult,
                                        "stop_loss": orb_high,
                                    })

                # --- VWAP mean reversion signals (10:30 - 15:30) ---
                if ((sim_hour == 10 and sim_minute >= 30) or
                    (11 <= sim_hour <= 14) or
                    (sim_hour == 15 and sim_minute <= 30)):
                    vwap_data = self._compute_vwap(bars_so_far)
                    if vwap_data:
                        vwap, upper_band, lower_band = vwap_data
                        mr_rr = getattr(config, "MR_MIN_RR_RATIO", 1.5)
                        # Price below lower band → long reversion to VWAP
                        if current_close < lower_band and vwap > 0:
                            sl_dist = lower_band - current_close + (upper_band - lower_band) * 0.2
                            signals.append({
                                "symbol": symbol,
                                "strategy": "VWAP",
                                "side": "buy",
                                "entry_price": current_close,
                                "take_profit": vwap,
                                "stop_loss": current_close - sl_dist,
                            })
                        # Price above upper band → short reversion to VWAP
                        elif current_close > upper_band and vwap > 0:
                            sl_dist = current_close - upper_band + (upper_band - lower_band) * 0.2
                            signals.append({
                                "symbol": symbol,
                                "strategy": "VWAP",
                                "side": "sell",
                                "entry_price": current_close,
                                "take_profit": vwap,
                                "stop_loss": current_close + sl_dist,
                            })

        except Exception as e:
            logger.warning(f"Signal generation error at {sim_time}: {e}")
        return signals

    # ----------------------------------------------------------
    # Exit checking
    # ----------------------------------------------------------

    def _check_exits(self, sim_time: datetime, open_trades: dict[str, _SimTrade],
                     bars: dict[str, pd.DataFrame]) -> list[_SimTrade]:
        """Check if any open trades hit TP/SL using bar high/low (not just close)."""
        closed = []
        for symbol, trade in list(open_trades.items()):
            try:
                bar_data = self._get_bar_at_time(symbol, sim_time, bars)
                if bar_data is None:
                    continue

                bar_high, bar_low, bar_close = bar_data

                if trade.side == "buy":
                    # Check stop loss first (conservative: assume SL hit before TP in same bar)
                    if bar_low <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = sim_time
                        trade.exit_reason = "stop_loss"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                    elif bar_high >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = sim_time
                        trade.exit_reason = "take_profit"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                else:  # sell/short
                    if bar_high >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = sim_time
                        trade.exit_reason = "stop_loss"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
                    elif bar_low <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.exit_time = sim_time
                        trade.exit_reason = "take_profit"
                        trade.pnl = self._calc_pnl(trade)
                        closed.append(trade)
            except Exception as e:
                logger.warning(f"Exit check error for {symbol}: {e}")
        return closed

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _apply_overrides(self, overrides: dict) -> dict:
        """Apply config overrides, return original values for restoration."""
        originals = {}
        for key, value in overrides.items():
            if hasattr(config, key):
                originals[key] = getattr(config, key)
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
        return originals

    def _restore_overrides(self, originals: dict):
        """Restore original config values."""
        for key, value in originals.items():
            setattr(config, key, value)

    def _generate_scan_times(self, date: str) -> list[datetime]:
        """Generate scan timestamps from 9:30 to 16:00 ET."""
        try:
            base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ET)
        except ValueError:
            return []

        times = []
        current = base.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = base.replace(hour=16, minute=0, second=0, microsecond=0)

        while current <= market_close:
            times.append(current)
            current += timedelta(minutes=self._scan_interval_minutes)
        return times

    def _bars_up_to(self, df: pd.DataFrame, sim_time: datetime) -> pd.DataFrame:
        """Return bars up to and including sim_time."""
        try:
            # Handle timezone-aware vs naive index
            idx = df.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                cutoff = sim_time if sim_time.tzinfo else sim_time.replace(tzinfo=ET)
            else:
                cutoff = sim_time.replace(tzinfo=None) if sim_time.tzinfo else sim_time
            return df[df.index <= cutoff]
        except Exception:
            return df

    def _get_bar_at_time(self, symbol: str, sim_time: datetime,
                         bars: dict[str, pd.DataFrame]) -> tuple[float, float, float] | None:
        """Get (high, low, close) for the bar nearest to sim_time."""
        df = bars.get(symbol)
        if df is None or df.empty:
            return None

        try:
            # Handle timezone
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                target = sim_time if sim_time.tzinfo else sim_time.replace(tzinfo=ET)
            else:
                target = sim_time.replace(tzinfo=None) if sim_time.tzinfo else sim_time

            # Find nearest bar within scan interval
            idx = df.index.get_indexer([target], method="nearest")[0]
            if idx < 0 or idx >= len(df):
                return None
            row = df.iloc[idx]
            return (float(row["high"]), float(row["low"]), float(row["close"]))
        except Exception:
            return None

    def _get_price_at_time(self, symbol: str, sim_time: datetime,
                           bars: dict[str, pd.DataFrame]) -> float | None:
        """Get the close price for the bar nearest to sim_time."""
        result = self._get_bar_at_time(symbol, sim_time, bars)
        return result[2] if result else None

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> tuple[float, float, float] | None:
        """Compute VWAP and +-1 std bands from bar data."""
        try:
            if df.empty or "volume" not in df.columns or "close" not in df.columns:
                return None
            vol = df["volume"].values
            typical = (df["high"].values + df["low"].values + df["close"].values) / 3.0
            cum_vol = vol.cumsum()
            cum_vp = (typical * vol).cumsum()
            if cum_vol[-1] == 0:
                return None
            vwap = cum_vp[-1] / cum_vol[-1]
            # Standard deviation band
            cum_vp2 = (typical ** 2 * vol).cumsum()
            variance = cum_vp2[-1] / cum_vol[-1] - vwap ** 2
            std = math.sqrt(max(variance, 0))
            return (vwap, vwap + std, vwap - std)
        except Exception:
            return None

    @staticmethod
    def _calc_pnl(trade: _SimTrade) -> float:
        """Calculate P&L for a trade (per-share)."""
        if trade.exit_price is None:
            return 0.0
        if trade.side == "buy":
            return trade.exit_price - trade.entry_price
        else:
            return trade.entry_price - trade.exit_price

    @staticmethod
    def _calc_win_rate(trades: list[dict]) -> float:
        """Calculate win rate from trade dicts."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        return wins / len(trades)

    @staticmethod
    def _calc_sharpe(trades: list[dict]) -> float:
        """Calculate Sharpe ratio from trade P&Ls."""
        pnls = [t.get("pnl", 0.0) for t in trades]
        if len(pnls) < 2:
            return 0.0
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return mean_pnl / std

    @staticmethod
    def _trade_to_dict(trade: _SimTrade) -> dict:
        """Convert a _SimTrade to a serializable dict."""
        return {
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "exit_reason": trade.exit_reason,
            "pnl": trade.pnl,
        }
