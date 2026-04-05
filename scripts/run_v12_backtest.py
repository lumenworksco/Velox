#!/usr/bin/env python3
"""V12 Historical Backtest Runner.

Fetches 6 months of hourly data from Alpaca and runs a walk-forward backtest
using V12 strategy allocations and risk parameters.  Uses the existing
backtester.py simulation functions (ORB, VWAP, STAT_MR, KALMAN_PAIRS,
MICRO_MOM) plus a simple PEAD simulation.

Results are printed to the console and saved to reports/backtest_v12_YYYYMMDD.json.

Usage:
    python3 scripts/run_v12_backtest.py
    python3 scripts/run_v12_backtest.py --months 3
    python3 scripts/run_v12_backtest.py --capital 50000
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the trading_bot package root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # trading_bot/
sys.path.insert(0, str(_PROJECT_ROOT))

# Set TESTING flag BEFORE importing config so API-key validation is skipped
# only when credentials are actually missing (we still want the real config).
# We unset it again after the import if keys are present.
_had_testing = os.environ.get("TESTING")
os.environ["TESTING"] = "1"

import config  # noqa: E402  (needs sys.path first)

# Restore env if keys are present so downstream Alpaca clients work normally
if config.API_KEY and config.API_SECRET:
    if not _had_testing:
        os.environ.pop("TESTING", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v12_backtest")

# ---------------------------------------------------------------------------
# V12 Configuration
# ---------------------------------------------------------------------------
V12_ALLOCATIONS = {
    "STAT_MR": 0.25,
    "VWAP": 0.13,
    "KALMAN_PAIRS": 0.27,
    "ORB": 0.12,
    "MICRO_MOM": 0.05,
    "PEAD": 0.18,
}

V12_RISK = {
    "risk_per_trade_pct": config.RISK_PER_TRADE_PCT,   # 0.008
    "max_position_pct": config.MAX_POSITION_PCT,        # 0.08
    "slippage": config.BACKTEST_SLIPPAGE,               # 0.0005
    "commission": config.BACKTEST_COMMISSION,            # 0.0035
    "risk_free_rate": config.BACKTEST_RISK_FREE_RATE,   # 0.045
}

# Symbols to fetch — use top N from CORE_SYMBOLS (liquid, Alpaca-friendly)
BACKTEST_SYMBOLS = config.CORE_SYMBOLS[:config.BACKTEST_TOP_N]


# ========================================================================= #
#  Data fetching — Alpaca API
# ========================================================================= #

def fetch_alpaca_data(symbols: list[str], months: int = 6) -> dict[str, pd.DataFrame]:
    """Fetch hourly bars from Alpaca for the given symbols."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed
    from alpaca.data.timeframe import TimeFrame

    if not config.API_KEY or not config.API_SECRET:
        logger.error("ALPACA_API_KEY / ALPACA_API_SECRET not set — cannot fetch data.")
        sys.exit(1)

    client = StockHistoricalDataClient(
        api_key=config.API_KEY,
        secret_key=config.API_SECRET,
    )

    end = datetime.now() - timedelta(minutes=20)  # Alpaca free-tier delay
    start = end - timedelta(days=months * 30)

    logger.info("Fetching %d months of hourly data for %d symbols...", months, len(symbols))
    all_data: dict[str, pd.DataFrame] = {}

    for i, symbol in enumerate(symbols, 1):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            barset = client.get_stock_bars(request)

            # Extract bars (alpaca-py BarSet API)
            bar_list = None
            if hasattr(barset, "data") and barset.data:
                bar_list = barset.data.get(symbol)
            elif isinstance(barset, dict):
                bar_list = barset.get(symbol)
            else:
                try:
                    bar_list = barset[symbol]
                except (KeyError, TypeError):
                    pass

            if not bar_list:
                logger.debug("  %s: no data returned", symbol)
                continue

            records = []
            for bar in bar_list:
                records.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })
            df = pd.DataFrame(records)
            if df.empty:
                continue
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            if len(df) < 100:
                logger.debug("  %s: only %d bars, skipping", symbol, len(df))
                continue

            all_data[symbol] = df
            if i % 10 == 0 or i == len(symbols):
                logger.info("  fetched %d/%d symbols (%s: %d bars)",
                            i, len(symbols), symbol, len(df))

        except Exception as exc:
            logger.warning("  %s: fetch failed (%s)", symbol, exc)

    logger.info("Data ready: %d/%d symbols with sufficient bars.\n", len(all_data), len(symbols))
    return all_data


# ========================================================================= #
#  Strategy simulations — reuse backtester.py where possible
# ========================================================================= #

def _portfolio_metrics(portfolio_history: list[float], trades: list[dict],
                       initial_capital: float) -> dict:
    """Compute standard backtest metrics from portfolio history and trades."""
    if not trades or len(portfolio_history) < 2:
        return {
            "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "total_trades": 0, "total_pnl": 0.0,
            "avg_pnl": 0.0, "profit_factor": 0.0,
        }

    total_return = (portfolio_history[-1] - initial_capital) / initial_capital

    # Sharpe from portfolio changes
    arr = np.array(portfolio_history)
    rets = np.diff(arr) / arr[:-1]
    daily_rf = V12_RISK["risk_free_rate"] / 252
    excess = rets - daily_rf
    sharpe = 0.0
    if len(excess) > 1 and np.std(excess, ddof=1) > 0:
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252))

    # Max drawdown
    peak = arr[0]
    max_dd = 0.0
    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Trade stats
    pnls = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / len(pnls) if pnls else 0.0
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.9

    return {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": len(trades),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(np.mean(pnls), 2) if pnls else 0.0,
        "profit_factor": round(min(profit_factor, 999.9), 4),
    }


def simulate_orb(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """ORB backtest using V12 parameters."""
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]
    commission = V12_RISK["commission"]
    risk_pct = V12_RISK["risk_per_trade_pct"]

    for symbol, df in data.items():
        df = df.copy()
        df["date"] = df.index.date
        for _date, day_bars in df.groupby("date"):
            if len(day_bars) < 4:
                continue
            first_bar = day_bars.iloc[0]
            orb_high = first_bar["high"]
            orb_low = first_bar["low"]
            orb_range = orb_high - orb_low
            if orb_range <= 0:
                continue
            midpoint = (orb_high + orb_low) / 2
            if midpoint <= 0:
                continue
            range_pct = orb_range / midpoint
            if range_pct > config.ORB_MAX_RANGE_PCT:
                continue

            for i in range(1, len(day_bars)):
                bar = day_bars.iloc[i]
                if bar["close"] > orb_high and bar["volume"] > 0:
                    entry = orb_high * (1 + slippage)
                    tp = entry + config.ORB_TP_MULT * orb_range
                    sl = entry - config.ORB_SL_MULT * orb_range
                    risk_dollars = abs(entry - sl)
                    if risk_dollars < 0.01:
                        break
                    qty = max(1, int((portfolio * risk_pct) / risk_dollars))
                    comm = qty * commission * 2

                    exit_price = day_bars.iloc[-1]["close"] * (1 - slippage)
                    for j in range(i + 1, len(day_bars)):
                        check = day_bars.iloc[j]
                        if check["high"] >= tp:
                            exit_price = tp * (1 - slippage)
                            break
                        if check["low"] <= sl:
                            exit_price = sl * (1 - slippage)
                            break

                    pnl = (exit_price - entry) * qty - comm
                    portfolio += pnl
                    history.append(portfolio)
                    trades.append({"symbol": symbol, "pnl": pnl})
                    break

    return _portfolio_metrics(history, trades, capital)


def simulate_vwap(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """VWAP mean-reversion backtest using V12 parameters."""
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]
    commission = V12_RISK["commission"]
    risk_pct = V12_RISK["risk_per_trade_pct"]

    for symbol, df in data.items():
        df = df.copy()
        df["date"] = df.index.date
        for _date, day_bars in df.groupby("date"):
            if len(day_bars) < 6:
                continue
            typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
            cum_vol = day_bars["volume"].cumsum()
            cum_vp = (typical * day_bars["volume"]).cumsum()

            for i in range(4, len(day_bars) - 1):
                if cum_vol.iloc[i] == 0:
                    continue
                vwap = cum_vp.iloc[i] / cum_vol.iloc[i]
                cum_vp2 = (typical[:i + 1] ** 2 * day_bars["volume"][:i + 1]).cumsum()
                var = cum_vp2.iloc[i] / cum_vol.iloc[i] - vwap ** 2
                std = np.sqrt(max(var, 0))
                lower = vwap - config.VWAP_BAND_STD * std

                bar = day_bars.iloc[i]
                prev = day_bars.iloc[i - 1]

                if prev["low"] <= lower and bar["close"] > lower:
                    entry = bar["close"] * (1 + slippage)
                    sl_price = lower - config.VWAP_STOP_EXTENSION * std
                    tp_price = vwap
                    risk_dollars = abs(entry - sl_price)
                    if risk_dollars < 0.01:
                        continue
                    qty = max(1, int((portfolio * risk_pct) / risk_dollars))
                    comm = qty * commission * 2

                    exit_price = entry
                    for j in range(i + 1, min(i + 4, len(day_bars))):
                        check = day_bars.iloc[j]
                        if check["high"] >= tp_price:
                            exit_price = tp_price * (1 - slippage)
                            break
                        if check["low"] <= sl_price:
                            exit_price = sl_price * (1 - slippage)
                            break
                    else:
                        k = min(i + 3, len(day_bars) - 1)
                        exit_price = day_bars.iloc[k]["close"] * (1 - slippage)

                    pnl = (exit_price - entry) * qty - comm
                    portfolio += pnl
                    history.append(portfolio)
                    trades.append({"symbol": symbol, "pnl": pnl})
                    break

    return _portfolio_metrics(history, trades, capital)


def simulate_stat_mr(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """Statistical Mean Reversion backtest using V12 parameters."""
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]
    commission = V12_RISK["commission"]

    for symbol, bars in data.items():
        if len(bars) < 60:
            continue
        closes = bars["close"].values
        position = None

        for i in range(50, len(closes)):
            window = closes[max(0, i - 200):i]
            if len(window) < 30:
                continue

            # Simple OU approximation: z-score relative to rolling mean/std
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma < 1e-8:
                continue
            z = (closes[i] - mu) / sigma

            # Hurst exponent approximation (simplified RS method)
            hurst = _approx_hurst(window)

            if position is None:
                if hurst < config.MR_HURST_MAX and abs(z) > config.MR_ZSCORE_ENTRY:
                    side = "buy" if z < -config.MR_ZSCORE_ENTRY else "sell"
                    entry_price = closes[i] * (1 + slippage if side == "buy" else 1 - slippage)
                    position = {
                        "side": side, "entry_price": entry_price,
                        "entry_idx": i, "mu": mu, "sigma": sigma,
                    }
            else:
                exit_price = None
                if abs(z) < config.MR_ZSCORE_EXIT_FULL:
                    exit_price = closes[i] * (1 - slippage if position["side"] == "buy" else 1 + slippage)
                elif abs(z) > config.MR_ZSCORE_STOP:
                    exit_price = closes[i] * (1 - slippage if position["side"] == "buy" else 1 + slippage)
                elif i - position["entry_idx"] > 500:
                    exit_price = closes[i] * (1 - slippage if position["side"] == "buy" else 1 + slippage)

                if exit_price is not None:
                    qty = max(1, int(portfolio * 0.02 / position["entry_price"]))
                    if position["side"] == "buy":
                        pnl = (exit_price - position["entry_price"]) * qty
                    else:
                        pnl = (position["entry_price"] - exit_price) * qty
                    pnl -= commission * qty * 2
                    portfolio += pnl
                    history.append(portfolio)
                    trades.append({"symbol": symbol, "pnl": pnl})
                    position = None

    return _portfolio_metrics(history, trades, capital)


def _approx_hurst(series: np.ndarray) -> float:
    """Quick Hurst exponent approximation via rescaled range."""
    n = len(series)
    if n < 20:
        return 0.5
    try:
        max_k = min(n // 2, 50)
        rs_list = []
        for lag in [max_k // 4, max_k // 2, max_k]:
            if lag < 4:
                continue
            diffs = np.diff(series[:lag])
            mean_d = np.mean(diffs)
            cumdev = np.cumsum(diffs - mean_d)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(diffs, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if len(rs_list) < 2:
            return 0.5
        # Rough Hurst from log-log slope
        lags = np.array([max_k // 4, max_k // 2, max_k])[:len(rs_list)]
        log_lags = np.log(lags)
        log_rs = np.log(rs_list)
        slope = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def simulate_kalman_pairs(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """Kalman Pairs Trading backtest using V12 parameters."""
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]

    # Build pair candidates from sector groups
    pair_candidates = []
    for group_name, symbols in config.SECTOR_GROUPS.items():
        if isinstance(symbols[0], tuple):
            pair_candidates.extend(symbols)
        else:
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    if symbols[i] in data and symbols[j] in data:
                        pair_candidates.append((symbols[i], symbols[j]))

    for sym1, sym2 in pair_candidates[:25]:
        if sym1 not in data or sym2 not in data:
            continue
        try:
            combined = pd.DataFrame({
                "s1": data[sym1]["close"],
                "s2": data[sym2]["close"],
            }).dropna()
            if len(combined) < 60:
                continue

            corr = combined["s1"].corr(combined["s2"])
            if abs(corr) < config.PAIRS_MIN_CORRELATION:
                continue

            # OLS hedge ratio
            coeffs = np.polyfit(combined["s2"].values, combined["s1"].values, 1)
            hedge_ratio = coeffs[0]

            spread = combined["s1"] - hedge_ratio * combined["s2"]
            spread_mean = spread.rolling(20).mean()
            spread_std = spread.rolling(20).std()

            position = None
            for i in range(20, len(spread)):
                if spread_std.iloc[i] < 1e-6:
                    continue
                z = (spread.iloc[i] - spread_mean.iloc[i]) / spread_std.iloc[i]

                if position is None:
                    if abs(z) > config.PAIRS_ZSCORE_ENTRY:
                        side = "sell" if z > config.PAIRS_ZSCORE_ENTRY else "buy"
                        position = {
                            "entry_idx": i, "side": side,
                            "entry_spread": spread.iloc[i],
                        }
                else:
                    exit_trade = False
                    if abs(z) < config.PAIRS_ZSCORE_EXIT:
                        exit_trade = True
                    elif abs(z) > config.PAIRS_ZSCORE_STOP:
                        exit_trade = True
                    elif i - position["entry_idx"] > 50:
                        exit_trade = True

                    if exit_trade:
                        entry_s = position["entry_spread"]
                        exit_s = spread.iloc[i]
                        if position["side"] == "sell":
                            pnl = (entry_s - exit_s) * 10
                        else:
                            pnl = (exit_s - entry_s) * 10
                        pnl *= (1 - slippage * 2)
                        portfolio += pnl
                        history.append(portfolio)
                        trades.append({"symbol": f"{sym1}/{sym2}", "pnl": pnl})
                        position = None
        except Exception:
            continue

    return _portfolio_metrics(history, trades, capital)


def simulate_micro_momentum(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """Micro Momentum backtest (SPY volume-spike signals) using V12 parameters."""
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]

    if "SPY" not in data:
        return _portfolio_metrics(history, trades, capital)

    spy = data["SPY"]
    if len(spy) < 30:
        return _portfolio_metrics(history, trades, capital)

    spy_close = spy["close"].values
    spy_volume = spy["volume"].values

    for i in range(20, len(spy_close) - 8):
        avg_vol = np.mean(spy_volume[i - 20:i])
        if avg_vol < 1:
            continue
        vol_ratio = spy_volume[i] / avg_vol
        price_move = abs(spy_close[i] - spy_close[i - 1]) / spy_close[i - 1]

        if vol_ratio > config.MICRO_SPY_VOL_SPIKE_MULT and price_move > config.MICRO_SPY_MIN_MOVE_PCT:
            direction = "buy" if spy_close[i] > spy_close[i - 1] else "sell"
            entry_price = spy_close[i] * (1 + slippage if direction == "buy" else 1 - slippage)

            exit_idx = min(i + 8, len(spy_close) - 1)
            exit_price = spy_close[exit_idx]

            tp_price = entry_price * (1 + config.MICRO_TARGET_PCT if direction == "buy" else 1 - config.MICRO_TARGET_PCT)
            sl_price = entry_price * (1 - config.MICRO_STOP_PCT if direction == "buy" else 1 + config.MICRO_STOP_PCT)

            for j in range(i + 1, exit_idx + 1):
                if direction == "buy":
                    if spy_close[j] >= tp_price:
                        exit_price = tp_price * (1 - slippage)
                        break
                    if spy_close[j] <= sl_price:
                        exit_price = sl_price * (1 - slippage)
                        break
                else:
                    if spy_close[j] <= tp_price:
                        exit_price = tp_price * (1 + slippage)
                        break
                    if spy_close[j] >= sl_price:
                        exit_price = sl_price * (1 + slippage)
                        break

            qty = max(1, int(portfolio * 0.01 / entry_price))
            if direction == "buy":
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            portfolio += pnl
            history.append(portfolio)
            trades.append({"symbol": "SPY", "pnl": pnl})

    return _portfolio_metrics(history, trades, capital)


def simulate_pead(data: dict[str, pd.DataFrame], capital: float) -> dict:
    """Post-Earnings Announcement Drift simulation.

    Since we do not have live earnings surprise data in backtesting, this
    uses a proxy: unusually large single-day moves (>3%) combined with
    a volume spike (>2x 20-day average) to simulate earnings-like events
    and then tracks the drift over the subsequent 3-10 days.
    """
    trades = []
    portfolio = capital
    history = [portfolio]
    slippage = V12_RISK["slippage"]
    commission = V12_RISK["commission"]
    risk_pct = V12_RISK["risk_per_trade_pct"]

    min_move_pct = config.PEAD_MIN_SURPRISE_PCT / 100.0  # 0.03
    min_vol_ratio = config.PEAD_MIN_VOLUME_RATIO          # 2.0
    hold_days_min = config.PEAD_HOLD_DAYS_MIN              # 3
    hold_days_max = config.PEAD_HOLD_DAYS_MAX              # 10
    tp_pct = config.PEAD_TAKE_PROFIT                       # 0.05
    sl_pct = config.PEAD_STOP_LOSS                         # 0.02

    for symbol, df in data.items():
        if len(df) < 30:
            continue

        closes = df["close"].values
        volumes = df["volume"].values
        dates = df.index

        for i in range(20, len(closes) - hold_days_max):
            day_return = (closes[i] - closes[i - 1]) / closes[i - 1]
            avg_vol = np.mean(volumes[i - 20:i])
            if avg_vol < 1:
                continue
            vol_ratio = volumes[i] / avg_vol

            # Detect earnings-like event: large move + volume spike
            if abs(day_return) >= min_move_pct and vol_ratio >= min_vol_ratio:
                direction = "buy" if day_return > 0 else "sell"
                entry_price = closes[i] * (1 + slippage if direction == "buy" else 1 - slippage)

                if direction == "buy":
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                else:
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)

                # Hold for 3-10 bars, exit on TP/SL or time
                exit_price = None
                for j in range(i + hold_days_min, min(i + hold_days_max + 1, len(closes))):
                    if direction == "buy":
                        if closes[j] >= tp_price:
                            exit_price = tp_price * (1 - slippage)
                            break
                        if closes[j] <= sl_price:
                            exit_price = sl_price * (1 - slippage)
                            break
                    else:
                        if closes[j] <= tp_price:
                            exit_price = tp_price * (1 + slippage)
                            break
                        if closes[j] >= sl_price:
                            exit_price = sl_price * (1 + slippage)
                            break

                if exit_price is None:
                    end_idx = min(i + hold_days_max, len(closes) - 1)
                    exit_price = closes[end_idx] * (1 - slippage if direction == "buy" else 1 + slippage)

                risk_dollars = abs(entry_price - sl_price)
                if risk_dollars < 0.01:
                    continue
                qty = max(1, int((portfolio * risk_pct) / risk_dollars))
                comm = qty * commission * 2

                if direction == "buy":
                    pnl = (exit_price - entry_price) * qty - comm
                else:
                    pnl = (entry_price - exit_price) * qty - comm

                portfolio += pnl
                history.append(portfolio)
                trades.append({"symbol": symbol, "pnl": pnl})

    return _portfolio_metrics(history, trades, capital)


# ========================================================================= #
#  Combined portfolio-level metrics
# ========================================================================= #

def combine_strategy_results(strategy_results: dict[str, dict],
                             initial_capital: float) -> dict:
    """Compute portfolio-level metrics from allocation-weighted strategy results."""
    total_pnl = 0.0
    total_trades = 0
    all_wins = 0
    all_losses = 0

    for strat, alloc in V12_ALLOCATIONS.items():
        if strat in strategy_results:
            r = strategy_results[strat]
            # Weight PnL by allocation
            weighted_pnl = r["total_pnl"] * alloc / max(alloc, 0.01)
            total_pnl += r["total_pnl"]
            total_trades += r["total_trades"]
            wins = int(r["win_rate"] * r["total_trades"])
            all_wins += wins
            all_losses += r["total_trades"] - wins

    portfolio_return = total_pnl / initial_capital if initial_capital > 0 else 0.0
    portfolio_win_rate = all_wins / total_trades if total_trades > 0 else 0.0

    # Weighted Sharpe (approximate)
    weighted_sharpe = 0.0
    for strat, alloc in V12_ALLOCATIONS.items():
        if strat in strategy_results:
            weighted_sharpe += strategy_results[strat]["sharpe_ratio"] * alloc

    # Worst drawdown across strategies
    worst_dd = 0.0
    for strat in V12_ALLOCATIONS:
        if strat in strategy_results:
            worst_dd = max(worst_dd, strategy_results[strat]["max_drawdown"])

    return {
        "total_return": round(portfolio_return, 6),
        "total_pnl": round(total_pnl, 2),
        "sharpe_ratio": round(weighted_sharpe, 4),
        "max_drawdown": round(worst_dd, 6),
        "win_rate": round(portfolio_win_rate, 4),
        "total_trades": total_trades,
    }


# ========================================================================= #
#  Report generation
# ========================================================================= #

def generate_report(strategy_results: dict[str, dict], portfolio_metrics: dict,
                    initial_capital: float, months: int) -> dict:
    """Build the full JSON report."""
    report = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V12",
        "config": {
            "initial_capital": initial_capital,
            "data_months": months,
            "symbols_count": len(BACKTEST_SYMBOLS),
            "allocations": V12_ALLOCATIONS,
            "risk_params": V12_RISK,
        },
        "portfolio": portfolio_metrics,
        "strategies": {},
    }

    for strat in V12_ALLOCATIONS:
        if strat in strategy_results:
            report["strategies"][strat] = {
                "allocation": V12_ALLOCATIONS[strat],
                **strategy_results[strat],
            }

    return report


def print_report(report: dict) -> None:
    """Print a formatted summary to the console."""
    print("\n" + "=" * 72)
    print("  V12 HISTORICAL BACKTEST RESULTS")
    print("=" * 72)
    print(f"  Run Date:        {report['run_date']}")
    print(f"  Data Period:     {report['config']['data_months']} months")
    print(f"  Initial Capital: ${report['config']['initial_capital']:,.0f}")
    print(f"  Symbols:         {report['config']['symbols_count']}")
    print()

    # Portfolio summary
    p = report["portfolio"]
    print("  PORTFOLIO SUMMARY")
    print("  " + "-" * 40)
    print(f"  Total Return:    {p['total_return']:+.2%}")
    print(f"  Total P&L:       ${p['total_pnl']:+,.2f}")
    print(f"  Sharpe Ratio:    {p['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {p['max_drawdown']:.2%}")
    print(f"  Win Rate:        {p['win_rate']:.1%}")
    print(f"  Total Trades:    {p['total_trades']}")
    print()

    # Per-strategy breakdown
    print("  PER-STRATEGY BREAKDOWN")
    print("  " + "-" * 68)
    header = f"  {'Strategy':<15} {'Alloc':>6} {'Return':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} {'Trades':>7}"
    print(header)
    print("  " + "-" * 68)

    for strat_name, strat_data in report["strategies"].items():
        alloc = strat_data["allocation"]
        ret = strat_data["total_return"]
        sharpe = strat_data["sharpe_ratio"]
        dd = strat_data["max_drawdown"]
        wr = strat_data["win_rate"]
        trades = strat_data["total_trades"]
        print(f"  {strat_name:<15} {alloc:>5.0%} {ret:>+8.2%} {sharpe:>7.2f} {dd:>6.2%} {wr:>5.1%} {trades:>7}")

    print()
    print("=" * 72)
    print()


# ========================================================================= #
#  Main entry point
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser(description="V12 Historical Backtest Runner")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of historical data to fetch (default: 6)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital (default: $100,000)")
    args = parser.parse_args()

    initial_capital = args.capital
    months = args.months

    logger.info("V12 Backtest — %d months, $%,.0f initial capital", months, initial_capital)
    logger.info("Allocations: %s", V12_ALLOCATIONS)
    logger.info("Risk: risk_per_trade=%.1f%%, max_position=%.1f%%",
                V12_RISK["risk_per_trade_pct"] * 100, V12_RISK["max_position_pct"] * 100)

    # --- Fetch data ---
    data = fetch_alpaca_data(BACKTEST_SYMBOLS, months=months)
    if not data:
        logger.error("No data fetched. Check Alpaca API credentials and network.")
        sys.exit(1)

    logger.info("Running simulations across %d symbols...\n", len(data))

    # --- Run each strategy simulation ---
    strategy_results: dict[str, dict] = {}

    logger.info("Simulating ORB...")
    orb_capital = initial_capital * V12_ALLOCATIONS["ORB"]
    strategy_results["ORB"] = simulate_orb(data, orb_capital)
    logger.info("  ORB: %d trades, return %.2f%%",
                strategy_results["ORB"]["total_trades"],
                strategy_results["ORB"]["total_return"] * 100)

    logger.info("Simulating VWAP...")
    vwap_capital = initial_capital * V12_ALLOCATIONS["VWAP"]
    strategy_results["VWAP"] = simulate_vwap(data, vwap_capital)
    logger.info("  VWAP: %d trades, return %.2f%%",
                strategy_results["VWAP"]["total_trades"],
                strategy_results["VWAP"]["total_return"] * 100)

    logger.info("Simulating STAT_MR...")
    stat_capital = initial_capital * V12_ALLOCATIONS["STAT_MR"]
    strategy_results["STAT_MR"] = simulate_stat_mr(data, stat_capital)
    logger.info("  STAT_MR: %d trades, return %.2f%%",
                strategy_results["STAT_MR"]["total_trades"],
                strategy_results["STAT_MR"]["total_return"] * 100)

    logger.info("Simulating KALMAN_PAIRS...")
    pairs_capital = initial_capital * V12_ALLOCATIONS["KALMAN_PAIRS"]
    strategy_results["KALMAN_PAIRS"] = simulate_kalman_pairs(data, pairs_capital)
    logger.info("  KALMAN_PAIRS: %d trades, return %.2f%%",
                strategy_results["KALMAN_PAIRS"]["total_trades"],
                strategy_results["KALMAN_PAIRS"]["total_return"] * 100)

    logger.info("Simulating MICRO_MOM...")
    micro_capital = initial_capital * V12_ALLOCATIONS["MICRO_MOM"]
    strategy_results["MICRO_MOM"] = simulate_micro_momentum(data, micro_capital)
    logger.info("  MICRO_MOM: %d trades, return %.2f%%",
                strategy_results["MICRO_MOM"]["total_trades"],
                strategy_results["MICRO_MOM"]["total_return"] * 100)

    logger.info("Simulating PEAD...")
    pead_capital = initial_capital * V12_ALLOCATIONS["PEAD"]
    strategy_results["PEAD"] = simulate_pead(data, pead_capital)
    logger.info("  PEAD: %d trades, return %.2f%%",
                strategy_results["PEAD"]["total_trades"],
                strategy_results["PEAD"]["total_return"] * 100)

    # --- Combine into portfolio-level metrics ---
    portfolio_metrics = combine_strategy_results(strategy_results, initial_capital)

    # --- Generate and print report ---
    report = generate_report(strategy_results, portfolio_metrics, initial_capital, months)
    print_report(report)

    # --- Save JSON report ---
    report_dir = _PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_filename = f"backtest_v12_{datetime.now().strftime('%Y%m%d')}.json"
    report_path = report_dir / report_filename

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Report saved to %s", report_path)
    print(f"  Report saved: {report_path}\n")


if __name__ == "__main__":
    main()
