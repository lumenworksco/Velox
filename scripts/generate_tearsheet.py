#!/usr/bin/env python3
"""Performance tear sheet generator for Velox Trading Bot.

Reads trade history and daily snapshots from bot.db, computes comprehensive
performance metrics, and produces:
  1. Rich-formatted terminal output
  2. Self-contained HTML report (inline CSS + inline SVG charts)

Usage:
    python3 scripts/generate_tearsheet.py              # Full history
    python3 scripts/generate_tearsheet.py --days 90    # Last 90 days
    python3 scripts/generate_tearsheet.py --output custom_report.html
"""

import argparse
import base64
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports -- degrade gracefully
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.045  # 4.5% annual
DB_PATH = Path(__file__).resolve().parent.parent / "bot.db"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# ============================================================================
# Data loading
# ============================================================================

def load_trades(db_path: str, days: int | None = None) -> list[dict]:
    """Load closed trades from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM trades ORDER BY exit_time ASC"
    rows = conn.execute(query).fetchall()
    conn.close()

    trades = [dict(r) for r in rows]

    if days and trades:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        trades = [t for t in trades if (t.get("exit_time") or "") >= cutoff]

    return trades


def load_snapshots(db_path: str, days: int | None = None) -> list[dict]:
    """Load daily portfolio snapshots from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM daily_snapshots ORDER BY date ASC"
    rows = conn.execute(query).fetchall()
    conn.close()

    snapshots = [dict(r) for r in rows]

    if days and snapshots:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        snapshots = [s for s in snapshots if s["date"] >= cutoff]

    return snapshots


def fetch_spy_benchmark(start_date: str, end_date: str) -> list[dict]:
    """Fetch SPY daily closes from yfinance for benchmark overlay."""
    if not HAS_YFINANCE:
        return []
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if spy.empty:
            return []
        results = []
        for date_idx, row in spy.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            close_val = row["Close"]
            # Handle both scalar and Series (yfinance version differences)
            if hasattr(close_val, "item"):
                close_val = close_val.item()
            elif hasattr(close_val, "iloc"):
                close_val = close_val.iloc[0]
            results.append({"date": date_str, "close": float(close_val)})
        return results
    except Exception:
        return []


# ============================================================================
# Metric computations
# ============================================================================

def compute_daily_returns(snapshots: list[dict]) -> np.ndarray:
    """Extract daily return percentages from snapshots."""
    if len(snapshots) < 2:
        return np.array([])
    return np.array([s["day_pnl_pct"] for s in snapshots[1:] if s["day_pnl_pct"] is not None])


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 99.9 if np.mean(excess) > 0 else 0.0
    ds_std = np.std(downside, ddof=1)
    if ds_std == 0:
        return 0.0
    return float(np.mean(excess) / ds_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calmar_ratio(ann_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return ann_return / abs(max_dd)


def max_drawdown_series(portfolio_values: list[float]) -> tuple[float, int, list[float]]:
    """Returns (max_dd_pct, duration_in_days, drawdown_series)."""
    if len(portfolio_values) < 2:
        return 0.0, 0, []

    arr = np.array(portfolio_values, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdown = (arr - running_max) / running_max
    max_dd = float(np.min(drawdown))

    # Compute duration of worst drawdown
    duration = 0
    current_duration = 0
    max_duration = 0
    in_drawdown = False
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0

    return max_dd, max_duration, drawdown.tolist()


def profit_factor(trades: list[dict]) -> float:
    gross_profit = sum(t["pnl"] for t in trades if (t.get("pnl") or 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if (t.get("pnl") or 0) < 0))
    if gross_loss == 0:
        return 99.9 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    return winners / len(trades) * 100


def avg_hold_time(trades: list[dict]) -> timedelta:
    """Average holding time across trades."""
    durations = []
    for t in trades:
        entry = t.get("entry_time")
        exit_ = t.get("exit_time")
        if entry and exit_:
            try:
                # Parse ISO format (may have timezone info)
                e = datetime.fromisoformat(entry)
                x = datetime.fromisoformat(exit_)
                durations.append((x - e).total_seconds())
            except (ValueError, TypeError):
                continue
    if not durations:
        return timedelta(0)
    return timedelta(seconds=sum(durations) / len(durations))


def consecutive_streaks(trades: list[dict]) -> tuple[int, int]:
    """Returns (max_win_streak, max_loss_streak)."""
    if not trades:
        return 0, 0
    max_win = max_loss = 0
    cur_win = cur_loss = 0
    for t in trades:
        pnl = t.get("pnl") or 0
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def compute_var_95(daily_returns: np.ndarray) -> float:
    """Historical VaR at 95% confidence (parametric)."""
    if len(daily_returns) < 5:
        return 0.0
    return float(np.percentile(daily_returns, 5))


def compute_monthly_returns(snapshots: list[dict]) -> dict[tuple[int, int], float]:
    """Compute monthly returns from snapshots. Key = (year, month)."""
    if len(snapshots) < 2:
        return {}

    monthly = {}
    # Group snapshots by year-month
    by_month: dict[tuple[int, int], list[dict]] = {}
    for s in snapshots:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        key = (d.year, d.month)
        by_month.setdefault(key, []).append(s)

    for key, month_snaps in by_month.items():
        month_snaps.sort(key=lambda s: s["date"])
        first_val = month_snaps[0]["portfolio_value"]
        last_val = month_snaps[-1]["portfolio_value"]
        if first_val and first_val > 0:
            monthly[key] = (last_val - first_val) / first_val * 100
        else:
            monthly[key] = 0.0

    return monthly


def compute_strategy_breakdown(trades: list[dict], daily_returns: np.ndarray) -> list[dict]:
    """Per-strategy performance breakdown."""
    by_strategy: dict[str, list[dict]] = {}
    for t in trades:
        strat = t.get("strategy", "Unknown")
        by_strategy.setdefault(strat, []).append(t)

    results = []
    for strat, strat_trades in by_strategy.items():
        pnls = [t.get("pnl") or 0 for t in strat_trades]
        winners = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)

        # Strategy-level Sharpe approximation from trade P&L percentages
        pnl_pcts = [t.get("pnl_pct") or 0 for t in strat_trades]
        strat_sharpe = 0.0
        if len(pnl_pcts) >= 2:
            arr = np.array(pnl_pcts)
            std = np.std(arr, ddof=1)
            if std > 0:
                strat_sharpe = float(np.mean(arr) / std * np.sqrt(TRADING_DAYS_PER_YEAR))

        results.append({
            "strategy": strat,
            "trades": len(strat_trades),
            "win_rate": winners / len(strat_trades) * 100 if strat_trades else 0,
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "sharpe": strat_sharpe,
            "total_pnl": total_pnl,
        })

    results.sort(key=lambda r: r["total_pnl"], reverse=True)
    return results


# ============================================================================
# Terminal output (Rich)
# ============================================================================

def _fmt_pct(val: float, decimals: int = 2) -> str:
    """Format percentage with sign."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def _fmt_dollar(val: float) -> str:
    """Format dollar amount with sign."""
    sign = "+" if val > 0 else ""
    return f"{sign}${val:,.2f}"


def _fmt_duration(td: timedelta) -> str:
    """Format timedelta as human-readable."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "0m"
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def print_terminal_report(
    trades: list[dict],
    snapshots: list[dict],
    daily_returns: np.ndarray,
    strategy_breakdown: list[dict],
    monthly_returns: dict,
    max_dd: float,
    dd_duration: int,
    days_filter: int | None,
):
    """Print the tear sheet to terminal using Rich."""
    if not HAS_RICH:
        print("[WARNING] Install 'rich' for formatted terminal output: pip install rich")
        print(f"Total trades: {len(trades)}")
        print(f"Snapshots: {len(snapshots)}")
        return

    console = Console()
    console.print()

    # Title
    title_text = "VELOX TRADING BOT -- PERFORMANCE TEAR SHEET"
    if days_filter:
        title_text += f"  (Last {days_filter} days)"
    console.print(Panel(f"[bold white]{title_text}[/]", border_style="cyan", expand=True))

    if not trades:
        console.print("[yellow]No trades found in the database.[/]")
        return

    # Dates
    start_date = snapshots[0]["date"] if snapshots else "N/A"
    end_date = snapshots[-1]["date"] if snapshots else "N/A"
    total_pnl = sum(t.get("pnl") or 0 for t in trades)

    # Portfolio values for return calc
    if snapshots:
        initial_val = snapshots[0]["portfolio_value"]
        final_val = snapshots[-1]["portfolio_value"]
        total_return = (final_val - initial_val) / initial_val * 100 if initial_val else 0
        n_days = len(snapshots)
        ann_factor = TRADING_DAYS_PER_YEAR / max(n_days, 1)
        ann_return = ((1 + total_return / 100) ** ann_factor - 1) * 100
    else:
        total_return = ann_return = 0.0

    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)
    calmar = calmar_ratio(ann_return / 100, max_dd)
    pf = profit_factor(trades)
    wr = win_rate(trades)
    avg_pnl = total_pnl / len(trades) if trades else 0
    hold = avg_hold_time(trades)

    # --- Section 1: Summary ---
    summary = Table(title="Summary Statistics", show_header=True, header_style="bold cyan",
                    border_style="dim", title_style="bold white")
    summary.add_column("Metric", style="white", width=28)
    summary.add_column("Value", justify="right", width=20)

    summary.add_row("Trading Period", f"{start_date}  --  {end_date}")
    summary.add_row("Total Return", _fmt_pct(total_return))
    summary.add_row("Annualized Return", _fmt_pct(ann_return))
    summary.add_row("Sharpe Ratio", f"{sharpe:.2f}")
    summary.add_row("Sortino Ratio", f"{sortino:.2f}")
    summary.add_row("Calmar Ratio", f"{calmar:.2f}")
    summary.add_row("Max Drawdown", f"{max_dd * 100:.2f}%  ({dd_duration}d)")
    summary.add_row("Win Rate", f"{wr:.1f}%")
    summary.add_row("Profit Factor", f"{pf:.2f}")
    summary.add_row("Total Trades", str(len(trades)))
    summary.add_row("Avg Trade P&L", _fmt_dollar(avg_pnl))
    summary.add_row("Avg Hold Time", _fmt_duration(hold))

    console.print(summary)
    console.print()

    # --- Section 2: Monthly Returns ---
    if monthly_returns:
        years = sorted(set(k[0] for k in monthly_returns.keys()))
        months_table = Table(title="Monthly Returns (%)", show_header=True,
                             header_style="bold cyan", border_style="dim",
                             title_style="bold white")
        months_table.add_column("Year", style="white", width=6)
        for m in range(1, 13):
            months_table.add_column(datetime(2000, m, 1).strftime("%b"), justify="right", width=7)
        months_table.add_column("YTD", justify="right", width=8, style="bold")

        for year in years:
            row = [str(year)]
            ytd = 0.0
            for m in range(1, 13):
                val = monthly_returns.get((year, m))
                if val is not None:
                    color = "green" if val >= 0 else "red"
                    row.append(f"[{color}]{val:+.1f}[/]")
                    ytd += val
                else:
                    row.append("[dim]--[/]")
            color = "green" if ytd >= 0 else "red"
            row.append(f"[{color}]{ytd:+.1f}[/]")
            months_table.add_row(*row)

        console.print(months_table)
        console.print()

    # --- Section 3: Strategy Breakdown ---
    if strategy_breakdown:
        strat_table = Table(title="Strategy Breakdown", show_header=True,
                            header_style="bold cyan", border_style="dim",
                            title_style="bold white")
        strat_table.add_column("Strategy", style="white", width=20)
        strat_table.add_column("Trades", justify="right", width=8)
        strat_table.add_column("Win Rate", justify="right", width=10)
        strat_table.add_column("Avg P&L", justify="right", width=12)
        strat_table.add_column("Sharpe", justify="right", width=8)
        strat_table.add_column("Total P&L", justify="right", width=14)

        for s in strategy_breakdown:
            pnl_color = "green" if s["total_pnl"] >= 0 else "red"
            strat_table.add_row(
                s["strategy"],
                str(s["trades"]),
                f"{s['win_rate']:.1f}%",
                _fmt_dollar(s["avg_pnl"]),
                f"{s['sharpe']:.2f}",
                f"[{pnl_color}]{_fmt_dollar(s['total_pnl'])}[/]",
            )

        console.print(strat_table)
        console.print()

    # --- Section 5: Risk Metrics ---
    var_95 = compute_var_95(daily_returns)
    vol = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    best_day = float(np.max(daily_returns)) if len(daily_returns) > 0 else 0.0
    worst_day = float(np.min(daily_returns)) if len(daily_returns) > 0 else 0.0
    best_trade = max((t.get("pnl") or 0 for t in trades), default=0)
    worst_trade = min((t.get("pnl") or 0 for t in trades), default=0)
    win_streak, loss_streak = consecutive_streaks(trades)

    # Average exposure
    if snapshots:
        exposures = []
        for s in snapshots:
            pv = s.get("portfolio_value") or 0
            cash = s.get("cash") or 0
            if pv > 0:
                exposures.append((pv - cash) / pv * 100)
        avg_exposure = sum(exposures) / len(exposures) if exposures else 0
    else:
        avg_exposure = 0

    risk_table = Table(title="Risk Metrics", show_header=True, header_style="bold cyan",
                       border_style="dim", title_style="bold white")
    risk_table.add_column("Metric", style="white", width=28)
    risk_table.add_column("Value", justify="right", width=20)

    risk_table.add_row("Daily VaR (95%)", _fmt_pct(var_95 * 100))
    risk_table.add_row("Daily P&L Volatility", _fmt_pct(vol * 100))
    risk_table.add_row("Best Day", _fmt_pct(best_day * 100))
    risk_table.add_row("Worst Day", _fmt_pct(worst_day * 100))
    risk_table.add_row("Best Trade", _fmt_dollar(best_trade))
    risk_table.add_row("Worst Trade", _fmt_dollar(worst_trade))
    risk_table.add_row("Max Win Streak", f"{win_streak} trades")
    risk_table.add_row("Max Loss Streak", f"{loss_streak} trades")
    risk_table.add_row("Avg Exposure", f"{avg_exposure:.1f}%")

    console.print(risk_table)
    console.print()


# ============================================================================
# SVG chart generation
# ============================================================================

def _generate_equity_svg(
    snapshots: list[dict],
    drawdown_series: list[float],
    spy_data: list[dict],
    width: int = 900,
    height: int = 400,
) -> str:
    """Generate inline SVG with equity curve, drawdown, and SPY overlay."""
    if not snapshots:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="100"><text x="20" y="50" fill="#888" font-family="monospace">No snapshot data available</text></svg>'

    dates = [s["date"] for s in snapshots]
    values = [s["portfolio_value"] for s in snapshots]
    n = len(values)

    # Chart dimensions
    margin_left = 80
    margin_right = 30
    margin_top = 30
    margin_bottom = 50
    eq_height = 250
    dd_height = 100
    total_height = eq_height + dd_height + margin_top + margin_bottom + 40

    chart_w = width - margin_left - margin_right
    chart_h = eq_height

    # Scale portfolio values
    min_val = min(values) * 0.999
    max_val = max(values) * 1.001
    val_range = max_val - min_val if max_val != min_val else 1

    def x_pos(i):
        return margin_left + (i / max(n - 1, 1)) * chart_w

    def y_pos(v):
        return margin_top + chart_h - ((v - min_val) / val_range) * chart_h

    # Build equity polyline
    eq_points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))

    # Gradient fill under equity line
    fill_points = (
        f"{x_pos(0):.1f},{margin_top + chart_h} "
        + eq_points
        + f" {x_pos(n - 1):.1f},{margin_top + chart_h}"
    )

    # SPY overlay (normalized to same start)
    spy_svg = ""
    if spy_data and values:
        spy_dates = {s["date"]: s["close"] for s in spy_data}
        spy_aligned = []
        for s in snapshots:
            if s["date"] in spy_dates:
                spy_aligned.append((s["date"], spy_dates[s["date"]]))

        if len(spy_aligned) > 1:
            spy_start = spy_aligned[0][1]
            if spy_start > 0:
                spy_normalized = [values[0] * (c / spy_start) for _, c in spy_aligned]
                spy_indices = []
                for date_str, _ in spy_aligned:
                    for i, s in enumerate(snapshots):
                        if s["date"] == date_str:
                            spy_indices.append(i)
                            break

                spy_points = " ".join(
                    f"{x_pos(spy_indices[j]):.1f},{y_pos(spy_normalized[j]):.1f}"
                    for j in range(len(spy_normalized))
                )
                spy_svg = f'<polyline points="{spy_points}" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>'

    # Drawdown chart
    dd_top = margin_top + chart_h + 40
    dd_chart_h = dd_height
    dd_svg = ""
    if drawdown_series:
        min_dd = min(drawdown_series) if drawdown_series else -0.01
        if min_dd >= 0:
            min_dd = -0.01  # Ensure some range
        dd_points_list = []
        for i, dd in enumerate(drawdown_series):
            dx = x_pos(i)
            dy = dd_top + dd_chart_h - (dd / min_dd) * dd_chart_h
            dd_points_list.append(f"{dx:.1f},{dy:.1f}")

        dd_points = " ".join(dd_points_list)
        dd_fill = (
            f"{x_pos(0):.1f},{dd_top} "
            + dd_points
            + f" {x_pos(len(drawdown_series) - 1):.1f},{dd_top}"
        )
        dd_svg = f"""
        <polygon points="{dd_fill}" fill="url(#ddGrad)" opacity="0.6"/>
        <polyline points="{dd_points}" fill="none" stroke="#ef4444" stroke-width="1.5"/>
        <line x1="{margin_left}" y1="{dd_top}" x2="{margin_left + chart_w}" y2="{dd_top}" stroke="#555" stroke-width="0.5"/>
        <text x="{margin_left - 10}" y="{dd_top + 5}" fill="#888" font-size="10" text-anchor="end" font-family="monospace">0%</text>
        <text x="{margin_left - 10}" y="{dd_top + dd_chart_h}" fill="#888" font-size="10" text-anchor="end" font-family="monospace">{min_dd * 100:.1f}%</text>
        <text x="{margin_left + chart_w / 2}" y="{dd_top + dd_chart_h + 18}" fill="#888" font-size="11" text-anchor="middle" font-family="monospace">Drawdown</text>
        """

    # X-axis labels (show ~6 date labels)
    x_labels = ""
    label_count = min(6, n)
    for i in range(label_count):
        idx = int(i * (n - 1) / max(label_count - 1, 1))
        x_labels += f'<text x="{x_pos(idx):.1f}" y="{margin_top + chart_h + 18}" fill="#888" font-size="10" text-anchor="middle" font-family="monospace">{dates[idx]}</text>'

    # Y-axis labels
    y_labels = ""
    for i in range(5):
        v = min_val + (i / 4) * val_range
        y_labels += f'<text x="{margin_left - 10}" y="{y_pos(v) + 4:.1f}" fill="#888" font-size="10" text-anchor="end" font-family="monospace">${v:,.0f}</text>'
        y_labels += f'<line x1="{margin_left}" y1="{y_pos(v):.1f}" x2="{margin_left + chart_w}" y2="{y_pos(v):.1f}" stroke="#333" stroke-width="0.5" stroke-dasharray="2,4"/>'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {total_height}" width="{width}" height="{total_height}">
    <defs>
        <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#22c55e" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#22c55e" stop-opacity="0.02"/>
        </linearGradient>
        <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#ef4444" stop-opacity="0.1"/>
            <stop offset="100%" stop-color="#ef4444" stop-opacity="0.4"/>
        </linearGradient>
    </defs>

    <!-- Grid and axes -->
    {y_labels}
    {x_labels}

    <!-- Equity fill -->
    <polygon points="{fill_points}" fill="url(#eqGrad)"/>

    <!-- Equity line -->
    <polyline points="{eq_points}" fill="none" stroke="#22c55e" stroke-width="2"/>

    <!-- SPY benchmark -->
    {spy_svg}

    <!-- Drawdown -->
    {dd_svg}

    <!-- Legend -->
    <line x1="{margin_left + 10}" y1="{margin_top - 12}" x2="{margin_left + 30}" y2="{margin_top - 12}" stroke="#22c55e" stroke-width="2"/>
    <text x="{margin_left + 35}" y="{margin_top - 8}" fill="#ccc" font-size="11" font-family="monospace">Portfolio</text>
    <line x1="{margin_left + 120}" y1="{margin_top - 12}" x2="{margin_left + 140}" y2="{margin_top - 12}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="{margin_left + 145}" y="{margin_top - 8}" fill="#ccc" font-size="11" font-family="monospace">SPY Benchmark</text>
</svg>"""
    return svg


# ============================================================================
# HTML report generation
# ============================================================================

def generate_html_report(
    trades: list[dict],
    snapshots: list[dict],
    daily_returns: np.ndarray,
    strategy_breakdown: list[dict],
    monthly_returns: dict,
    max_dd: float,
    dd_duration: int,
    drawdown_series: list[float],
    spy_data: list[dict],
    days_filter: int | None,
) -> str:
    """Generate self-contained HTML tear sheet."""
    # Compute all metrics
    total_pnl = sum(t.get("pnl") or 0 for t in trades)

    if snapshots:
        initial_val = snapshots[0]["portfolio_value"]
        final_val = snapshots[-1]["portfolio_value"]
        total_return = (final_val - initial_val) / initial_val * 100 if initial_val else 0
        n_days = len(snapshots)
        ann_factor = TRADING_DAYS_PER_YEAR / max(n_days, 1)
        ann_return = ((1 + total_return / 100) ** ann_factor - 1) * 100
        start_date = snapshots[0]["date"]
        end_date = snapshots[-1]["date"]
    else:
        total_return = ann_return = 0.0
        start_date = end_date = "N/A"

    sharpe = sharpe_ratio(daily_returns)
    sortino = sortino_ratio(daily_returns)
    calmar = calmar_ratio(ann_return / 100, max_dd)
    pf = profit_factor(trades)
    wr = win_rate(trades)
    avg_pnl = total_pnl / len(trades) if trades else 0
    hold = avg_hold_time(trades)
    var_95 = compute_var_95(daily_returns)
    vol = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    best_day = float(np.max(daily_returns)) if len(daily_returns) > 0 else 0.0
    worst_day = float(np.min(daily_returns)) if len(daily_returns) > 0 else 0.0
    best_trade = max((t.get("pnl") or 0 for t in trades), default=0)
    worst_trade = min((t.get("pnl") or 0 for t in trades), default=0)
    win_streak, loss_streak = consecutive_streaks(trades)

    if snapshots:
        exposures = []
        for s in snapshots:
            pv = s.get("portfolio_value") or 0
            cash = s.get("cash") or 0
            if pv > 0:
                exposures.append((pv - cash) / pv * 100)
        avg_exposure = sum(exposures) / len(exposures) if exposures else 0
    else:
        avg_exposure = 0

    # Generate SVG chart
    equity_svg = _generate_equity_svg(snapshots, drawdown_series, spy_data)

    # Build monthly returns table HTML
    monthly_html = ""
    if monthly_returns:
        years = sorted(set(k[0] for k in monthly_returns.keys()))
        monthly_html += '<table class="data-table monthly-table"><thead><tr><th>Year</th>'
        for m in range(1, 13):
            monthly_html += f"<th>{datetime(2000, m, 1).strftime('%b')}</th>"
        monthly_html += "<th>YTD</th></tr></thead><tbody>"

        for year in years:
            monthly_html += f"<tr><td class='label'>{year}</td>"
            ytd = 0.0
            for m in range(1, 13):
                val = monthly_returns.get((year, m))
                if val is not None:
                    css_class = "positive" if val >= 0 else "negative"
                    monthly_html += f'<td class="{css_class}">{val:+.1f}</td>'
                    ytd += val
                else:
                    monthly_html += '<td class="na">--</td>'
            css_class = "positive" if ytd >= 0 else "negative"
            monthly_html += f'<td class="{css_class} ytd">{ytd:+.1f}</td>'
            monthly_html += "</tr>"
        monthly_html += "</tbody></table>"

    # Build strategy breakdown table HTML
    strat_html = ""
    if strategy_breakdown:
        strat_html += """<table class="data-table"><thead><tr>
            <th>Strategy</th><th>Trades</th><th>Win Rate</th>
            <th>Avg P&L</th><th>Sharpe</th><th>Total P&L</th>
        </tr></thead><tbody>"""
        for s in strategy_breakdown:
            pnl_class = "positive" if s["total_pnl"] >= 0 else "negative"
            strat_html += f"""<tr>
                <td class="label">{s['strategy']}</td>
                <td>{s['trades']}</td>
                <td>{s['win_rate']:.1f}%</td>
                <td class="{'positive' if s['avg_pnl'] >= 0 else 'negative'}">{_fmt_dollar(s['avg_pnl'])}</td>
                <td>{s['sharpe']:.2f}</td>
                <td class="{pnl_class}">{_fmt_dollar(s['total_pnl'])}</td>
            </tr>"""
        strat_html += "</tbody></table>"

    # Subtitle
    subtitle = ""
    if days_filter:
        subtitle = f'<span class="subtitle">Last {days_filter} days</span>'

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Velox Performance Tear Sheet</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace;
        background: #0a0e17;
        color: #e2e8f0;
        line-height: 1.6;
        padding: 40px;
        max-width: 1100px;
        margin: 0 auto;
    }}

    .header {{
        text-align: center;
        margin-bottom: 40px;
        padding-bottom: 24px;
        border-bottom: 1px solid #1e293b;
    }}

    .header h1 {{
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #f8fafc;
        margin-bottom: 4px;
    }}

    .header .subtitle {{
        font-size: 14px;
        color: #64748b;
    }}

    .header .period {{
        font-size: 13px;
        color: #94a3b8;
        margin-top: 8px;
    }}

    .header .generated {{
        font-size: 11px;
        color: #475569;
        margin-top: 4px;
    }}

    h2 {{
        font-size: 16px;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 36px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }}

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 36px;
    }}

    .kpi {{
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }}

    .kpi .kpi-label {{
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }}

    .kpi .kpi-value {{
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
    }}

    .kpi .kpi-value.positive {{ color: #22c55e; }}
    .kpi .kpi-value.negative {{ color: #ef4444; }}

    .chart-container {{
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 36px;
        overflow-x: auto;
    }}

    .chart-container svg {{
        display: block;
        margin: 0 auto;
        max-width: 100%;
        height: auto;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 24px;
        font-size: 13px;
    }}

    .data-table th {{
        background: #111827;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 11px;
        padding: 10px 12px;
        text-align: right;
        border-bottom: 2px solid #1e293b;
    }}

    .data-table th:first-child {{ text-align: left; }}

    .data-table td {{
        padding: 8px 12px;
        text-align: right;
        border-bottom: 1px solid #1e293b;
        color: #cbd5e1;
    }}

    .data-table td.label {{
        text-align: left;
        color: #f8fafc;
        font-weight: 500;
    }}

    .data-table td.positive {{ color: #22c55e; }}
    .data-table td.negative {{ color: #ef4444; }}
    .data-table td.na {{ color: #334155; }}
    .data-table td.ytd {{ font-weight: 700; }}

    .data-table tbody tr:hover {{
        background: #0f172a;
    }}

    .monthly-table th, .monthly-table td {{
        text-align: center;
        padding: 8px 6px;
    }}

    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}

    .metric-list {{
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
    }}

    .metric-row {{
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1a2332;
    }}

    .metric-row:last-child {{ border-bottom: none; }}

    .metric-row .metric-label {{
        color: #94a3b8;
        font-size: 13px;
    }}

    .metric-row .metric-value {{
        color: #f8fafc;
        font-weight: 600;
        font-size: 13px;
    }}

    .metric-row .metric-value.positive {{ color: #22c55e; }}
    .metric-row .metric-value.negative {{ color: #ef4444; }}

    .footer {{
        text-align: center;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid #1e293b;
        color: #475569;
        font-size: 11px;
    }}

    @media print {{
        body {{ background: #0a0e17; padding: 20px; }}
        .kpi-grid {{ grid-template-columns: repeat(4, 1fr); }}
    }}

    @media (max-width: 768px) {{
        body {{ padding: 16px; }}
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .two-col {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<div class="header">
    <h1>VELOX PERFORMANCE TEAR SHEET</h1>
    {subtitle}
    <div class="period">{start_date}  &mdash;  {end_date}</div>
    <div class="generated">Generated {generated}</div>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
    <div class="kpi">
        <div class="kpi-label">Total Return</div>
        <div class="kpi-value {'positive' if total_return >= 0 else 'negative'}">{_fmt_pct(total_return)}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Annualized Return</div>
        <div class="kpi-value {'positive' if ann_return >= 0 else 'negative'}">{_fmt_pct(ann_return)}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value">{sharpe:.2f}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value negative">{max_dd * 100:.2f}%</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value">{wr:.1f}%</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Profit Factor</div>
        <div class="kpi-value">{pf:.2f}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Total Trades</div>
        <div class="kpi-value">{len(trades)}</div>
    </div>
    <div class="kpi">
        <div class="kpi-label">Avg Trade P&L</div>
        <div class="kpi-value {'positive' if avg_pnl >= 0 else 'negative'}">{_fmt_dollar(avg_pnl)}</div>
    </div>
</div>

<!-- Summary Statistics -->
<h2>Summary Statistics</h2>
<div class="two-col">
    <div class="metric-list">
        <div class="metric-row">
            <span class="metric-label">Total Return</span>
            <span class="metric-value {'positive' if total_return >= 0 else 'negative'}">{_fmt_pct(total_return)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Annualized Return</span>
            <span class="metric-value {'positive' if ann_return >= 0 else 'negative'}">{_fmt_pct(ann_return)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Sharpe Ratio</span>
            <span class="metric-value">{sharpe:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Sortino Ratio</span>
            <span class="metric-value">{sortino:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Calmar Ratio</span>
            <span class="metric-value">{calmar:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Max Drawdown</span>
            <span class="metric-value negative">{max_dd * 100:.2f}% ({dd_duration}d)</span>
        </div>
    </div>
    <div class="metric-list">
        <div class="metric-row">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value">{wr:.1f}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Profit Factor</span>
            <span class="metric-value">{pf:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Total Trades</span>
            <span class="metric-value">{len(trades)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Trade P&L</span>
            <span class="metric-value {'positive' if avg_pnl >= 0 else 'negative'}">{_fmt_dollar(avg_pnl)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Hold Time</span>
            <span class="metric-value">{_fmt_duration(hold)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Trading Period</span>
            <span class="metric-value">{start_date} &mdash; {end_date}</span>
        </div>
    </div>
</div>

<!-- Equity Curve -->
<h2>Equity Curve &amp; Drawdown</h2>
<div class="chart-container">
    {equity_svg}
</div>

<!-- Monthly Returns -->
<h2>Monthly Returns (%)</h2>
{monthly_html if monthly_html else '<p style="color:#475569;">Insufficient data for monthly breakdown.</p>'}

<!-- Strategy Breakdown -->
<h2>Strategy Breakdown</h2>
{strat_html if strat_html else '<p style="color:#475569;">No strategy data available.</p>'}

<!-- Risk Metrics -->
<h2>Risk Metrics</h2>
<div class="two-col">
    <div class="metric-list">
        <div class="metric-row">
            <span class="metric-label">Daily VaR (95%)</span>
            <span class="metric-value negative">{_fmt_pct(var_95 * 100)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Daily P&L Volatility</span>
            <span class="metric-value">{_fmt_pct(vol * 100)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Best Day</span>
            <span class="metric-value positive">{_fmt_pct(best_day * 100)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Worst Day</span>
            <span class="metric-value negative">{_fmt_pct(worst_day * 100)}</span>
        </div>
    </div>
    <div class="metric-list">
        <div class="metric-row">
            <span class="metric-label">Best Trade</span>
            <span class="metric-value positive">{_fmt_dollar(best_trade)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Worst Trade</span>
            <span class="metric-value negative">{_fmt_dollar(worst_trade)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Max Win Streak</span>
            <span class="metric-value">{win_streak} trades</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Max Loss Streak</span>
            <span class="metric-value">{loss_streak} trades</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Exposure</span>
            <span class="metric-value">{avg_exposure:.1f}%</span>
        </div>
    </div>
</div>

<div class="footer">
    Velox V12 Algorithmic Trading System &mdash; Generated {generated}<br>
    Past performance is not indicative of future results.
</div>

</body>
</html>"""
    return html


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Velox performance tear sheet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Limit to last N days (default: full history)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path for HTML report",
    )
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH),
        help=f"Path to database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--no-spy", action="store_true",
        help="Skip SPY benchmark fetch",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    # Load data
    trades = load_trades(db_path, args.days)
    snapshots = load_snapshots(db_path, args.days)

    if not trades and not snapshots:
        print("No data found in database. Run the trading bot first.")
        sys.exit(0)

    # Compute metrics
    daily_returns = compute_daily_returns(snapshots)
    portfolio_values = [s["portfolio_value"] for s in snapshots] if snapshots else []
    max_dd, dd_duration, drawdown_series = max_drawdown_series(portfolio_values)
    strategy_breakdown = compute_strategy_breakdown(trades, daily_returns)
    monthly_returns = compute_monthly_returns(snapshots)

    # Fetch SPY benchmark
    spy_data = []
    if not args.no_spy and snapshots:
        start_date = snapshots[0]["date"]
        end_date = snapshots[-1]["date"]
        print("Fetching SPY benchmark data...")
        spy_data = fetch_spy_benchmark(start_date, end_date)
        if spy_data:
            print(f"  SPY data: {len(spy_data)} days loaded")
        else:
            print("  SPY data unavailable (continuing without benchmark)")

    # Terminal output
    print_terminal_report(
        trades, snapshots, daily_returns, strategy_breakdown,
        monthly_returns, max_dd, dd_duration, args.days,
    )

    # HTML output
    html = generate_html_report(
        trades, snapshots, daily_returns, strategy_breakdown,
        monthly_returns, max_dd, dd_duration, drawdown_series,
        spy_data, args.days,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = REPORTS_DIR / f"tearsheet_{date_str}.html"

    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report saved to: {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
