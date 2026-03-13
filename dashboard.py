"""Rich terminal dashboard V6 — Velox V6 strategy metrics, risk state, consistency score."""

import logging
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

import config
from risk import RiskManager

logger = logging.getLogger(__name__)
console = Console()


def format_duration(start: datetime, now: datetime) -> str:
    delta = now - start
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def format_pnl(pnl: float) -> str:
    if pnl >= 0:
        return f"[green]+${pnl:.0f}[/green]"
    return f"[red]-${abs(pnl):.0f}[/red]"


def format_pnl_pct(pct: float) -> str:
    if pct >= 0:
        return f"[green]+{pct:.1%}[/green]"
    return f"[red]{pct:.1%}[/red]"


def build_dashboard(
    risk: RiskManager,
    regime: str,
    start_time: datetime,
    now: datetime,
    last_scan_time: datetime | None,
    num_symbols: int,
    analytics: dict | None = None,
    pnl_lock_state: str = "NORMAL",
    vol_scalar: float = 1.0,
    portfolio_beta: float = 0.0,
    consistency_score: float = 0.0,
) -> Panel:
    """Build the full dashboard layout as a Rich Panel."""

    mode = "PAPER" if config.PAPER_MODE else "[bold red]LIVE[/bold red]"
    uptime = format_duration(start_time, now)

    regime_color = "green" if regime == "BULLISH" else "red" if regime == "BEARISH" else "yellow"
    regime_text = f"[{regime_color}]{regime}[/{regime_color}]"

    # PnL Lock state display
    if pnl_lock_state == "LOSS_HALT":
        lock_text = "[bold red blink]LOSS HALT[/bold red blink]"
    elif pnl_lock_state == "GAIN_LOCK":
        lock_text = "[bold yellow]GAIN LOCK[/bold yellow]"
    else:
        lock_text = "[green]NORMAL[/green]"

    cb_text = (
        "[bold red]ACTIVE - NO NEW TRADES[/bold red]"
        if risk.circuit_breaker_active
        else "[green]INACTIVE[/green]"
    )

    # Header
    header = f"  VELOX V6 | {mode} MODE | Regime: {regime_text} | PnL Lock: {lock_text} | Up: {uptime}"

    # Portfolio section
    day_pnl_dollars = risk.day_pnl * risk.starting_equity if risk.starting_equity else 0
    portfolio_lines = [
        f"  Value:    ${risk.current_equity:,.0f}",
        f"  Cash:     ${risk.current_cash:,.0f}",
        f"  Day P&L:  {format_pnl(day_pnl_dollars)} {format_pnl_pct(risk.day_pnl)}",
    ]

    # Add week P&L if analytics available
    if analytics and analytics.get("week_pnl") is not None:
        week_pnl = analytics["week_pnl"]
        week_pct = analytics.get("week_pnl_pct", 0)
        portfolio_lines.append(f"  Week P&L: {format_pnl(week_pnl)} {format_pnl_pct(week_pct)}")

    # --- V6: Risk State panel ---
    vol_color = "green" if 0.7 <= vol_scalar <= 1.3 else "yellow" if 0.5 <= vol_scalar <= 1.5 else "red"
    beta_color = "green" if abs(portfolio_beta) <= 0.3 else "yellow" if abs(portfolio_beta) <= 0.5 else "red"
    cs_color = "green" if consistency_score >= 70 else "yellow" if consistency_score >= 40 else "red"

    risk_state_lines = [
        f"  Vol Scalar:       [{vol_color}]{vol_scalar:.2f}[/{vol_color}]",
        f"  Portfolio Beta:   [{beta_color}]{portfolio_beta:+.2f}[/{beta_color}]",
        f"  PnL Lock:         {lock_text}",
        f"  Consistency:      [{cs_color}]{consistency_score:.0f}/100[/{cs_color}]",
    ]

    # --- V6: Strategy Allocation panel ---
    alloc_map = config.STRATEGY_ALLOCATIONS
    # Count trades per strategy
    strat_counts = {}
    for trade in risk.open_trades.values():
        strat_counts[trade.strategy] = strat_counts.get(trade.strategy, 0) + 1

    alloc_lines = []
    for strat_name, display_name in [("STAT_MR", "MR"), ("KALMAN_PAIRS", "PAIRS"), ("MICRO_MOM", "MICRO")]:
        weight = alloc_map.get(strat_name, 0)
        count = strat_counts.get(strat_name, 0)
        bar_len = int(weight * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        alloc_lines.append(f"  {display_name:<6} {bar} {weight:>3.0%}  ({count} open)")

    # Add beta hedge count if any
    hedge_count = strat_counts.get("BETA_HEDGE", 0)
    if hedge_count > 0:
        alloc_lines.append(f"  HEDGE                                ({hedge_count} open)")

    # Open positions
    open_count = len(risk.open_trades)
    pos_header = f"OPEN POSITIONS ({open_count}/{config.MAX_POSITIONS})"

    pos_lines = []
    for symbol, trade in risk.open_trades.items():
        held_time = format_duration(trade.entry_time, now)
        hold_tag = ""
        if trade.hold_type == "swing":
            hold_tag = " [cyan]SWING[/cyan]"
        side_tag = ""
        if trade.side == "sell":
            side_tag = " [magenta]SHORT[/magenta]"

        # V6: strategy-specific display
        strat_display = trade.strategy
        extra = ""
        if trade.strategy == "STAT_MR":
            strat_display = "[blue]MR[/blue]"
            # z-score would be in trade metadata if available
        elif trade.strategy == "KALMAN_PAIRS":
            strat_display = "[cyan]PAIR[/cyan]"
            if trade.pair_id:
                extra = f" [{trade.pair_id[:8]}]"
        elif trade.strategy == "MICRO_MOM":
            strat_display = "[yellow]MICRO[/yellow]"
        elif trade.strategy == "BETA_HEDGE":
            strat_display = "[magenta]HEDGE[/magenta]"

        pos_lines.append(
            f"  {symbol:<6} {strat_display:<12} "
            f"entry=${trade.entry_price:<8.2f} "
            f"qty={trade.qty:<4} {held_time}{hold_tag}{side_tag}{extra}"
        )

    if not pos_lines:
        pos_lines = ["  (none)"]

    # Metrics panel
    metrics_lines = []
    if analytics:
        s7 = analytics.get("sharpe_7d", 0)
        wr = analytics.get("win_rate", 0)
        pf = analytics.get("profit_factor", 0)
        md = analytics.get("max_drawdown", 0)
        sharpe_color = "green" if s7 > 1 else "yellow" if s7 > 0.5 else "red"
        metrics_lines = [
            f"  Sharpe (7d):   [{sharpe_color}]{s7:.2f}[/{sharpe_color}]",
            f"  Win Rate:      {wr:.0%}",
            f"  Profit Factor: {pf:.2f}",
            f"  Max Drawdown:  {md:.1%}",
        ]

    # Strategy breakdown (from analytics)
    strat_lines = []
    if analytics and analytics.get("strategy_breakdown"):
        breakdown = analytics["strategy_breakdown"]
        for strat, data in breakdown.items():
            trades_n = data.get("trades", 0)
            wr_s = data.get("win_rate", 0)
            pnl_s = data.get("pnl", 0)
            strat_lines.append(
                f"  {strat:<14} {trades_n:>3} trades  {wr_s:>3.0%} win  {format_pnl(pnl_s)}"
            )

    # Recent trades
    recent = risk.closed_trades[-6:] if risk.closed_trades else []
    recent_lines = []
    for trade in reversed(recent):
        icon = "[green]W[/green]" if trade.pnl > 0 else "[red]L[/red]"
        time_str = trade.exit_time.strftime("%H:%M") if trade.exit_time else "??:??"
        pnl_pct = trade.pnl / (trade.entry_price * trade.qty) if trade.entry_price * trade.qty > 0 else 0
        reason = f" ({trade.exit_reason})" if trade.exit_reason else ""
        side_str = trade.side.upper()
        recent_lines.append(
            f"  {time_str} {side_str:<5} {trade.symbol:<6} "
            f"{trade.strategy:<12} {format_pnl(trade.pnl)} "
            f"{format_pnl_pct(pnl_pct)} {icon}{reason}"
        )
    if not recent_lines:
        recent_lines = ["  (no trades yet)"]

    # Feature flags
    feat_parts = ["MR60%", "PAIRS25%", "MICRO15%"]
    if config.ALLOW_SHORT:
        feat_parts.append("Short")
    if config.TELEGRAM_ENABLED:
        feat_parts.append("TG")
    if config.WEB_DASHBOARD_ENABLED:
        feat_parts.append("Web")
    if config.WEBSOCKET_MONITORING:
        feat_parts.append("WS")
    feat_str = f" | {'+'.join(feat_parts)}"

    # Footer
    scan_time_str = last_scan_time.strftime("%H:%M:%S") if last_scan_time else "---"
    footer = (
        f"  Last scan: {scan_time_str} | "
        f"{num_symbols} symbols"
        f"{feat_str}\n"
        f"  Circuit breaker: {cb_text}"
    )

    # --- Assemble ---
    sep = f"{'---' * 23}"
    content = f"[bold]{header}[/bold]\n{sep}\n"

    # Portfolio + Risk State
    content += " PORTFOLIO\n" + "\n".join(portfolio_lines) + "\n"
    content += f"{sep}\n RISK STATE\n" + "\n".join(risk_state_lines) + "\n"

    if metrics_lines:
        content += f"{sep}\n METRICS (this week)\n" + "\n".join(metrics_lines) + "\n"

    content += f"{sep}\n STRATEGY ALLOCATION\n" + "\n".join(alloc_lines) + "\n"
    content += f"{sep}\n {pos_header}\n" + "\n".join(pos_lines) + "\n"

    if strat_lines:
        content += f"{sep}\n STRATEGY BREAKDOWN (this week)\n" + "\n".join(strat_lines) + "\n"

    content += f"{sep}\n RECENT TRADES\n" + "\n".join(recent_lines) + "\n"

    # Trade Analysis (kept from V5)
    try:
        from database import get_exit_reason_breakdown, get_filter_block_summary

        analysis_lines = []

        exit_data = get_exit_reason_breakdown(days=7)
        if exit_data:
            analysis_lines.append("[bold]Exit Reasons (7d):[/bold]")
            for row in exit_data:
                reason = row["exit_reason"] or "unknown"
                count = row["count"]
                avg = row["avg_pnl"] or 0
                color = "green" if avg >= 0 else "red"
                analysis_lines.append(f"  {reason:<20} {count:>3} trades  avg [{color}]{avg:+.2f}[/{color}]")

        blocks = get_filter_block_summary()
        if blocks:
            analysis_lines.append("")
            analysis_lines.append("[bold]Filter Blocks (today):[/bold]")
            for reason, cnt in list(blocks.items())[:5]:
                analysis_lines.append(f"  {reason:<25} {cnt:>4} blocked")

        if analysis_lines:
            analysis_text = "\n".join(analysis_lines)
            content += f"{sep}\n TRADE ANALYSIS\n{analysis_text}\n"
    except Exception:
        pass

    content += f"{sep}\n{footer}"

    return Panel(
        content,
        title="[bold cyan]VELOX V6 -- Autonomous Algorithmic Trading System[/bold cyan]",
        border_style="cyan",
    )


def print_day_summary(summary: dict, consistency_score: float = 0.0):
    """Print the end-of-day summary."""
    if summary.get("trades", 0) == 0:
        console.print("\n[yellow]=== DAY SUMMARY ===[/yellow]")
        console.print("  No trades today.")
        if consistency_score > 0:
            console.print(f"  Consistency Score: {consistency_score:.0f}/100")
        console.print("[yellow]===================[/yellow]\n")
        return

    console.print("\n[bold yellow]=== DAY SUMMARY ===[/bold yellow]")
    console.print(f"  Trades today:     {summary['trades']}")
    console.print(f"  Winners:          {summary['winners']}  ({summary['win_rate']:.0%})")
    console.print(f"  Losers:           {summary['losers']}  ({1 - summary['win_rate']:.0%})")
    console.print(f"  Day P&L:         ${summary['total_pnl']:+.0f} ({summary['pnl_pct']:+.1%})")
    console.print(f"  Best trade:       {summary['best_trade']}")
    console.print(f"  Worst trade:      {summary['worst_trade']}")

    # Per-strategy win rates
    for key, val in summary.items():
        if key.endswith("_win_rate"):
            strat = key.replace("_win_rate", "").upper()
            console.print(f"  {strat} win rate:  {val}")

    if consistency_score > 0:
        cs_color = "green" if consistency_score >= 70 else "yellow" if consistency_score >= 40 else "red"
        console.print(f"  Consistency:      [{cs_color}]{consistency_score:.0f}/100[/{cs_color}]")

    console.print("[bold yellow]===================[/bold yellow]\n")


def print_startup_info(info: dict):
    """Print startup connectivity info."""
    mode = "PAPER" if info["paper"] else "[bold red]LIVE[/bold red]"
    market = "[green]OPEN[/green]" if info["market_open"] else "[yellow]CLOSED[/yellow]"

    console.print(Panel(
        f"  Account: {info['account_id']}\n"
        f"  Mode:    {mode}\n"
        f"  Equity:  ${info['equity']:,.2f}\n"
        f"  Cash:    ${info['cash']:,.2f}\n"
        f"  Market:  {market}\n"
        f"  Next:    {info.get('next_open', 'N/A')}",
        title="[bold green]VELOX V6 -- CONNECTION VERIFIED[/bold green]",
        border_style="green",
    ))
