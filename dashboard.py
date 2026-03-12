"""Rich terminal dashboard V3 — metrics, strategy breakdown, capital allocation, week P&L."""

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
    earnings_excluded: int = 0,
    strategy_weights: dict | None = None,
) -> Panel:
    """Build the full dashboard layout as a Rich Panel."""

    mode = "PAPER" if config.PAPER_MODE else "[bold red]LIVE[/bold red]"
    uptime = format_duration(start_time, now)

    regime_color = "green" if regime == "BULLISH" else "red" if regime == "BEARISH" else "yellow"
    regime_text = f"[{regime_color}]{regime}[/{regime_color}]"

    cb_text = (
        "[bold red]ACTIVE - NO NEW TRADES[/bold red]"
        if risk.circuit_breaker_active
        else "[green]INACTIVE[/green]"
    )

    # V4: VIX display
    vix_str = ""
    if config.VIX_RISK_SCALING_ENABLED:
        try:
            from risk import get_vix_level, get_vix_risk_scalar
            vix = get_vix_level()
            scalar = get_vix_risk_scalar()
            if vix >= 40:
                vix_str = f" | VIX: [bold red blink]{vix:.0f} HALT[/bold red blink]"
            elif vix >= 30:
                vix_str = f" | VIX: [bold red]{vix:.0f}[/bold red] ({scalar:.0%})"
            elif vix >= 25:
                vix_str = f" | VIX: [yellow]{vix:.0f}[/yellow] ({scalar:.0%})"
            else:
                vix_str = f" | VIX: [green]{vix:.0f}[/green]"
        except Exception:
            pass

    # Header
    header = f"  ALGO BOT V4 | {mode} MODE | Regime: {regime_text}{vix_str} | Up: {uptime}"

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
        strat_tag = trade.strategy
        if trade.strategy == "GAP_GO":
            strat_tag = "[yellow]GAP[/yellow]"
        pos_lines.append(
            f"  {symbol:<6} {strat_tag:<5} "
            f"entry=${trade.entry_price:<8.2f} "
            f"qty={trade.qty:<4} {held_time}{hold_tag}{side_tag}"
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

    # Strategy breakdown
    strat_lines = []
    if analytics and analytics.get("strategy_breakdown"):
        breakdown = analytics["strategy_breakdown"]
        for strat, data in breakdown.items():
            trades_n = data.get("trades", 0)
            wr_s = data.get("win_rate", 0)
            pnl_s = data.get("pnl", 0)
            strat_lines.append(
                f"  {strat:<10} {trades_n:>3} trades  {wr_s:>3.0%} win  {format_pnl(pnl_s)}"
            )

    # V3: Capital allocation weights
    alloc_lines = []
    if strategy_weights:
        for strat, weight in strategy_weights.items():
            bar_len = int(weight * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            alloc_lines.append(f"  {strat:<10} {bar} {weight:.0%}")

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
            f"{trade.strategy:<5} {format_pnl(trade.pnl)} "
            f"{format_pnl_pct(pnl_pct)} {icon}{reason}"
        )
    if not recent_lines:
        recent_lines = ["  (no trades yet)"]

    # V3 features status line
    v3_status_parts = []
    if config.WEBSOCKET_MONITORING:
        v3_status_parts.append("WS")
    if config.USE_ML_FILTER:
        v3_status_parts.append("ML")
    if config.ALLOW_SHORT:
        v3_status_parts.append("SHORT")
    if config.GAP_GO_ENABLED:
        v3_status_parts.append("GAP")
    if config.USE_RS_FILTER:
        v3_status_parts.append("RS")
    v3_str = f" | V3: {'+'.join(v3_status_parts)}" if v3_status_parts else ""

    # Footer
    scan_time_str = last_scan_time.strftime("%H:%M:%S") if last_scan_time else "---"
    earnings_str = f" | Earnings: {earnings_excluded} excl" if earnings_excluded > 0 else ""
    footer = (
        f"  Last scan: {scan_time_str} | "
        f"{num_symbols} symbols | "
        f"Signals today: {risk.signals_today}"
        f"{earnings_str}{v3_str}\n"
        f"  Circuit breaker: {cb_text}"
    )

    # Assemble
    sep = f"{'─' * 68}"
    content = f"[bold]{header}[/bold]\n{sep}\n"

    # Portfolio + Metrics
    content += " PORTFOLIO\n" + "\n".join(portfolio_lines) + "\n"

    if metrics_lines:
        content += f"{sep}\n METRICS (this week)\n" + "\n".join(metrics_lines) + "\n"

    content += f"{sep}\n {pos_header}\n" + "\n".join(pos_lines) + "\n"

    if strat_lines:
        content += f"{sep}\n STRATEGY BREAKDOWN (this week)\n" + "\n".join(strat_lines) + "\n"

    if alloc_lines:
        content += f"{sep}\n CAPITAL ALLOCATION\n" + "\n".join(alloc_lines) + "\n"

    content += f"{sep}\n RECENT TRADES\n" + "\n".join(recent_lines) + "\n"
    content += f"{sep}\n{footer}"

    return Panel(content, title="[bold cyan]ALGO TRADING BOT V4[/bold cyan]", border_style="cyan")


def print_day_summary(summary: dict):
    """Print the end-of-day summary."""
    if summary.get("trades", 0) == 0:
        console.print("\n[yellow]=== DAY SUMMARY ===[/yellow]")
        console.print("  No trades today.")
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
        title="[bold green]CONNECTION VERIFIED[/bold green]",
        border_style="green",
    ))
