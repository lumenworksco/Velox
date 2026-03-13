"""V3: WhatsApp notifications — trade alerts, daily summaries, warnings."""

import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _send_whatsapp(message: str):
    """Send a message via WhatsApp Cloud API. Synchronous (fire-and-forget)."""
    if not config.WHATSAPP_ENABLED:
        return
    if not config.WHATSAPP_ACCESS_TOKEN or not config.WHATSAPP_PHONE_NUMBER_ID:
        return
    if not config.WHATSAPP_RECIPIENT_NUMBER:
        return

    try:
        import httpx
        url = (
            f"https://graph.facebook.com/v21.0/"
            f"{config.WHATSAPP_PHONE_NUMBER_ID}/messages"
        )
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {config.WHATSAPP_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": config.WHATSAPP_RECIPIENT_NUMBER,
                    "type": "text",
                    "text": {"body": message},
                },
            )
            if resp.status_code != 200:
                logger.warning(f"WhatsApp send failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"WhatsApp notification failed: {e}")


def send_message(message: str):
    """Public helper — send a plain text message."""
    _send_whatsapp(message)


def notify_trade_opened(trade_or_symbol, side=None, strategy=None,
                        entry_price=None, qty=None,
                        take_profit=None, stop_loss=None):
    """Notify when a new position is opened. Accepts a TradeRecord or individual fields."""
    if hasattr(trade_or_symbol, "symbol"):
        t = trade_or_symbol
        symbol, side, strategy = t.symbol, t.side, t.strategy
        entry_price, qty = t.entry_price, t.qty
        take_profit, stop_loss = t.take_profit, t.stop_loss
    else:
        symbol = trade_or_symbol

    arrow = "\U0001f4c8" if side == "buy" else "\U0001f4c9"
    _send_whatsapp(
        f"{arrow} *OPENED* {side.upper()} {symbol}\n"
        f"Strategy: {strategy}\n"
        f"Entry: ${entry_price:.2f} | Size: {qty} shares\n"
        f"TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}"
    )


def notify_trade_closed(trade_or_symbol, pnl=None, pnl_pct=None,
                        exit_reason=None, hold_time=""):
    """Notify when a position is closed. Accepts a TradeRecord or individual fields."""
    if hasattr(trade_or_symbol, "symbol"):
        t = trade_or_symbol
        symbol = t.symbol
        pnl = t.pnl if pnl is None else pnl
        pnl_pct = (t.pnl / (t.entry_price * t.qty) if t.entry_price * t.qty > 0 else 0) if pnl_pct is None else pnl_pct
        exit_reason = t.exit_reason if exit_reason is None else exit_reason
    else:
        symbol = trade_or_symbol

    pnl = pnl or 0
    pnl_pct = pnl_pct or 0
    exit_reason = exit_reason or ""
    icon = "\u2705" if pnl > 0 else "\u274c"
    _send_whatsapp(
        f"{icon} *CLOSED* {symbol}\n"
        f"P&L: ${pnl:+.2f} ({pnl_pct:+.2%})\n"
        f"Reason: {exit_reason}"
        + (f" | Hold: {hold_time}" if hold_time else "")
    )


def notify_circuit_breaker(day_pnl_pct: float):
    """Notify when circuit breaker triggers."""
    _send_whatsapp(
        f"\U0001f6a8 *CIRCUIT BREAKER TRIGGERED*\n"
        f"Daily P&L: {day_pnl_pct:.2%}\n"
        f"No new trades for remainder of day."
    )


def notify_daily_summary(summary_or_pnl, equity=None, day_pnl_pct=None,
                         n_trades=None, win_rate=None,
                         best_trade=None, worst_trade=None):
    """Notify with end-of-day summary. Accepts a summary dict or individual fields."""
    if isinstance(summary_or_pnl, dict):
        s = summary_or_pnl
        day_pnl = s.get("total_pnl", 0)
        day_pnl_pct = s.get("pnl_pct", 0)
        n_trades = s.get("trades", 0)
        win_rate = s.get("win_rate", 0)
        best_trade = s.get("best_trade", "N/A")
        worst_trade = s.get("worst_trade", "N/A")
    else:
        day_pnl = summary_or_pnl

    _send_whatsapp(
        f"\U0001f4ca *DAILY SUMMARY*\n"
        f"P&L: ${day_pnl:+.2f} ({day_pnl_pct:+.2%})\n"
        f"Trades: {n_trades} | Win rate: {win_rate:.0%}\n"
        f"Best: {best_trade} | Worst: {worst_trade}"
        + (f"\nEquity: ${equity:,.0f}" if equity else "")
    )


def notify_drawdown_warning(drawdown_pct: float):
    """Notify when portfolio drawdown exceeds threshold."""
    _send_whatsapp(
        f"\u26a0\ufe0f *DRAWDOWN WARNING*\n"
        f"Portfolio down {drawdown_pct:.2%} from peak.\n"
        f"Position sizing reduced."
    )


def notify_ml_retrain(results: dict):
    """Notify about ML model retraining results."""
    lines = ["\U0001f9e0 *ML MODEL RETRAINED*"]
    for strategy, metrics in results.items():
        status = "\u2705" if metrics["active"] else "\u274c"
        lines.append(
            f"{status} {strategy}: precision={metrics['precision']:.1%}, "
            f"n={metrics['train_samples']}"
        )
    _send_whatsapp("\n".join(lines))


def notify_vix_alert(vix_level: float, risk_scalar: float):
    """V4: Notify when VIX crosses a significant threshold."""
    if vix_level >= 40:
        _send_whatsapp(
            f"\U0001f6a8 *VIX EXTREME: {vix_level:.1f}*\n"
            f"ALL NEW POSITIONS HALTED\n"
            f"Managing existing positions only."
        )
    elif vix_level >= 30:
        _send_whatsapp(
            f"\u26a0\ufe0f *VIX HIGH: {vix_level:.1f}*\n"
            f"Risk scalar: {risk_scalar:.0%}\n"
            f"Position sizes severely reduced."
        )
    elif vix_level >= 25:
        _send_whatsapp(
            f"\U0001f536 *VIX ELEVATED: {vix_level:.1f}*\n"
            f"Risk scalar: {risk_scalar:.0%}\n"
            f"Position sizes reduced."
        )


def notify_strategy_demoted(strategy: str, sharpe: float):
    """Send notification when a strategy is auto-demoted to shadow mode."""
    _send_whatsapp(
        f"\u26a0\ufe0f *Strategy Demoted*\n\n"
        f"Strategy: {strategy}\n"
        f"30d Sharpe: {sharpe:.2f}\n"
        f"Action: Demoted to shadow mode\n"
        f"Reason: Sharpe below minimum threshold"
    )


def notify_optimization(strategy: str, old_sharpe: float, new_sharpe: float,
                        params: dict):
    """Notify when strategy parameters are optimized."""
    _send_whatsapp(
        f"\U0001f527 *{strategy} PARAMS UPDATED*\n"
        f"Sharpe: {old_sharpe:.2f} \u2192 {new_sharpe:.2f}\n"
        f"New params: {params}"
    )
