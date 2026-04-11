"""Telegram notifications — trade alerts, daily summaries, risk warnings.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Set env vars: TELEGRAM_ENABLED=true, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _send_telegram(message: str, parse_mode: str = "Markdown"):
    """Send a message via Telegram Bot API. Synchronous (fire-and-forget).

    V12 HOTFIX: Telegram's legacy Markdown parser chokes on characters like
    '=', '<', '>', '(', ')' embedded in exit reasons (e.g. "MR z-stop z=-4.04
    (< -2.5)") and returns HTTP 400 "can't parse entities". If the first
    send fails with a parse error, retry once as plain text.
    """
    if not config.TELEGRAM_ENABLED:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return

    try:
        import httpx
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"

        def _post(text: str, mode):
            with httpx.Client(timeout=10) as client:
                return client.post(
                    url,
                    json={
                        "chat_id": config.TELEGRAM_CHAT_ID,
                        "text": text,
                        "parse_mode": mode,
                        "disable_web_page_preview": True,
                    },
                )

        resp = _post(message, parse_mode)
        if resp.status_code == 400 and parse_mode and "parse" in resp.text.lower():
            # Retry without markdown, stripping the markdown markers
            plain = (message.replace("*", "").replace("`", "")
                     .replace("_", ""))
            resp = _post(plain, None)
        if resp.status_code != 200:
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")


def send_message(message: str):
    """Public helper — send a plain text message."""
    _send_telegram(message, parse_mode=None)


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
    _send_telegram(
        f"{arrow} *OPENED* {side.upper()} `{symbol}`\n"
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
    _send_telegram(
        f"{icon} *CLOSED* `{symbol}`\n"
        f"P&L: ${pnl:+.2f} ({pnl_pct:+.2%})\n"
        f"Reason: {exit_reason}"
        + (f" | Hold: {hold_time}" if hold_time else "")
    )


def notify_circuit_breaker(day_pnl_pct: float):
    """Notify when circuit breaker triggers."""
    _send_telegram(
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

    _send_telegram(
        f"\U0001f4ca *DAILY SUMMARY*\n"
        f"P&L: ${day_pnl:+.2f} ({day_pnl_pct:+.2%})\n"
        f"Trades: {n_trades} | Win rate: {win_rate:.0%}\n"
        f"Best: {best_trade} | Worst: {worst_trade}"
        + (f"\nEquity: ${equity:,.0f}" if equity else "")
    )


def notify_drawdown_warning(drawdown_pct: float):
    """Notify when portfolio drawdown exceeds threshold."""
    _send_telegram(
        f"\u26a0\ufe0f *DRAWDOWN WARNING*\n"
        f"Portfolio down {drawdown_pct:.2%} from peak.\n"
        f"Position sizing reduced."
    )


def notify_pnl_lock(state: str, day_pnl_pct: float):
    """V6: Notify when PnL lock state changes."""
    if state == "GAIN_LOCK":
        _send_telegram(
            f"\U0001f512 *GAIN LOCK ACTIVATED*\n"
            f"Daily P&L: {day_pnl_pct:+.2%}\n"
            f"Position sizing reduced to 30%."
        )
    elif state == "LOSS_HALT":
        _send_telegram(
            f"\U0001f6a8 *LOSS HALT ACTIVATED*\n"
            f"Daily P&L: {day_pnl_pct:+.2%}\n"
            f"No new trades for remainder of day."
        )


def notify_beta_hedge(action: str, spy_qty: int, portfolio_beta: float):
    """V6: Notify when beta neutralization triggers a hedge."""
    _send_telegram(
        f"\U0001f6e1 *BETA HEDGE*\n"
        f"Action: {action} {abs(spy_qty)} SPY\n"
        f"Portfolio beta: {portfolio_beta:+.2f} \u2192 ~0.00"
    )


def notify_optimization(strategy: str, old_sharpe: float, new_sharpe: float,
                        params: dict):
    """Notify when strategy parameters are optimized."""
    _send_telegram(
        f"\U0001f527 *{strategy} PARAMS UPDATED*\n"
        f"Sharpe: {old_sharpe:.2f} \u2192 {new_sharpe:.2f}\n"
        f"New params: {params}"
    )
