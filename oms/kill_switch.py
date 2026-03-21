"""V10 OMS — Emergency kill switch: cancel all orders + close all positions."""

import logging
import time as _time
from datetime import datetime

import config

from engine.event_log import log_event, EventType

logger = logging.getLogger(__name__)

# MED-031: Batch size and delay for position closes to avoid API rate limits
KILL_SWITCH_BATCH_SIZE = getattr(config, "KILL_SWITCH_BATCH_SIZE", 5)
KILL_SWITCH_BATCH_DELAY_SEC = getattr(config, "KILL_SWITCH_BATCH_DELAY_SEC", 0.5)


class KillSwitch:
    """Emergency halt: cancel all orders, close all positions, disable new trading.

    Can be activated via:
    - API endpoint (web dashboard)
    - CLI command
    - Auto-trigger on extreme drawdown (configurable)
    """

    def __init__(self):
        self.active = False
        self.activated_at: datetime | None = None
        self.reason: str = ""

    def activate(self, reason: str = "manual", risk_manager=None, order_manager=None):
        """Activate kill switch: cancel all orders and close all positions.

        Args:
            reason: Why the kill switch was activated
            risk_manager: RiskManager instance for position data
            order_manager: OrderManager instance for order cancellation
        """
        if self.active:
            logger.warning("Kill switch already active")
            return

        self.active = True
        self.activated_at = datetime.now(config.ET)
        self.reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        log_event(EventType.KILL_SWITCH, "kill_switch",
                  details=f"reason={reason}", severity="CRITICAL")

        # 1. Cancel all pending/active orders
        if order_manager:
            cancelled = order_manager.cancel_all()
            logger.info(f"Kill switch: cancelled {len(cancelled)} orders")

        # 2. Close all positions via broker (MED-031: batch to avoid API rate limits)
        failed_closes = []
        if risk_manager:
            from execution import close_position
            symbols = list(risk_manager.open_trades.keys())
            for batch_start in range(0, len(symbols), KILL_SWITCH_BATCH_SIZE):
                batch = symbols[batch_start:batch_start + KILL_SWITCH_BATCH_SIZE]
                for symbol in batch:
                    try:
                        close_position(symbol, reason="kill_switch")
                        trade = risk_manager.open_trades.get(symbol)
                        if trade:
                            risk_manager.close_trade(
                                symbol, trade.entry_price,
                                datetime.now(config.ET),
                                exit_reason="kill_switch",
                            )
                        logger.info(f"Kill switch: closed {symbol}")
                    except Exception as e:
                        failed_closes.append(symbol)
                        logger.error(f"Kill switch: failed to close {symbol}: {e}")
                # Sleep between batches to avoid API rate limits (skip after last batch)
                if batch_start + KILL_SWITCH_BATCH_SIZE < len(symbols):
                    _time.sleep(KILL_SWITCH_BATCH_DELAY_SEC)

            if failed_closes:
                logger.critical(
                    f"KILL SWITCH: {len(failed_closes)} positions FAILED to close: "
                    f"{failed_closes}. MANUAL INTERVENTION REQUIRED."
                )

        # 3. Send notification
        try:
            import notifications
            if config.TELEGRAM_ENABLED:
                notifications.send_alert(
                    f"KILL SWITCH ACTIVATED: {reason}\n"
                    f"All positions closed, all orders cancelled."
                )
        except Exception:
            pass

    def deactivate(self):
        """Deactivate kill switch, allowing trading to resume."""
        if not self.active:
            return
        self.active = False
        logger.info(f"Kill switch deactivated (was active since {self.activated_at})")
        self.activated_at = None
        self.reason = ""

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (kill switch not active)."""
        return not self.active

    @property
    def status(self) -> dict:
        return {
            "active": self.active,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "reason": self.reason,
        }
