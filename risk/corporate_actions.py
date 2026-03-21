"""RISK-004: Corporate actions detection and position adjustment.

Detects stock splits and dividends, then adjusts position records and
model parameters accordingly:
    - Stock split: Adjust position qty, entry price, stop loss, take profit.
    - Ex-dividend: Adjust price targets downward by the dividend amount.

Data sources (tried in order, fail-open):
    1. Alpaca corporate actions API
    2. yfinance as fallback

Usage:
    detector = CorporateActionDetector()
    actions = detector.check_actions(["AAPL", "TSLA", "MSFT"])
    adjustments = detector.apply_adjustments(actions, open_trades)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any

import config

logger = logging.getLogger(__name__)

# How often to re-check (seconds)
CHECK_INTERVAL_SEC = 300  # 5 minutes


class ActionType(Enum):
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"


@dataclass
class CorporateAction:
    """A detected corporate action."""
    symbol: str
    action_type: ActionType
    effective_date: date
    # Split fields
    split_ratio: float = 1.0       # e.g., 4.0 for a 4:1 split, 0.5 for 1:2 reverse
    # Dividend fields
    dividend_amount: float = 0.0   # Per-share cash dividend
    ex_date: date | None = None
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class PositionAdjustment:
    """Adjustment to apply to an open position."""
    symbol: str
    action_type: ActionType
    # Quantity adjustment (for splits)
    new_qty: int | None = None
    old_qty: int | None = None
    # Price adjustments
    new_entry_price: float | None = None
    new_stop_loss: float | None = None
    new_take_profit: float | None = None
    reason: str = ""


class CorporateActionDetector:
    """Detect and process corporate actions for open positions.

    Thread-safe. Fail-open: if detection fails, no adjustments are made.
    """

    def __init__(self, check_interval_sec: float = CHECK_INTERVAL_SEC):
        self._check_interval = check_interval_sec
        self._last_check: datetime | None = None
        self._processed_actions: set[str] = set()  # "SYMBOL_TYPE_DATE" keys
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_actions(
        self,
        symbols: list[str],
        reference_date: date | None = None,
    ) -> list[CorporateAction]:
        """Check for corporate actions affecting the given symbols.

        Args:
            symbols: List of stock symbols to check.
            reference_date: Date to check around (default: today).

        Returns:
            List of CorporateAction objects for newly detected actions.
        """
        if not symbols:
            return []

        if reference_date is None:
            reference_date = date.today()

        actions: list[CorporateAction] = []

        # Try Alpaca first, then yfinance as fallback
        for symbol in symbols:
            try:
                symbol_actions = self._check_alpaca(symbol, reference_date)
                if not symbol_actions:
                    symbol_actions = self._check_yfinance(symbol, reference_date)
                actions.extend(symbol_actions)
            except Exception as e:
                logger.debug(f"RISK-004: Corporate action check failed for {symbol} (fail-open): {e}")

        # Filter out already-processed actions
        new_actions = []
        with self._lock:
            for action in actions:
                key = f"{action.symbol}_{action.action_type.value}_{action.effective_date}"
                if key not in self._processed_actions:
                    new_actions.append(action)

        if new_actions:
            logger.info(
                f"RISK-004: Detected {len(new_actions)} new corporate action(s): "
                + ", ".join(f"{a.symbol} {a.action_type.value}" for a in new_actions)
            )

        return new_actions

    def apply_adjustments(
        self,
        actions: list[CorporateAction],
        open_trades: dict[str, Any],
    ) -> list[PositionAdjustment]:
        """Apply corporate action adjustments to open positions.

        Args:
            actions: List of detected corporate actions.
            open_trades: Dict of symbol -> TradeRecord (mutable; will be modified in place).

        Returns:
            List of PositionAdjustment records describing what was changed.
        """
        adjustments: list[PositionAdjustment] = []

        for action in actions:
            if action.symbol not in open_trades:
                continue

            trade = open_trades[action.symbol]

            try:
                if action.action_type in (ActionType.STOCK_SPLIT, ActionType.REVERSE_SPLIT):
                    adj = self._apply_split(trade, action)
                elif action.action_type == ActionType.DIVIDEND:
                    adj = self._apply_dividend(trade, action)
                else:
                    continue

                if adj:
                    adjustments.append(adj)
                    # Mark as processed
                    with self._lock:
                        key = f"{action.symbol}_{action.action_type.value}_{action.effective_date}"
                        self._processed_actions.add(key)

            except Exception as e:
                logger.error(
                    f"RISK-004: Failed to apply {action.action_type.value} "
                    f"for {action.symbol} (fail-open): {e}"
                )

        return adjustments

    # ------------------------------------------------------------------
    # Split adjustment
    # ------------------------------------------------------------------

    def _apply_split(
        self,
        trade: Any,
        action: CorporateAction,
    ) -> PositionAdjustment | None:
        """Adjust position for a stock split or reverse split."""
        ratio = action.split_ratio
        if ratio <= 0 or ratio == 1.0:
            return None

        old_qty = getattr(trade, "qty", 0)
        old_entry = getattr(trade, "entry_price", 0.0)
        old_stop = getattr(trade, "stop_loss", 0.0)
        old_tp = getattr(trade, "take_profit", 0.0)

        # For a 4:1 split: qty *= 4, prices /= 4
        new_qty = int(old_qty * ratio)
        new_entry = old_entry / ratio
        new_stop = old_stop / ratio
        new_tp = old_tp / ratio

        # Apply to trade object
        trade.qty = new_qty
        trade.entry_price = new_entry
        trade.stop_loss = new_stop
        trade.take_profit = new_tp

        # Adjust tracking prices
        if hasattr(trade, "highest_price_seen"):
            trade.highest_price_seen = trade.highest_price_seen / ratio
        if hasattr(trade, "lowest_price_seen"):
            trade.lowest_price_seen = trade.lowest_price_seen / ratio

        logger.info(
            f"RISK-004: Split adjustment for {action.symbol}: "
            f"ratio={ratio}, qty {old_qty}->{new_qty}, "
            f"entry ${old_entry:.2f}->${new_entry:.2f}"
        )

        return PositionAdjustment(
            symbol=action.symbol,
            action_type=action.action_type,
            new_qty=new_qty,
            old_qty=old_qty,
            new_entry_price=new_entry,
            new_stop_loss=new_stop,
            new_take_profit=new_tp,
            reason=f"{ratio}:1 split applied",
        )

    # ------------------------------------------------------------------
    # Dividend adjustment
    # ------------------------------------------------------------------

    def _apply_dividend(
        self,
        trade: Any,
        action: CorporateAction,
    ) -> PositionAdjustment | None:
        """Adjust price targets on ex-dividend date."""
        div_amount = action.dividend_amount
        if div_amount <= 0:
            return None

        side = getattr(trade, "side", "buy")
        old_tp = getattr(trade, "take_profit", 0.0)
        old_stop = getattr(trade, "stop_loss", 0.0)

        # On ex-date, stock opens lower by ~dividend amount.
        # For longs: adjust TP and SL down by dividend amount.
        # For shorts: adjust TP and SL down (favorable for shorts).
        new_tp = old_tp - div_amount
        new_stop = old_stop - div_amount

        trade.take_profit = new_tp
        trade.stop_loss = new_stop

        logger.info(
            f"RISK-004: Dividend adjustment for {action.symbol}: "
            f"${div_amount:.4f}/share, TP ${old_tp:.2f}->${new_tp:.2f}, "
            f"SL ${old_stop:.2f}->${new_stop:.2f}"
        )

        return PositionAdjustment(
            symbol=action.symbol,
            action_type=action.action_type,
            new_take_profit=new_tp,
            new_stop_loss=new_stop,
            reason=f"Ex-div ${div_amount:.4f} price adjustment",
        )

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _check_alpaca(
        self,
        symbol: str,
        reference_date: date,
    ) -> list[CorporateAction]:
        """Check Alpaca corporate actions API."""
        try:
            from broker.alpaca_client import get_trading_client

            client = get_trading_client()
            # Alpaca corporate_actions endpoint (if available)
            # This is a best-effort check; exact API varies by Alpaca SDK version
            announcements = client.get_corporate_actions(
                symbols=[symbol],
                date_from=str(reference_date - timedelta(days=1)),
                date_to=str(reference_date + timedelta(days=1)),
            )

            actions = []
            for ann in announcements:
                if ann.ca_type == "dividend":
                    actions.append(CorporateAction(
                        symbol=symbol,
                        action_type=ActionType.DIVIDEND,
                        effective_date=ann.ex_date,
                        dividend_amount=float(ann.cash),
                        ex_date=ann.ex_date,
                    ))
                elif ann.ca_type in ("forward_split", "stock_split"):
                    ratio = float(ann.new_rate) / float(ann.old_rate)
                    actions.append(CorporateAction(
                        symbol=symbol,
                        action_type=ActionType.STOCK_SPLIT,
                        effective_date=ann.ex_date,
                        split_ratio=ratio,
                    ))
                elif ann.ca_type == "reverse_split":
                    ratio = float(ann.new_rate) / float(ann.old_rate)
                    actions.append(CorporateAction(
                        symbol=symbol,
                        action_type=ActionType.REVERSE_SPLIT,
                        effective_date=ann.ex_date,
                        split_ratio=ratio,
                    ))

            return actions

        except Exception as e:
            logger.debug(f"RISK-004: Alpaca corporate actions unavailable for {symbol}: {e}")
            return []

    def _check_yfinance(
        self,
        symbol: str,
        reference_date: date,
    ) -> list[CorporateAction]:
        """Check yfinance for splits and dividends (fallback)."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)

            actions = []

            # Check splits
            splits = ticker.splits
            if splits is not None and not splits.empty:
                # Filter to recent splits (within 2 days of reference date)
                for ts, ratio in splits.items():
                    split_date = ts.date() if hasattr(ts, "date") else ts
                    if abs((split_date - reference_date).days) <= 1 and ratio != 1.0:
                        action_type = ActionType.STOCK_SPLIT if ratio > 1.0 else ActionType.REVERSE_SPLIT
                        actions.append(CorporateAction(
                            symbol=symbol,
                            action_type=action_type,
                            effective_date=split_date,
                            split_ratio=float(ratio),
                        ))

            # Check dividends
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                for ts, amount in dividends.items():
                    div_date = ts.date() if hasattr(ts, "date") else ts
                    if abs((div_date - reference_date).days) <= 1 and amount > 0:
                        actions.append(CorporateAction(
                            symbol=symbol,
                            action_type=ActionType.DIVIDEND,
                            effective_date=div_date,
                            dividend_amount=float(amount),
                            ex_date=div_date,
                        ))

            return actions

        except Exception as e:
            logger.debug(f"RISK-004: yfinance corporate action check failed for {symbol}: {e}")
            return []

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "processed_actions_count": len(self._processed_actions),
                "last_check": self._last_check.isoformat() if self._last_check else None,
                "check_interval_sec": self._check_interval,
            }
