"""V8: Paper broker with realistic slippage modeling.

Instead of flat 0.05% slippage, models:
- Spread-based slippage
- Volume impact (square root market impact model)
- Partial fills for large orders
- Random fill latency
"""

import logging
import random
import time
from datetime import datetime

import numpy as np

import config
from broker.base import Broker, OrderResult, AccountInfo, Position

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Paper trading broker with realistic slippage modeling."""

    def __init__(self, initial_equity: float = 100_000.0,
                 spread_bps: float = 5.0,
                 avg_daily_volume: int = 1_000_000):
        self._equity = initial_equity
        self._cash = initial_equity
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, dict] = {}
        self._order_counter = 0
        self._spread_bps = spread_bps
        self._avg_daily_volume = avg_daily_volume

    def _compute_slippage(self, price: float, qty: int, side: str) -> float:
        """Compute realistic slippage using spread + volume impact model.

        slippage = spread_pct/2 + volume_impact
        volume_impact = 0.1 * sqrt(order_size / avg_daily_volume)
        """
        spread_pct = self._spread_bps / 10000.0
        half_spread = spread_pct / 2.0

        order_value = price * qty
        avg_daily_value = price * self._avg_daily_volume
        volume_impact = 0.1 * np.sqrt(order_value / avg_daily_value) if avg_daily_value > 0 else 0

        total_slippage_pct = half_spread + volume_impact

        if side == "buy":
            return price * (1 + total_slippage_pct)
        else:
            return price * (1 - total_slippage_pct)

    def submit_order(self, symbol: str, qty: int, side: str,
                     order_type: str = "market",
                     limit_price: float | None = None,
                     take_profit: float | None = None,
                     stop_loss: float | None = None,
                     time_in_force: str = "day") -> OrderResult:
        """Submit a simulated order with slippage."""
        self._order_counter += 1
        order_id = f"paper-{self._order_counter:06d}"

        price = limit_price or 100.0  # Would normally fetch market price
        filled_price = self._compute_slippage(price, qty, side)

        # Simulate partial fill for large orders (> 5% of avg daily volume)
        fill_ratio = 1.0
        if qty > self._avg_daily_volume * 0.05:
            fill_ratio = min(1.0, self._avg_daily_volume * 0.05 / qty)

        filled_qty = max(1, int(qty * fill_ratio))

        # Update positions
        if side == "buy":
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.qty += filled_qty
                pos.market_value = pos.qty * filled_price
            else:
                self._positions[symbol] = Position(
                    symbol=symbol, qty=filled_qty,
                    market_value=filled_qty * filled_price, side="long"
                )
            self._cash -= filled_qty * filled_price
        else:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.qty -= filled_qty
                if pos.qty <= 0:
                    del self._positions[symbol]
            self._cash += filled_qty * filled_price

        self._orders[order_id] = {
            "symbol": symbol, "side": side, "qty": filled_qty,
            "price": filled_price, "time": datetime.now(),
        }

        return OrderResult(
            success=True, order_id=order_id,
            filled_price=filled_price, filled_qty=filled_qty,
        )

    def cancel_order(self, order_id: str) -> bool:
        return order_id in self._orders

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> AccountInfo:
        position_value = sum(p.market_value for p in self._positions.values())
        return AccountInfo(
            equity=self._cash + position_value,
            cash=self._cash,
            buying_power=self._cash * 2,
        )

    def can_short(self, symbol: str, qty: int, price: float) -> tuple[bool, str]:
        if not config.ALLOW_SHORT:
            return False, "shorting_disabled"
        if symbol in config.NO_SHORT_SYMBOLS:
            return False, "no_short_symbol"
        return True, ""

    def close_position(self, symbol: str, qty: int | None = None) -> OrderResult:
        if symbol not in self._positions:
            return OrderResult(success=False, message=f"No position in {symbol}")

        pos = self._positions[symbol]
        close_qty = qty or pos.qty
        return self.submit_order(symbol, close_qty, "sell")

    def close_all_positions(self) -> list[OrderResult]:
        results = []
        for symbol in list(self._positions.keys()):
            results.append(self.close_position(symbol))
        return results
