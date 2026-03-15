"""V8: Abstract broker interface.

Abstracts the broker behind a common interface so paper trading can have
realistic slippage modeling and future broker swaps are easy.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: str = ""
    filled_price: float = 0.0
    filled_qty: int = 0
    message: str = ""


@dataclass
class AccountInfo:
    """Account information."""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0


@dataclass
class Position:
    """Open position info."""
    symbol: str = ""
    qty: int = 0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    side: str = "long"


class Broker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def submit_order(self, symbol: str, qty: int, side: str,
                     order_type: str = "market",
                     limit_price: float | None = None,
                     take_profit: float | None = None,
                     stop_loss: float | None = None,
                     time_in_force: str = "day") -> OrderResult:
        """Submit an order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account information."""
        ...

    @abstractmethod
    def can_short(self, symbol: str, qty: int, price: float) -> tuple[bool, str]:
        """Check if shorting is allowed for this symbol."""
        ...

    @abstractmethod
    def close_position(self, symbol: str, qty: int | None = None) -> OrderResult:
        """Close a position (full or partial)."""
        ...

    @abstractmethod
    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions."""
        ...
