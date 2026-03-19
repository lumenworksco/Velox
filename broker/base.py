"""V10: Abstract broker interface.

Abstracts the broker behind a common interface for multi-broker support:
- Alpaca (current, via data.py + execution.py)
- Paper/Simulated (broker/paper_broker.py)
- Interactive Brokers (future, Phase 5)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: str = ""
    filled_price: float = 0.0
    filled_qty: int = 0
    message: str = ""
    submitted_at: datetime | None = None
    filled_at: datetime | None = None


@dataclass
class AccountInfo:
    """Account information."""
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    day_trade_count: int = 0
    pattern_day_trader: bool = False


@dataclass
class Position:
    """Open position info."""
    symbol: str = ""
    qty: int = 0
    avg_entry_price: float = 0.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    side: str = "long"


@dataclass
class Snapshot:
    """Market data snapshot for a symbol."""
    symbol: str = ""
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    timestamp: datetime | None = None


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

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        """Get a market data snapshot. Optional."""
        return None

    def get_filled_exit_info(self, symbol: str, side: str = "buy") -> tuple[float | None, str]:
        """Get fill price and reason for a recently closed position. Optional."""
        return None, ""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BrokerError(Exception):
    """Error from broker operations."""
    pass
