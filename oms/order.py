"""V10 OMS — Order dataclass with state machine."""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from tz_utils import now_et

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order lifecycle states."""
    PENDING = "pending"           # Created, not yet submitted
    SUBMITTED = "submitted"       # Sent to broker
    PARTIAL_FILL = "partial_fill" # Partially filled
    FILLED = "filled"             # Fully filled
    CANCELLED = "cancelled"       # Cancelled by us or broker
    REJECTED = "rejected"         # Rejected by broker
    EXPIRED = "expired"           # Expired (DAY orders at EOD)
    FAILED = "failed"             # Submission failed


# Valid state transitions
_TRANSITIONS = {
    OrderState.PENDING: {OrderState.SUBMITTED, OrderState.FAILED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {OrderState.FILLED, OrderState.PARTIAL_FILL, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED},
    OrderState.PARTIAL_FILL: {OrderState.FILLED, OrderState.CANCELLED},
    OrderState.FILLED: set(),      # Terminal
    OrderState.CANCELLED: set(),   # Terminal
    OrderState.REJECTED: set(),    # Terminal
    OrderState.EXPIRED: set(),     # Terminal
    OrderState.FAILED: set(),      # Terminal
}


@dataclass
class Order:
    """Represents a single order in the OMS."""
    # Identity
    oms_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    broker_order_id: str = ""
    idempotency_key: str = ""

    # Order details
    symbol: str = ""
    strategy: str = ""
    side: str = ""           # "buy" or "sell"
    order_type: str = ""     # "market", "limit", "bracket"
    qty: int = 0
    limit_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0

    # State
    state: OrderState = OrderState.PENDING
    filled_qty: int = 0
    filled_avg_price: float = 0.0

    # Timestamps
    # BUG-009: Use timezone-aware ET datetime instead of naive datetime.now()
    created_at: datetime = field(default_factory=now_et)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Metadata
    pair_id: str = ""
    parent_oms_id: str = ""  # For child orders (bracket legs)

    def transition(self, new_state: OrderState) -> bool:
        """Attempt a state transition. Returns True if valid."""
        if new_state in _TRANSITIONS.get(self.state, set()):
            old = self.state
            self.state = new_state
            # BUG-009: Use timezone-aware ET datetime
            if new_state == OrderState.SUBMITTED:
                self.submitted_at = now_et()
            elif new_state == OrderState.FILLED:
                self.filled_at = now_et()
            elif new_state == OrderState.CANCELLED:
                self.cancelled_at = now_et()
            logger.debug(f"Order {self.oms_id} ({self.symbol}): {old.value} -> {new_state.value}")
            return True
        else:
            logger.warning(
                f"Invalid order transition: {self.oms_id} ({self.symbol}) "
                f"{self.state.value} -> {new_state.value}"
            )
            return False

    @property
    def is_terminal(self) -> bool:
        return len(_TRANSITIONS.get(self.state, set())) == 0

    @property
    def is_active(self) -> bool:
        return self.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL_FILL)
