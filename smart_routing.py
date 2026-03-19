"""Smart Order Routing — choose order type and timing based on signal context."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime

import config
from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderParams:
    """Parameters decided by the smart router for order submission."""
    order_type: str  # "market", "limit", "ioc"
    limit_price: float | None
    urgency: str  # "high", "medium", "low"
    use_twap: bool
    twap_slices: int
    twap_interval_sec: int


# Strategies considered time-sensitive (speed > price improvement)
_TIME_SENSITIVE = {"ORB", "MICRO_MOM"}

# Strategies considered mean-reversion (patient limit orders)
_MEAN_REVERSION = {"STAT_MR", "VWAP", "KALMAN_PAIRS"}


class SmartOrderRouter:
    """Choose order type and timing based on signal urgency, liquidity, and spread."""

    def route(
        self,
        signal: Signal,
        qty: int,
        spread_pct: float = 0.0,
        equity: float = 100_000.0,
    ) -> OrderParams:
        """Decision tree for order routing.

        1. If disabled globally, return plain market order.
        2. If spread > SPREAD_THRESHOLD_PCT: limit at mid-price, medium urgency.
        3. If time-sensitive (ORB, MICRO_MOM): IOC market order, high urgency.
        4. If large order (qty * price > equity * 0.03): TWAP with adaptive slices.
        5. If mean-reversion (STAT_MR, VWAP, KALMAN_PAIRS): limit at entry_price, low urgency.
        6. If PEAD: limit at entry +/- 0.3%, medium urgency.
        Default: market order.
        """
        if not config.SMART_ROUTING_ENABLED:
            return OrderParams(
                order_type="market",
                limit_price=None,
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 1: Wide spread -> limit at mid-price
        if spread_pct > config.SPREAD_THRESHOLD_PCT:
            return OrderParams(
                order_type="limit",
                limit_price=round(signal.entry_price, 2),
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 2: Time-sensitive strategies -> IOC market
        if signal.strategy in _TIME_SENSITIVE:
            return OrderParams(
                order_type="ioc",
                limit_price=None,
                urgency="high",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 3: Large orders -> TWAP
        order_value = qty * signal.entry_price
        if config.ADAPTIVE_TWAP_ENABLED and order_value > equity * 0.03:
            urgency = self._infer_urgency(signal)
            slices, interval = self.compute_adaptive_twap(signal, qty, urgency)
            return OrderParams(
                order_type="market",
                limit_price=None,
                urgency=urgency,
                use_twap=True,
                twap_slices=slices,
                twap_interval_sec=interval,
            )

        # Rule 4: Mean-reversion -> limit at entry price
        if signal.strategy in _MEAN_REVERSION:
            return OrderParams(
                order_type="limit",
                limit_price=round(signal.entry_price, 2),
                urgency="low",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 5: PEAD -> limit at entry +/- 0.3%
        if signal.strategy == "PEAD":
            offset = 0.003
            if signal.side == "buy":
                price = round(signal.entry_price * (1 + offset), 2)
            else:
                price = round(signal.entry_price * (1 - offset), 2)
            return OrderParams(
                order_type="limit",
                limit_price=price,
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Default: market order
        return OrderParams(
            order_type="market",
            limit_price=None,
            urgency="medium",
            use_twap=False,
            twap_slices=0,
            twap_interval_sec=0,
        )

    def compute_adaptive_twap(
        self, signal: Signal, qty: int, urgency: str
    ) -> tuple[int, int]:
        """Compute (n_slices, interval_sec) based on urgency.

        High urgency:   3 slices, 15 sec
        Medium urgency: 5 slices, 30 sec
        Low urgency:    8 slices, 60 sec
        """
        table = {
            "high": (3, 15),
            "medium": (5, 30),
            "low": (8, 60),
        }
        return table.get(urgency, (5, 30))

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_urgency(signal: Signal) -> str:
        """Infer urgency from the signal's strategy."""
        if signal.strategy in _TIME_SENSITIVE:
            return "high"
        if signal.strategy in _MEAN_REVERSION:
            return "low"
        return "medium"


class FillMonitor:
    """Monitors pending orders and tracks fill quality."""

    def __init__(self):
        self._pending: dict[str, dict] = {}  # order_id -> {signal, submit_time, qty}
        self._fill_stats: dict[str, list] = {}  # strategy -> [slippage_pcts]
        self._lock = threading.Lock()

    def register_order(
        self, order_id: str, signal: Signal, submit_time: datetime, qty: int
    ):
        """Register a new pending order for monitoring."""
        self._pending[order_id] = {
            "signal": signal,
            "submit_time": submit_time,
            "qty": qty,
        }
        logger.debug(f"FillMonitor: registered order {order_id} for {signal.symbol}")

    def check_pending(self, now: datetime) -> list[dict]:
        """Check pending orders and return recommended actions.

        Returns list of actions:
        - After CHASE_AFTER_SECONDS:          {action: "chase", order_id, new_price}
        - After CHASE_CONVERT_MARKET_AFTER:   {action: "convert_market", order_id}

        Fail-open: returns empty list on error.
        """
        try:
            actions: list[dict] = []
            for order_id, info in list(self._pending.items()):
                elapsed = (now - info["submit_time"]).total_seconds()
                signal: Signal = info["signal"]

                if elapsed >= config.CHASE_CONVERT_MARKET_AFTER:
                    actions.append({
                        "action": "convert_market",
                        "order_id": order_id,
                    })
                elif elapsed >= config.CHASE_AFTER_SECONDS:
                    # Chase: move limit closer to current price (use entry as proxy)
                    if signal.side == "buy":
                        new_price = round(signal.entry_price * 1.001, 2)
                    else:
                        new_price = round(signal.entry_price * 0.999, 2)
                    actions.append({
                        "action": "chase",
                        "order_id": order_id,
                        "new_price": new_price,
                    })

            return actions
        except Exception:
            logger.exception("FillMonitor.check_pending failed — fail-open")
            return []

    def remove_order(self, order_id: str):
        """Remove an order from pending tracking (after fill or cancel)."""
        self._pending.pop(order_id, None)

    def record_fill(
        self, order_id: str, fill_price: float, expected_price: float, strategy: str
    ):
        """Record fill quality for analytics."""
        if expected_price == 0:
            return
        slippage_pct = (fill_price - expected_price) / expected_price
        with self._lock:
            self._fill_stats.setdefault(strategy, []).append(slippage_pct)
        self.remove_order(order_id)
        logger.info(
            f"Fill recorded: order={order_id} strategy={strategy} "
            f"slippage={slippage_pct:.4%}"
        )

    def get_slippage_stats(self) -> dict[str, float]:
        """Return average slippage per strategy."""
        with self._lock:
            result: dict[str, float] = {}
            for strategy, slippages in self._fill_stats.items():
                if slippages:
                    result[strategy] = sum(slippages) / len(slippages)
            return result

    @property
    def pending_count(self) -> int:
        return len(self._pending)
