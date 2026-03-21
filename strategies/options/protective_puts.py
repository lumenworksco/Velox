"""PROD-015: Protective Put Buyer strategy framework.

Buys OTM put options to hedge downside risk on long equity positions.
Targets positions with significant unrealized gains or during elevated
volatility regimes where tail risk is heightened.

This is a framework stub — full options execution requires broker API
support for options order types.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProtectivePutCandidate:
    """A candidate position for protective put buying."""
    symbol: str
    current_price: float
    shares_held: int
    strategy: str
    unrealized_pnl_pct: float = 0.0
    strike_price: float = 0.0
    expiration: Optional[datetime] = None
    estimated_cost: float = 0.0
    delta: float = 0.0
    protection_pct: float = 0.0  # How much downside is protected


class ProtectivePutBuyer:
    """PROD-015: Framework for buying protective puts on long equity positions.

    Triggered when:
    - A position has significant unrealized gains (protect profits).
    - Volatility regime is elevated (VIX > threshold).
    - Portfolio-level tail risk exceeds limits (CVaR too negative).

    Targets OTM puts (delta -0.20 to -0.35) with 2-6 week expiry.
    """

    def __init__(
        self,
        target_delta: float = -0.25,
        max_cost_pct: float = 0.02,
        min_unrealized_gain_pct: float = 0.03,
        max_days_to_expiry: int = 45,
        min_days_to_expiry: int = 14,
        vix_trigger: float = 25.0,
    ):
        """Initialize the protective put buyer.

        Args:
            target_delta: Target put delta (-0.20 to -0.35 = OTM).
            max_cost_pct: Maximum premium cost as % of position value.
            min_unrealized_gain_pct: Minimum unrealized gain to trigger protection.
            max_days_to_expiry: Maximum DTE for selected contracts.
            min_days_to_expiry: Minimum DTE for selected contracts.
            vix_trigger: VIX level above which all large positions get puts.
        """
        self._target_delta = target_delta
        self._max_cost_pct = max_cost_pct
        self._min_gain_pct = min_unrealized_gain_pct
        self._max_dte = max_days_to_expiry
        self._min_dte = min_days_to_expiry
        self._vix_trigger = vix_trigger

        logger.info(
            "PROD-015: ProtectivePutBuyer initialized (delta=%.2f, max_cost=%.1f%%, "
            "min_gain=%.1f%%, vix_trigger=%.0f)",
            target_delta, max_cost_pct * 100, min_unrealized_gain_pct * 100, vix_trigger,
        )

    def scan_candidates(
        self,
        open_positions: Dict[str, dict],
        current_prices: Dict[str, float],
        current_time: datetime,
        vix_level: Optional[float] = None,
    ) -> List[ProtectivePutCandidate]:
        """Scan open positions for protective put candidates.

        Args:
            open_positions: Dict of symbol -> position dict.
            current_prices: Dict of symbol -> current price.
            current_time: Current datetime.
            vix_level: Current VIX level (optional).

        Returns:
            List of ProtectivePutCandidate objects.
        """
        candidates = []
        force_hedge = vix_level is not None and vix_level >= self._vix_trigger

        for symbol, pos in open_positions.items():
            try:
                # Must be long
                if pos.get("side", "") != "buy":
                    continue

                qty = pos.get("qty", 0)
                if qty < 100:
                    continue

                price = current_prices.get(symbol)
                entry_price = pos.get("entry_price", 0)
                if price is None or price <= 0 or entry_price <= 0:
                    continue

                unrealized_pnl_pct = (price - entry_price) / entry_price

                # Only hedge if significant gains OR VIX is elevated
                if not force_hedge and unrealized_pnl_pct < self._min_gain_pct:
                    continue

                # Estimate put parameters
                strike = self._estimate_strike(price, self._target_delta)
                cost = self._estimate_premium(price, strike, self._target_delta)
                position_value = price * qty

                if cost * (qty // 100) * 100 / position_value > self._max_cost_pct:
                    continue

                protection_pct = (price - strike) / price

                candidate = ProtectivePutCandidate(
                    symbol=symbol,
                    current_price=price,
                    shares_held=int(qty),
                    strategy=pos.get("strategy", ""),
                    unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
                    strike_price=strike,
                    expiration=current_time + timedelta(days=self._max_dte),
                    estimated_cost=cost,
                    delta=self._target_delta,
                    protection_pct=round(protection_pct, 4),
                )
                candidates.append(candidate)

            except Exception as e:
                logger.warning("PROD-015: Error evaluating %s for protective put: %s", symbol, e)

        logger.info(
            "PROD-015: Found %d protective put candidates (vix=%.1f, force_hedge=%s)",
            len(candidates), vix_level or 0, force_hedge,
        )
        return candidates

    def _estimate_strike(self, price: float, target_delta: float) -> float:
        """Estimate OTM put strike price for target delta.

        Stub — in production, query options chain data.
        """
        otm_pct = (0.5 + target_delta) * 0.2  # delta=-0.25 -> ~5% OTM
        return round(price * (1 - otm_pct), 2)

    def _estimate_premium(self, price: float, strike: float, delta: float) -> float:
        """Estimate put premium.

        Stub — in production, use Black-Scholes or broker quotes.
        """
        return round(price * abs(delta) * 0.04, 2)

    def generate_orders(self, candidates: List[ProtectivePutCandidate]) -> List[dict]:
        """Generate order instructions for protective puts.

        Currently a stub — actual execution requires options API support.
        """
        orders = []
        for c in candidates:
            contracts = c.shares_held // 100
            orders.append({
                "action": "BUY_TO_OPEN",
                "symbol": c.symbol,
                "option_type": "PUT",
                "strike": c.strike_price,
                "expiration": c.expiration.strftime("%Y-%m-%d") if c.expiration else None,
                "contracts": contracts,
                "estimated_cost": c.estimated_cost * contracts * 100,
                "order_type": "LIMIT",
                "limit_price": c.estimated_cost,
                "status": "SIMULATED",
            })

        logger.info("PROD-015: Generated %d protective put orders (simulated)", len(orders))
        return orders
