"""PROD-015: Gamma Scalping strategy framework.

Delta-neutral strategy that profits from realized volatility exceeding
implied volatility. Buys straddles/strangles and dynamically hedges delta
by trading the underlying, capturing gamma profits from large price moves.

This is a framework stub — full options execution requires broker API
support for options order types and real-time Greeks data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GammaPosition:
    """Tracks a gamma scalping position (options + underlying hedge)."""
    symbol: str
    call_strike: float
    put_strike: float
    expiration: Optional[datetime] = None
    contracts: int = 0
    entry_time: Optional[datetime] = None
    underlying_hedge_qty: int = 0  # Shares of underlying for delta hedge
    hedge_side: str = ""  # "buy" or "sell"
    entry_iv: float = 0.0
    current_delta: float = 0.0
    current_gamma: float = 0.0
    realized_pnl: float = 0.0
    hedge_trades: int = 0


@dataclass
class GammaCandidate:
    """A candidate symbol for gamma scalping."""
    symbol: str
    current_price: float
    implied_vol: float
    realized_vol: float
    vol_spread: float  # realized - implied (positive = favorable)
    atr_pct: float = 0.0
    avg_daily_volume: int = 0


class GammaScalper:
    """PROD-015: Framework for delta-neutral gamma scalping.

    Strategy logic:
    1. Screen for symbols where realized vol > implied vol (vol is underpriced).
    2. Buy ATM straddle or near-ATM strangle.
    3. Delta-hedge by trading underlying when delta drifts beyond threshold.
    4. Profit comes from gamma: each hedge trade locks in a small gain.
    5. Close position before expiry to avoid assignment risk.

    Requires:
    - Real-time options Greeks (delta, gamma, IV) from broker or data provider.
    - Options order execution capability.
    - Frequent delta re-hedging (intraday).
    """

    def __init__(
        self,
        min_vol_spread: float = 0.05,
        delta_hedge_threshold: float = 0.10,
        max_positions: int = 3,
        target_dte: int = 30,
        min_dte_exit: int = 5,
        max_cost_pct: float = 0.03,
        min_daily_volume: int = 1_000_000,
    ):
        """Initialize the gamma scalper.

        Args:
            min_vol_spread: Minimum (realized_vol - implied_vol) to enter.
            delta_hedge_threshold: Re-hedge when |delta| exceeds this.
            max_positions: Maximum concurrent gamma positions.
            target_dte: Target days to expiry for option purchases.
            min_dte_exit: Close position when DTE falls below this.
            max_cost_pct: Maximum straddle cost as % of underlying price.
            min_daily_volume: Minimum average daily volume for underlying.
        """
        self._min_vol_spread = min_vol_spread
        self._delta_threshold = delta_hedge_threshold
        self._max_positions = max_positions
        self._target_dte = target_dte
        self._min_dte_exit = min_dte_exit
        self._max_cost_pct = max_cost_pct
        self._min_volume = min_daily_volume

        # Active positions
        self._positions: Dict[str, GammaPosition] = {}

        logger.info(
            "PROD-015: GammaScalper initialized (min_vol_spread=%.1f%%, delta_threshold=%.2f, "
            "max_positions=%d, target_dte=%d)",
            min_vol_spread * 100, delta_hedge_threshold, max_positions, target_dte,
        )

    def scan_candidates(
        self,
        symbols: List[str],
        current_prices: Dict[str, float],
        realized_vols: Dict[str, float],
        implied_vols: Optional[Dict[str, float]] = None,
        volumes: Optional[Dict[str, int]] = None,
    ) -> List[GammaCandidate]:
        """Scan for gamma scalping candidates.

        Args:
            symbols: List of symbols to evaluate.
            current_prices: Current prices.
            realized_vols: Realized volatility per symbol (annualized).
            implied_vols: Implied volatility per symbol (from options data).
                          If None, no candidates are returned (IV data required).
            volumes: Average daily volume per symbol.

        Returns:
            List of GammaCandidate objects, sorted by vol spread (best first).
        """
        if implied_vols is None:
            logger.debug(
                "PROD-015: No implied vol data available — gamma scalp scan skipped"
            )
            return []

        candidates = []

        for symbol in symbols:
            try:
                price = current_prices.get(symbol)
                rv = realized_vols.get(symbol)
                iv = implied_vols.get(symbol)
                vol = (volumes or {}).get(symbol, 0)

                if price is None or rv is None or iv is None:
                    continue
                if price <= 0 or iv <= 0:
                    continue

                # Volume filter
                if vol < self._min_volume:
                    continue

                # Already have a position
                if symbol in self._positions:
                    continue

                vol_spread = rv - iv
                if vol_spread < self._min_vol_spread:
                    continue

                candidates.append(GammaCandidate(
                    symbol=symbol,
                    current_price=price,
                    implied_vol=iv,
                    realized_vol=rv,
                    vol_spread=round(vol_spread, 4),
                    avg_daily_volume=vol,
                ))

            except Exception as e:
                logger.warning("PROD-015: Error evaluating %s for gamma scalp: %s", symbol, e)

        candidates.sort(key=lambda c: c.vol_spread, reverse=True)
        logger.info(
            "PROD-015: Found %d gamma scalp candidates from %d symbols",
            len(candidates), len(symbols),
        )
        return candidates[:self._max_positions]

    def check_delta_hedges(
        self,
        current_prices: Dict[str, float],
        current_greeks: Optional[Dict[str, dict]] = None,
    ) -> List[dict]:
        """Check positions and generate delta hedge orders if needed.

        Args:
            current_prices: Current prices for all symbols.
            current_greeks: Optional dict of symbol -> {"delta": float, "gamma": float}.

        Returns:
            List of hedge order dicts (simulated).
        """
        hedge_orders = []

        for symbol, pos in self._positions.items():
            try:
                price = current_prices.get(symbol)
                if price is None:
                    continue

                # Get current Greeks (stub: would come from options data provider)
                greeks = (current_greeks or {}).get(symbol, {})
                current_delta = greeks.get("delta", 0.0)
                pos.current_delta = current_delta
                pos.current_gamma = greeks.get("gamma", 0.0)

                # Check if delta exceeds threshold
                if abs(current_delta) > self._delta_threshold:
                    # Calculate hedge quantity
                    hedge_shares = -int(current_delta * pos.contracts * 100)
                    if hedge_shares == 0:
                        continue

                    hedge_side = "buy" if hedge_shares > 0 else "sell"
                    hedge_orders.append({
                        "action": "DELTA_HEDGE",
                        "symbol": symbol,
                        "side": hedge_side,
                        "qty": abs(hedge_shares),
                        "price": price,
                        "current_delta": current_delta,
                        "status": "SIMULATED",
                    })

                    pos.hedge_trades += 1
                    logger.info(
                        "PROD-015: Gamma hedge %s %s %d shares (delta=%.3f, gamma=%.4f)",
                        hedge_side.upper(), symbol, abs(hedge_shares),
                        current_delta, pos.current_gamma,
                    )

            except Exception as e:
                logger.warning("PROD-015: Error checking delta hedge for %s: %s", symbol, e)

        return hedge_orders

    def generate_entry_orders(self, candidates: List[GammaCandidate]) -> List[dict]:
        """Generate straddle entry orders for gamma scalping candidates.

        Currently a stub — actual execution requires options API support.
        """
        orders = []
        for c in candidates:
            if len(self._positions) >= self._max_positions:
                break

            strike = round(c.current_price)  # ATM strike
            estimated_premium = c.current_price * c.implied_vol * 0.08  # Rough straddle price

            if estimated_premium / c.current_price > self._max_cost_pct:
                continue

            orders.append({
                "action": "BUY_STRADDLE",
                "symbol": c.symbol,
                "call_strike": strike,
                "put_strike": strike,
                "contracts": 1,
                "estimated_cost": round(estimated_premium * 100, 2),  # Per straddle
                "vol_spread": c.vol_spread,
                "status": "SIMULATED",
            })

        logger.info("PROD-015: Generated %d gamma scalp entries (simulated)", len(orders))
        return orders

    @property
    def open_positions(self) -> Dict[str, GammaPosition]:
        """Get currently open gamma positions."""
        return dict(self._positions)

    def stats(self) -> dict:
        """Return gamma scalper statistics."""
        return {
            "open_positions": len(self._positions),
            "max_positions": self._max_positions,
            "min_vol_spread": self._min_vol_spread,
            "delta_threshold": self._delta_threshold,
            "positions": {
                sym: {
                    "delta": pos.current_delta,
                    "gamma": pos.current_gamma,
                    "hedge_trades": pos.hedge_trades,
                    "realized_pnl": pos.realized_pnl,
                }
                for sym, pos in self._positions.items()
            },
        }
