"""Strategy — Dynamic universe expansion based on regime and opportunity.

Instead of a static symbol list, dynamically selects symbols based on:
1. Volatility regime (more symbols in high-vol for mean reversion)
2. Sector momentum (add hot sectors, remove cold ones)
3. Volume filter (minimum ADV to ensure liquidity)
4. Opportunity scoring (prioritize symbols with strongest signal potential)

Usage:
    universe = DynamicUniverse()
    symbols = universe.select(regime="high_vol", equity=100000)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class UniverseSelection:
    """Result of a universe selection."""
    symbols: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    regime: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class DynamicUniverse:
    """Dynamic symbol universe manager.

    Expands/contracts the trading universe based on market conditions,
    liquidity, and opportunity scoring.
    """

    def __init__(
        self,
        base_symbols: list[str] | None = None,
        expansion_pool: list[str] | None = None,
        max_symbols: int | None = None,
        min_adv: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ):
        self.base_symbols = base_symbols or getattr(config, "CORE_SYMBOLS", config.SYMBOLS[:20])
        self.expansion_pool = expansion_pool or getattr(config, "EXPANSION_SYMBOLS", config.SYMBOLS)
        self.max_symbols = max_symbols or getattr(config, "MAX_UNIVERSE_SIZE", 50)
        self.min_adv = min_adv or getattr(config, "MIN_ADV_DOLLARS", 5_000_000)
        self.min_price = min_price or getattr(config, "MIN_STOCK_PRICE", 10.0)
        self.max_price = max_price or getattr(config, "MAX_STOCK_PRICE", 500.0)

        self._current_universe: list[str] = list(self.base_symbols)
        self._symbol_scores: dict[str, float] = {}
        self._last_selection: UniverseSelection = UniverseSelection()

    def select(
        self,
        regime: str = "normal",
        snapshots: dict = None,
        sector_momentum: dict[str, float] = None,
    ) -> UniverseSelection:
        """Select the trading universe based on current conditions.

        Args:
            regime: Market regime (low_vol, normal, high_vol, crisis)
            snapshots: Symbol -> snapshot dict with price/volume data
            sector_momentum: Sector -> momentum score dict

        Returns:
            UniverseSelection with the chosen symbols
        """
        # Start with base symbols (always included)
        candidates = set(self.base_symbols)

        # Determine target size based on regime
        regime_multipliers = {
            "low_vol": 0.7,    # Fewer symbols in calm markets
            "normal": 1.0,
            "high_vol": 1.3,   # More symbols = more mean reversion opportunities
            "crisis": 0.5,     # Minimal universe in crisis
        }
        target_size = min(
            self.max_symbols,
            int(len(self.base_symbols) * regime_multipliers.get(regime, 1.0))
        )

        # Score expansion pool candidates
        scored: list[tuple[str, float]] = []
        for symbol in self.expansion_pool:
            if symbol in candidates:
                continue

            score = self._score_symbol(symbol, regime, snapshots, sector_momentum)
            if score > 0:
                scored.append((symbol, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Add top-scored symbols up to target
        for symbol, score in scored:
            if len(candidates) >= target_size:
                break
            candidates.add(symbol)
            self._symbol_scores[symbol] = score

        new_universe = sorted(candidates)
        old_universe = set(self._current_universe)

        added = [s for s in new_universe if s not in old_universe]
        removed = [s for s in old_universe if s not in new_universe and s not in self.base_symbols]

        self._current_universe = new_universe

        selection = UniverseSelection(
            symbols=new_universe,
            added=added,
            removed=removed,
            scores={s: self._symbol_scores.get(s, 1.0) for s in new_universe},
            regime=regime,
        )
        self._last_selection = selection

        if added or removed:
            logger.info(
                f"Universe updated: {len(new_universe)} symbols "
                f"(+{len(added)}, -{len(removed)}) regime={regime}"
            )

        return selection

    def _score_symbol(
        self,
        symbol: str,
        regime: str,
        snapshots: dict | None,
        sector_momentum: dict[str, float] | None,
    ) -> float:
        """Score a symbol for inclusion in the universe.

        Higher score = more desirable. Returns 0 if filtered out.
        """
        score = 0.5  # Base score
        price = 0.0

        if snapshots and symbol in snapshots:
            snap = snapshots[symbol]
            latest_trade = getattr(snap, 'latest_trade', None)
            if latest_trade and hasattr(latest_trade, 'price'):
                price = float(latest_trade.price)
                if price < self.min_price or price > self.max_price:
                    return 0.0
                # Prefer mid-range prices (better for retail sizing)
                if 20 <= price <= 200:
                    score += 0.2

            # Volume filter
            vol = getattr(snap, 'daily_bar', None)
            if vol:
                daily_volume = float(vol.volume) * price if price else 0
                if daily_volume < self.min_adv:
                    return 0.0
                # Higher volume = better liquidity = higher score
                score += min(0.3, daily_volume / (self.min_adv * 10))

        # Sector momentum bonus
        if sector_momentum:
            # Look up this symbol's sector from config if available
            sector_map = getattr(config, "SYMBOL_SECTORS", {})
            sector = sector_map.get(symbol, "unknown")
            if sector in sector_momentum:
                mom = sector_momentum[sector]
                if regime == "high_vol" and mom < 0:
                    score += 0.3  # Mean reversion loves beaten-down sectors
                elif regime in ("normal", "low_vol") and mom > 0:
                    score += 0.2  # Momentum loves trending sectors

        return score

    @property
    def current_symbols(self) -> list[str]:
        return list(self._current_universe)

    @property
    def status(self) -> dict:
        return {
            "universe_size": len(self._current_universe),
            "base_size": len(self.base_symbols),
            "max_size": self.max_symbols,
            "last_regime": self._last_selection.regime,
            "last_added": self._last_selection.added[:5],
            "last_removed": self._last_selection.removed[:5],
        }
