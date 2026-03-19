"""V10 Risk — Tiered intraday circuit breaker.

Replaces the single-threshold circuit breaker with 4 progressive tiers:
- Tier 1 (Yellow): Reduce new position sizes by 50%
- Tier 2 (Orange): Stop all new entries, manage existing only
- Tier 3 (Red):    Close all day-trade positions
- Tier 4 (Black):  Kill switch — close everything

Each tier is configurable via config constants.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import config

logger = logging.getLogger(__name__)


class CircuitTier(IntEnum):
    NORMAL = 0
    YELLOW = 1   # Reduce sizing
    ORANGE = 2   # Stop new entries
    RED = 3      # Close day trades
    BLACK = 4    # Kill switch


@dataclass
class TierConfig:
    threshold_pct: float     # Negative P&L % that triggers this tier
    size_multiplier: float   # Position size multiplier (1.0 = normal, 0 = blocked)
    allow_new_entries: bool  # Whether new trades are allowed
    close_day_trades: bool   # Whether to close all day-hold positions
    close_all: bool          # Whether to close ALL positions


# Default tier thresholds (negative values = loss)
DEFAULT_TIERS = {
    CircuitTier.NORMAL: TierConfig(0.0, 1.0, True, False, False),
    CircuitTier.YELLOW: TierConfig(-0.01, 0.5, True, False, False),    # -1%
    CircuitTier.ORANGE: TierConfig(-0.02, 0.0, False, False, False),   # -2%
    CircuitTier.RED: TierConfig(-0.03, 0.0, False, True, False),       # -3%
    CircuitTier.BLACK: TierConfig(-0.04, 0.0, False, False, True),     # -4%
}


class TieredCircuitBreaker:
    """Progressive circuit breaker with 4 severity tiers."""

    def __init__(self, tiers: dict[CircuitTier, TierConfig] | None = None):
        self.tiers = tiers or DEFAULT_TIERS
        self.current_tier = CircuitTier.NORMAL
        self.last_update: datetime | None = None
        self.tier_history: list[tuple[datetime, CircuitTier]] = []

    def update(self, day_pnl_pct: float, now: datetime | None = None) -> CircuitTier:
        """Evaluate current P&L and determine the appropriate tier.

        Args:
            day_pnl_pct: Today's P&L as a decimal (e.g., -0.02 = -2%)
            now: Current timestamp

        Returns:
            The current CircuitTier
        """
        if now is None:
            now = datetime.now(config.ET)

        old_tier = self.current_tier

        # Determine tier based on P&L (most severe matching tier wins)
        new_tier = CircuitTier.NORMAL
        for tier in sorted(self.tiers.keys(), reverse=True):
            if tier == CircuitTier.NORMAL:
                continue
            cfg = self.tiers[tier]
            if day_pnl_pct <= cfg.threshold_pct:
                new_tier = tier
                break

        if new_tier != old_tier:
            self.current_tier = new_tier
            self.tier_history.append((now, new_tier))
            if new_tier > old_tier:
                logger.warning(
                    f"Circuit breaker ESCALATED: {old_tier.name} -> {new_tier.name} "
                    f"(day P&L: {day_pnl_pct:.2%})"
                )
            else:
                logger.info(
                    f"Circuit breaker de-escalated: {old_tier.name} -> {new_tier.name} "
                    f"(day P&L: {day_pnl_pct:.2%})"
                )

        self.last_update = now
        return self.current_tier

    @property
    def config(self) -> TierConfig:
        """Get the configuration for the current tier."""
        return self.tiers[self.current_tier]

    @property
    def size_multiplier(self) -> float:
        """Get the position size multiplier for the current tier."""
        return self.config.size_multiplier

    @property
    def allow_new_entries(self) -> bool:
        """Whether new entries are allowed at the current tier."""
        return self.config.allow_new_entries

    @property
    def should_close_day_trades(self) -> bool:
        """Whether day-trade positions should be closed."""
        return self.config.close_day_trades

    @property
    def should_close_all(self) -> bool:
        """Whether ALL positions should be closed (kill switch)."""
        return self.config.close_all

    def reset_daily(self):
        """Reset at start of new trading day."""
        self.current_tier = CircuitTier.NORMAL
        self.tier_history.clear()
        logger.info("Circuit breaker reset for new day")

    @property
    def status(self) -> dict:
        return {
            "tier": self.current_tier.name,
            "tier_value": int(self.current_tier),
            "size_multiplier": self.size_multiplier,
            "allow_new_entries": self.allow_new_entries,
            "should_close_day_trades": self.should_close_day_trades,
            "should_close_all": self.should_close_all,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
