"""V8: Kelly Criterion position sizing engine.

Replaces flat RISK_PER_TRADE_PCT with per-strategy Kelly-optimal sizing
based on historical win rate and avg win/loss ratio.
"""

import logging
from datetime import datetime

import config
import database

logger = logging.getLogger(__name__)


class KellyEngine:
    """Compute half-Kelly fractions per strategy from trade history."""

    def __init__(self):
        self._fractions: dict[str, float] = {}
        self._params: dict[str, dict] = {}
        self._last_computed: datetime | None = None

    def compute_fractions(self):
        """Recalculate Kelly fractions from trade database. Call daily at market open."""
        strategies = list(config.STRATEGY_ALLOCATIONS.keys())

        for strategy in strategies:
            try:
                trades = database.get_recent_trades_by_strategy(strategy, days=365)
                # Use last KELLY_LOOKBACK trades
                trades = trades[-config.KELLY_LOOKBACK:] if len(trades) > config.KELLY_LOOKBACK else trades

                if len(trades) < config.KELLY_MIN_TRADES:
                    self._fractions[strategy] = config.RISK_PER_TRADE_PCT
                    logger.debug(f"Kelly {strategy}: insufficient trades ({len(trades)}/{config.KELLY_MIN_TRADES}), using flat {config.RISK_PER_TRADE_PCT}")
                    continue

                wins = [t for t in trades if t.get("pnl", 0) > 0]
                losses = [t for t in trades if t.get("pnl", 0) <= 0]

                win_rate = len(wins) / len(trades) if trades else 0
                avg_win = sum(abs(t.get("pnl_pct", 0)) for t in wins) / len(wins) if wins else 0
                avg_loss = sum(abs(t.get("pnl_pct", 0)) for t in losses) / len(losses) if losses else 1e-6

                if avg_loss < 1e-8:
                    avg_loss = 1e-6

                win_loss_ratio = avg_win / avg_loss

                # Guard against near-zero win/loss ratio
                if win_loss_ratio < 1e-6:
                    self._fractions[strategy] = config.KELLY_MIN_RISK
                    logger.warning(f"Kelly {strategy}: near-zero win/loss ratio={win_loss_ratio:.6f}, using min risk")
                    continue

                # Kelly fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
                kelly_f = win_rate - ((1 - win_rate) / win_loss_ratio)

                # Half-Kelly for safety
                half_kelly = kelly_f * config.KELLY_FRACTION_MULT

                # Clamp to [KELLY_MIN_RISK, KELLY_MAX_RISK]
                half_kelly = max(config.KELLY_MIN_RISK, min(config.KELLY_MAX_RISK, half_kelly))

                # If Kelly is negative (losing strategy), use minimum
                if kelly_f <= 0:
                    half_kelly = config.KELLY_MIN_RISK
                    logger.warning(f"Kelly {strategy}: negative kelly_f={kelly_f:.4f}, using min risk {config.KELLY_MIN_RISK}")

                self._fractions[strategy] = half_kelly
                self._params[strategy] = {
                    "win_rate": round(win_rate, 4),
                    "avg_win_loss": round(win_loss_ratio, 4),
                    "kelly_f": round(kelly_f, 4),
                    "half_kelly_f": round(half_kelly, 4),
                    "sample_size": len(trades),
                }

                logger.info(f"Kelly {strategy}: wr={win_rate:.2%} w/l={win_loss_ratio:.2f} kelly={kelly_f:.4f} half={half_kelly:.4f} (n={len(trades)})")

                # Save to database
                try:
                    database.save_kelly_params(
                        strategy=strategy,
                        win_rate=win_rate,
                        avg_win_loss=win_loss_ratio,
                        kelly_f=kelly_f,
                        half_kelly_f=half_kelly,
                        sample_size=len(trades),
                    )
                except Exception as e:
                    logger.debug(f"Failed to save kelly params for {strategy}: {e}")

            except Exception as e:
                logger.warning(f"Kelly computation failed for {strategy}: {e}")
                self._fractions[strategy] = config.RISK_PER_TRADE_PCT

        self._last_computed = datetime.now(config.ET)

    def get_fraction(self, strategy: str) -> float:
        """Get the Kelly fraction for a strategy.

        Returns half-Kelly if sufficient data, else falls back to RISK_PER_TRADE_PCT.
        """
        if not config.KELLY_ENABLED:
            return config.RISK_PER_TRADE_PCT

        return self._fractions.get(strategy, config.RISK_PER_TRADE_PCT)

    @property
    def params(self) -> dict[str, dict]:
        """Return computed Kelly parameters for all strategies."""
        return self._params.copy()

    @property
    def last_computed(self) -> datetime | None:
        return self._last_computed
