"""Walk-forward validation for weekly strategy parameter checks.

Splits recent trade history into in-sample / out-of-sample halves
and computes OOS Sharpe to recommend promote / maintain / demote.
"""

import logging

import numpy as np

import config
import database

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Weekly walk-forward parameter validation using real trade records."""

    # ------------------------------------------------------------------ #
    #  Core metric
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_sharpe(returns: list[float]) -> float:
        """Annualised Sharpe ratio from a list of per-trade returns.

        Returns 0.0 if fewer than 5 data points or zero standard deviation.
        """
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns, dtype=float)
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(252))

    # ------------------------------------------------------------------ #
    #  Single-strategy validation
    # ------------------------------------------------------------------ #

    def validate_strategy(self, strategy_name: str,
                          trades_30d: list[dict]) -> dict:
        """Validate a strategy using in-sample / out-of-sample split.

        Args:
            strategy_name: e.g. 'STAT_MR', 'VWAP', 'ORB'.
            trades_30d: Last 30 days of completed trade dicts.  Each dict
                must contain at least ``pnl_pct``, ``strategy``,
                ``entry_time``, ``exit_time``.

        Returns:
            dict with keys: sharpe, win_rate, total_trades, recommendation.
            recommendation is one of 'promote', 'maintain', 'demote'.
        """
        # Filter to the requested strategy
        trades = [t for t in trades_30d if t.get('strategy') == strategy_name]

        if len(trades) < 5:
            logger.info(
                "WF: %s — only %d trades, insufficient for validation",
                strategy_name, len(trades),
            )
            return {
                'sharpe': 0.0,
                'win_rate': 0.0,
                'total_trades': len(trades),
                'recommendation': 'maintain',
            }

        # Split 50/50: first half = in-sample, second half = out-of-sample
        mid = len(trades) // 2
        oos_trades = trades[mid:]

        oos_returns = [t['pnl_pct'] for t in oos_trades]
        sharpe = self.compute_sharpe(oos_returns)
        wins = sum(1 for r in oos_returns if r > 0)
        win_rate = wins / len(oos_returns) if oos_returns else 0.0

        # Recommendation logic
        if sharpe < config.WALK_FORWARD_MIN_SHARPE:
            recommendation = 'demote'
        elif sharpe > 1.0:
            recommendation = 'promote'
        else:
            recommendation = 'maintain'

        logger.info(
            "WF: %s — OOS Sharpe=%.2f  WR=%.1f%%  trades=%d -> %s",
            strategy_name, sharpe, win_rate * 100, len(oos_trades),
            recommendation,
        )

        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': len(oos_trades),
            'recommendation': recommendation,
        }

    # ------------------------------------------------------------------ #
    #  Run validation for all strategies
    # ------------------------------------------------------------------ #

    def run_weekly_validation(
        self, strategies: list[str] | None = None,
    ) -> dict[str, dict]:
        """Run walk-forward validation for each strategy.

        BUG-021: Includes ALL strategies found in recent trades (including
        demoted ones) to avoid survivorship bias. The `strategies` parameter
        provides a baseline list; any additional strategies found in the trade
        history are also validated. Demotion status is tracked separately.

        Fetches the last 30 days of trades from the database, validates
        each strategy, logs results, and returns a mapping of
        strategy -> validation result dict.
        """
        trades_30d = database.get_recent_trades(days=30)
        results: dict[str, dict] = {}

        # BUG-021: Discover all strategies in trade history, not just active ones
        strategies_in_trades = {t.get('strategy') for t in trades_30d if t.get('strategy')}
        all_strategies = set(strategies or []) | strategies_in_trades

        for strat in sorted(all_strategies):
            result = self.validate_strategy(strat, trades_30d)
            # BUG-021: Track whether strategy is currently active or demoted
            result['is_active'] = strat in (strategies or [])
            results[strat] = result

        active_count = sum(1 for r in results.values() if r.get('is_active'))
        demoted_count = len(results) - active_count
        logger.info("Walk-forward validation complete for %d strategies "
                     "(%d active, %d demoted/historical)",
                     len(results), active_count, demoted_count)
        return results
