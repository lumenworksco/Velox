"""Walk-forward validation for weekly strategy parameter checks.

Splits recent trade history into in-sample / out-of-sample halves
and computes OOS Sharpe to recommend promote / maintain / demote.
"""

import logging

import numpy as np

import config
import database

logger = logging.getLogger(__name__)

# WIRE-010: Monte Carlo robustness gate (fail-open)
_mc_tester = None
try:
    from backtesting.monte_carlo import MonteCarloTester as _MCT
    _mc_tester = _MCT()
except ImportError:
    _MCT = None


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
        """Validate a strategy using expanding-window walk-forward.

        HIGH-016: Uses 4-6 expanding-window segments instead of a simple
        50/50 train/test split.  Each segment uses all prior trades as
        in-sample and the next segment as out-of-sample, producing
        multiple OOS Sharpe estimates for a more robust assessment.

        Args:
            strategy_name: e.g. 'STAT_MR', 'VWAP', 'ORB'.
            trades_30d: Last 30 days of completed trade dicts.  Each dict
                must contain at least ``pnl_pct``, ``strategy``,
                ``entry_time``, ``exit_time``.

        Returns:
            dict with keys: sharpe, win_rate, total_trades, recommendation,
            segment_sharpes.
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
                'segment_sharpes': [],
            }

        # Determine number of segments (4-6, depending on trade count)
        n_segments = max(4, min(6, len(trades) // 3))
        segment_size = len(trades) // n_segments

        # Expanding-window walk-forward:
        # Segment 0..k-1 = training, segment k = test
        # Start with at least 1 segment of training
        segment_sharpes = []
        all_oos_returns = []

        for k in range(1, n_segments):
            # Training: all trades up to segment k
            train_end = k * segment_size
            # Test: segment k
            test_start = train_end
            test_end = (k + 1) * segment_size if k < n_segments - 1 else len(trades)

            oos_trades = trades[test_start:test_end]
            if len(oos_trades) < 2:
                continue

            oos_returns = [t['pnl_pct'] for t in oos_trades]
            seg_sharpe = self.compute_sharpe(oos_returns)
            segment_sharpes.append(seg_sharpe)
            all_oos_returns.extend(oos_returns)

        # Aggregate OOS metrics across all segments
        if not all_oos_returns:
            return {
                'sharpe': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'recommendation': 'maintain',
                'segment_sharpes': [],
            }

        sharpe = self.compute_sharpe(all_oos_returns)
        wins = sum(1 for r in all_oos_returns if r > 0)
        win_rate = wins / len(all_oos_returns)

        # Recommendation: use median segment Sharpe to be robust to outliers
        median_seg_sharpe = float(np.median(segment_sharpes)) if segment_sharpes else sharpe

        if median_seg_sharpe < config.WALK_FORWARD_MIN_SHARPE:
            recommendation = 'demote'
        elif median_seg_sharpe > 1.0:
            recommendation = 'promote'
        else:
            recommendation = 'maintain'

        # WIRE-010: Monte Carlo robustness gate (fail-open)
        # If MC 5th percentile Sharpe <= 0, demote regardless
        try:
            if _mc_tester is not None and recommendation != 'demote' and trades:
                mc_result = _mc_tester.run_simulations(
                    trades=trades,
                    initial_capital=getattr(config, 'STARTING_EQUITY', 100_000),
                )
                if hasattr(mc_result, 'sharpe_5th_pct') and mc_result.sharpe_5th_pct <= 0:
                    recommendation = 'demote'
                    logger.warning(
                        "WIRE-010: %s demoted by MC gate — 5th pct Sharpe=%.2f",
                        strategy_name, mc_result.sharpe_5th_pct,
                    )
        except Exception as _e:
            logger.debug("WIRE-010: MC test failed for %s (fail-open): %s", strategy_name, _e)

        logger.info(
            "WF: %s — OOS Sharpe=%.2f (median seg=%.2f)  WR=%.1f%%  "
            "trades=%d  segments=%d -> %s",
            strategy_name, sharpe, median_seg_sharpe, win_rate * 100,
            len(all_oos_returns), len(segment_sharpes), recommendation,
        )

        return {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': len(all_oos_returns),
            'recommendation': recommendation,
            'segment_sharpes': segment_sharpes,
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
