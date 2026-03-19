"""Kalman Pairs Trader — 25% of capital allocation.

Trades cointegrated pairs within sector groups using a Kalman filter
for dynamic hedge ratio estimation. Dollar-neutral positioning.

Target: 0.3-0.5% per trade, high win rate (70%+).
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame

import config
from data import get_intraday_bars, get_daily_bars
from strategies.base import Signal
from database import (
    save_kalman_pair, get_active_kalman_pairs,
    deactivate_kalman_pair, deactivate_all_kalman_pairs,
)

logger = logging.getLogger(__name__)


class KalmanPairsTrader:
    """Trade cointegrated pairs with Kalman-estimated hedge ratios.

    Workflow:
    1. select_pairs_weekly() — test cointegration within sector groups, select top pairs
    2. scan() every 2 min — update Kalman filter, compute spread z-score, enter at |z| > 2.0
    3. check_exits() every cycle — convergence at |z| < 0.2, stop at |z| > 3.0, max hold 10 days

    CRITICAL: Kalman state persists across scans (theta, P matrices per pair).
    """

    def __init__(self):
        self.active_pairs: list[dict] = []  # [{symbol1, symbol2, hedge_ratio, ...}]
        self.kalman_state: dict[str, dict] = {}  # "SYM1_SYM2" -> {theta, P, spread_mean, spread_std}
        self._pairs_ready = False

    def reset_daily(self):
        """Don't clear pairs — they persist for up to a week."""
        pass  # Only reset on weekly pair selection

    def select_pairs_weekly(self, now: datetime) -> list[dict]:
        """Select cointegrated pairs from sector groups.

        Called weekly (Sunday night/Monday morning).
        Tests all possible pairs within each sector group.

        For each pair:
        1. Get 60 days of daily close data
        2. Test cointegration (Engle-Granger: OLS residuals -> ADF test)
        3. Check correlation > PAIRS_MIN_CORRELATION (0.80)
        4. Check cointegration p-value < PAIRS_COINT_PVALUE (0.05)
        5. Initialize Kalman filter state
        6. Save to database
        """
        deactivate_all_kalman_pairs()
        self.active_pairs = []
        self.kalman_state = {}

        all_candidates = []

        for group_name, members in config.SECTOR_GROUPS.items():
            # Handle ETF pairs (already tuples)
            if group_name == 'etf_pairs':
                pairs_to_test = members  # Already (sym1, sym2) tuples
            else:
                # Generate all pairs within group
                pairs_to_test = [
                    (members[i], members[j])
                    for i in range(len(members))
                    for j in range(i + 1, len(members))
                ]

            for sym1, sym2 in pairs_to_test:
                try:
                    result = self._test_pair(sym1, sym2, group_name)
                    if result:
                        all_candidates.append(result)
                except Exception as e:
                    logger.debug(f"Pair test error {sym1}/{sym2}: {e}")

        # Sort by cointegration quality (low p-value)
        all_candidates.sort(key=lambda c: c['coint_pvalue'])
        top_pairs = all_candidates[:config.PAIRS_MAX_ACTIVE]

        for pair in top_pairs:
            # Initialize Kalman state
            pair_key = f"{pair['symbol1']}_{pair['symbol2']}"
            self.kalman_state[pair_key] = {
                'theta': np.array([pair['hedge_ratio'], 0.0]),  # [hedge_ratio, intercept]
                'P': np.eye(2) * 1.0,  # Covariance matrix
                'spread_mean': pair['spread_mean'],
                'spread_std': pair['spread_std'],
            }

            # Estimate half-life from spread autocorrelation
            half_life = self._estimate_half_life(pair.get('_spread'))

            # Save to DB
            save_kalman_pair(
                pair['symbol1'], pair['symbol2'],
                pair['hedge_ratio'], pair['spread_mean'],
                pair['spread_std'], pair['correlation'],
                pair['coint_pvalue'], half_life,
                pair.get('sector_group', 'unknown'),
            )

            self.active_pairs.append(pair)

        self._pairs_ready = True
        logger.info(f"Pairs selected: {len(self.active_pairs)} from {len(all_candidates)} candidates")
        return self.active_pairs

    @staticmethod
    def _estimate_half_life(spread: np.ndarray | None) -> float:
        """Estimate mean-reversion half-life from spread series."""
        if spread is None or len(spread) < 10:
            return 5.0  # Default
        spread_lag = spread[:-1]
        spread_delta = np.diff(spread)
        if np.std(spread_lag) < 1e-8:
            return 5.0
        beta = np.sum(spread_lag * spread_delta) / np.sum(spread_lag ** 2)
        if beta >= 0:
            return 10.0  # Not mean-reverting, return max
        return max(1.0, -np.log(2) / beta)

    def _test_pair(self, sym1: str, sym2: str, sector_group: str = "unknown") -> dict | None:
        """Test if two symbols form a cointegrated pair."""
        bars1 = get_daily_bars(sym1, days=60)
        bars2 = get_daily_bars(sym2, days=60)

        if bars1 is None or bars2 is None:
            return None
        if len(bars1) < 40 or len(bars2) < 40:
            return None

        # Align dates
        close1 = bars1["close"]
        close2 = bars2["close"]

        # Ensure same length (take the shorter)
        min_len = min(len(close1), len(close2))
        close1 = close1.iloc[-min_len:]
        close2 = close2.iloc[-min_len:]

        # 1. Correlation check
        corr = close1.corr(close2)
        if abs(corr) < config.PAIRS_MIN_CORRELATION:
            return None

        # 2. OLS hedge ratio: close1 = beta * close2 + alpha + epsilon
        X = np.column_stack([close2.values, np.ones(len(close2))])
        params = np.linalg.lstsq(X, close1.values, rcond=None)[0]
        hedge_ratio = params[0]
        intercept = params[1]

        # 3. Compute spread
        spread = close1.values - hedge_ratio * close2.values - intercept
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        if spread_std < 1e-6:
            return None

        # 4. V10: Proper ADF test using statsmodels (replaces hand-rolled approximation)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread, maxlag=None, autolag='AIC')
            t_stat = adf_result[0]
            approx_pvalue = adf_result[1]
        except Exception:
            return None

        if approx_pvalue > config.PAIRS_COINT_PVALUE:
            return None

        return {
            'symbol1': sym1,
            'symbol2': sym2,
            'hedge_ratio': float(hedge_ratio),
            'correlation': float(corr),
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'coint_pvalue': approx_pvalue,
            'sector_group': sector_group,
            '_spread': spread,  # Kept for half-life estimation, not persisted
        }

    def _update_kalman(self, pair_key: str, price1: float, price2: float) -> float:
        """Update Kalman filter and return current spread z-score.

        Kalman filter tracks dynamic hedge ratio:
        observation: price1 = theta[0] * price2 + theta[1] + noise

        Returns z-score of the spread.
        """
        state = self.kalman_state.get(pair_key)
        if not state:
            return 0.0

        theta = state['theta']
        P = state['P']

        # Prediction step
        # State transition: theta doesn't change (random walk model)
        Q = np.eye(2) * config.KALMAN_DELTA  # Process noise
        P = P + Q

        # Observation
        x = np.array([price2, 1.0])  # Observation vector
        y = price1  # Observed value

        # Innovation
        y_hat = x @ theta
        e = y - y_hat  # Spread (innovation)

        # Innovation covariance
        R = config.KALMAN_OBS_NOISE  # Observation noise
        S = x @ P @ x.T + R
        S = max(S, 1e-8)  # V10: prevent division by near-zero

        # Kalman gain
        K = P @ x / S

        # V10: Clamp Kalman gain to prevent divergence
        k_norm = np.linalg.norm(K)
        if k_norm > 1.0:
            K = K / k_norm

        # Update step
        theta = theta + K * e
        P = P - np.outer(K, x) @ P

        # V10: Enforce P symmetry and positive semi-definiteness
        P = (P + P.T) / 2
        eigvals = np.linalg.eigvalsh(P)
        if np.any(eigvals < 0):
            P += np.eye(P.shape[0]) * (abs(min(eigvals)) + 1e-8)

        # Store updated state
        state['theta'] = theta
        state['P'] = P

        # Update running mean/std of spread
        alpha = 0.02  # Exponential decay for running stats
        state['spread_mean'] = (1 - alpha) * state['spread_mean'] + alpha * e
        state['spread_std'] = max(
            0.001,
            np.sqrt((1 - alpha) * state['spread_std'] ** 2 + alpha * e ** 2)
        )

        self.kalman_state[pair_key] = state

        # Z-score of spread
        zscore = (e - state['spread_mean']) / state['spread_std']
        return float(zscore)

    def scan(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan active pairs for entry signals.

        Called every SCAN_INTERVAL_SEC (120s).

        Entry: |z-score| > PAIRS_ZSCORE_ENTRY (2.0)
        Generates TWO linked signals (dollar-neutral pair):
        - If spread too wide (z > 2): short symbol1, long symbol2
        - If spread too narrow (z < -2): long symbol1, short symbol2
        """
        signals = []

        if not self._pairs_ready or not self.active_pairs:
            return signals

        for pair in self.active_pairs:
            sym1, sym2 = pair['symbol1'], pair['symbol2']
            pair_key = f"{sym1}_{sym2}"

            try:
                # Get latest prices (use 2-min bars, last bar)
                lookback = now - timedelta(minutes=10)
                bars1 = get_intraday_bars(sym1, TimeFrame(2, "Min"), start=lookback, end=now)
                bars2 = get_intraday_bars(sym2, TimeFrame(2, "Min"), start=lookback, end=now)

                if bars1 is None or bars2 is None or bars1.empty or bars2.empty:
                    continue

                price1 = bars1["close"].iloc[-1]
                price2 = bars2["close"].iloc[-1]

                # Update Kalman and get z-score
                zscore = self._update_kalman(pair_key, price1, price2)

                state = self.kalman_state.get(pair_key, {})
                hedge_ratio = state.get('theta', [1.0, 0.0])[0]

                # --- Spread too wide (z > entry): short sym1, long sym2
                if zscore > config.PAIRS_ZSCORE_ENTRY:
                    pair_id = f"PAIR_{sym1}_{sym2}_{now.strftime('%H%M')}"

                    # Short leg (sym1)
                    if config.ALLOW_SHORT and sym1 not in config.NO_SHORT_SYMBOLS:
                        signals.append(Signal(
                            symbol=sym1,
                            strategy="KALMAN_PAIRS",
                            side="sell",
                            entry_price=round(price1, 2),
                            take_profit=round(price1 * (1 - config.PAIRS_TP_PCT), 2),
                            stop_loss=round(price1 * (1 + config.PAIRS_SL_PCT), 2),
                            reason=f"Pairs short z={zscore:.2f} vs {sym2}",
                            hold_type="day",
                            pair_id=pair_id,
                        ))

                    # Long leg (sym2)
                    signals.append(Signal(
                        symbol=sym2,
                        strategy="KALMAN_PAIRS",
                        side="buy",
                        entry_price=round(price2, 2),
                        take_profit=round(price2 * (1 + config.PAIRS_TP_PCT), 2),
                        stop_loss=round(price2 * (1 - config.PAIRS_SL_PCT), 2),
                        reason=f"Pairs long z={zscore:.2f} vs {sym1}",
                        hold_type="day",
                        pair_id=pair_id,
                    ))

                # --- Spread too narrow (z < -entry): long sym1, short sym2
                elif zscore < -config.PAIRS_ZSCORE_ENTRY:
                    pair_id = f"PAIR_{sym1}_{sym2}_{now.strftime('%H%M')}"

                    # Long leg (sym1)
                    signals.append(Signal(
                        symbol=sym1,
                        strategy="KALMAN_PAIRS",
                        side="buy",
                        entry_price=round(price1, 2),
                        take_profit=round(price1 * (1 + config.PAIRS_TP_PCT), 2),
                        stop_loss=round(price1 * (1 - config.PAIRS_SL_PCT), 2),
                        reason=f"Pairs long z={zscore:.2f} vs {sym2}",
                        hold_type="day",
                        pair_id=pair_id,
                    ))

                    # Short leg (sym2)
                    if config.ALLOW_SHORT and sym2 not in config.NO_SHORT_SYMBOLS:
                        signals.append(Signal(
                            symbol=sym2,
                            strategy="KALMAN_PAIRS",
                            side="sell",
                            entry_price=round(price2, 2),
                            take_profit=round(price2 * (1 - config.PAIRS_TP_PCT), 2),
                            stop_loss=round(price2 * (1 + config.PAIRS_SL_PCT), 2),
                            reason=f"Pairs short z={zscore:.2f} vs {sym1}",
                            hold_type="day",
                            pair_id=pair_id,
                        ))

            except Exception as e:
                logger.debug(f"Pairs scan error for {sym1}/{sym2}: {e}")

        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check pairs trades for exit conditions.

        Returns list of exit actions with pair_id for coordinated exits.

        Exit conditions:
        - Convergence: |z| < PAIRS_ZSCORE_EXIT (0.2)
        - Divergence stop: |z| > PAIRS_ZSCORE_STOP (3.0)
        - Max hold: PAIRS_MAX_HOLD_DAYS (10 days)
        """
        exits = []
        checked_pairs = set()

        for symbol, trade in open_trades.items():
            if trade.strategy != "KALMAN_PAIRS":
                continue
            if not trade.pair_id or trade.pair_id in checked_pairs:
                continue

            checked_pairs.add(trade.pair_id)

            # Find both legs of the pair
            parts = trade.pair_id.split('_')  # PAIR_SYM1_SYM2_HHMM
            if len(parts) < 4:
                continue

            sym1, sym2 = parts[1], parts[2]
            pair_key = f"{sym1}_{sym2}"

            try:
                # Get current prices
                lookback = now - timedelta(minutes=10)
                bars1 = get_intraday_bars(sym1, TimeFrame(2, "Min"), start=lookback, end=now)
                bars2 = get_intraday_bars(sym2, TimeFrame(2, "Min"), start=lookback, end=now)

                if bars1 is None or bars2 is None or bars1.empty or bars2.empty:
                    continue

                price1 = bars1["close"].iloc[-1]
                price2 = bars2["close"].iloc[-1]

                zscore = self._update_kalman(pair_key, price1, price2)

                # Max hold check
                if hasattr(trade, 'entry_time') and trade.entry_time:
                    days_held = (now - trade.entry_time).total_seconds() / 86400
                    if days_held > config.PAIRS_MAX_HOLD_DAYS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"Pairs max hold ({days_held:.1f}d)",
                            "pair_id": trade.pair_id,
                        })
                        continue

                # Convergence exit
                if abs(zscore) < config.PAIRS_ZSCORE_EXIT:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Pairs converged z={zscore:.2f}",
                        "pair_id": trade.pair_id,
                    })

                # Divergence stop
                elif abs(zscore) > config.PAIRS_ZSCORE_STOP:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Pairs diverged z={zscore:.2f}",
                        "pair_id": trade.pair_id,
                    })

            except Exception as e:
                logger.debug(f"Pairs exit check error for {pair_key}: {e}")

        return exits
