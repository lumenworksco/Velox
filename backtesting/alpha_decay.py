"""BACKTEST-005: Alpha Decay Analysis.

Measures how quickly a strategy's predictive signal loses value over time.
Tracks the expected return at various horizons after signal generation
and monitors decay rate to recommend strategy demotion.

Usage:
    analyzer = AlphaDecayAnalyzer()
    curve = analyzer.compute_decay_curve(signals, prices)
    print(curve.horizons, curve.expected_returns)

    should_demote = analyzer.should_demote_strategy("STAT_MR")
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Data classes
# ===================================================================== #

@dataclass
class DecayCurve:
    """Alpha decay curve for a strategy or signal.

    The decay curve shows expected return at each time horizon after
    signal generation. A healthy signal has positive expected return
    at early horizons that gradually decays toward zero.
    """
    strategy: str = ""

    # Horizons in minutes
    horizons: list[int] = field(default_factory=lambda: [0, 1, 5, 15, 60])

    # Expected return at each horizon (same length as horizons)
    expected_returns: list[float] = field(default_factory=list)

    # Standard error at each horizon
    std_errors: list[float] = field(default_factory=list)

    # Number of observations at each horizon
    n_observations: list[int] = field(default_factory=list)

    # Half-life: minutes until alpha decays to 50% of peak
    half_life_minutes: float = 0.0

    # Decay rate: exponential decay constant (higher = faster decay)
    decay_rate: float = 0.0

    # Peak alpha: maximum expected return across horizons
    peak_alpha: float = 0.0
    peak_horizon_minutes: int = 0

    # Current health
    is_healthy: bool = True
    health_score: float = 1.0   # 0.0 = dead, 1.0 = strong alpha


@dataclass
class _SignalRecord:
    """Internal record of a signal and subsequent price path."""
    symbol: str
    strategy: str
    signal_time: datetime
    signal_direction: float   # +1 for long, -1 for short
    signal_strength: float    # 0-1 confidence
    entry_price: float
    # Prices at various horizons (filled in as time passes)
    horizon_prices: dict[int, float] = field(default_factory=dict)


class AlphaDecayAnalyzer:
    """Analyzes alpha decay across strategies and signals.

    Tracks signals over time and measures how expected return evolves
    at standard horizons (t=0, t+1min, t+5min, t+15min, t+1hr).

    Args:
        horizons_minutes: List of horizon points in minutes.
        min_observations: Minimum signals needed for reliable decay estimate.
        demote_sharpe_threshold: Sharpe below which strategy should be demoted.
        demote_half_life_min: Half-life below which alpha decays too fast.
        lookback_days: Number of days of signal history to consider.
    """

    DEFAULT_HORIZONS = [0, 1, 5, 15, 60]

    def __init__(
        self,
        horizons_minutes: Optional[list[int]] = None,
        min_observations: int = 30,
        demote_sharpe_threshold: float = 0.3,
        demote_half_life_min: float = 2.0,
        lookback_days: int = 30,
    ) -> None:
        self._horizons = horizons_minutes or self.DEFAULT_HORIZONS
        self._min_obs = min_observations
        self._demote_sharpe = demote_sharpe_threshold
        self._demote_half_life = demote_half_life_min
        self._lookback_days = lookback_days

        # Signal storage by strategy
        self._signals: dict[str, list[_SignalRecord]] = defaultdict(list)
        self._decay_curves: dict[str, DecayCurve] = {}

        # Historical decay tracking (strategy -> list of (date, half_life))
        self._decay_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    #  Signal recording
    # ------------------------------------------------------------------ #

    def record_signal(
        self,
        strategy: str,
        symbol: str,
        signal_time: datetime,
        signal_direction: float,
        entry_price: float,
        signal_strength: float = 1.0,
    ) -> None:
        """Record a new signal for decay tracking.

        Args:
            strategy: Strategy name.
            symbol: Ticker symbol.
            signal_time: When the signal was generated.
            signal_direction: +1 for long, -1 for short.
            entry_price: Price at signal time.
            signal_strength: Confidence 0-1 (default 1.0).
        """
        record = _SignalRecord(
            symbol=symbol,
            strategy=strategy,
            signal_time=signal_time,
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            entry_price=entry_price,
        )
        self._signals[strategy].append(record)

    def update_prices(
        self,
        strategy: str,
        symbol: str,
        current_price: float,
        current_time: datetime,
    ) -> None:
        """Update horizon prices for pending signals.

        Call this periodically to fill in the price at each horizon
        after signal generation.

        Args:
            strategy: Strategy name.
            symbol: Ticker symbol.
            current_price: Current market price.
            current_time: Current timestamp.
        """
        if strategy not in self._signals:
            return

        for record in self._signals[strategy]:
            if record.symbol != symbol:
                continue

            elapsed_minutes = (current_time - record.signal_time).total_seconds() / 60.0

            for horizon in self._horizons:
                if horizon not in record.horizon_prices:
                    # Allow a small tolerance window for matching
                    if abs(elapsed_minutes - horizon) < max(0.5, horizon * 0.1):
                        record.horizon_prices[horizon] = current_price

    # ------------------------------------------------------------------ #
    #  Decay curve computation
    # ------------------------------------------------------------------ #

    def compute_decay_curve(
        self,
        signals: Optional[list[dict]] = None,
        prices: Optional[dict[str, pd.DataFrame]] = None,
        strategy: Optional[str] = None,
    ) -> DecayCurve:
        """Compute the alpha decay curve.

        Can be called in two modes:
        1. With pre-recorded signals (using record_signal + update_prices)
           by passing strategy name.
        2. With raw signal/price data passed directly.

        Args:
            signals: List of signal dicts with keys: symbol, time, direction,
                price, and optionally strategy. Used for batch computation.
            prices: Dict of symbol -> DataFrame with close prices indexed
                by datetime. Used with signals parameter.
            strategy: Strategy name to compute from recorded signals.

        Returns:
            DecayCurve with expected returns at each horizon.
        """
        if signals is not None and prices is not None:
            return self._compute_from_raw(signals, prices)

        if strategy is not None:
            return self._compute_from_recorded(strategy)

        # Default: return empty curve
        return DecayCurve()

    def _compute_from_raw(
        self,
        signals: list[dict],
        prices: dict[str, pd.DataFrame],
    ) -> DecayCurve:
        """Compute decay from raw signal and price data."""
        if not signals:
            return DecayCurve()

        strategy_name = signals[0].get("strategy", "UNKNOWN")
        horizon_returns: dict[int, list[float]] = {h: [] for h in self._horizons}

        for sig in signals:
            symbol = sig.get("symbol", "")
            sig_time = sig.get("time") or sig.get("signal_time")
            direction = sig.get("direction", 1.0)
            entry_price = sig.get("price") or sig.get("entry_price", 0.0)

            if symbol not in prices or entry_price <= 0 or sig_time is None:
                continue

            df = prices[symbol]

            for horizon in self._horizons:
                target_time = sig_time + timedelta(minutes=horizon)

                try:
                    # Find the closest price at or after the target time
                    future_prices = df.loc[df.index >= target_time]
                    if len(future_prices) > 0:
                        future_price = float(future_prices.iloc[0]["close"])
                        ret = direction * (future_price - entry_price) / entry_price
                        horizon_returns[horizon].append(ret)
                except (KeyError, IndexError, TypeError):
                    continue

        return self._build_curve(strategy_name, horizon_returns)

    def _compute_from_recorded(self, strategy: str) -> DecayCurve:
        """Compute decay from internally recorded signals."""
        if strategy not in self._signals:
            return DecayCurve(strategy=strategy)

        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=self._lookback_days)
        signals = [
            s for s in self._signals[strategy]
            if s.signal_time >= cutoff
        ]

        if len(signals) < self._min_obs:
            logger.debug(
                "Insufficient signals for %s: %d < %d",
                strategy, len(signals), self._min_obs,
            )
            return DecayCurve(strategy=strategy)

        horizon_returns: dict[int, list[float]] = {h: [] for h in self._horizons}

        for sig in signals:
            if sig.entry_price <= 0:
                continue

            for horizon in self._horizons:
                if horizon in sig.horizon_prices:
                    ret = (
                        sig.signal_direction *
                        (sig.horizon_prices[horizon] - sig.entry_price) /
                        sig.entry_price
                    )
                    horizon_returns[horizon].append(ret)

        curve = self._build_curve(strategy, horizon_returns)

        # Store in cache and history
        self._decay_curves[strategy] = curve
        self._decay_history[strategy].append(
            (datetime.now(), curve.half_life_minutes)
        )

        return curve

    def _build_curve(
        self,
        strategy: str,
        horizon_returns: dict[int, list[float]],
    ) -> DecayCurve:
        """Build a DecayCurve from computed horizon returns."""
        curve = DecayCurve(
            strategy=strategy,
            horizons=list(self._horizons),
        )

        expected_returns = []
        std_errors = []
        n_observations = []

        for horizon in self._horizons:
            rets = horizon_returns.get(horizon, [])
            n = len(rets)
            n_observations.append(n)

            if n > 0:
                mean_ret = float(np.mean(rets))
                std_ret = float(np.std(rets, ddof=1)) if n > 1 else 0.0
                se = std_ret / np.sqrt(n) if n > 0 else 0.0
                expected_returns.append(mean_ret)
                std_errors.append(se)
            else:
                expected_returns.append(0.0)
                std_errors.append(0.0)

        curve.expected_returns = expected_returns
        curve.std_errors = std_errors
        curve.n_observations = n_observations

        # Peak alpha
        if expected_returns:
            peak_idx = int(np.argmax(np.abs(expected_returns)))
            curve.peak_alpha = expected_returns[peak_idx]
            curve.peak_horizon_minutes = self._horizons[peak_idx]

        # Half-life estimation via exponential fit
        curve.half_life_minutes, curve.decay_rate = self._estimate_half_life(
            self._horizons, expected_returns
        )

        # Health assessment
        curve.health_score = self._compute_health_score(curve)
        curve.is_healthy = curve.health_score > 0.3

        return curve

    def _estimate_half_life(
        self,
        horizons: list[int],
        returns: list[float],
    ) -> tuple[float, float]:
        """Estimate half-life from exponential decay fit.

        Returns (half_life_minutes, decay_rate).
        """
        if len(returns) < 2 or all(r == 0 for r in returns):
            return 0.0, 0.0

        # Use absolute returns for decay estimation
        abs_returns = np.abs(returns)
        peak = max(abs_returns)
        if peak <= 0:
            return 0.0, 0.0

        # Normalize
        normalized = abs_returns / peak

        # Find where it drops below 0.5
        for i in range(len(normalized)):
            if normalized[i] < 0.5 and horizons[i] > 0:
                # Interpolate
                if i > 0:
                    frac = (0.5 - normalized[i]) / (normalized[i - 1] - normalized[i])
                    half_life = horizons[i] - frac * (horizons[i] - horizons[i - 1])
                else:
                    half_life = float(horizons[i])

                # Decay rate: lambda = ln(2) / half_life
                decay_rate = np.log(2) / max(half_life, 0.01)
                return float(half_life), float(decay_rate)

        # Alpha hasn't decayed to 50% within our horizons
        return float(horizons[-1] * 2), np.log(2) / (horizons[-1] * 2)

    def _compute_health_score(self, curve: DecayCurve) -> float:
        """Compute a 0-1 health score for the decay curve.

        Factors:
        - Peak alpha magnitude (higher = healthier)
        - Half-life (longer = healthier, up to a point)
        - Early horizons positive (signal works immediately)
        """
        score = 0.0

        # Peak alpha contribution (0-0.4)
        if abs(curve.peak_alpha) > 0.005:
            score += min(0.4, abs(curve.peak_alpha) * 40)

        # Half-life contribution (0-0.3)
        if curve.half_life_minutes > 0:
            # Optimal half-life: 5-30 minutes
            if curve.half_life_minutes >= 5:
                score += min(0.3, curve.half_life_minutes / 100)
            else:
                score += curve.half_life_minutes / 5 * 0.15

        # Early horizon return sign (0-0.3)
        if len(curve.expected_returns) >= 2:
            early_positive = sum(
                1 for r in curve.expected_returns[:3] if r > 0
            )
            score += early_positive / 3 * 0.3

        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------ #
    #  Strategy health and demotion
    # ------------------------------------------------------------------ #

    def should_demote_strategy(self, strategy: str) -> bool:
        """Determine if a strategy should be demoted due to alpha decay.

        A strategy should be demoted if:
        1. The decay curve health score is below threshold, OR
        2. The half-life has been shrinking over time, OR
        3. Peak alpha has declined significantly

        Args:
            strategy: Strategy name.

        Returns:
            True if the strategy should be demoted.
        """
        curve = self._decay_curves.get(strategy)

        if curve is None:
            # Try to compute
            curve = self._compute_from_recorded(strategy)

        if curve is None or not curve.expected_returns:
            logger.debug("No decay data for %s, cannot assess", strategy)
            return False

        # Check 1: Health score
        if curve.health_score < 0.2:
            logger.info(
                "DEMOTE %s: health score %.2f < 0.20", strategy, curve.health_score
            )
            return True

        # Check 2: Half-life too short
        if curve.half_life_minutes < self._demote_half_life and curve.peak_alpha > 0:
            logger.info(
                "DEMOTE %s: half-life %.1f min < %.1f min threshold",
                strategy, curve.half_life_minutes, self._demote_half_life,
            )
            return True

        # Check 3: Declining half-life trend
        history = self._decay_history.get(strategy, [])
        if len(history) >= 5:
            recent_hl = [h for _, h in history[-5:]]
            if all(recent_hl[i] < recent_hl[i - 1] for i in range(1, len(recent_hl))):
                logger.info(
                    "DEMOTE %s: half-life declining for 5 consecutive checks",
                    strategy,
                )
                return True

        # Check 4: No positive expected return at any horizon
        if all(r <= 0 for r in curve.expected_returns):
            logger.info("DEMOTE %s: no positive expected return at any horizon", strategy)
            return True

        return False

    def get_decay_rate(self, strategy: str) -> float:
        """Get the exponential decay rate for a strategy.

        The decay rate (lambda) measures how quickly alpha dissipates.
        Higher values mean faster decay.  Related to half-life by:
            half_life = ln(2) / decay_rate

        Args:
            strategy: Strategy name.

        Returns:
            Decay rate (float). Returns 0.0 if no curve is available.
        """
        curve = self._decay_curves.get(strategy)
        if curve is None:
            curve = self._compute_from_recorded(strategy)
        return curve.decay_rate if curve else 0.0

    def should_demote(self, strategy: str, threshold: float = 0.3) -> bool:
        """Determine if a strategy should be demoted due to alpha decay.

        Convenience wrapper around should_demote_strategy that also
        accepts a configurable health-score threshold.

        Args:
            strategy: Strategy name.
            threshold: Health score below which the strategy is demoted
                (default 0.3).  Lower = more tolerant.

        Returns:
            True if the strategy should be demoted.
        """
        curve = self._decay_curves.get(strategy)
        if curve is None:
            curve = self._compute_from_recorded(strategy)

        if curve is None or not curve.expected_returns:
            return False

        # Use the provided threshold for health score check
        if curve.health_score < threshold:
            logger.info(
                "DEMOTE %s: health score %.2f < %.2f threshold",
                strategy, curve.health_score, threshold,
            )
            return True

        # Delegate remaining checks to the full method
        return self.should_demote_strategy(strategy)

    def get_decay_trend(self, strategy: str) -> Optional[float]:
        """Get the trend in half-life over time.

        Returns:
            Slope of half-life trend. Negative = decaying faster over time.
            None if insufficient history.
        """
        history = self._decay_history.get(strategy, [])
        if len(history) < 3:
            return None

        half_lives = [h for _, h in history[-10:]]
        x = np.arange(len(half_lives))
        coeffs = np.polyfit(x, half_lives, 1)
        return float(coeffs[0])

    # ------------------------------------------------------------------ #
    #  Housekeeping
    # ------------------------------------------------------------------ #

    def get_all_curves(self) -> dict[str, DecayCurve]:
        """Return all computed decay curves."""
        return dict(self._decay_curves)

    def reset(self, strategy: Optional[str] = None) -> None:
        """Clear stored signals and curves."""
        if strategy is not None:
            self._signals.pop(strategy, None)
            self._decay_curves.pop(strategy, None)
            self._decay_history.pop(strategy, None)
        else:
            self._signals.clear()
            self._decay_curves.clear()
            self._decay_history.clear()
        logger.debug("AlphaDecayAnalyzer reset (strategy=%s)", strategy)

    def __repr__(self) -> str:
        return (
            f"AlphaDecayAnalyzer(strategies={len(self._signals)}, "
            f"horizons={self._horizons})"
        )
