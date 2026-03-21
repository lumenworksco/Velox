"""RISK-007: Data-driven stop losses using split conformal prediction intervals.

Computes prediction intervals for price movements based on recent historical
data, without distributional assumptions. Sets stop-loss levels at the lower
boundary of the prediction interval.

Method (split conformal prediction):
    1. Split recent price returns into calibration and test sets.
    2. Fit a simple model (rolling mean) on calibration data.
    3. Compute residuals (nonconformity scores) on calibration set.
    4. Use the (1 - alpha) quantile of residuals as the prediction width.
    5. Stop loss = current_price - prediction_width (for longs).

This provides distribution-free coverage guarantees at the specified
confidence level (e.g., 95% means the true price will stay within the
interval 95% of the time on exchangeable data).

Usage:
    engine = ConformalStopEngine()
    stop = engine.compute_stop(
        prices=[100.0, 101.0, 99.5, ...],
        current_price=102.0,
        side="buy",
        confidence=0.95,
    )
    print(f"Stop loss: ${stop.stop_price:.2f} (width: {stop.interval_width:.2f})")
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_CONFIDENCE = 0.95
DEFAULT_CALIBRATION_WINDOW = 60  # Days of calibration data
DEFAULT_MIN_SAMPLES = 20         # Minimum samples needed


@dataclass
class ConformalStop:
    """Result of conformal stop-loss computation."""
    stop_price: float
    interval_width: float          # Width of prediction interval in dollars
    interval_width_pct: float      # Width as fraction of current price
    confidence: float              # Confidence level used
    calibration_samples: int       # Number of calibration residuals
    current_price: float
    side: str                      # "buy" or "sell"
    method: str = "split_conformal"
    computed_at: datetime = field(default_factory=datetime.now)


class ConformalStopEngine:
    """Compute data-driven stop losses using split conformal prediction.

    Fail-open: if computation fails (insufficient data, numerical issues),
    returns None so the caller can fall back to a default stop method.
    """

    def __init__(
        self,
        calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        self._calibration_window = calibration_window
        self._min_samples = min_samples
        self._cache: dict[str, ConformalStop] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_stop(
        self,
        prices: list[float] | np.ndarray,
        current_price: float,
        side: str = "buy",
        confidence: float = DEFAULT_CONFIDENCE,
        symbol: str = "",
    ) -> ConformalStop | None:
        """Compute a conformal prediction-based stop loss.

        Args:
            prices: Historical price series (oldest first). Needs at least
                    min_samples + 10 observations.
            current_price: Current market price.
            side: "buy" (long) or "sell" (short).
            confidence: Prediction interval confidence level (e.g., 0.95).
            symbol: Optional symbol for logging/caching.

        Returns:
            ConformalStop with the computed stop price, or None if
            insufficient data or computation fails (fail-open).
        """
        try:
            return self._compute_stop_inner(prices, current_price, side, confidence, symbol)
        except Exception as e:
            logger.warning(
                f"RISK-007: Conformal stop computation failed for {symbol or 'unknown'} "
                f"(fail-open): {e}"
            )
            return None

    def compute_stops_batch(
        self,
        positions: dict[str, Any],
        price_data: dict[str, list[float]],
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> dict[str, ConformalStop]:
        """Compute conformal stops for multiple positions.

        Args:
            positions: Dict of symbol -> position (TradeRecord or dict).
            price_data: Dict of symbol -> historical price list.
            confidence: Confidence level for all computations.

        Returns:
            Dict of symbol -> ConformalStop (only for successful computations).
        """
        results: dict[str, ConformalStop] = {}

        for symbol, pos in positions.items():
            prices = price_data.get(symbol)
            if prices is None:
                continue

            current_price = self._get_attr(pos, "entry_price", 0.0)
            side = self._get_attr(pos, "side", "buy")

            if current_price <= 0:
                continue

            stop = self.compute_stop(
                prices=prices,
                current_price=current_price,
                side=side,
                confidence=confidence,
                symbol=symbol,
            )

            if stop is not None:
                results[symbol] = stop

        return results

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_stop_inner(
        self,
        prices: list[float] | np.ndarray,
        current_price: float,
        side: str,
        confidence: float,
        symbol: str,
    ) -> ConformalStop | None:
        """Internal split conformal stop computation."""
        prices = np.asarray(prices, dtype=np.float64)

        if len(prices) < self._min_samples + 10:
            logger.debug(
                f"RISK-007: Insufficient price data for {symbol}: "
                f"{len(prices)} < {self._min_samples + 10}"
            )
            return None

        if current_price <= 0:
            return None

        # Compute log returns
        returns = np.diff(np.log(prices))
        returns = returns[np.isfinite(returns)]

        if len(returns) < self._min_samples:
            return None

        # Split into calibration (first 70%) and holdout (last 30%)
        split_idx = int(len(returns) * 0.7)
        cal_returns = returns[:split_idx]
        # holdout_returns = returns[split_idx:]  # Not used for stop computation

        if len(cal_returns) < self._min_samples:
            return None

        # Simple prediction model: rolling mean of recent returns
        # We predict next return = mean of last N returns
        window = min(20, len(cal_returns) // 2)
        predictions = np.array([
            np.mean(cal_returns[max(0, i - window):i])
            for i in range(window, len(cal_returns))
        ])
        actuals = cal_returns[window:]

        if len(predictions) == 0 or len(actuals) == 0:
            return None

        # Nonconformity scores: absolute prediction errors
        residuals = np.abs(actuals - predictions)

        # Conformal quantile: (1 - alpha) quantile of residuals
        # For finite-sample validity: use ceil((n+1)(1-alpha))/n quantile
        alpha = 1.0 - confidence
        n = len(residuals)
        quantile_level = min(np.ceil((n + 1) * confidence) / n, 1.0)
        prediction_width_return = float(np.quantile(residuals, quantile_level))

        # Convert return-space width to dollar-space
        # For small returns: price_width ~= current_price * return_width
        interval_width = current_price * prediction_width_return
        interval_width_pct = prediction_width_return

        # Set stop based on side
        if side == "buy":
            stop_price = current_price - interval_width
        else:
            stop_price = current_price + interval_width

        # Sanity: stop must be positive and reasonable
        if stop_price <= 0:
            stop_price = current_price * 0.90  # Fallback: 10% stop

        result = ConformalStop(
            stop_price=round(stop_price, 2),
            interval_width=round(interval_width, 2),
            interval_width_pct=round(interval_width_pct, 4),
            confidence=confidence,
            calibration_samples=n,
            current_price=current_price,
            side=side,
        )

        # Cache
        if symbol:
            with self._lock:
                self._cache[symbol] = result

        logger.debug(
            f"RISK-007: Conformal stop for {symbol}: ${result.stop_price:.2f} "
            f"(width={result.interval_width_pct:.2%}, n={n}, conf={confidence})"
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        """Get attribute from object or dict."""
        if hasattr(obj, name):
            return getattr(obj, name, default)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def cached_stops(self) -> dict[str, ConformalStop]:
        with self._lock:
            return dict(self._cache)

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "cached_symbols": list(self._cache.keys()),
                "cached_count": len(self._cache),
                "calibration_window": self._calibration_window,
                "min_samples": self._min_samples,
            }
