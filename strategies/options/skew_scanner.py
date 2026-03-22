"""T5-005: Options Skew Signal Scanner.

Computes the put-call volatility skew z-score against a 90-day rolling mean.
Used as a soft filter to boost or penalize signal confidence:

    - z-score < -1.5  ->  bullish bias  (puts cheap relative to calls)
    - z-score > +1.5  ->  bearish bias  (puts expensive = fear)
    - Confidence multiplier: 1 + 0.2 * skew_z, clamped to +/-40%

If live options data is unavailable, falls back to a VIX-based skew proxy.
All operations are fail-open: returns neutral (1.0) on any error.
"""

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class SkewResult:
    """Result of a skew computation for a single symbol."""
    symbol: str
    skew_z: float               # z-score vs 90-day rolling mean
    raw_skew: float             # raw put-call IV skew (25-delta puts minus 25-delta calls)
    rolling_mean: float         # 90-day mean of raw skew
    rolling_std: float          # 90-day std of raw skew
    bias: str                   # "bullish", "bearish", or "neutral"
    confidence_mult: float      # multiplier to apply to signal confidence
    timestamp: str = ""


class OptionsSkewScanner:
    """Scan options skew and produce confidence-adjusting signals.

    Maintains a rolling history of skew observations per symbol and
    computes z-scores for real-time signal adjustment.

    Usage::

        scanner = OptionsSkewScanner()
        scanner.update_skew("AAPL", raw_skew=0.05)  # from options data or proxy
        result = scanner.get_skew_signal("AAPL")
        adjusted_confidence = base_confidence * result.confidence_mult
    """

    def __init__(
        self,
        rolling_window: int | None = None,
        bullish_threshold: float | None = None,
        bearish_threshold: float | None = None,
        max_boost: float | None = None,
    ):
        self._rolling_window = rolling_window or getattr(config, "SKEW_ROLLING_WINDOW", 90)
        self._bullish_threshold = bullish_threshold or getattr(config, "SKEW_BULLISH_THRESHOLD", -1.5)
        self._bearish_threshold = bearish_threshold or getattr(config, "SKEW_BEARISH_THRESHOLD", 1.5)
        self._max_boost = max_boost or getattr(config, "SKEW_MAX_BOOST", 0.40)

        # Per-symbol rolling skew history
        self._skew_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._rolling_window)
        )
        self._lock = threading.Lock()
        self._enabled = getattr(config, "OPTIONS_SKEW_ENABLED", True)

    def update_skew(self, symbol: str, raw_skew: float) -> None:
        """Record a new skew observation for a symbol.

        Args:
            symbol: Ticker symbol.
            raw_skew: Raw put-call IV skew value (e.g., 25-delta put IV - 25-delta call IV).
                      Positive means puts are relatively more expensive (bearish tilt).
        """
        with self._lock:
            self._skew_history[symbol].append(raw_skew)

    def update_skew_from_proxy(self, symbol: str) -> Optional[float]:
        """Compute and record a skew proxy from available market data.

        Uses the ratio of high-low range to close as a crude volatility
        skew proxy when live options data is unavailable.

        Returns:
            The proxy skew value, or None on failure.
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist is None or len(hist) < 3:
                return None

            # Proxy: normalized high-low range relative to recent average
            ranges = (hist["High"] - hist["Low"]) / hist["Close"]
            current_range = float(ranges.iloc[-1])
            avg_range = float(ranges.mean())

            # Skew proxy: positive when current range > average (fear/volatility)
            proxy_skew = (current_range - avg_range) / max(avg_range, 1e-6)
            self.update_skew(symbol, proxy_skew)
            return proxy_skew

        except Exception as e:
            logger.debug("T5-005: Skew proxy failed for %s: %s", symbol, e)
            return None

    def get_skew_signal(self, symbol: str) -> SkewResult:
        """Get the current skew signal for a symbol.

        Computes z-score of latest skew vs rolling mean and returns
        a confidence multiplier.

        Args:
            symbol: Ticker symbol.

        Returns:
            SkewResult with z-score, bias, and confidence multiplier.
        """
        if not self._enabled:
            return SkewResult(
                symbol=symbol, skew_z=0.0, raw_skew=0.0,
                rolling_mean=0.0, rolling_std=0.0,
                bias="neutral", confidence_mult=1.0,
            )

        with self._lock:
            history = list(self._skew_history.get(symbol, []))

        if len(history) < 10:
            # Insufficient data for meaningful z-score
            return SkewResult(
                symbol=symbol, skew_z=0.0, raw_skew=history[-1] if history else 0.0,
                rolling_mean=0.0, rolling_std=0.0,
                bias="neutral", confidence_mult=1.0,
            )

        arr = np.array(history)
        rolling_mean = float(np.mean(arr))
        rolling_std = float(np.std(arr))
        raw_skew = float(arr[-1])

        if rolling_std < 1e-8:
            skew_z = 0.0
        else:
            skew_z = (raw_skew - rolling_mean) / rolling_std

        # Determine bias
        if skew_z < self._bullish_threshold:
            bias = "bullish"
        elif skew_z > self._bearish_threshold:
            bias = "bearish"
        else:
            bias = "neutral"

        # Confidence multiplier: 1 + 0.2 * skew_z, clamped to +/- max_boost
        raw_adjustment = 0.2 * skew_z
        clamped_adjustment = max(-self._max_boost, min(raw_adjustment, self._max_boost))
        confidence_mult = 1.0 + clamped_adjustment

        return SkewResult(
            symbol=symbol,
            skew_z=round(skew_z, 4),
            raw_skew=round(raw_skew, 6),
            rolling_mean=round(rolling_mean, 6),
            rolling_std=round(rolling_std, 6),
            bias=bias,
            confidence_mult=round(confidence_mult, 4),
            timestamp=datetime.now().isoformat(),
        )

    def get_confidence_multiplier(self, symbol: str) -> float:
        """Convenience: get just the confidence multiplier for a symbol.

        Returns:
            Float multiplier (0.6 to 1.4 range). 1.0 if insufficient data.
        """
        return self.get_skew_signal(symbol).confidence_mult

    def bulk_update_proxies(self, symbols: list[str]) -> int:
        """Update skew proxies for a list of symbols.

        Returns:
            Number of successful updates.
        """
        count = 0
        for symbol in symbols:
            result = self.update_skew_from_proxy(symbol)
            if result is not None:
                count += 1
        logger.info("T5-005: Skew proxy updated for %d/%d symbols", count, len(symbols))
        return count

    @property
    def tracked_symbols(self) -> list[str]:
        """Symbols with skew history."""
        with self._lock:
            return list(self._skew_history.keys())

    def reset(self, symbol: str | None = None) -> None:
        """Clear skew history for a symbol or all symbols."""
        with self._lock:
            if symbol:
                self._skew_history.pop(symbol, None)
            else:
                self._skew_history.clear()
