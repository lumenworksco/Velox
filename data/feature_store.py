"""DATA-002: Feature Store — online/offline feature computation and caching.

Provides a unified feature computation layer for all strategies.
Computes technical indicators (RSI, ATR, Bollinger, OBV, MFI, etc.),
volatility measures, return horizons, and price-relative-to-MA features.

Thread-safe with proper locking for concurrent strategy access.
"""

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature metadata registry
# ---------------------------------------------------------------------------

@dataclass
class FeatureMeta:
    """Metadata for a registered feature."""
    name: str
    version: int = 1
    description: str = ""
    staleness_threshold_sec: float = 300.0  # 5 minutes default
    dependencies: list[str] = field(default_factory=list)
    compute_fn: Optional[Callable] = None  # set during registration


# ---------------------------------------------------------------------------
# Pure computation helpers (stateless, operate on numpy/pandas)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> float:
    """Compute RSI for the most recent bar."""
    if len(close) < period + 1:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.iloc[-period:].mean()
    avg_loss = loss.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    """Average True Range for most recent bar."""
    if len(close) < period + 1:
        return np.nan
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[float, float, float]:
    """Return (upper, middle, lower) Bollinger Bands for the latest bar."""
    if len(close) < period:
        return np.nan, np.nan, np.nan
    window = close.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std(ddof=1))
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _bollinger_pct_b(close: pd.Series, period: int = 20, num_std: float = 2.0) -> float:
    """%B = (price - lower) / (upper - lower)."""
    upper, _mid, lower = _bollinger(close, period, num_std)
    if np.isnan(upper) or upper == lower:
        return np.nan
    return float((close.iloc[-1] - lower) / (upper - lower))


def _bollinger_bandwidth(close: pd.Series, period: int = 20, num_std: float = 2.0) -> float:
    """Bandwidth = (upper - lower) / middle."""
    upper, mid, lower = _bollinger(close, period, num_std)
    if np.isnan(upper) or mid == 0:
        return np.nan
    return float((upper - lower) / mid)


def _vwap_deviation(close: pd.Series, volume: pd.Series, vwap: Optional[pd.Series]) -> float:
    """Deviation of latest close from VWAP, in percent."""
    if vwap is not None and len(vwap) > 0 and not np.isnan(vwap.iloc[-1]):
        vwap_val = vwap.iloc[-1]
    else:
        # Compute cumulative VWAP from available data
        if len(close) == 0 or len(volume) == 0:
            return np.nan
        cum_vol = volume.cumsum()
        cum_vp = (close * volume).cumsum()
        if cum_vol.iloc[-1] == 0:
            return np.nan
        vwap_val = float(cum_vp.iloc[-1] / cum_vol.iloc[-1])
    if vwap_val == 0:
        return np.nan
    return float((close.iloc[-1] - vwap_val) / vwap_val)


def _volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    """Current bar volume / average volume over lookback."""
    if len(volume) < lookback + 1:
        return np.nan
    avg = volume.iloc[-(lookback + 1):-1].mean()
    if avg == 0:
        return np.nan
    return float(volume.iloc[-1] / avg)


def _obv(close: pd.Series, volume: pd.Series) -> float:
    """On-Balance Volume (cumulative, returns latest value)."""
    if len(close) < 2:
        return np.nan
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return float(obv.iloc[-1])


def _money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series, period: int = 14) -> float:
    """Money Flow Index (volume-weighted RSI)."""
    if len(close) < period + 1:
        return np.nan
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    tp_diff = typical_price.diff()

    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    pos_sum = positive_flow.iloc[-period:].sum()
    neg_sum = negative_flow.iloc[-period:].sum()

    if neg_sum == 0:
        return 100.0
    ratio = pos_sum / neg_sum
    return float(100.0 - 100.0 / (1.0 + ratio))


def _returns(close: pd.Series, horizon: int) -> float:
    """Log return over *horizon* bars ending at the latest bar."""
    if len(close) <= horizon:
        return np.nan
    ret = float(np.log(close.iloc[-1] / close.iloc[-1 - horizon]))
    # MED-018: Sanity check — flag extreme returns (|return| > 50% per bar)
    if abs(ret) > 0.50:
        logger.warning(
            "Extreme return detected: %.4f over %d bars (close %.2f -> %.2f) — possible bad data",
            ret, horizon, close.iloc[-1 - horizon], close.iloc[-1],
        )
    return ret


def _realized_volatility(close: pd.Series, window: int = 20) -> float:
    """Annualised realised volatility (close-to-close)."""
    if len(close) < window + 1:
        return np.nan
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return np.nan
    return float(log_ret.iloc[-window:].std() * np.sqrt(252))


def _parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> float:
    """Parkinson volatility estimator (uses high-low range)."""
    if len(high) < window:
        return np.nan
    hl_log = np.log(high / low)
    parkinson_var = (1.0 / (4.0 * np.log(2))) * (hl_log ** 2)
    return float(np.sqrt(parkinson_var.iloc[-window:].mean() * 252))


def _garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
                              close: pd.Series, window: int = 20) -> float:
    """Garman-Klass volatility estimator (uses OHLC)."""
    if len(close) < window:
        return np.nan
    hl = np.log(high / low)
    co = np.log(close / open_)
    gk_var = 0.5 * hl ** 2 - (2.0 * np.log(2) - 1.0) * co ** 2
    return float(np.sqrt(gk_var.iloc[-window:].mean() * 252))


def _price_relative_to_ma(close: pd.Series, period: int) -> float:
    """(close - MA) / MA as a fraction."""
    if len(close) < period:
        return np.nan
    ma = close.iloc[-period:].mean()
    if ma == 0:
        return np.nan
    return float((close.iloc[-1] - ma) / ma)


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------

class FeatureStore:
    """Centralised feature computation and caching.

    Supports:
    - **Online store**: in-memory dict, refreshed every scan cycle.
    - **Offline store**: optional file/DB persistence for backtesting.
    - Thread-safe reads and writes.
    """

    # Default feature definitions — name -> (compute_fn, version, description, staleness_sec)
    _BUILTIN_FEATURES: list[tuple[str, int, str, float]] = [
        ("rsi_7", 1, "RSI with 7-bar lookback", 300),
        ("rsi_14", 1, "RSI with 14-bar lookback", 300),
        ("vwap_deviation", 1, "Price deviation from VWAP (%)", 120),
        ("atr_14", 1, "Average True Range (14-bar)", 300),
        ("bollinger_pct_b", 1, "Bollinger %B (20,2)", 300),
        ("bollinger_bandwidth", 1, "Bollinger Band width (20,2)", 300),
        ("volume_ratio", 1, "Current volume / 20-bar avg volume", 120),
        ("obv", 1, "On-Balance Volume (cumulative)", 300),
        ("money_flow_index", 1, "Money Flow Index (14-bar)", 300),
        ("return_1", 1, "1-bar log return", 120),
        ("return_5", 1, "5-bar log return", 120),
        ("return_20", 1, "20-bar log return", 300),
        ("return_60", 1, "60-bar log return", 600),
        ("volatility_realized", 1, "Realized vol (20-bar, annualised)", 300),
        ("volatility_parkinson", 1, "Parkinson vol (20-bar, annualised)", 300),
        ("volatility_garman_klass", 1, "Garman-Klass vol (20-bar, annualised)", 300),
        ("price_rel_ma_5", 1, "Price relative to 5-bar MA", 120),
        ("price_rel_ma_10", 1, "Price relative to 10-bar MA", 120),
        ("price_rel_ma_20", 1, "Price relative to 20-bar MA", 300),
        ("price_rel_ma_50", 1, "Price relative to 50-bar MA", 600),
        ("price_rel_ma_200", 1, "Price relative to 200-bar MA", 600),
    ]

    def __init__(self, offline_path: Optional[str] = None):
        self._lock = threading.Lock()

        # Online store: symbol -> {feature_name: value}
        self._online: Dict[str, Dict[str, float]] = {}
        # Timestamps: symbol -> {feature_name: epoch}
        self._timestamps: Dict[str, Dict[str, float]] = {}

        # Feature registry: name -> FeatureMeta
        self._registry: Dict[str, FeatureMeta] = {}

        # Offline path (parquet/sqlite for backtest snapshots)
        self._offline_path = offline_path

        # Stats
        self._compute_count = 0
        self._error_count = 0

        # MED-042: Track bar counts for incremental computation
        self._last_bar_count: Dict[str, int] = {}

        # PROD-003: Track last computed index per symbol for true incremental computation.
        # Maps symbol -> index (row count) of the last bar that was computed.
        # Only recomputes features when new bars arrive beyond this index.
        self._last_computed: Dict[str, int] = {}

        # Register built-in features
        self._register_builtins()

        logger.info("FeatureStore initialised with %d built-in features", len(self._registry))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_builtins(self):
        """Register all built-in feature definitions."""
        for name, version, desc, staleness in self._BUILTIN_FEATURES:
            meta = FeatureMeta(
                name=name, version=version,
                description=desc, staleness_threshold_sec=staleness,
            )
            self._registry[name] = meta

    def register_feature(self, name: str, compute_fn: Callable,
                         version: int = 1, description: str = "",
                         staleness_threshold_sec: float = 300.0):
        """Register a custom feature with its computation function.

        Args:
            name: Unique feature name.
            compute_fn: Callable(bars: pd.DataFrame) -> float.
            version: Feature version (bump on breaking changes).
            description: Human-readable description.
            staleness_threshold_sec: Max age before recompute.
        """
        meta = FeatureMeta(
            name=name, version=version, description=description,
            staleness_threshold_sec=staleness_threshold_sec,
            compute_fn=compute_fn,
        )
        with self._lock:
            self._registry[name] = meta
        logger.debug("Registered custom feature: %s (v%d)", name, version)

    @property
    def feature_names(self) -> List[str]:
        """Return list of all registered feature names."""
        with self._lock:
            return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_builtin(self, name: str, bars: pd.DataFrame) -> float:
        """Compute a single built-in feature from bars DataFrame.

        Expects bars with columns: open, high, low, close, volume
        and optionally 'vwap'.
        """
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        open_ = bars["open"]
        volume = bars["volume"]
        # CRIT-007: Use column check instead of .get() for DataFrame safety
        vwap = bars["vwap"] if "vwap" in bars.columns else None

        try:
            if name == "rsi_7":
                return _rsi(close, 7)
            elif name == "rsi_14":
                return _rsi(close, 14)
            elif name == "vwap_deviation":
                return _vwap_deviation(close, volume, vwap)
            elif name == "atr_14":
                return _atr(high, low, close, 14)
            elif name == "bollinger_pct_b":
                return _bollinger_pct_b(close)
            elif name == "bollinger_bandwidth":
                return _bollinger_bandwidth(close)
            elif name == "volume_ratio":
                return _volume_ratio(volume)
            elif name == "obv":
                return _obv(close, volume)
            elif name == "money_flow_index":
                return _money_flow_index(high, low, close, volume)
            elif name == "return_1":
                return _returns(close, 1)
            elif name == "return_5":
                return _returns(close, 5)
            elif name == "return_20":
                return _returns(close, 20)
            elif name == "return_60":
                return _returns(close, 60)
            elif name == "volatility_realized":
                return _realized_volatility(close)
            elif name == "volatility_parkinson":
                return _parkinson_volatility(high, low)
            elif name == "volatility_garman_klass":
                return _garman_klass_volatility(open_, high, low, close)
            elif name == "price_rel_ma_5":
                return _price_relative_to_ma(close, 5)
            elif name == "price_rel_ma_10":
                return _price_relative_to_ma(close, 10)
            elif name == "price_rel_ma_20":
                return _price_relative_to_ma(close, 20)
            elif name == "price_rel_ma_50":
                return _price_relative_to_ma(close, 50)
            elif name == "price_rel_ma_200":
                return _price_relative_to_ma(close, 200)
            else:
                return np.nan
        except Exception as e:
            logger.debug("Error computing feature %s: %s", name, e)
            return np.nan

    def compute_features(self, symbol: str, bars: pd.DataFrame) -> Dict[str, float]:
        """Compute all registered features for a symbol.

        Args:
            symbol: Ticker symbol.
            bars: OHLCV DataFrame (must have open, high, low, close, volume columns).

        Returns:
            Dict mapping feature name to computed value.
        """
        if bars is None or bars.empty:
            logger.warning("FeatureStore: empty bars for %s, returning empty features", symbol)
            return {}

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(bars.columns)
        if missing:
            logger.error("FeatureStore: bars for %s missing columns: %s", symbol, missing)
            return {}

        now = _time.time()
        features: Dict[str, float] = {}

        for name, meta in self._registry.items():
            try:
                if meta.compute_fn is not None:
                    # Custom feature
                    value = float(meta.compute_fn(bars))
                else:
                    # Built-in feature
                    value = self._compute_builtin(name, bars)
                features[name] = value
            except Exception as e:
                self._error_count += 1
                logger.debug("FeatureStore: error computing %s for %s: %s", name, symbol, e)
                features[name] = np.nan

        self._compute_count += 1

        # Update online store
        with self._lock:
            self._online[symbol] = features
            self._timestamps[symbol] = {name: now for name in features}
            self._last_bar_count[symbol] = len(bars)

        return features

    def compute_features_incremental(self, symbol: str, bars: pd.DataFrame) -> Dict[str, float]:
        """PROD-003 / MED-042: Incremental feature computation — skip if no new bars.

        Tracks the last computed index per symbol via `_last_computed` dict.
        Only recomputes features when new bars arrive beyond the previously
        computed index. This avoids redundant full recomputation when bars
        haven't changed.

        For most technical indicators the full lookback window is needed,
        so we still pass the full bars DataFrame to compute functions.
        The savings come from skipping the entire computation when no new
        data has arrived.

        Args:
            symbol: Ticker symbol.
            bars: OHLCV DataFrame.

        Returns:
            Dict mapping feature name to computed value (cached or fresh).
        """
        if bars is None or bars.empty:
            return self.get_all_features(symbol)

        current_count = len(bars)
        with self._lock:
            prev_computed = self._last_computed.get(symbol, 0)
            cached = self._online.get(symbol)

        # PROD-003: If no new bars since last computation and we have cache, return it
        if current_count <= prev_computed and cached:
            logger.debug(
                "PROD-003: Skipping recompute for %s (bars=%d, last_computed=%d)",
                symbol, current_count, prev_computed,
            )
            return dict(cached)

        # New bars arrived — full recompute (indicators need full lookback)
        features = self.compute_features(symbol, bars)

        # Update the last computed index
        with self._lock:
            self._last_computed[symbol] = current_count

        return features

    def get_feature(self, symbol: str, feature_name: str) -> float:
        """Get a single cached feature value.

        Args:
            symbol: Ticker symbol.
            feature_name: Registered feature name.

        Returns:
            Cached feature value, or NaN if unavailable/stale.
        """
        with self._lock:
            sym_features = self._online.get(symbol)
            if sym_features is None:
                return np.nan

            value = sym_features.get(feature_name, np.nan)

            # Check staleness
            meta = self._registry.get(feature_name)
            ts = self._timestamps.get(symbol, {}).get(feature_name, 0)
            if meta and (_time.time() - ts) > meta.staleness_threshold_sec:
                logger.debug("FeatureStore: stale feature %s for %s (age=%.0fs)",
                             feature_name, symbol, _time.time() - ts)
                return np.nan

            return value

    def compute_all(self, bars_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Compute features for multiple symbols at once.

        Args:
            bars_dict: Mapping of symbol -> OHLCV DataFrame.

        Returns:
            Mapping of symbol -> {feature_name: value}.
        """
        results: Dict[str, Dict[str, float]] = {}
        for symbol, bars in bars_dict.items():
            try:
                results[symbol] = self.compute_features(symbol, bars)
            except Exception as e:
                logger.error("FeatureStore: failed to compute features for %s: %s", symbol, e)
                results[symbol] = {}
        return results

    def get_all_features(self, symbol: str) -> Dict[str, float]:
        """Return all cached features for a symbol (no recompute)."""
        with self._lock:
            return dict(self._online.get(symbol, {}))

    # ------------------------------------------------------------------
    # Offline persistence
    # ------------------------------------------------------------------

    def save_snapshot(self, timestamp: Optional[str] = None):
        """Persist current online store to offline storage (parquet file).

        Args:
            timestamp: ISO timestamp label. Defaults to current time.
        """
        if not self._offline_path:
            logger.debug("FeatureStore: no offline_path configured, skipping snapshot")
            return

        with self._lock:
            if not self._online:
                return
            data = self._online.copy()

        try:
            import os
            os.makedirs(os.path.dirname(self._offline_path) or ".", exist_ok=True)

            df = pd.DataFrame.from_dict(data, orient="index")
            if timestamp:
                df["_snapshot_ts"] = timestamp
            else:
                from datetime import datetime, timezone
                df["_snapshot_ts"] = datetime.now(timezone.utc).isoformat()
            df.index.name = "symbol"

            # Append to existing file or create new
            if os.path.exists(self._offline_path):
                existing = pd.read_parquet(self._offline_path)
                df = pd.concat([existing, df.reset_index()], ignore_index=True)
                df.to_parquet(self._offline_path, index=False)
            else:
                df.reset_index().to_parquet(self._offline_path, index=False)

            logger.info("FeatureStore: saved snapshot (%d symbols) to %s",
                        len(data), self._offline_path)
        except Exception as e:
            logger.error("FeatureStore: failed to save snapshot: %s", e)

    def load_snapshot(self, timestamp: str) -> Dict[str, Dict[str, float]]:
        """Load a historical feature snapshot from offline storage.

        Args:
            timestamp: ISO timestamp label to load.

        Returns:
            Dict of symbol -> features, or empty dict on failure.
        """
        if not self._offline_path:
            return {}
        try:
            import os
            if not os.path.exists(self._offline_path):
                return {}

            df = pd.read_parquet(self._offline_path)
            snapshot = df[df["_snapshot_ts"] == timestamp]
            if snapshot.empty:
                return {}

            feature_cols = [c for c in snapshot.columns if c not in ("symbol", "_snapshot_ts")]
            result = {}
            for _, row in snapshot.iterrows():
                sym = row["symbol"]
                result[sym] = {col: float(row[col]) for col in feature_cols
                               if pd.notna(row[col])}
            return result
        except Exception as e:
            logger.error("FeatureStore: failed to load snapshot: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Invalidation & stats
    # ------------------------------------------------------------------

    def invalidate(self, symbol: Optional[str] = None):
        """Clear cached features for a symbol, or all symbols if None."""
        with self._lock:
            if symbol:
                self._online.pop(symbol, None)
                self._timestamps.pop(symbol, None)
                self._last_computed.pop(symbol, None)
            else:
                self._online.clear()
                self._timestamps.clear()
                self._last_computed.clear()

    def stats(self) -> dict:
        """Return feature store statistics."""
        with self._lock:
            return {
                "registered_features": len(self._registry),
                "symbols_cached": len(self._online),
                "total_computes": self._compute_count,
                "total_errors": self._error_count,
                "offline_path": self._offline_path,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_feature_store: Optional[FeatureStore] = None
_init_lock = threading.Lock()


def get_feature_store(offline_path: Optional[str] = None) -> FeatureStore:
    """Get or create the global FeatureStore singleton."""
    global _feature_store
    with _init_lock:
        if _feature_store is None:
            _feature_store = FeatureStore(offline_path=offline_path)
    return _feature_store
