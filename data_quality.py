"""V8: Data quality validation checks.

Pre-trade data quality validation to catch halted symbols, stale quotes,
anomalous moves, and bad data before it corrupts OU fitting or triggers
false signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class DataQualityResult:
    """Result of data quality checks."""
    is_clean: bool
    issues: list[str] = field(default_factory=list)

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_clean = False


def check_bar_quality(bars: pd.DataFrame, symbol: str,
                      now: datetime | None = None,
                      max_staleness_sec: int = 300,
                      max_single_move: float = 0.15,
                      max_zero_vol_pct: float = 0.10,
                      min_bars: int = 50) -> DataQualityResult:
    """Run all quality checks on a bar DataFrame.

    Args:
        bars: DataFrame with OHLCV data (indexed by datetime)
        symbol: Ticker symbol (for logging)
        now: Current time for staleness check
        max_staleness_sec: Max age of latest bar in seconds
        max_single_move: Max allowed single-bar percentage move
        max_zero_vol_pct: Max fraction of zero-volume bars
        min_bars: Minimum number of bars required

    Returns:
        DataQualityResult with is_clean flag and list of issues
    """
    result = DataQualityResult(is_clean=True)

    if bars is None or bars.empty:
        result.add_issue(f"{symbol}: No bar data available")
        return result

    # 1. Minimum bars check
    if len(bars) < min_bars:
        result.add_issue(f"{symbol}: Only {len(bars)} bars (need {min_bars})")

    # 2. Stale data check
    if now is not None and hasattr(bars.index, 'tz'):
        try:
            last_bar_time = bars.index[-1]
            if hasattr(last_bar_time, 'timestamp'):
                age = (now - last_bar_time).total_seconds()
                if age > max_staleness_sec:
                    result.add_issue(f"{symbol}: Stale data, last bar {age:.0f}s old")
        except Exception:
            pass

    # 3. Price anomaly check (single-bar moves > threshold)
    if "close" in bars.columns and len(bars) > 1:
        returns = bars["close"].pct_change().dropna()
        if len(returns) > 0:
            max_move = returns.abs().max()
            if max_move > max_single_move:
                result.add_issue(f"{symbol}: Price anomaly, {max_move:.1%} single-bar move")

    # 4. Volume anomaly check (zero volume = possible halt)
    if "volume" in bars.columns:
        zero_vol_pct = (bars["volume"] == 0).mean()
        if zero_vol_pct > max_zero_vol_pct:
            result.add_issue(f"{symbol}: {zero_vol_pct:.0%} zero-volume bars (possible halt)")

    # 5. OHLC consistency check
    if all(c in bars.columns for c in ["open", "high", "low", "close"]):
        invalid_high = (bars["high"] < bars[["open", "close"]].max(axis=1)).any()
        invalid_low = (bars["low"] > bars[["open", "close"]].min(axis=1)).any()
        if invalid_high or invalid_low:
            result.add_issue(f"{symbol}: OHLC inconsistency detected")

    # 6. Duplicate timestamps
    if bars.index.duplicated().any():
        dup_count = bars.index.duplicated().sum()
        result.add_issue(f"{symbol}: {dup_count} duplicate timestamps")

    return result
