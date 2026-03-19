"""Correlation filter — prevent opening positions too correlated with existing ones."""

import logging
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Cache: correlation matrix refreshed once per day
_corr_lock = threading.Lock()
_corr_matrix: pd.DataFrame | None = None
_cache_date: str = ""
_returns_cache: dict[str, pd.Series] = {}


def load_correlation_cache(symbols: list[str]):
    """Preload 30-day daily returns for all symbols. Call once per day."""
    global _corr_matrix, _cache_date, _returns_cache

    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    with _corr_lock:
        if _cache_date == today and _corr_matrix is not None:
            return

    logger.info(f"Loading correlation data for {len(symbols)} symbols...")
    _returns_cache.clear()

    try:
        import yfinance as yf

        end = datetime.now(config.ET)
        start = end - timedelta(days=45)  # Extra buffer for weekends/holidays
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        # Suppress yfinance's noisy logging during bulk download
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        # Download in batches to avoid DNS thread exhaustion
        BATCH_SIZE = 40
        all_closes = []

        try:
            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i + BATCH_SIZE]
                data = yf.download(
                    batch,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=True,
                )

                if data.empty:
                    continue

                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"]
                else:
                    closes = data

                all_closes.append(closes)
        finally:
            yf_logger.setLevel(prev_level)

        if not all_closes:
            logger.warning("No correlation data downloaded")
            _corr_matrix = pd.DataFrame()
            _cache_date = today
            return

        # Merge all batches on date index
        combined = pd.concat(all_closes, axis=1)

        # Drop columns (symbols) with mostly missing data (< 50% present)
        min_rows = len(combined) * 0.5
        combined = combined.dropna(axis=1, thresh=int(min_rows))

        if combined.empty or len(combined.columns) < 2:
            logger.warning("Not enough symbols with data for correlation")
            _corr_matrix = pd.DataFrame()
            _cache_date = today
            return

        # Calculate daily returns — drop only leading NaN per column
        returns = combined.pct_change()

        # Keep last 30 trading days
        returns = returns.tail(30)

        if len(returns) < 10:
            logger.warning("Not enough data for correlation calculation")
            _corr_matrix = pd.DataFrame()
            _cache_date = today
            return

        # Use min_periods to tolerate a few NaN per pair
        _corr_matrix = returns.corr(min_periods=15)

        # Cache individual return series for later use
        for col in returns.columns:
            _returns_cache[col] = returns[col]

        _cache_date = today
        logger.info(f"Correlation matrix loaded: {len(_corr_matrix.columns)} symbols")

    except Exception as e:
        logger.error(f"Failed to load correlation data: {e}")
        _corr_matrix = pd.DataFrame()
        _cache_date = today


def is_too_correlated(new_symbol: str, open_symbols: list[str],
                      threshold: float = None) -> bool:
    """Check if new_symbol is too correlated with any open position.

    Returns True if any correlation > threshold (should skip trade).
    """
    if threshold is None:
        threshold = config.CORRELATION_THRESHOLD

    if _corr_matrix is None or _corr_matrix.empty:
        return False  # Can't check, allow trade

    if new_symbol not in _corr_matrix.columns:
        return False  # No data, allow trade

    for open_sym in open_symbols:
        if open_sym not in _corr_matrix.columns:
            continue
        corr = _corr_matrix.loc[new_symbol, open_sym]
        if abs(corr) > threshold:
            logger.info(
                f"Correlation filter: {new_symbol} correlated {corr:.2f} "
                f"with open position {open_sym} (threshold={threshold})"
            )
            return True

    return False


def get_correlation(symbol_a: str, symbol_b: str) -> float | None:
    """Get correlation between two symbols. Returns None if unavailable."""
    if _corr_matrix is None or _corr_matrix.empty:
        return None
    if symbol_a not in _corr_matrix.columns or symbol_b not in _corr_matrix.columns:
        return None
    return float(_corr_matrix.loc[symbol_a, symbol_b])
