"""V8: Advanced performance metrics — Sortino ratio and helpers."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_sortino(returns, risk_free_rate: float = 0.045,
                    periods_per_year: int = 252) -> float:
    """Compute annualized Sortino ratio.

    Unlike Sharpe, Sortino only penalizes downside volatility, making it
    more appropriate for asymmetric return strategies.

    Args:
        returns: Array-like of period returns (daily, per-trade, etc.)
        risk_free_rate: Annual risk-free rate (default 4.5%)
        periods_per_year: Annualization factor (252 for daily)

    Returns:
        Annualized Sortino ratio (float)
    """
    returns = np.array(returns, dtype=float)

    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]

    if len(downside) == 0:
        # No downside — infinite Sortino, cap at a large number
        return 10.0

    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std < 1e-8:
        return 10.0

    sortino = (np.mean(excess) / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def compute_sharpe(returns, risk_free_rate: float = 0.045,
                   periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio.

    Provided for convenience alongside Sortino.
    """
    returns = np.array(returns, dtype=float)

    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate / periods_per_year
    std = np.std(returns, ddof=1)

    if std < 1e-8:
        return 0.0

    return float((np.mean(excess) / std) * np.sqrt(periods_per_year))
