"""Analytics — performance metrics, statistical tools, and dashboard computations."""

from analytics.performance import (
    compute_analytics,
    sharpe_ratio,
    sortino_ratio,
    profit_factor,
    max_drawdown,
    win_rate,
    benchmark_comparison,
    strategy_attribution,
)

__all__ = [
    "compute_analytics",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "max_drawdown",
    "win_rate",
    "benchmark_comparison",
    "strategy_attribution",
]
