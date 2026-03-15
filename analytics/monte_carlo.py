"""V8: Monte Carlo tail risk engine.

Runs bootstrap simulations on historical daily returns to compute
VaR (Value at Risk) and CVaR (Expected Shortfall) at 95% and 99%.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def monte_carlo_var(daily_returns: np.ndarray, num_simulations: int = 10000,
                    horizon_days: int = 21, confidence: float = 0.95,
                    seed: int | None = None) -> dict:
    """Bootstrap Monte Carlo VaR/CVaR.

    Args:
        daily_returns: Array of historical daily returns
        num_simulations: Number of paths to simulate
        horizon_days: Forward-looking horizon in trading days
        confidence: VaR confidence level
        seed: Random seed for reproducibility

    Returns:
        Dict with var_95, var_99, cvar_95, cvar_99, median_return,
        best_case_95, worst_path
    """
    if len(daily_returns) < 5:
        return {
            "var_95": 0.0, "var_99": 0.0,
            "cvar_95": 0.0, "cvar_99": 0.0,
            "median_return": 0.0, "best_case_95": 0.0,
            "worst_path": 0.0,
        }

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    returns = np.array(daily_returns, dtype=float)

    terminal_values = np.zeros(num_simulations)
    for i in range(num_simulations):
        sampled = rng.choice(returns, size=horizon_days, replace=True)
        terminal_values[i] = np.prod(1 + sampled) - 1

    sorted_vals = np.sort(terminal_values)

    var_idx_95 = int(num_simulations * 0.05)
    var_idx_99 = int(num_simulations * 0.01)

    return {
        "var_95": float(sorted_vals[var_idx_95]),
        "var_99": float(sorted_vals[var_idx_99]),
        "cvar_95": float(np.mean(sorted_vals[:var_idx_95])) if var_idx_95 > 0 else 0.0,
        "cvar_99": float(np.mean(sorted_vals[:var_idx_99])) if var_idx_99 > 0 else 0.0,
        "median_return": float(np.median(terminal_values)),
        "best_case_95": float(sorted_vals[int(num_simulations * 0.95)]),
        "worst_path": float(sorted_vals[0]),
    }
