"""Tests for V8 Sortino ratio and performance metrics."""

import pytest
import numpy as np


class TestComputeSortino:

    def test_positive_returns_high_sortino(self):
        from analytics.metrics import compute_sortino
        # All positive returns — high Sortino
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01] * 20)
        sortino = compute_sortino(returns)
        assert sortino > 2.0

    def test_negative_returns_negative_sortino(self):
        from analytics.metrics import compute_sortino
        # All negative returns
        returns = np.array([-0.01, -0.02, -0.015, -0.005, -0.01] * 20)
        sortino = compute_sortino(returns)
        assert sortino < 0

    def test_mixed_returns(self):
        from analytics.metrics import compute_sortino
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        sortino = compute_sortino(returns)
        assert isinstance(sortino, float)

    def test_no_downside_capped(self):
        from analytics.metrics import compute_sortino
        # All positive, no downside deviation
        returns = np.array([0.01, 0.02, 0.03])
        sortino = compute_sortino(returns)
        assert sortino == 10.0

    def test_insufficient_data(self):
        from analytics.metrics import compute_sortino
        sortino = compute_sortino([0.01])
        assert sortino == 0.0

    def test_empty_returns(self):
        from analytics.metrics import compute_sortino
        sortino = compute_sortino([])
        assert sortino == 0.0

    def test_custom_risk_free(self):
        from analytics.metrics import compute_sortino
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        s1 = compute_sortino(returns, risk_free_rate=0.0)
        s2 = compute_sortino(returns, risk_free_rate=0.10)
        # Higher risk-free rate → lower Sortino
        assert s1 > s2

    def test_asymmetric_strategy(self):
        from analytics.metrics import compute_sortino, compute_sharpe
        # Strategy with occasional big ups, small frequent downs
        # (like mean reversion with tight stops)
        np.random.seed(42)
        returns = np.concatenate([
            np.array([0.05, 0.04, 0.03, 0.06, 0.05]),  # Big wins
            np.array([-0.01, -0.005, -0.008, -0.01, -0.005] * 4),  # Small losses
        ])
        sortino = compute_sortino(returns)
        sharpe = compute_sharpe(returns)
        # Sortino should be higher than Sharpe for this kind of distribution
        # because upside vol isn't penalized
        assert sortino > sharpe


class TestComputeSharpe:

    def test_positive_sharpe(self):
        from analytics.metrics import compute_sharpe
        returns = np.array([0.01, 0.02, 0.005, 0.01, 0.015] * 20)
        sharpe = compute_sharpe(returns)
        assert sharpe > 0

    def test_zero_std(self):
        from analytics.metrics import compute_sharpe
        returns = np.array([0.001, 0.001, 0.001])
        sharpe = compute_sharpe(returns)
        assert sharpe == 0.0  # std is 0

    def test_insufficient_data(self):
        from analytics.metrics import compute_sharpe
        assert compute_sharpe([]) == 0.0
        assert compute_sharpe([0.01]) == 0.0
