"""Tests for V8 Monte Carlo tail risk engine."""

import pytest
import numpy as np


class TestMonteCarloVaR:

    def test_basic_simulation(self):
        from analytics.monte_carlo import monte_carlo_var
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 60)
        result = monte_carlo_var(returns, num_simulations=1000, seed=42)

        assert "var_95" in result
        assert "var_99" in result
        assert "cvar_95" in result
        assert "cvar_99" in result
        assert result["var_99"] <= result["var_95"]  # 99% VaR worse than 95%

    def test_positive_returns_positive_median(self):
        from analytics.monte_carlo import monte_carlo_var
        returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01] * 12)
        result = monte_carlo_var(returns, num_simulations=1000, seed=42)
        assert result["median_return"] > 0

    def test_negative_returns_negative_median(self):
        from analytics.monte_carlo import monte_carlo_var
        returns = np.array([-0.01, -0.02, -0.015, -0.005, -0.01] * 12)
        result = monte_carlo_var(returns, num_simulations=1000, seed=42)
        assert result["median_return"] < 0

    def test_insufficient_data(self):
        from analytics.monte_carlo import monte_carlo_var
        result = monte_carlo_var(np.array([0.01, 0.02]), seed=42)
        assert result["var_95"] == 0.0

    def test_cvar_worse_than_var(self):
        from analytics.monte_carlo import monte_carlo_var
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 60)
        result = monte_carlo_var(returns, num_simulations=5000, seed=42)
        assert result["cvar_95"] <= result["var_95"]
        assert result["cvar_99"] <= result["var_99"]

    def test_worst_path(self):
        from analytics.monte_carlo import monte_carlo_var
        returns = np.random.normal(0.0, 0.01, 60)
        result = monte_carlo_var(returns, num_simulations=1000, seed=42)
        assert result["worst_path"] <= result["var_99"]

    def test_best_case(self):
        from analytics.monte_carlo import monte_carlo_var
        returns = np.random.normal(0.001, 0.01, 60)
        result = monte_carlo_var(returns, num_simulations=1000, seed=42)
        assert result["best_case_95"] > result["median_return"]

    def test_reproducibility(self):
        from analytics.monte_carlo import monte_carlo_var
        returns = np.random.normal(0.001, 0.01, 60)
        r1 = monte_carlo_var(returns, seed=123)
        r2 = monte_carlo_var(returns, seed=123)
        assert r1["var_95"] == r2["var_95"]

    def test_monte_carlo_db_save(self, in_memory_db):
        import database
        database.save_monte_carlo_result(
            date="2026-03-13", var_95=-0.03, var_99=-0.05,
            cvar_95=-0.04, cvar_99=-0.06,
            horizon_days=21, simulations=10000,
        )
        rows = in_memory_db.execute("SELECT * FROM monte_carlo_results").fetchall()
        assert len(rows) == 1
        assert rows[0]["var_99"] == -0.05
