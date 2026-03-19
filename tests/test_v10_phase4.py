"""Tests for V10 Phase 4 — VaR monitor, correlation limiter, dynamic universe."""

import pytest
import numpy as np

from risk.var_monitor import VaRMonitor, VaRResult
from risk.correlation_limiter import CorrelationLimiter, ConcentrationResult
from strategies.dynamic_universe import DynamicUniverse, UniverseSelection


class TestVaRMonitor:
    """Test Value-at-Risk monitor."""

    def test_insufficient_data(self):
        monitor = VaRMonitor()
        result = monitor.update([0.01, -0.01], 100000)
        assert result.var_95 == 0.0
        assert result.sample_size == 2

    def test_parametric_var(self):
        monitor = VaRMonitor()
        # Generate 25 days of returns (not enough for historical, uses parametric)
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.01, 25))
        result = monitor.update(returns, 100000)
        assert result.method == "parametric"
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95

    def test_historical_var(self):
        monitor = VaRMonitor()
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.015, 60))
        result = monitor.update(returns, 100000)
        assert result.method == "historical"
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95

    def test_monte_carlo_var(self):
        monitor = VaRMonitor(mc_simulations=1000)
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.01, 30))
        monitor.update(returns, 100000)
        result = monitor.monte_carlo_var(horizon_days=5)
        assert result.method == "monte_carlo"
        assert result.var_95 > 0

    def test_risk_budget(self):
        monitor = VaRMonitor(max_var_pct=0.02)
        np.random.seed(42)
        # Low volatility = lots of risk budget
        returns = list(np.random.normal(0.001, 0.005, 60))
        monitor.update(returns, 100000)
        assert monitor.risk_budget_remaining > 0.5

    def test_risk_budget_exhausted(self):
        monitor = VaRMonitor(max_var_pct=0.01)
        np.random.seed(42)
        # High volatility = risk budget exhausted
        returns = list(np.random.normal(-0.005, 0.03, 60))
        monitor.update(returns, 100000)
        assert monitor.risk_budget_remaining < 0.5
        assert monitor.size_multiplier < 1.0

    def test_size_multiplier_bounds(self):
        monitor = VaRMonitor()
        assert 0.0 <= monitor.size_multiplier <= 1.0

    def test_lookback_window(self):
        monitor = VaRMonitor(lookback_days=30)
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 100))
        monitor.update(returns, 100000)
        # Should only use last 30 days
        assert monitor.result.sample_size == 30

    def test_status_dict(self):
        monitor = VaRMonitor()
        status = monitor.status
        assert "var_95" in status
        assert "risk_budget_remaining" in status
        assert "max_var_pct" in status


class TestCorrelationLimiter:
    """Test correlation-based position limiter."""

    def test_empty_portfolio_allows_entry(self):
        limiter = CorrelationLimiter()
        result = limiter.check_new_position("AAPL", [])
        assert not result.too_concentrated

    def test_high_correlation_blocks(self):
        limiter = CorrelationLimiter(max_pairwise_corr=0.70)
        # Set a high correlation
        limiter.update_correlation("AAPL", "MSFT", 0.85)
        result = limiter.check_new_position("MSFT", ["AAPL"])
        assert result.too_concentrated
        assert "high_corr" in result.reason

    def test_low_correlation_allows(self):
        limiter = CorrelationLimiter(max_pairwise_corr=0.70)
        limiter.update_correlation("AAPL", "XOM", 0.15)
        # Set different sectors to avoid sector concentration trigger
        limiter.set_sector_map({"AAPL": "tech", "XOM": "energy"})
        result = limiter.check_new_position("XOM", ["AAPL"])
        assert not result.too_concentrated

    def test_effective_bets(self):
        limiter = CorrelationLimiter(min_effective_bets=2.0)
        # All highly correlated = low effective bets
        limiter.update_correlation("AAPL", "MSFT", 0.65)
        limiter.update_correlation("AAPL", "GOOGL", 0.60)
        limiter.update_correlation("MSFT", "GOOGL", 0.65)
        result = limiter.check_new_position("GOOGL", ["AAPL", "MSFT"])
        # Should have effective bets < 3 since they're correlated
        assert result.effective_bets < 3.0

    def test_sector_concentration(self):
        limiter = CorrelationLimiter(max_sector_weight=0.50)
        limiter.set_sector_map({
            "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech",
            "XOM": "energy",
        })
        # Portfolio already has 2 tech stocks, adding a 3rd makes tech > 50%
        result = limiter.check_new_position(
            "GOOGL", ["AAPL", "MSFT", "XOM"],
            correlations={},  # Low correlations
        )
        # 3 tech out of 4 = 75% > 50% threshold
        assert result.too_concentrated
        assert "sector_tech" in result.reason

    def test_sector_diversified_allows(self):
        limiter = CorrelationLimiter(max_sector_weight=0.50)
        limiter.set_sector_map({
            "AAPL": "tech", "XOM": "energy", "JPM": "finance", "PFE": "health",
        })
        result = limiter.check_new_position(
            "PFE", ["AAPL", "XOM", "JPM"],
            correlations={},
        )
        # 25% each sector = fine
        assert not result.too_concentrated

    def test_external_correlations_override(self):
        limiter = CorrelationLimiter(max_pairwise_corr=0.70)
        # Cache says low correlation
        limiter.update_correlation("AAPL", "MSFT", 0.30)
        # But external data says high
        external = {("AAPL", "MSFT"): 0.90}
        result = limiter.check_new_position("MSFT", ["AAPL"], correlations=external)
        assert result.too_concentrated

    def test_status_dict(self):
        limiter = CorrelationLimiter()
        limiter.update_correlation("A", "B", 0.5)
        limiter.set_sector_map({"A": "tech", "B": "energy"})
        status = limiter.status
        assert status["cached_correlations"] == 1
        assert status["tracked_sectors"] == 2


class TestDynamicUniverse:
    """Test dynamic universe expansion."""

    def test_base_symbols_always_included(self):
        universe = DynamicUniverse(
            base_symbols=["AAPL", "MSFT", "GOOGL"],
            expansion_pool=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
        )
        selection = universe.select(regime="normal")
        assert "AAPL" in selection.symbols
        assert "MSFT" in selection.symbols
        assert "GOOGL" in selection.symbols

    def test_high_vol_expands(self):
        universe = DynamicUniverse(
            base_symbols=["AAPL", "MSFT"],
            expansion_pool=["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
            max_symbols=10,
        )
        normal = universe.select(regime="normal")
        universe_hv = DynamicUniverse(
            base_symbols=["AAPL", "MSFT"],
            expansion_pool=["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
            max_symbols=10,
        )
        high_vol = universe_hv.select(regime="high_vol")
        # high_vol should have >= normal symbols (1.3x multiplier)
        assert len(high_vol.symbols) >= len(normal.symbols)

    def test_crisis_contracts(self):
        universe = DynamicUniverse(
            base_symbols=["AAPL", "MSFT", "GOOGL", "META"],
            expansion_pool=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
            max_symbols=10,
        )
        crisis = universe.select(regime="crisis")
        # Crisis = 0.5x, but base symbols always included
        assert len(crisis.symbols) >= len(universe.base_symbols)

    def test_max_symbols_cap(self):
        universe = DynamicUniverse(
            base_symbols=list(f"SYM{i}" for i in range(5)),
            expansion_pool=list(f"SYM{i}" for i in range(50)),
            max_symbols=10,
        )
        selection = universe.select(regime="high_vol")
        assert len(selection.symbols) <= 10

    def test_tracks_added_removed(self):
        universe = DynamicUniverse(
            base_symbols=["AAPL", "MSFT"],
            expansion_pool=["AAPL", "MSFT", "GOOGL"],
            max_symbols=10,
        )
        s1 = universe.select(regime="high_vol")

        # Change universe (simulate base change)
        universe.base_symbols = ["AAPL", "TSLA"]
        universe.expansion_pool = ["AAPL", "TSLA", "AMZN"]
        s2 = universe.select(regime="high_vol")

        # Should track changes
        assert isinstance(s2.added, list)
        assert isinstance(s2.removed, list)

    def test_status_dict(self):
        universe = DynamicUniverse(base_symbols=["AAPL", "MSFT"])
        universe.select(regime="normal")
        status = universe.status
        assert status["universe_size"] >= 2
        assert status["base_size"] == 2

    def test_current_symbols_property(self):
        universe = DynamicUniverse(base_symbols=["AAPL"])
        universe.select(regime="normal")
        assert "AAPL" in universe.current_symbols
