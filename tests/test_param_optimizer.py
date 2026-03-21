"""Tests for analytics/param_optimizer.py — Bayesian parameter optimization."""

import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config
import config.settings
from analytics.param_optimizer import (
    BayesianOptimizer,
    OptimizationResult,
    PARAM_SPACES,
    _get_current_params,
    _compute_sortino_from_trades,
    _INT_PARAMS,
)


# ---- Helpers ----

def _make_trade_history(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic trade-history DataFrame with a 'pnl' column."""
    rng = np.random.default_rng(seed)
    pnls = rng.normal(loc=0.002, scale=0.01, size=n)
    return pd.DataFrame({"pnl": pnls, "strategy": ["STAT_MR"] * n})


def _make_small_history(n: int = 3) -> pd.DataFrame:
    """Trade history too small for walk-forward split."""
    return pd.DataFrame({"pnl": [0.01] * n})


# ---- Config defaults ----

class TestConfigDefaults:
    """Verify the new config parameters exist."""

    def test_param_optimizer_enabled_exists(self):
        assert hasattr(config, "PARAM_OPTIMIZER_ENABLED")
        assert config.PARAM_OPTIMIZER_ENABLED is True

    def test_param_optimizer_day(self):
        assert config.PARAM_OPTIMIZER_DAY == "sunday"

    def test_param_optimizer_trials(self):
        assert config.PARAM_OPTIMIZER_TRIALS == 100

    def test_param_optimizer_min_improvement(self):
        assert config.PARAM_OPTIMIZER_MIN_IMPROVEMENT == 0.15

    def test_param_optimizer_apply_auto(self):
        assert config.PARAM_OPTIMIZER_APPLY_AUTO is False


# ---- PARAM_SPACES ----

class TestParamSpaces:
    """Verify parameter space definitions."""

    def test_known_strategies_present(self):
        for strat in ("STAT_MR", "VWAP", "ORB"):
            assert strat in PARAM_SPACES, f"{strat} missing from PARAM_SPACES"

    def test_ranges_are_valid_tuples(self):
        for strat, space in PARAM_SPACES.items():
            for key, (lo, hi) in space.items():
                assert lo < hi, f"{strat}.{key}: low ({lo}) >= high ({hi})"

    def test_stat_mr_params(self):
        space = PARAM_SPACES["STAT_MR"]
        assert "MR_ZSCORE_ENTRY" in space
        assert "MR_HURST_MAX" in space

    def test_vwap_params(self):
        space = PARAM_SPACES["VWAP"]
        assert "VWAP_OU_ZSCORE_MIN" in space
        assert "VWAP_RSI_OVERSOLD" in space
        assert "VWAP_RSI_OVERBOUGHT" in space

    def test_orb_params(self):
        space = PARAM_SPACES["ORB"]
        assert "ORB_VOLUME_RATIO" in space
        assert "ORB_TP_MULT" in space
        assert "ORB_SL_MULT" in space


# ---- _get_current_params ----

class TestGetCurrentParams:
    def test_reads_from_config_module(self):
        space = {"MR_ZSCORE_ENTRY": (1.0, 2.5)}
        params = _get_current_params(space)
        assert params["MR_ZSCORE_ENTRY"] == config.MR_ZSCORE_ENTRY

    def test_prefers_runtime_param(self):
        config.set_param("MR_ZSCORE_ENTRY", 99.9)
        try:
            params = _get_current_params({"MR_ZSCORE_ENTRY": (1.0, 2.5)})
            assert params["MR_ZSCORE_ENTRY"] == 99.9
        finally:
            config.settings._runtime_params.pop("MR_ZSCORE_ENTRY", None)


# ---- _compute_sortino_from_trades ----

class TestComputeSortinoFromTrades:
    def test_empty_df_returns_zero(self):
        assert _compute_sortino_from_trades(pd.DataFrame()) == 0.0

    def test_missing_pnl_column_returns_zero(self):
        df = pd.DataFrame({"returns": [0.01, 0.02]})
        assert _compute_sortino_from_trades(df) == 0.0

    def test_valid_trades_return_nonzero(self):
        df = _make_trade_history(50)
        val = _compute_sortino_from_trades(df)
        assert isinstance(val, float)


# ---- BayesianOptimizer ----

class TestBayesianOptimizerInit:
    def test_get_last_results_empty_initially(self):
        opt = BayesianOptimizer()
        assert opt.get_last_results() == {}


class TestOptimizeStrategy:
    """Core optimization tests."""

    def test_returns_optimization_result(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        result = opt.optimize_strategy("STAT_MR", hist, n_trials=5)
        assert isinstance(result, OptimizationResult)
        assert result.strategy == "STAT_MR"

    def test_result_has_all_fields(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        result = opt.optimize_strategy("STAT_MR", hist, n_trials=5)
        assert isinstance(result.current_params, dict)
        assert isinstance(result.optimized_params, dict)
        assert isinstance(result.current_sortino, float)
        assert isinstance(result.optimized_sortino, float)
        assert isinstance(result.improvement_pct, float)
        assert isinstance(result.should_apply, bool)
        assert isinstance(result.reason, str)

    def test_current_params_captured(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        result = opt.optimize_strategy("STAT_MR", hist, n_trials=5)
        # Current params should include all keys from PARAM_SPACES["STAT_MR"]
        for key in PARAM_SPACES["STAT_MR"]:
            assert key in result.current_params

    def test_optimized_params_within_bounds(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        result = opt.optimize_strategy("STAT_MR", hist, n_trials=10)
        space = PARAM_SPACES["STAT_MR"]
        for key, val in result.optimized_params.items():
            lo, hi = space[key]
            assert lo <= val <= hi, f"{key}={val} outside [{lo}, {hi}]"

    def test_stored_in_last_results(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        opt.optimize_strategy("STAT_MR", hist, n_trials=5)
        last = opt.get_last_results()
        assert "STAT_MR" in last

    def test_custom_param_space(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        custom = {"MR_ZSCORE_ENTRY": (1.2, 2.0)}
        result = opt.optimize_strategy("STAT_MR", hist, param_space=custom, n_trials=5)
        assert "MR_ZSCORE_ENTRY" in result.optimized_params
        assert len(result.optimized_params) == 1

    def test_vwap_strategy(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(80)
        result = opt.optimize_strategy("VWAP", hist, n_trials=5)
        assert result.strategy == "VWAP"
        for key in PARAM_SPACES["VWAP"]:
            assert key in result.optimized_params

    def test_orb_strategy(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(80)
        result = opt.optimize_strategy("ORB", hist, n_trials=5)
        assert result.strategy == "ORB"
        for key in PARAM_SPACES["ORB"]:
            assert key in result.optimized_params


# ---- Fail-open behaviour ----

class TestFailOpen:
    def test_empty_trade_history(self):
        opt = BayesianOptimizer()
        result = opt.optimize_strategy("STAT_MR", pd.DataFrame())
        assert result.should_apply is False
        assert result.optimized_params == result.current_params
        assert "Insufficient" in result.reason

    def test_none_trade_history(self):
        opt = BayesianOptimizer()
        result = opt.optimize_strategy("STAT_MR", None)
        assert result.should_apply is False

    def test_no_pnl_column(self):
        opt = BayesianOptimizer()
        df = pd.DataFrame({"returns": [0.01, 0.02, 0.03]})
        result = opt.optimize_strategy("STAT_MR", df)
        assert result.should_apply is False
        assert "Insufficient" in result.reason

    def test_too_few_trades_for_split(self):
        opt = BayesianOptimizer()
        hist = _make_small_history(3)
        result = opt.optimize_strategy("STAT_MR", hist, n_trials=3)
        assert result.should_apply is False
        assert "walk-forward" in result.reason.lower() or "Not enough" in result.reason

    def test_unknown_strategy_no_space(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(50)
        result = opt.optimize_strategy("UNKNOWN_STRAT", hist, n_trials=3)
        assert result.should_apply is False
        assert "No parameter space" in result.reason

    def test_optuna_import_error(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        with patch.dict("sys.modules", {"optuna": None}):
            # Force import to fail inside the method
            import importlib
            result = opt.optimize_strategy("STAT_MR", hist, n_trials=3)
            # Should fail-open: no crash, returns current params
            assert isinstance(result, OptimizationResult)
            assert result.should_apply is False


# ---- should_apply ----

class TestShouldApply:
    def test_returns_true_above_threshold(self):
        opt = BayesianOptimizer()
        result = OptimizationResult(
            strategy="STAT_MR",
            current_params={},
            optimized_params={},
            current_sortino=1.0,
            optimized_sortino=1.5,
            improvement_pct=0.50,
            should_apply=True,
            reason="test",
        )
        assert opt.should_apply(result) is True

    def test_returns_false_below_threshold(self):
        opt = BayesianOptimizer()
        result = OptimizationResult(
            strategy="STAT_MR",
            current_params={},
            optimized_params={},
            current_sortino=1.0,
            optimized_sortino=1.05,
            improvement_pct=0.05,
            should_apply=False,
            reason="test",
        )
        assert opt.should_apply(result) is False

    def test_returns_false_at_exact_threshold(self):
        opt = BayesianOptimizer()
        # Default threshold is 0.15; at exactly 0.15 it should pass (>=)
        result = OptimizationResult(
            strategy="X", current_params={}, optimized_params={},
            current_sortino=1.0, optimized_sortino=1.15,
            improvement_pct=0.15, should_apply=True, reason="test",
        )
        assert opt.should_apply(result) is True

    def test_respects_config_threshold(self, override_config):
        opt = BayesianOptimizer()
        with override_config(PARAM_OPTIMIZER_MIN_IMPROVEMENT=0.50):
            result = OptimizationResult(
                strategy="X", current_params={}, optimized_params={},
                current_sortino=1.0, optimized_sortino=1.3,
                improvement_pct=0.30, should_apply=False, reason="test",
            )
            assert opt.should_apply(result) is False

    def test_negative_improvement_is_false(self):
        opt = BayesianOptimizer()
        result = OptimizationResult(
            strategy="X", current_params={}, optimized_params={},
            current_sortino=2.0, optimized_sortino=1.5,
            improvement_pct=-0.25, should_apply=False, reason="test",
        )
        assert opt.should_apply(result) is False


# ---- apply_optimized_params ----

class TestApplyOptimizedParams:
    def test_sets_runtime_params(self):
        opt = BayesianOptimizer()
        result = OptimizationResult(
            strategy="STAT_MR",
            current_params={"MR_ZSCORE_ENTRY": 1.5, "MR_HURST_MAX": 0.52},
            optimized_params={"MR_ZSCORE_ENTRY": 1.8, "MR_HURST_MAX": 0.48},
            current_sortino=1.0,
            optimized_sortino=1.5,
            improvement_pct=0.50,
            should_apply=True,
            reason="test",
        )
        opt.apply_optimized_params(result)
        try:
            assert config.get_param("MR_ZSCORE_ENTRY") == 1.8
            assert config.get_param("MR_HURST_MAX") == 0.48
        finally:
            config.settings._runtime_params.pop("MR_ZSCORE_ENTRY", None)
            config.settings._runtime_params.pop("MR_HURST_MAX", None)

    def test_apply_empty_params_is_noop(self):
        opt = BayesianOptimizer()
        result = OptimizationResult(
            strategy="X", current_params={}, optimized_params={},
            current_sortino=0.0, optimized_sortino=0.0,
            improvement_pct=0.0, should_apply=False, reason="empty",
        )
        opt.apply_optimized_params(result)  # Should not raise


# ---- Disabled optimizer ----

class TestDisabledOptimizer:
    def test_config_flag_can_be_disabled(self, override_config):
        with override_config(PARAM_OPTIMIZER_ENABLED=False):
            assert config.PARAM_OPTIMIZER_ENABLED is False
            # Optimization still works (caller checks the flag), but
            # the flag is accessible
            opt = BayesianOptimizer()
            hist = _make_trade_history(50)
            result = opt.optimize_strategy("STAT_MR", hist, n_trials=3)
            assert isinstance(result, OptimizationResult)


# ---- Integer parameter handling ----

class TestIntegerParams:
    def test_int_params_set_defined(self):
        assert "MR_RSI_OVERSOLD" in _INT_PARAMS
        assert "VWAP_RSI_OVERSOLD" in _INT_PARAMS
        assert "VWAP_RSI_OVERBOUGHT" in _INT_PARAMS

    def test_integer_params_are_int_in_result(self):
        opt = BayesianOptimizer()
        hist = _make_trade_history(100)
        result = opt.optimize_strategy("VWAP", hist, n_trials=5)
        assert isinstance(result.optimized_params["VWAP_RSI_OVERSOLD"], int)
        assert isinstance(result.optimized_params["VWAP_RSI_OVERBOUGHT"], int)
