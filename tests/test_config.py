"""Tests for config.py — validate all critical parameters."""

import pytest
from datetime import time


class TestStrategyAllocations:
    """Tests for STRATEGY_ALLOCATIONS."""

    def test_allocations_sum_to_one(self):
        """All strategy allocations (including 0.0 disabled weights) must
        sum to approximately 1.0.

        2026-04-17: disabled strategies keep their key in the dict with
        weight 0.0. When a strategy is disabled its weight is redistributed
        to the remaining actives so the total still sums to 1.0 — cash
        drag from under-commitment is not acceptable.
        """
        import config
        total = sum(config.STRATEGY_ALLOCATIONS.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Allocations sum to {total}, expected 1.0"
        )

    def test_allocations_are_non_negative(self):
        """Every allocation weight must be non-negative.

        2026-04-17: allow 0.0 for disabled strategies (previously required
        strictly positive).
        """
        import config
        disabled = getattr(config, 'DISABLED_STRATEGIES', set()) or set()
        for strategy, weight in config.STRATEGY_ALLOCATIONS.items():
            assert weight >= 0, f"Allocation for {strategy} is negative: {weight}"
            if strategy not in disabled:
                assert weight > 0, (
                    f"Active strategy {strategy} has zero weight — "
                    f"add it to DISABLED_STRATEGIES if intended"
                )

    def test_allocations_are_floats(self):
        """Allocation weights must be numeric (int or float)."""
        import config
        for strategy, weight in config.STRATEGY_ALLOCATIONS.items():
            assert isinstance(weight, (int, float)), (
                f"Allocation for {strategy} is not numeric: {type(weight)}"
            )

    def test_allocations_keys_are_strings(self):
        """Strategy keys must be non-empty strings."""
        import config
        for key in config.STRATEGY_ALLOCATIONS:
            assert isinstance(key, str) and key, f"Bad strategy key: {key!r}"


class TestRiskParams:
    """Tests for risk-related configuration parameters."""

    def test_risk_per_trade_pct_in_range(self):
        """RISK_PER_TRADE_PCT must be between 0 (exclusive) and 1 (exclusive)."""
        import config
        assert 0 < config.RISK_PER_TRADE_PCT < 1, (
            f"RISK_PER_TRADE_PCT={config.RISK_PER_TRADE_PCT} out of range"
        )

    def test_daily_loss_halt_is_negative(self):
        """DAILY_LOSS_HALT must be negative (it is a loss threshold)."""
        import config
        assert config.DAILY_LOSS_HALT < 0, (
            f"DAILY_LOSS_HALT={config.DAILY_LOSS_HALT} should be negative"
        )

    def test_max_portfolio_deploy_in_range(self):
        """MAX_PORTFOLIO_DEPLOY must be between 0 and 1."""
        import config
        assert 0 < config.MAX_PORTFOLIO_DEPLOY <= 1.0, (
            f"MAX_PORTFOLIO_DEPLOY={config.MAX_PORTFOLIO_DEPLOY} out of range"
        )

    def test_max_position_pct_in_range(self):
        """MAX_POSITION_PCT must be between 0 and 1."""
        import config
        assert 0 < config.MAX_POSITION_PCT <= 1.0, (
            f"MAX_POSITION_PCT={config.MAX_POSITION_PCT} out of range"
        )

    def test_short_hard_stop_pct_positive(self):
        """SHORT_HARD_STOP_PCT must be positive."""
        import config
        assert config.SHORT_HARD_STOP_PCT > 0, (
            f"SHORT_HARD_STOP_PCT={config.SHORT_HARD_STOP_PCT} must be positive"
        )

    def test_pnl_loss_halt_is_negative(self):
        """PNL_LOSS_HALT_PCT must be negative."""
        import config
        assert config.PNL_LOSS_HALT_PCT < 0, (
            f"PNL_LOSS_HALT_PCT={config.PNL_LOSS_HALT_PCT} should be negative"
        )

    def test_pnl_gain_lock_is_positive(self):
        """PNL_GAIN_LOCK_PCT must be positive."""
        import config
        assert config.PNL_GAIN_LOCK_PCT > 0, (
            f"PNL_GAIN_LOCK_PCT={config.PNL_GAIN_LOCK_PCT} must be positive"
        )


class TestMaxPositions:
    """Tests for position count limits."""

    def test_max_positions_positive(self):
        """MAX_POSITIONS must be a positive integer."""
        import config
        assert config.MAX_POSITIONS > 0
        assert isinstance(config.MAX_POSITIONS, int)



class TestMarketHours:
    """Tests for market timing configuration."""

    def test_market_open_before_trading_start(self):
        """MARKET_OPEN must be before TRADING_START."""
        import config
        assert config.MARKET_OPEN < config.TRADING_START, (
            f"MARKET_OPEN={config.MARKET_OPEN} is not before TRADING_START={config.TRADING_START}"
        )

    def test_trading_start_before_market_close(self):
        """TRADING_START must be before MARKET_CLOSE."""
        import config
        assert config.TRADING_START < config.MARKET_CLOSE, (
            f"TRADING_START={config.TRADING_START} is not before MARKET_CLOSE={config.MARKET_CLOSE}"
        )

    def test_market_open_before_market_close(self):
        """MARKET_OPEN must be before MARKET_CLOSE."""
        import config
        assert config.MARKET_OPEN < config.MARKET_CLOSE

    def test_orb_exit_before_market_close(self):
        """ORB_EXIT_TIME must be before MARKET_CLOSE."""
        import config
        assert config.ORB_EXIT_TIME < config.MARKET_CLOSE, (
            f"ORB_EXIT_TIME={config.ORB_EXIT_TIME} is not before MARKET_CLOSE={config.MARKET_CLOSE}"
        )

    def test_market_hours_are_time_objects(self):
        """All market hour settings must be datetime.time instances."""
        import config
        for attr in ("MARKET_OPEN", "TRADING_START", "ORB_EXIT_TIME", "MARKET_CLOSE"):
            val = getattr(config, attr)
            assert isinstance(val, time), f"{attr} is not a datetime.time: {type(val)}"

    def test_scan_interval_positive(self):
        """SCAN_INTERVAL_SEC must be positive."""
        import config
        assert config.SCAN_INTERVAL_SEC > 0

    def test_state_save_interval_positive(self):
        """STATE_SAVE_INTERVAL_SEC must be positive."""
        import config
        assert config.STATE_SAVE_INTERVAL_SEC > 0

    def test_regime_check_interval_positive(self):
        """REGIME_CHECK_INTERVAL_MIN must be positive."""
        import config
        assert config.REGIME_CHECK_INTERVAL_MIN > 0


class TestBooleanFlags:
    """Tests that all _ENABLED flags and boolean settings are actual booleans."""

    def test_enabled_flags_are_booleans(self):
        """All _ENABLED config attributes must be bool."""
        import config
        enabled_flags = [attr for attr in dir(config) if attr.endswith("_ENABLED")]
        assert len(enabled_flags) > 0, "No _ENABLED flags found in config"
        for attr in enabled_flags:
            val = getattr(config, attr)
            assert isinstance(val, bool), (
                f"config.{attr}={val!r} is not a bool (type: {type(val).__name__})"
            )

    def test_paper_mode_is_bool(self):
        """PAPER_MODE must be a bool."""
        import config
        assert isinstance(config.PAPER_MODE, bool)

    def test_allow_short_is_bool(self):
        """ALLOW_SHORT must be a bool."""
        import config
        assert isinstance(config.ALLOW_SHORT, bool)

    def test_dynamic_allocation_is_bool(self):
        """DYNAMIC_ALLOCATION must be a bool."""
        import config
        assert isinstance(config.DYNAMIC_ALLOCATION, bool)

    def test_websocket_monitoring_is_bool(self):
        """WEBSOCKET_MONITORING must be a bool."""
        import config
        assert isinstance(config.WEBSOCKET_MONITORING, bool)


class TestSymbolUniverse:
    """Tests for the trading symbol universe."""

    def test_symbols_is_non_empty_list(self):
        """SYMBOLS must be a non-empty list."""
        import config
        assert isinstance(config.SYMBOLS, list)
        assert len(config.SYMBOLS) > 0, "SYMBOLS list is empty"

    def test_core_symbols_is_non_empty_list(self):
        """CORE_SYMBOLS must be a non-empty list."""
        import config
        assert isinstance(config.CORE_SYMBOLS, list)
        assert len(config.CORE_SYMBOLS) > 0

    def test_symbols_are_strings(self):
        """Every symbol must be a non-empty uppercase string."""
        import config
        for sym in config.SYMBOLS:
            assert isinstance(sym, str) and sym, f"Invalid symbol: {sym!r}"
            assert sym == sym.upper(), f"Symbol not uppercase: {sym!r}"

    def test_no_duplicate_symbols(self):
        """SYMBOLS list must contain no duplicates."""
        import config
        assert len(config.SYMBOLS) == len(set(config.SYMBOLS)), (
            "SYMBOLS list has duplicates"
        )

    def test_standard_symbols_excludes_leveraged(self):
        """STANDARD_SYMBOLS must not contain any LEVERAGED_ETFS."""
        import config
        overlap = set(config.STANDARD_SYMBOLS) & config.LEVERAGED_ETFS
        assert not overlap, f"STANDARD_SYMBOLS contains leveraged ETFs: {overlap}"

    def test_leveraged_etfs_is_set(self):
        """LEVERAGED_ETFS must be a set."""
        import config
        assert isinstance(config.LEVERAGED_ETFS, (set, frozenset))


class TestTimingConfigs:
    """Tests for timing-related numeric configs."""

    def test_vix_cache_seconds_positive(self):
        """VIX_CACHE_SECONDS must be positive."""
        import config
        assert config.VIX_CACHE_SECONDS > 0

    def test_websocket_reconnect_positive(self):
        """WEBSOCKET_RECONNECT_SEC must be positive."""
        import config
        assert config.WEBSOCKET_RECONNECT_SEC > 0

    def test_mtf_cache_seconds_positive(self):
        """MTF_CACHE_SECONDS must be positive."""
        import config
        assert config.MTF_CACHE_SECONDS > 0

    def test_allocation_lookback_positive(self):
        """ALLOCATION_LOOKBACK_DAYS must be positive."""
        import config
        assert config.ALLOCATION_LOOKBACK_DAYS > 0

    def test_kelly_lookback_positive(self):
        """KELLY_LOOKBACK must be positive."""
        import config
        assert config.KELLY_LOOKBACK > 0

    def test_kelly_min_trades_positive(self):
        """KELLY_MIN_TRADES must be positive."""
        import config
        assert config.KELLY_MIN_TRADES > 0

    def test_orb_time_stop_positive(self):
        """ORB_TIME_STOP_HOURS must be positive."""
        import config
        assert config.ORB_TIME_STOP_HOURS > 0


class TestKellyParams:
    """Tests for Kelly criterion configuration."""

    def test_kelly_min_less_than_max_risk(self):
        """KELLY_MIN_RISK must be strictly less than KELLY_MAX_RISK."""
        import config
        assert config.KELLY_MIN_RISK < config.KELLY_MAX_RISK, (
            f"KELLY_MIN_RISK={config.KELLY_MIN_RISK} >= KELLY_MAX_RISK={config.KELLY_MAX_RISK}"
        )

    def test_kelly_min_risk_positive(self):
        """KELLY_MIN_RISK must be positive."""
        import config
        assert config.KELLY_MIN_RISK > 0

    def test_kelly_max_risk_positive(self):
        """KELLY_MAX_RISK must be positive."""
        import config
        assert config.KELLY_MAX_RISK > 0

    def test_kelly_fraction_mult_in_range(self):
        """KELLY_FRACTION_MULT must be between 0 and 1 inclusive."""
        import config
        assert 0 < config.KELLY_FRACTION_MULT <= 1.0, (
            f"KELLY_FRACTION_MULT={config.KELLY_FRACTION_MULT} out of range (0, 1]"
        )

    def test_kelly_enabled_is_bool(self):
        """KELLY_ENABLED must be a bool."""
        import config
        assert isinstance(config.KELLY_ENABLED, bool)


class TestRuntimeParams:
    """Tests for runtime-mutable parameter helpers."""

    def test_get_param_returns_default_when_missing(self):
        """get_param returns the default when key is not set."""
        import config
        val = config.get_param("nonexistent_key_xyz", default=42)
        assert val == 42

    def test_set_and_get_param_roundtrip(self):
        """set_param stores a value that get_param retrieves."""
        import config
        config.set_param("test_key_abc", 99)
        assert config.get_param("test_key_abc") == 99
        # Cleanup via config.settings (underscore names not exported by wildcard import)
        import config.settings as _settings
        _settings._runtime_params.pop("test_key_abc", None)

    def test_get_param_returns_none_default(self):
        """get_param default is None when not specified."""
        import config
        val = config.get_param("another_missing_key_xyz")
        assert val is None
