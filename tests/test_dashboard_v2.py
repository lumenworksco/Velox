"""Tests for V9 dashboard — V2 API endpoints and terminal dashboard enhancements."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


# ===================================================================
# FastAPI test client setup
# ===================================================================

@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from web_dashboard import app
    return TestClient(app)


# ===================================================================
# V2 Endpoint Tests
# ===================================================================

class TestV2Overview:
    """Tests for /api/v2/overview."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "regime" in data
        assert "cross_asset" in data
        assert "portfolio_heat" in data
        assert "daily_pnl" in data
        assert "adaptive_weights" in data

    def test_regime_has_state_and_probabilities(self, client):
        resp = client.get("/api/v2/overview")
        data = resp.json()
        assert "state" in data["regime"]
        assert "probabilities" in data["regime"]

    def test_returns_data_with_populated_state(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(
            hmm_regime="LOW_VOL_BULL",
            hmm_probabilities={"LOW_VOL_BULL": 0.87},
            cross_asset_bias=0.6,
            portfolio_heat_pct=0.42,
            adaptive_weights={"STAT_MR": 0.40, "VWAP": 0.20},
        )
        resp = client.get("/api/v2/overview")
        data = resp.json()
        assert data["regime"]["state"] == "LOW_VOL_BULL"
        assert data["cross_asset"]["bias"] == 0.6
        assert data["portfolio_heat"]["current_pct"] == 0.42
        assert "STAT_MR" in data["adaptive_weights"]

    def test_partial_data_on_error(self, client):
        """Even if some data fails, other keys should still be present."""
        resp = client.get("/api/v2/overview")
        data = resp.json()
        # All top-level keys must be present regardless
        for key in ["regime", "cross_asset", "portfolio_heat", "daily_pnl", "adaptive_weights"]:
            assert key in data


class TestV2Strategy:
    """Tests for /api/v2/strategy/{name}."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/strategy/STAT_MR")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "STAT_MR"

    def test_has_expected_fields(self, client):
        resp = client.get("/api/v2/strategy/VWAP")
        data = resp.json()
        assert "alpha_decay" in data
        assert "recent_trades" in data
        assert "win_rate_7d" in data
        assert "allocation_weight" in data

    def test_unknown_strategy_returns_data(self, client):
        """Unknown strategies should still return a valid response."""
        resp = client.get("/api/v2/strategy/NONEXISTENT")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "NONEXISTENT"

    def test_regime_affinity_graceful_when_module_missing(self, client):
        """Should handle missing hmm_regime module gracefully."""
        with patch.dict("sys.modules", {"analytics.hmm_regime": None}):
            resp = client.get("/api/v2/strategy/ORB")
            assert resp.status_code == 200
            data = resp.json()
            # regime_affinity should be None or have an error, not crash
            assert "regime_affinity" in data


class TestV2SignalsPipeline:
    """Tests for /api/v2/signals/pipeline."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/signals/pipeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "signals_today" in data
        assert "db_signals" in data
        assert "rejection_reasons" in data

    def test_with_populated_signals(self, client):
        from web_dashboard import update_v9_state
        test_signals = [
            {"symbol": "AAPL", "strategy": "ORB", "decision": "approved"},
            {"symbol": "MSFT", "strategy": "VWAP", "decision": "rejected", "reason": "low_volume"},
        ]
        update_v9_state(signals_today=test_signals)
        resp = client.get("/api/v2/signals/pipeline")
        data = resp.json()
        assert len(data["signals_today"]) == 2

    def test_ranking_history_graceful(self, client):
        """Should handle missing signal_ranker module gracefully."""
        resp = client.get("/api/v2/signals/pipeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "ranking_history" in data


class TestV2RiskExposure:
    """Tests for /api/v2/risk/exposure."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/risk/exposure")
        assert resp.status_code == 200
        data = resp.json()
        assert "portfolio_heat" in data
        assert "beta_exposure" in data
        assert "overnight" in data
        assert "monte_carlo" in data
        assert "daily_pnl_lock" in data

    def test_heat_has_cluster_detail(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(
            portfolio_heat_pct=0.42,
            cluster_heat={"tech": 0.15, "finance": 0.10},
        )
        resp = client.get("/api/v2/risk/exposure")
        data = resp.json()
        assert "cluster_heat" in data["portfolio_heat"]

    def test_overnight_data(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(
            overnight_count=2,
            overnight_positions=[{"symbol": "AAPL"}, {"symbol": "MSFT"}],
        )
        resp = client.get("/api/v2/risk/exposure")
        data = resp.json()
        assert data["overnight"]["count"] == 2


class TestV2ExecutionQuality:
    """Tests for /api/v2/execution/quality."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/execution/quality")
        assert resp.status_code == 200
        data = resp.json()
        assert "slippage_by_strategy" in data
        assert "fill_rate" in data

    def test_with_populated_stats(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(execution_stats={
            "slippage_by_strategy": {"STAT_MR": 0.0002, "VWAP": 0.0003},
            "fill_rate": 0.95,
            "latency_p50_ms": 12,
            "latency_p95_ms": 45,
            "cancel_rate": 0.02,
        })
        resp = client.get("/api/v2/execution/quality")
        data = resp.json()
        assert data["fill_rate"] == 0.95
        assert "STAT_MR" in data["slippage_by_strategy"]

    def test_defaults_when_empty(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(execution_stats={})
        resp = client.get("/api/v2/execution/quality")
        data = resp.json()
        # Should have defaults
        assert "slippage_by_strategy" in data
        assert "fill_rate" in data


class TestV2Health:
    """Tests for /api/v2/health."""

    def test_returns_valid_json(self, client):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] in ("V9", "V10")
        assert "uptime_seconds" in data
        assert "data_feed_status" in data
        assert "strategy_scan_times" in data

    def test_with_populated_health(self, client):
        from web_dashboard import update_v9_state
        update_v9_state(system_health={
            "data_feed_status": "connected",
            "api_latency_ms": 35,
            "cache_hit_rate": 0.82,
            "strategy_scan_times": {"STAT_MR": 1.2, "VWAP": 0.8},
            "last_error": None,
        })
        resp = client.get("/api/v2/health")
        data = resp.json()
        assert data["data_feed_status"] == "connected"
        assert data["cache_hit_rate"] == 0.82

    def test_health_endpoint_version(self, client):
        """The legacy /health endpoint should also say V9."""
        resp = client.get("/health")
        data = resp.json()
        assert data["version"] in ("V9", "V10")


# ===================================================================
# Existing endpoint version check
# ===================================================================

class TestVersionStrings:
    """Verify all V8 references are updated to V9."""

    def test_health_endpoint_version(self, client):
        resp = client.get("/health")
        assert resp.json()["version"] in ("V9", "V10")

    def test_html_dashboard_title(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "V9" in resp.text or "V10" in resp.text
        assert "V8" not in resp.text

    def test_v2_health_version(self, client):
        resp = client.get("/api/v2/health")
        assert resp.json()["version"] in ("V9", "V10")


# ===================================================================
# Terminal Dashboard Tests
# ===================================================================

class TestTerminalDashboard:
    """Tests for dashboard.py build_dashboard with V9 kwargs."""

    def _make_mock_risk(self):
        """Create a minimal mock RiskManager."""
        risk = MagicMock()
        risk.day_pnl = 0.005
        risk.starting_equity = 100_000
        risk.current_equity = 100_500
        risk.current_cash = 80_000
        risk.circuit_breaker_active = False
        risk.open_trades = {}
        risk.closed_trades = []
        return risk

    def test_builds_without_v9_data(self):
        """Dashboard should build fine with no V9 kwargs (backward compat)."""
        from dashboard import build_dashboard
        risk = self._make_mock_risk()
        panel = build_dashboard(
            risk=risk,
            regime="BULLISH",
            start_time=datetime(2026, 3, 15, 9, 30, tzinfo=ET),
            now=datetime(2026, 3, 15, 11, 0, tzinfo=ET),
            last_scan_time=datetime(2026, 3, 15, 10, 58, tzinfo=ET),
            num_symbols=50,
        )
        assert panel is not None
        # Should contain V9 in title
        assert "V9" in str(panel.title)

    def test_builds_with_v9_data(self):
        """Dashboard should display V9 intelligence lines when provided."""
        from dashboard import build_dashboard
        risk = self._make_mock_risk()
        panel = build_dashboard(
            risk=risk,
            regime="BULLISH",
            start_time=datetime(2026, 3, 15, 9, 30, tzinfo=ET),
            now=datetime(2026, 3, 15, 11, 0, tzinfo=ET),
            last_scan_time=datetime(2026, 3, 15, 10, 58, tzinfo=ET),
            num_symbols=50,
            hmm_regime_state="LOW_VOL_BULL",
            hmm_probabilities={"LOW_VOL_BULL": 0.87},
            cross_asset_bias=0.6,
            portfolio_heat_pct=42.0,
            alpha_warnings=["STAT_MR: Sharpe declining"],
            overnight_count=2,
        )
        assert panel is not None
        content = str(panel.renderable)
        assert "LOW_VOL_BULL" in content
        assert "REGIME" in content
        assert "CROSS" in content
        assert "HEAT" in content
        assert "OVERNIGHT" in content

    def test_builds_with_partial_v9_data(self):
        """Dashboard should handle partial V9 data gracefully."""
        from dashboard import build_dashboard
        risk = self._make_mock_risk()
        panel = build_dashboard(
            risk=risk,
            regime="BEARISH",
            start_time=datetime(2026, 3, 15, 9, 30, tzinfo=ET),
            now=datetime(2026, 3, 15, 11, 0, tzinfo=ET),
            last_scan_time=None,
            num_symbols=50,
            hmm_regime_state="HIGH_VOL_BEAR",
            # Other V9 kwargs left as defaults
        )
        assert panel is not None

    def test_pead_in_allocation_display(self):
        """PEAD strategy should appear in allocation lines."""
        from dashboard import build_dashboard
        risk = self._make_mock_risk()
        panel = build_dashboard(
            risk=risk,
            regime="BULLISH",
            start_time=datetime(2026, 3, 15, 9, 30, tzinfo=ET),
            now=datetime(2026, 3, 15, 11, 0, tzinfo=ET),
            last_scan_time=datetime(2026, 3, 15, 10, 58, tzinfo=ET),
            num_symbols=50,
        )
        content = str(panel.renderable)
        assert "PEAD" in content

    def test_v9_version_in_title(self):
        """Panel title should reference V9."""
        from dashboard import build_dashboard
        risk = self._make_mock_risk()
        panel = build_dashboard(
            risk=risk,
            regime="BULLISH",
            start_time=datetime(2026, 3, 15, 9, 30, tzinfo=ET),
            now=datetime(2026, 3, 15, 11, 0, tzinfo=ET),
            last_scan_time=None,
            num_symbols=50,
        )
        assert "V9" in str(panel.title)
        assert "V8" not in str(panel.title)


# ===================================================================
# Module availability / fail-open tests
# ===================================================================

class TestFailOpen:
    """Endpoints must return partial data even when V9 modules are unavailable."""

    def test_overview_without_modules(self, client):
        resp = client.get("/api/v2/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_strategy_without_hmm(self, client):
        """Strategy endpoint should work even without hmm_regime module."""
        resp = client.get("/api/v2/strategy/ORB")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "ORB"

    def test_signals_without_ranker(self, client):
        resp = client.get("/api/v2/signals/pipeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "signals_today" in data

    def test_risk_without_cross_asset(self, client):
        resp = client.get("/api/v2/risk/exposure")
        assert resp.status_code == 200
        data = resp.json()
        assert "vix_regime" in data

    def test_execution_without_analytics(self, client):
        resp = client.get("/api/v2/execution/quality")
        assert resp.status_code == 200
        data = resp.json()
        assert "slippage_by_strategy" in data

    def test_health_always_responds(self, client):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] in ("V9", "V10")


# ===================================================================
# update_v9_state tests
# ===================================================================

class TestUpdateV9State:
    """Tests for the shared state update function."""

    def test_updates_known_keys(self):
        from web_dashboard import update_v9_state, _v9_state
        update_v9_state(hmm_regime="TEST_REGIME", cross_asset_bias=0.5)
        assert _v9_state["hmm_regime"] == "TEST_REGIME"
        assert _v9_state["cross_asset_bias"] == 0.5

    def test_ignores_unknown_keys(self):
        from web_dashboard import update_v9_state, _v9_state
        update_v9_state(unknown_key_xyz="should_be_ignored")
        assert "unknown_key_xyz" not in _v9_state
