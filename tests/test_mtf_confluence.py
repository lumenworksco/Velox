"""Tests for V8 Multi-Timeframe Confluence Filter."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

ET = ZoneInfo("America/New_York")


class TestMTFConfluence:

    def setup_method(self):
        """Clear cache before each test."""
        from analytics.mtf_confluence import clear_cache
        clear_cache()

    def test_disabled_returns_neutral(self, override_config):
        from analytics.mtf_confluence import get_mtf_confluence
        with override_config(MTF_CONFLUENCE_ENABLED=False):
            score = get_mtf_confluence("AAPL", "buy")
            assert score == 0.5

    def test_check_filter_disabled_always_allows(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter
        with override_config(MTF_CONFLUENCE_ENABLED=False):
            allowed, reason = check_mtf_filter("AAPL", "ORB", "buy")
            assert allowed is True

    def test_breakout_needs_alignment(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter

        with override_config(MTF_CONFLUENCE_ENABLED=True,
                           MTF_MIN_CONFLUENCE_BREAKOUT=0.66,
                           MTF_MAX_CONFLUENCE_MEANREV=0.33):
            # Mock low confluence for ORB
            with patch("analytics.mtf_confluence.get_mtf_confluence", return_value=0.33):
                allowed, reason = check_mtf_filter("AAPL", "ORB", "buy")
                assert not allowed
                assert "mtf_confluence" in reason

    def test_breakout_passes_with_alignment(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter

        with override_config(MTF_CONFLUENCE_ENABLED=True,
                           MTF_MIN_CONFLUENCE_BREAKOUT=0.66,
                           MTF_MAX_CONFLUENCE_MEANREV=0.33):
            with patch("analytics.mtf_confluence.get_mtf_confluence", return_value=1.0):
                allowed, reason = check_mtf_filter("AAPL", "ORB", "buy")
                assert allowed

    def test_meanrev_wants_dislocation(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter

        with override_config(MTF_CONFLUENCE_ENABLED=True,
                           MTF_MIN_CONFLUENCE_BREAKOUT=0.66,
                           MTF_MAX_CONFLUENCE_MEANREV=0.33):
            # High confluence = bad for mean reversion
            with patch("analytics.mtf_confluence.get_mtf_confluence", return_value=0.66):
                allowed, reason = check_mtf_filter("AAPL", "STAT_MR", "buy")
                assert not allowed

    def test_meanrev_passes_with_dislocation(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter

        with override_config(MTF_CONFLUENCE_ENABLED=True,
                           MTF_MIN_CONFLUENCE_BREAKOUT=0.66,
                           MTF_MAX_CONFLUENCE_MEANREV=0.33):
            with patch("analytics.mtf_confluence.get_mtf_confluence", return_value=0.0):
                allowed, reason = check_mtf_filter("AAPL", "STAT_MR", "buy")
                assert allowed

    def test_kalman_pairs_no_filter(self, override_config):
        from analytics.mtf_confluence import check_mtf_filter

        with override_config(MTF_CONFLUENCE_ENABLED=True,
                           MTF_MIN_CONFLUENCE_BREAKOUT=0.66,
                           MTF_MAX_CONFLUENCE_MEANREV=0.33):
            with patch("analytics.mtf_confluence.get_mtf_confluence", return_value=0.0):
                allowed, reason = check_mtf_filter("AAPL", "KALMAN_PAIRS", "buy")
                assert allowed  # No filter for pairs

    def test_cache_hit(self, override_config):
        from analytics.mtf_confluence import get_mtf_confluence, _mtf_cache

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        _mtf_cache["AAPL_buy"] = (0.8, now)

        with override_config(MTF_CONFLUENCE_ENABLED=True, MTF_CACHE_SECONDS=300):
            score = get_mtf_confluence("AAPL", "buy", now)
            assert score == 0.8

    def test_clear_cache(self):
        from analytics.mtf_confluence import _mtf_cache, clear_cache
        _mtf_cache["test"] = (0.5, datetime.now())
        clear_cache()
        from analytics import mtf_confluence
        assert len(mtf_confluence._mtf_cache) == 0
