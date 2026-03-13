"""Tests for news_filter.py — news sentiment filtering."""

import time
from unittest.mock import patch, MagicMock

import pytest

from conftest import ET


class TestNewsFilterDisabled:
    def test_disabled_returns_neutral(self, override_config):
        """When NEWS_FILTER_ENABLED=False, get_sentiment returns NEUTRAL."""
        with override_config(NEWS_FILTER_ENABLED=False):
            from news_filter import NewsFilter
            nf = NewsFilter()
            assert nf.get_sentiment("AAPL") == "NEUTRAL"


class TestSentimentScoring:
    def test_bearish_blocks_buy(self, override_config):
        """Bearish sentiment blocks buy signals."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("BEARISH", time.time())

            blocked, reason = nf.should_block("AAPL", "buy")
            assert blocked is True
            assert "BEARISH" in reason

    def test_bullish_blocks_sell(self, override_config):
        """Bullish sentiment blocks sell signals."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("BULLISH", time.time())

            blocked, reason = nf.should_block("AAPL", "sell")
            assert blocked is True
            assert "BULLISH" in reason

    def test_neutral_allows_all(self, override_config):
        """Neutral sentiment allows both buy and sell."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("NEUTRAL", time.time())

            blocked_buy, _ = nf.should_block("AAPL", "buy")
            assert blocked_buy is False

            blocked_sell, _ = nf.should_block("AAPL", "sell")
            assert blocked_sell is False

    def test_bearish_allows_sell(self, override_config):
        """Bearish sentiment does NOT block sell signals."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("BEARISH", time.time())

            blocked, _ = nf.should_block("AAPL", "sell")
            assert blocked is False

    def test_bullish_allows_buy(self, override_config):
        """Bullish sentiment does NOT block buy signals."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("BULLISH", time.time())

            blocked, _ = nf.should_block("AAPL", "buy")
            assert blocked is False


class TestKeywordScoring:
    def test_bearish_keywords_counted(self):
        """_count_keywords counts bearish keywords (V5: catastrophic-only keywords)."""
        from news_filter import NewsFilter
        bearish, bullish = NewsFilter._count_keywords(
            "Company halted after fraud investigation and bankruptcy filing"
        )
        assert bearish >= 3
        assert bullish == 0

    def test_bullish_keywords_counted(self):
        """_count_keywords counts bullish keywords."""
        from news_filter import NewsFilter
        bearish, bullish = NewsFilter._count_keywords(
            "Company beats expectations with record growth and surge"
        )
        assert bullish >= 3
        assert bearish == 0


class TestNewsCache:
    def test_cache_hit(self, override_config):
        """Cached result returned without additional API call."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("NEUTRAL", time.time())

            mock_client = MagicMock()
            nf._client = mock_client

            sentiment = nf.get_sentiment("AAPL")
            assert sentiment == "NEUTRAL"
            mock_client.get_news.assert_not_called()

    def test_clear_cache(self, override_config):
        """clear_cache empties the cache dict."""
        with override_config(NEWS_FILTER_ENABLED=True):
            from news_filter import NewsFilter
            nf = NewsFilter()
            nf._cache["AAPL"] = ("BEARISH", time.time())
            nf._cache["MSFT"] = ("BULLISH", time.time())

            nf.clear_cache()
            assert len(nf._cache) == 0


class TestNewsFilterErrors:
    def test_error_returns_neutral(self, override_config):
        """API errors fail-open to NEUTRAL."""
        with override_config(NEWS_FILTER_ENABLED=True, NEWS_LOOKBACK_HOURS=24):
            from news_filter import NewsFilter
            nf = NewsFilter()

            mock_client = MagicMock()
            mock_client.get_news.side_effect = Exception("API error")
            nf._client = mock_client

            sentiment = nf.get_sentiment("AAPL")
            assert sentiment == "NEUTRAL"
