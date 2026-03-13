"""
News sentiment filter using the Alpaca News API.

Checks recent headlines and summaries for bearish/bullish keywords
to block trades that conflict with prevailing news sentiment.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import config

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

BEARISH_KEYWORDS: set[str] = {
    "bankruptcy", "fraud", "halt", "halted", "delisted",
    "sec investigation", "indictment", "default",
    "recall", "subpoena",
}

BULLISH_KEYWORDS: set[str] = {
    "upgrade", "beat", "beats", "above expectations", "record",
    "raises guidance", "raises forecast", "partnership", "acquisition",
    "buyback", "dividend", "strong", "growth", "surge", "rally", "wins",
    "expands", "approved", "fda approval", "breakthrough", "outperform",
    "buy rating", "price target raised", "guidance raised",
}


# ---------------------------------------------------------------------------
# NewsFilter
# ---------------------------------------------------------------------------

class NewsFilter:
    """Lazy-initialised Alpaca news sentiment checker with daily cache."""

    def __init__(self) -> None:
        self._client: NewsClient | None = None
        self._cache: dict[str, tuple[str, float]] = {}  # symbol -> (sentiment, timestamp)

    # -- internal helpers ----------------------------------------------------

    def _get_client(self) -> NewsClient:
        if self._client is None:
            self._client = NewsClient(
                api_key=config.API_KEY,
                secret_key=config.API_SECRET,
            )
        return self._client

    @staticmethod
    def _count_keywords(text: str) -> tuple[int, int]:
        """Return (bearish_count, bullish_count) for *text*."""
        lowered = text.lower()
        bearish = sum(1 for kw in BEARISH_KEYWORDS if kw in lowered)
        bullish = sum(1 for kw in BULLISH_KEYWORDS if kw in lowered)
        return bearish, bullish

    @staticmethod
    def _cache_is_today(ts: float) -> bool:
        """Return True if *ts* (epoch) falls on today's date (UTC)."""
        cached_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        return cached_date == datetime.now(tz=timezone.utc).date()

    # -- public API ----------------------------------------------------------

    def get_sentiment(self, symbol: str) -> str:
        """Return ``"BEARISH"``, ``"BULLISH"``, or ``"NEUTRAL"`` for *symbol*.

        Results are cached for the current trading day.  Any error is treated
        as ``"NEUTRAL"`` (fail-open).
        """
        if not config.NEWS_FILTER_ENABLED:
            return "NEUTRAL"

        # Check cache
        if symbol in self._cache:
            sentiment, ts = self._cache[symbol]
            if self._cache_is_today(ts):
                return sentiment

        try:
            now = datetime.now(tz=timezone.utc)
            start = now - timedelta(hours=config.NEWS_LOOKBACK_HOURS)

            request = NewsRequest(
                symbols=symbol,
                start=start,
                limit=10,
            )

            news = self._get_client().get_news(request)
            articles = news.news if news and hasattr(news, "news") else []

            total_bearish = 0
            total_bullish = 0

            for article in articles:
                text_parts: list[str] = []
                if hasattr(article, "headline") and article.headline:
                    text_parts.append(article.headline)
                if hasattr(article, "summary") and article.summary:
                    text_parts.append(article.summary)
                combined = " ".join(text_parts)

                b, bu = self._count_keywords(combined)
                total_bearish += b
                total_bullish += bu

            if total_bearish > total_bullish + 1:
                sentiment = "BEARISH"
            elif total_bullish > total_bearish + 1:
                sentiment = "BULLISH"
            else:
                sentiment = "NEUTRAL"

            self._cache[symbol] = (sentiment, time.time())
            logger.info(
                "News sentiment for %s: %s  (bearish=%d, bullish=%d, articles=%d)",
                symbol, sentiment, total_bearish, total_bullish, len(articles),
            )
            return sentiment

        except Exception:
            logger.exception("News sentiment check failed for %s — defaulting to NEUTRAL", symbol)
            return "NEUTRAL"

    def should_block(self, symbol: str, side: str) -> tuple[bool, str]:
        """Decide whether a trade should be blocked based on news sentiment.

        Returns ``(should_block, reason)`` where *reason* is a human-readable
        explanation (empty string when not blocked).
        """
        sentiment = self.get_sentiment(symbol)

        if side.lower() == "buy" and sentiment == "BEARISH":
            reason = f"Blocking BUY on {symbol}: recent news sentiment is BEARISH"
            logger.warning(reason)
            return True, reason

        if side.lower() == "sell" and sentiment == "BULLISH":
            reason = f"Blocking SELL on {symbol}: recent news sentiment is BULLISH"
            logger.warning(reason)
            return True, reason

        return False, ""

    def clear_cache(self) -> None:
        """Drop all cached sentiment results."""
        self._cache.clear()
        logger.info("News sentiment cache cleared")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

news_filter = NewsFilter()
