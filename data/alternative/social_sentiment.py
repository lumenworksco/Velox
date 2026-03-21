"""COMP-005: FinBERT-based sentiment from social media.

Framework for processing Twitter/Reddit/StockTwits mentions. Uses FinBERT
(transformers library) for sentiment scoring when available, with a
keyword-based fallback for environments without GPU/transformers.

Aggregates per-symbol sentiment scores with decay weighting (recent
mentions weighted more heavily). Produces a normalized sentiment signal
suitable for integration with trading strategies.

Fail-open: returns neutral sentiment if no data or models are available.

Usage::

    analyzer = SocialSentimentAnalyzer()
    analyzer.ingest_mentions("AAPL", [
        SocialMention(text="AAPL earnings beat expectations!", source="twitter", ...),
        SocialMention(text="Apple guidance looks weak", source="reddit", ...),
    ])
    result = analyzer.get_sentiment("AAPL")
    # SentimentResult(symbol="AAPL", score=0.3, label="bullish", ...)
"""

import logging
import math
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentiment score thresholds
BULLISH_THRESHOLD = 0.15
BEARISH_THRESHOLD = -0.15

# Time decay half-life in hours (mentions older than this get half weight)
DECAY_HALF_LIFE_HOURS = 4.0

# Minimum mentions required for a valid sentiment signal
MIN_MENTIONS = 5

# Maximum mentions to keep per symbol (rolling window)
MAX_MENTIONS_PER_SYMBOL = 1000

# Source credibility weights (higher = more trusted)
SOURCE_WEIGHTS = {
    "twitter": 0.6,
    "reddit": 0.8,
    "stocktwits": 0.5,
    "news": 1.0,
    "sec_filing": 1.0,
    "analyst": 0.9,
    "unknown": 0.3,
}

# FinBERT model identifier
FINBERT_MODEL = "ProsusAI/finbert"

# Keyword-based sentiment lexicon (fallback when FinBERT unavailable)
_BULLISH_KEYWORDS = {
    "buy", "long", "bullish", "moon", "rocket", "calls", "breakout",
    "upgrade", "beat", "strong", "growth", "rally", "surge", "soar",
    "outperform", "upside", "green", "accumulate", "oversold", "dip",
    "undervalued", "squeeze", "gamma", "tendies", "diamond", "hold",
    "support", "bounce", "recovery", "positive", "exceeds", "blowout",
}

_BEARISH_KEYWORDS = {
    "sell", "short", "bearish", "crash", "dump", "puts", "breakdown",
    "downgrade", "miss", "weak", "decline", "drop", "plunge", "tank",
    "underperform", "downside", "red", "distribute", "overbought", "top",
    "overvalued", "bubble", "bagholder", "loss", "resistance", "fade",
    "recession", "negative", "disappoints", "warning", "cut", "layoff",
}

_NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "don't", "doesn't", "isn't", "won't", "can't"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SocialMention:
    """A single social media mention of a symbol."""
    text: str
    source: str                         # "twitter", "reddit", "stocktwits", etc.
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = ""
    url: str = ""
    engagement: int = 0                 # likes + retweets + comments
    raw_sentiment: Optional[float] = None  # Pre-scored sentiment if available


@dataclass
class ScoredMention:
    """A mention with computed sentiment score."""
    mention: SocialMention
    score: float                        # -1.0 to +1.0
    confidence: float                   # 0.0 to 1.0
    method: str                         # "finbert" or "keyword"
    weighted_score: float = 0.0         # After applying source weight + decay


@dataclass
class SentimentResult:
    """Aggregate sentiment for a symbol."""
    symbol: str
    score: float                        # -1.0 (bearish) to +1.0 (bullish)
    label: SentimentLabel
    confidence: float                   # 0.0 to 1.0
    mention_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_engagement: float
    volume_zscore: float                # Mention volume vs. recent average
    source_breakdown: Dict[str, float] = field(default_factory=dict)
    top_mentions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# Neutral result returned on failure
_NEUTRAL_RESULT = SentimentResult(
    symbol="",
    score=0.0,
    label=SentimentLabel.NEUTRAL,
    confidence=0.0,
    mention_count=0,
    bullish_count=0,
    bearish_count=0,
    neutral_count=0,
    avg_engagement=0.0,
    volume_zscore=0.0,
)


# ---------------------------------------------------------------------------
# FinBERT scorer (lazy-loaded)
# ---------------------------------------------------------------------------

class _FinBERTScorer:
    """Lazy-loaded FinBERT sentiment model wrapper."""

    def __init__(self):
        self._pipeline = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if transformers/torch are installed."""
        if self._available is not None:
            return self._available
        try:
            import transformers  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.info("transformers not installed — using keyword-based sentiment fallback")
        return self._available

    def load(self) -> bool:
        """Load the FinBERT model. Returns True on success."""
        if self._pipeline is not None:
            return True
        if not self.is_available():
            return False
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=FINBERT_MODEL,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
            return True
        except Exception as exc:
            logger.warning("Failed to load FinBERT: %s — using keyword fallback", exc)
            self._available = False
            return False

    def score(self, text: str) -> Tuple[float, float]:
        """Score text sentiment. Returns (score, confidence).

        Score: -1.0 (negative) to +1.0 (positive).
        Confidence: 0.0 to 1.0.
        """
        if not self._pipeline:
            if not self.load():
                return 0.0, 0.0

        try:
            result = self._pipeline(text[:512])[0]
            label = result["label"].lower()
            conf = result["score"]

            if label == "positive":
                return conf, conf
            elif label == "negative":
                return -conf, conf
            else:
                return 0.0, conf
        except Exception as exc:
            logger.debug("FinBERT scoring failed: %s", exc)
            return 0.0, 0.0


# Global singleton (lazy)
_finbert = _FinBERTScorer()


# ---------------------------------------------------------------------------
# Keyword-based scorer (fallback)
# ---------------------------------------------------------------------------

def _keyword_score(text: str) -> Tuple[float, float]:
    """Score text sentiment using keyword matching.

    Returns (score, confidence) where score is -1.0 to +1.0 and
    confidence is based on the proportion of sentiment words found.
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0, 0.0

    bullish_hits = 0
    bearish_hits = 0
    negation_active = False

    for token in tokens:
        if token in _NEGATION_WORDS:
            negation_active = True
            continue

        if token in _BULLISH_KEYWORDS:
            if negation_active:
                bearish_hits += 1
            else:
                bullish_hits += 1
            negation_active = False
        elif token in _BEARISH_KEYWORDS:
            if negation_active:
                bullish_hits += 1
            else:
                bearish_hits += 1
            negation_active = False
        else:
            negation_active = False

    total_hits = bullish_hits + bearish_hits
    if total_hits == 0:
        return 0.0, 0.0

    score = (bullish_hits - bearish_hits) / total_hits
    # Confidence based on hit density
    confidence = min(total_hits / max(len(tokens) * 0.1, 1), 1.0)

    return score, confidence


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SocialSentimentAnalyzer:
    """Aggregate social media sentiment per symbol.

    Ingests mentions from any source, scores them using FinBERT (if
    available) or keyword matching, and produces a time-decayed
    aggregate sentiment signal.
    """

    def __init__(self, use_finbert: bool = True):
        self._use_finbert = use_finbert
        self._mentions: Dict[str, List[ScoredMention]] = defaultdict(list)
        self._volume_history: Dict[str, List[int]] = defaultdict(list)  # hourly counts
        self._last_cleanup: float = 0.0

    # ------------------------------------------------------------------
    # Mention ingestion
    # ------------------------------------------------------------------

    def ingest_mentions(
        self,
        symbol: str,
        mentions: List[SocialMention],
    ) -> int:
        """Ingest and score a batch of social mentions for a symbol.

        Args:
            symbol: Ticker symbol.
            mentions: List of SocialMention objects.

        Returns:
            Number of mentions successfully scored.
        """
        scored_count = 0
        for mention in mentions:
            try:
                scored = self._score_mention(mention)
                if scored:
                    self._mentions[symbol].append(scored)
                    scored_count += 1
            except Exception as exc:
                logger.debug("Failed to score mention for %s: %s", symbol, exc)

        # Trim to max window
        if len(self._mentions[symbol]) > MAX_MENTIONS_PER_SYMBOL:
            self._mentions[symbol] = self._mentions[symbol][-MAX_MENTIONS_PER_SYMBOL:]

        # Update volume history
        self._volume_history[symbol].append(len(mentions))
        if len(self._volume_history[symbol]) > 168:  # 1 week of hourly counts
            self._volume_history[symbol] = self._volume_history[symbol][-168:]

        logger.debug("Ingested %d/%d mentions for %s", scored_count, len(mentions), symbol)
        return scored_count

    def _score_mention(self, mention: SocialMention) -> Optional[ScoredMention]:
        """Score a single mention using FinBERT or keyword fallback."""
        if mention.raw_sentiment is not None:
            # Pre-scored (e.g., from a data provider)
            return ScoredMention(
                mention=mention,
                score=mention.raw_sentiment,
                confidence=0.7,
                method="pre_scored",
            )

        # Try FinBERT first
        if self._use_finbert and _finbert.is_available():
            score, confidence = _finbert.score(mention.text)
            method = "finbert"
        else:
            score, confidence = _keyword_score(mention.text)
            method = "keyword"

        if confidence < 0.01:
            return ScoredMention(
                mention=mention,
                score=0.0,
                confidence=0.0,
                method=method,
            )

        return ScoredMention(
            mention=mention,
            score=score,
            confidence=confidence,
            method=method,
        )

    # ------------------------------------------------------------------
    # Decay and weighting
    # ------------------------------------------------------------------

    def _compute_decay_weight(self, mention_time: datetime, now: datetime) -> float:
        """Exponential time decay weight. Half-life = DECAY_HALF_LIFE_HOURS."""
        age_hours = (now - mention_time).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0
        return math.exp(-0.693 * age_hours / DECAY_HALF_LIFE_HOURS)

    def _compute_weighted_score(self, scored: ScoredMention, now: datetime) -> float:
        """Compute final weighted score for a mention."""
        source_weight = SOURCE_WEIGHTS.get(scored.mention.source, 0.3)
        decay = self._compute_decay_weight(scored.mention.timestamp, now)
        engagement_boost = 1.0 + min(scored.mention.engagement / 1000.0, 1.0)

        return scored.score * scored.confidence * source_weight * decay * engagement_boost

    # ------------------------------------------------------------------
    # Sentiment aggregation
    # ------------------------------------------------------------------

    def get_sentiment(
        self,
        symbol: str,
        window_hours: float = 24.0,
    ) -> SentimentResult:
        """Compute aggregate sentiment for a symbol.

        Args:
            symbol: Ticker symbol.
            window_hours: Only consider mentions within this time window.

        Returns:
            SentimentResult with aggregate scores.
        """
        mentions = self._mentions.get(symbol, [])
        if not mentions:
            result = SentimentResult(
                symbol=symbol, score=0.0, label=SentimentLabel.NEUTRAL,
                confidence=0.0, mention_count=0, bullish_count=0,
                bearish_count=0, neutral_count=0, avg_engagement=0.0,
                volume_zscore=0.0,
            )
            return result

        now = datetime.now()
        cutoff = now - timedelta(hours=window_hours)

        # Filter to window
        recent = [m for m in mentions if m.mention.timestamp >= cutoff]
        if not recent:
            return SentimentResult(
                symbol=symbol, score=0.0, label=SentimentLabel.NEUTRAL,
                confidence=0.0, mention_count=0, bullish_count=0,
                bearish_count=0, neutral_count=0, avg_engagement=0.0,
                volume_zscore=0.0,
            )

        # Compute weighted scores
        weighted_scores: List[float] = []
        total_weight = 0.0
        bullish = 0
        bearish = 0
        neutral = 0
        source_scores: Dict[str, List[float]] = defaultdict(list)
        engagements: List[int] = []

        for scored in recent:
            w_score = self._compute_weighted_score(scored, now)
            scored.weighted_score = w_score
            weighted_scores.append(w_score)
            weight = abs(w_score) if w_score != 0 else 0.01
            total_weight += weight

            if scored.score > BULLISH_THRESHOLD:
                bullish += 1
            elif scored.score < BEARISH_THRESHOLD:
                bearish += 1
            else:
                neutral += 1

            source_scores[scored.mention.source].append(scored.score)
            engagements.append(scored.mention.engagement)

        # Aggregate score (weighted mean)
        if total_weight > 0:
            agg_score = sum(weighted_scores) / len(weighted_scores)
        else:
            agg_score = 0.0

        agg_score = max(-1.0, min(1.0, agg_score))

        # Confidence based on volume and agreement
        volume_factor = min(len(recent) / MIN_MENTIONS, 1.0)
        if bullish + bearish > 0:
            agreement = abs(bullish - bearish) / (bullish + bearish)
        else:
            agreement = 0.0
        confidence = volume_factor * 0.5 + agreement * 0.5

        # Label
        if agg_score > BULLISH_THRESHOLD:
            label = SentimentLabel.BULLISH
        elif agg_score < BEARISH_THRESHOLD:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        # Volume z-score
        volume_zscore = self._compute_volume_zscore(symbol, len(recent))

        # Source breakdown
        source_breakdown = {
            source: sum(scores) / len(scores) if scores else 0.0
            for source, scores in source_scores.items()
        }

        # Top mentions by absolute weighted score
        sorted_mentions = sorted(recent, key=lambda m: abs(m.weighted_score), reverse=True)
        top_texts = [m.mention.text[:100] for m in sorted_mentions[:5]]

        return SentimentResult(
            symbol=symbol,
            score=agg_score,
            label=label,
            confidence=confidence,
            mention_count=len(recent),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            avg_engagement=sum(engagements) / len(engagements) if engagements else 0.0,
            volume_zscore=volume_zscore,
            source_breakdown=source_breakdown,
            top_mentions=top_texts,
            timestamp=now,
        )

    def _compute_volume_zscore(self, symbol: str, current_count: int) -> float:
        """Compute z-score of current mention volume vs. recent history."""
        history = self._volume_history.get(symbol, [])
        if len(history) < 5:
            return 0.0

        import numpy as np
        arr = np.array(history[-24:], dtype=float)  # Last 24 periods
        mean = arr.mean()
        std = arr.std()
        if std < 1e-6:
            return 0.0
        return float((current_count - mean) / std)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def screen_symbols(
        self,
        symbols: List[str],
        min_confidence: float = 0.3,
        window_hours: float = 24.0,
    ) -> List[SentimentResult]:
        """Screen symbols for strong sentiment signals.

        Returns only symbols with confidence above threshold, sorted
        by absolute score descending.
        """
        results: List[SentimentResult] = []
        for symbol in symbols:
            try:
                result = self.get_sentiment(symbol, window_hours)
                if result.confidence >= min_confidence and result.mention_count >= MIN_MENTIONS:
                    results.append(result)
            except Exception as exc:
                logger.debug("Sentiment screening failed for %s: %s", symbol, exc)

        results.sort(key=lambda r: abs(r.score), reverse=True)
        return results

    def get_trending(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get symbols with highest mention volume z-scores (trending).

        Returns list of (symbol, zscore) tuples, sorted by zscore descending.
        """
        trending: List[Tuple[str, float]] = []
        for symbol in self._mentions:
            recent_count = len(self._mentions[symbol])
            zscore = self._compute_volume_zscore(symbol, recent_count)
            if zscore > 1.0:
                trending.append((symbol, zscore))

        trending.sort(key=lambda x: x[1], reverse=True)
        return trending[:top_n]

    def clear_data(self, symbol: Optional[str] = None) -> None:
        """Clear mention data for one or all symbols."""
        if symbol:
            self._mentions.pop(symbol, None)
            self._volume_history.pop(symbol, None)
        else:
            self._mentions.clear()
            self._volume_history.clear()
