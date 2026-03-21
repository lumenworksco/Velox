"""COMP-011: Real-time earnings call transcription scoring.

Framework for processing earnings call transcripts and scoring them on
multiple dimensions:
    - Overall sentiment (positive / negative / neutral)
    - Uncertainty level (hedging language, qualifiers)
    - Forward guidance quality (specificity of outlook)
    - Key metric mentions (revenue, margins, EPS, guidance)

Uses keyword-based scoring by default with an optional transformer
upgrade path (HuggingFace pipeline).

Usage:
    scorer = EarningsCallScorer()
    result = scorer.score_transcript(transcript_text)
    print(result.sentiment_score)   # [-1, 1]
    print(result.uncertainty_score) # [0, 1]
    print(result.guidance_quality)  # [0, 1]
    print(result.key_metrics)       # dict of mentioned metrics

Dependencies: numpy (required), transformers (optional).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional transformer backend
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
try:
    from transformers import pipeline as hf_pipeline

    _HAS_TRANSFORMERS = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS: List[str] = [
    "strong", "growth", "exceeded", "beat", "record", "robust",
    "accelerating", "improvement", "outperformed", "upside", "tailwind",
    "optimistic", "confident", "momentum", "expanding", "solid",
    "resilient", "favorable", "strengthen", "above expectations",
    "higher than expected", "well positioned", "ahead of schedule",
    "double digit", "margin expansion", "market share gains",
]

NEGATIVE_KEYWORDS: List[str] = [
    "weak", "decline", "miss", "missed", "below", "headwind",
    "challenging", "deteriorating", "disappointed", "downside", "pressure",
    "cautious", "concerned", "slowdown", "contraction", "soft",
    "underperformed", "unfavorable", "weaken", "below expectations",
    "lower than expected", "macro uncertainty", "behind schedule",
    "margin compression", "market share loss", "restructuring",
]

UNCERTAINTY_KEYWORDS: List[str] = [
    "uncertain", "might", "could", "possibly", "perhaps", "unclear",
    "depends on", "subject to", "volatile", "unpredictable", "risk",
    "contingent", "if conditions", "hard to predict", "visibility",
    "range of outcomes", "too early to tell", "we'll see", "evolving",
    "fluid situation", "monitor closely", "wait and see",
]

GUIDANCE_SPECIFIC_KEYWORDS: List[str] = [
    "expect revenue of", "guidance range", "target of", "forecast",
    "we anticipate", "full year outlook", "raising guidance",
    "reaffirming guidance", "lowering guidance", "quarterly guidance",
    "eps of", "revenue between", "margin target", "capex of",
]

GUIDANCE_VAGUE_KEYWORDS: List[str] = [
    "broadly in line", "approximately", "roughly", "in the ballpark",
    "trending towards", "directionally", "order of magnitude",
    "not providing specific", "withdrawing guidance", "suspending guidance",
]

KEY_METRIC_PATTERNS: Dict[str, str] = {
    "revenue": r"(?:revenue|sales|top\s*line)[\s:]*\$?([\d,.]+\s*(?:billion|million|B|M)?)",
    "eps": r"(?:eps|earnings\s+per\s+share)[\s:]*\$?([\d,.]+)",
    "margin_gross": r"(?:gross\s+margin)[\s:]*(\d+\.?\d*)\s*%?",
    "margin_operating": r"(?:operating\s+margin)[\s:]*(\d+\.?\d*)\s*%?",
    "margin_net": r"(?:net\s+margin)[\s:]*(\d+\.?\d*)\s*%?",
    "guidance_revenue": r"(?:revenue\s+(?:guidance|outlook|forecast))[\s:]*\$?([\d,.]+)",
    "guidance_eps": r"(?:eps\s+(?:guidance|outlook|forecast))[\s:]*\$?([\d,.]+)",
    "free_cash_flow": r"(?:free\s+cash\s+flow|fcf)[\s:]*\$?([\d,.]+\s*(?:billion|million|B|M)?)",
    "capex": r"(?:capex|capital\s+expenditure)[\s:]*\$?([\d,.]+\s*(?:billion|million|B|M)?)",
    "buyback": r"(?:buyback|repurchase)[\s:]*\$?([\d,.]+\s*(?:billion|million|B|M)?)",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EarningsScore:
    """Result of scoring an earnings call transcript."""

    sentiment_score: float = 0.0       # [-1, 1] negative to positive
    uncertainty_score: float = 0.0     # [0, 1] low to high uncertainty
    guidance_quality: float = 0.0      # [0, 1] vague to specific
    positive_count: int = 0
    negative_count: int = 0
    uncertainty_count: int = 0
    key_metrics: Dict[str, str] = field(default_factory=dict)
    guidance_direction: str = "neutral"  # "raised", "maintained", "lowered", "neutral"
    word_count: int = 0
    backend: str = "keyword"

    def to_signal(self) -> float:
        """Convert to a single trading signal in [-1, 1].

        Combines sentiment, guidance quality, and inverted uncertainty.
        """
        signal = (
            0.5 * self.sentiment_score
            + 0.3 * self.guidance_quality * (1.0 if self.guidance_direction != "lowered" else -1.0)
            - 0.2 * self.uncertainty_score
        )
        return float(np.clip(signal, -1.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sentiment_score": self.sentiment_score,
            "uncertainty_score": self.uncertainty_score,
            "guidance_quality": self.guidance_quality,
            "guidance_direction": self.guidance_direction,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "uncertainty_count": self.uncertainty_count,
            "key_metrics": self.key_metrics,
            "word_count": self.word_count,
            "signal": self.to_signal(),
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Keyword-based scorer
# ---------------------------------------------------------------------------


class KeywordScorer:
    """Score transcript text using keyword frequency analysis."""

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
        uncertainty_words: Optional[List[str]] = None,
    ) -> None:
        self.positive = [w.lower() for w in (positive_words or POSITIVE_KEYWORDS)]
        self.negative = [w.lower() for w in (negative_words or NEGATIVE_KEYWORDS)]
        self.uncertainty = [w.lower() for w in (uncertainty_words or UNCERTAINTY_KEYWORDS)]

    def count_matches(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text."""
        text_lower = text.lower()
        count = 0
        for kw in keywords:
            count += text_lower.count(kw)
        return count

    def score_sentiment(self, text: str) -> Tuple[float, int, int]:
        """Score sentiment and return (score, pos_count, neg_count)."""
        pos = self.count_matches(text, self.positive)
        neg = self.count_matches(text, self.negative)
        total = pos + neg
        if total == 0:
            return 0.0, 0, 0
        score = (pos - neg) / total
        return float(score), pos, neg

    def score_uncertainty(self, text: str) -> Tuple[float, int]:
        """Score uncertainty level. Returns (score, count)."""
        count = self.count_matches(text, self.uncertainty)
        words = len(text.split())
        if words == 0:
            return 0.0, 0
        # Normalize by word count, cap at 1.0
        density = count / (words / 100.0)  # per 100 words
        score = min(1.0, density / 5.0)  # 5+ per 100 words = max uncertainty
        return float(score), count

    def score_guidance(self, text: str) -> Tuple[float, str]:
        """Score guidance quality and direction.

        Returns (quality_score, direction).
        """
        text_lower = text.lower()
        specific = sum(1 for kw in GUIDANCE_SPECIFIC_KEYWORDS if kw in text_lower)
        vague = sum(1 for kw in GUIDANCE_VAGUE_KEYWORDS if kw in text_lower)

        total = specific + vague
        if total == 0:
            return 0.5, "neutral"

        quality = specific / total

        # Determine direction
        if "raising guidance" in text_lower or "raised guidance" in text_lower:
            direction = "raised"
        elif "lowering guidance" in text_lower or "lowered guidance" in text_lower:
            direction = "lowered"
        elif "reaffirming guidance" in text_lower or "maintaining guidance" in text_lower:
            direction = "maintained"
        elif "withdrawing guidance" in text_lower or "suspending guidance" in text_lower:
            direction = "withdrawn"
        else:
            direction = "neutral"

        return float(quality), direction


def extract_key_metrics(text: str) -> Dict[str, str]:
    """Extract key financial metrics from transcript text.

    Returns a dict mapping metric names to their extracted values.
    """
    metrics: Dict[str, str] = {}
    for name, pattern in KEY_METRIC_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[name] = match.group(1).strip()
    return metrics


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class EarningsCallScorer:
    """Score earnings call transcripts for trading signals.

    Parameters
    ----------
    backend : str
        ``"auto"`` uses transformer if available, else keyword.
        ``"keyword"`` forces keyword scoring.
        ``"transformer"`` forces transformer (raises if unavailable).
    transformer_model : str
        HuggingFace model for sentiment analysis.
    """

    def __init__(
        self,
        backend: str = "auto",
        transformer_model: str = "ProsusAI/finbert",
    ) -> None:
        self._keyword_scorer = KeywordScorer()
        self._transformer_pipeline = None
        self.backend = backend

        if backend == "transformer" and not _HAS_TRANSFORMERS:
            raise ImportError("transformers package required for transformer backend")

        if backend in ("auto", "transformer") and _HAS_TRANSFORMERS:
            try:
                self._transformer_pipeline = hf_pipeline(
                    "sentiment-analysis", model=transformer_model,
                )
                self.backend = "transformer"
                logger.info("EarningsCallScorer using transformer backend: %s", transformer_model)
            except Exception as e:
                logger.warning("Failed to load transformer model: %s. Using keyword fallback.", e)
                self.backend = "keyword"
        else:
            self.backend = "keyword"
            logger.info("EarningsCallScorer using keyword backend.")

    def score_transcript(self, text: str) -> EarningsScore:
        """Score an earnings call transcript.

        Parameters
        ----------
        text : str
            Full or partial transcript text.

        Returns
        -------
        EarningsScore
            Multi-dimensional scoring result.
        """
        if not text or not text.strip():
            logger.warning("Empty transcript provided.")
            return EarningsScore(backend=self.backend)

        try:
            word_count = len(text.split())

            # Keyword-based scores (always computed)
            sent_score, pos_count, neg_count = self._keyword_scorer.score_sentiment(text)
            unc_score, unc_count = self._keyword_scorer.score_uncertainty(text)
            guid_quality, guid_direction = self._keyword_scorer.score_guidance(text)
            metrics = extract_key_metrics(text)

            # Override sentiment with transformer if available
            if self._transformer_pipeline is not None:
                sent_score = self._transformer_sentiment(text)

            result = EarningsScore(
                sentiment_score=sent_score,
                uncertainty_score=unc_score,
                guidance_quality=guid_quality,
                positive_count=pos_count,
                negative_count=neg_count,
                uncertainty_count=unc_count,
                key_metrics=metrics,
                guidance_direction=guid_direction,
                word_count=word_count,
                backend=self.backend,
            )

            logger.info(
                "Earnings score: sentiment=%.2f, uncertainty=%.2f, "
                "guidance=%.2f (%s), metrics=%d, words=%d",
                result.sentiment_score, result.uncertainty_score,
                result.guidance_quality, result.guidance_direction,
                len(metrics), word_count,
            )
            return result

        except Exception as e:
            logger.error("Transcript scoring failed: %s — returning neutral", e)
            return EarningsScore(backend=self.backend)

    def score_section(
        self,
        text: str,
        section: str = "qa",
    ) -> EarningsScore:
        """Score a specific section of the earnings call.

        Parameters
        ----------
        text : str
            Section text (e.g., Q&A portion only).
        section : str
            Section identifier for logging.

        Returns
        -------
        EarningsScore
        """
        logger.debug("Scoring section: %s (%d chars)", section, len(text))
        return self.score_transcript(text)

    def _transformer_sentiment(self, text: str) -> float:
        """Compute sentiment using transformer pipeline.

        Handles long texts by chunking into 512-token segments.
        Returns score in [-1, 1].
        """
        try:
            # Chunk text for transformer (rough 512-token limit)
            max_chars = 2000
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            chunks = chunks[:10]  # Cap at 10 chunks

            scores = []
            for chunk in chunks:
                result = self._transformer_pipeline(chunk[:512])[0]
                label = result["label"].lower()
                confidence = result["score"]
                if label == "positive":
                    scores.append(confidence)
                elif label == "negative":
                    scores.append(-confidence)
                else:
                    scores.append(0.0)

            return float(np.mean(scores)) if scores else 0.0

        except Exception as e:
            logger.warning("Transformer sentiment failed: %s", e)
            return 0.0

    def batch_score(
        self, transcripts: Dict[str, str],
    ) -> Dict[str, EarningsScore]:
        """Score multiple transcripts.

        Parameters
        ----------
        transcripts : dict
            ``{ticker: transcript_text}`` mapping.

        Returns
        -------
        dict
            ``{ticker: EarningsScore}`` mapping.
        """
        results: Dict[str, EarningsScore] = {}
        for ticker, text in transcripts.items():
            results[ticker] = self.score_transcript(text)
        logger.info("Batch scored %d transcripts.", len(results))
        return results
