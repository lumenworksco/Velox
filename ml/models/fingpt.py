"""EDGE-007: FinGPT — Financial Text Sentiment Analysis.

Wraps a language model for sentiment analysis of earnings calls, SEC filings,
and general financial text.  Supports three backends in priority order:

  1. HuggingFace transformers (FinBERT / ProsusAI/finbert)
  2. OpenAI API (gpt-3.5-turbo or gpt-4)
  3. Keyword-based fallback (always available)

All ML library imports are conditional — the bot runs without them.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
_HAS_OPENAI = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    import openai  # type: ignore[import-untyped]
    _HAS_OPENAI = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Keyword lexicon for fallback sentiment
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset([
    "beat", "exceeded", "strong", "growth", "record", "upside", "improved",
    "outperform", "bullish", "accelerat", "surpass", "positive", "upgrade",
    "raised", "guidance", "momentum", "robust", "profit", "gain", "optimistic",
    "tailwind", "expand", "dividend", "buyback", "innovation", "breakthrough",
])

_NEGATIVE_WORDS = frozenset([
    "miss", "missed", "weak", "decline", "loss", "headwind", "downgrade",
    "restructur", "impair", "writedown", "bearish", "slowdown", "risk",
    "litigation", "default", "bankruptcy", "warning", "disappoint", "below",
    "shortfall", "deteriorat", "layoff", "restat", "concern", "volatil",
])


@dataclass
class FinGPTConfig:
    """Configuration for the FinGPT model."""

    backend: str = "auto"  # "transformers", "openai", "keyword", or "auto"
    model_name: str = "ProsusAI/finbert"  # HuggingFace model ID
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None
    max_length: int = 512  # max token length for transformer
    batch_size: int = 16
    device: str = "cpu"
    cache_results: bool = True


class FinGPT:
    """Financial text sentiment analysis engine.

    Provides a unified interface across transformer, API, and keyword backends.
    Follows the common model interface with fit() / predict() / score().

    Parameters
    ----------
    config : FinGPTConfig, optional
        Model configuration.  Defaults are sensible for most use-cases.
    """

    def __init__(self, config: Optional[FinGPTConfig] = None):
        self.config = config or FinGPTConfig()
        self._backend: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._cache: Dict[int, float] = {}
        self._fitted = False
        self._resolve_backend()

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> None:
        """Select the best available backend."""
        choice = self.config.backend
        if choice == "auto":
            if _HAS_TRANSFORMERS:
                choice = "transformers"
            elif _HAS_OPENAI and (self.config.openai_api_key or os.getenv("OPENAI_API_KEY")):
                choice = "openai"
            else:
                choice = "keyword"

        if choice == "transformers" and not _HAS_TRANSFORMERS:
            logger.warning("transformers not available, falling back to keyword backend")
            choice = "keyword"
        if choice == "openai" and not _HAS_OPENAI:
            logger.warning("openai not available, falling back to keyword backend")
            choice = "keyword"

        self._backend = choice
        logger.info("FinGPT using backend: %s", self._backend)

    def _load_transformer(self) -> None:
        """Lazy-load the HuggingFace model and tokenizer."""
        if self._tokenizer is not None:
            return
        logger.info("Loading transformer model %s ...", self.config.model_name)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
            )
            self._model.to(self.config.device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Failed to load transformer model: %s — falling back to keyword", exc)
            self._backend = "keyword"

    # ------------------------------------------------------------------
    # Common interface: fit / predict / score
    # ------------------------------------------------------------------

    def fit(self, texts: List[str], labels: Optional[List[float]] = None) -> "FinGPT":
        """Fit is largely a no-op for pre-trained LMs.

        If labels are supplied we could fine-tune, but for now we simply
        validate inputs and mark the model as fitted.
        """
        if labels is not None and len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        logger.info("FinGPT fit called with %d texts (backend=%s)", len(texts), self._backend)
        self._fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return sentiment scores in [-1, 1] for each text."""
        return np.array(self.batch_analyze(texts))

    def score(self, texts: List[str], labels: np.ndarray) -> Dict[str, float]:
        """Evaluate prediction quality against ground-truth labels."""
        preds = self.predict(texts)
        mse = float(np.mean((preds - labels) ** 2))
        corr = float(np.corrcoef(preds, labels)[0, 1]) if len(preds) > 2 else 0.0
        directional = float(np.mean(np.sign(preds) == np.sign(labels)))
        return {"mse": mse, "correlation": corr, "directional_accuracy": directional}

    # ------------------------------------------------------------------
    # Primary analysis methods
    # ------------------------------------------------------------------

    def analyze_earnings_call(self, transcript: str) -> float:
        """Analyse an earnings-call transcript and return a sentiment score.

        Returns a float in [-1, 1] where -1 is very bearish and 1 is very
        bullish.  Long transcripts are chunked and averaged.
        """
        chunks = self._chunk_text(transcript, max_chars=2000)
        scores = [self._score_single(c) for c in chunks]
        return float(np.mean(scores)) if scores else 0.0

    def analyze_filing(self, text: str) -> Dict[str, Any]:
        """Analyse an SEC filing (10-K, 10-Q, 8-K, etc.).

        Returns a dict with overall sentiment plus section-level scores when
        detectable section headers are present.
        """
        overall = self.analyze_earnings_call(text)
        sections = self._split_filing_sections(text)
        section_scores = {name: self.analyze_earnings_call(body) for name, body in sections.items()}
        return {
            "overall_sentiment": overall,
            "section_sentiments": section_scores,
            "n_sections": len(section_scores),
            "backend": self._backend,
        }

    def batch_analyze(self, texts: List[str]) -> List[float]:
        """Score a list of texts, returning sentiment in [-1, 1] for each."""
        if self._backend == "transformers":
            return self._batch_transformers(texts)
        return [self._score_single(t) for t in texts]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _score_single(self, text: str) -> float:
        """Score a single text snippet."""
        if self.config.cache_results:
            key = hash(text)
            if key in self._cache:
                return self._cache[key]

        if self._backend == "transformers":
            score = self._score_transformers(text)
        elif self._backend == "openai":
            score = self._score_openai(text)
        else:
            score = self._score_keyword(text)

        if self.config.cache_results:
            self._cache[hash(text)] = score
        return score

    def _score_transformers(self, text: str) -> float:
        """Score with HuggingFace FinBERT-style model."""
        self._load_transformer()
        if self._tokenizer is None or self._model is None:
            return self._score_keyword(text)
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            # FinBERT convention: [positive, negative, neutral]
            if len(probs) == 3:
                return float(probs[0] - probs[1])
            return float(probs[0] * 2 - 1)
        except Exception as exc:
            logger.warning("Transformer inference failed: %s — using keyword fallback", exc)
            return self._score_keyword(text)

    def _batch_transformers(self, texts: List[str]) -> List[float]:
        """Batch scoring with the transformer model."""
        self._load_transformer()
        if self._tokenizer is None or self._model is None:
            return [self._score_keyword(t) for t in texts]

        scores: List[float] = []
        bs = self.config.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            try:
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for p in probs:
                    if len(p) == 3:
                        scores.append(float(p[0] - p[1]))
                    else:
                        scores.append(float(p[0] * 2 - 1))
            except Exception as exc:
                logger.warning("Batch transformer failed: %s", exc)
                scores.extend(self._score_keyword(t) for t in batch)
        return scores

    def _score_openai(self, text: str) -> float:
        """Score using OpenAI chat completion API."""
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OpenAI API key — falling back to keyword")
            return self._score_keyword(text)
        try:
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment analyst. "
                            "Rate the following text on a scale from -1.0 (very bearish) "
                            "to 1.0 (very bullish). Reply ONLY with a single number."
                        ),
                    },
                    {"role": "user", "content": text[:3000]},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()  # type: ignore[union-attr]
            return float(np.clip(float(raw), -1.0, 1.0))
        except Exception as exc:
            logger.warning("OpenAI API call failed: %s — using keyword fallback", exc)
            return self._score_keyword(text)

    def _score_keyword(self, text: str) -> float:
        """Simple keyword-count sentiment (always-available fallback)."""
        words = re.findall(r"[a-z]+", text.lower())
        pos = sum(1 for w in words if any(w.startswith(p) for p in _POSITIVE_WORDS))
        neg = sum(1 for w in words if any(w.startswith(n) for n in _NEGATIVE_WORDS))
        total = pos + neg
        if total == 0:
            return 0.0
        return float(np.clip((pos - neg) / total, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2000) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > max_chars and current:
                chunks.append(current.strip())
                current = ""
            current += " " + sent
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text[:max_chars]]

    @staticmethod
    def _split_filing_sections(text: str) -> Dict[str, str]:
        """Attempt to split an SEC filing into named sections."""
        pattern = re.compile(
            r"(?:^|\n)((?:ITEM|Item|Part)\s+\d+[A-Z]?\.?\s*[-:]?\s*.+?)(?=\n)",
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return {}
        sections: Dict[str, str] = {}
        for i, m in enumerate(matches):
            name = m.group(1).strip()[:80]
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if len(body) > 50:
                sections[name] = body
        return sections
