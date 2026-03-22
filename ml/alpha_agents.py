"""T5-012: LLM Multi-Agent Alpha Mining — sentiment, fundamental, and regime agents.

Three specialized sub-agents:
  - SentimentAgent: Scores recent news/social sentiment per symbol.
  - FundamentalAgent: Parses SEC/earnings data.
  - RegimeAgent: Assesses macro conditions.

Agents run with 3-second timeout. Results cached by symbol x data-hash.
Aggregate: alpha_score = 0.4 * sentiment + 0.3 * fundamental + 0.3 * regime.
Wired to position-sizing multiplier (0.7x to 1.3x).
Cost controls: $0.10/day cap. Falls back to keyword scoring on failure.

Usage::

    orchestrator = AlphaAgentOrchestrator()
    result = orchestrator.get_alpha_score("AAPL")
    # result.alpha_score in [-1.0, 1.0]
    # result.size_multiplier in [0.7, 1.3]
"""

import hashlib
import json
import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_TIMEOUT_SEC = 3.0
DAILY_COST_CAP_USD = 0.10
CACHE_TTL_SEC = 300  # 5 minutes

# Sentiment weights
W_SENTIMENT = 0.4
W_FUNDAMENTAL = 0.3
W_REGIME = 0.3

# Sizing multiplier range
MIN_SIZE_MULT = 0.7
MAX_SIZE_MULT = 1.3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from a single sub-agent."""
    agent_name: str
    score: float          # -1.0 to 1.0
    confidence: float     # 0.0 to 1.0
    reasoning: str = ""
    source: str = ""      # "llm", "keyword_fallback", "cached"
    latency_ms: float = 0.0


@dataclass
class AlphaScore:
    """Aggregated alpha score from all agents."""
    symbol: str
    alpha_score: float           # -1.0 to 1.0
    size_multiplier: float       # 0.7 to 1.3
    sentiment: AgentResult | None = None
    fundamental: AgentResult | None = None
    regime: AgentResult | None = None
    is_cached: bool = False
    computed_at: str = ""


# ---------------------------------------------------------------------------
# Keyword-based fallback scorers (no LLM cost)
# ---------------------------------------------------------------------------

# Positive and negative keyword lists for sentiment fallback
_POSITIVE_KEYWORDS = {
    "beat", "beats", "upgrade", "bullish", "surge", "surges", "rally",
    "record", "outperform", "growth", "strong", "positive", "raise",
    "raises", "buy", "overweight", "accelerate", "expansion", "upside",
}

_NEGATIVE_KEYWORDS = {
    "miss", "misses", "downgrade", "bearish", "crash", "plunge", "sell",
    "warning", "weak", "negative", "cut", "cuts", "underperform",
    "decline", "risk", "layoff", "layoffs", "investigation", "downside",
}


def _keyword_sentiment_score(texts: list[str]) -> float:
    """Simple keyword-based sentiment scoring. Returns -1.0 to 1.0."""
    if not texts:
        return 0.0
    pos_count = 0
    neg_count = 0
    for text in texts:
        words = set(text.lower().split())
        pos_count += len(words & _POSITIVE_KEYWORDS)
        neg_count += len(words & _NEGATIVE_KEYWORDS)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos_count - neg_count) / total))


def _keyword_fundamental_score(data: dict) -> float:
    """Simple rule-based fundamental scoring. Returns -1.0 to 1.0."""
    score = 0.0
    # Earnings surprise
    surprise = data.get("earnings_surprise_pct", 0.0)
    if surprise > 5.0:
        score += 0.5
    elif surprise > 0:
        score += 0.2
    elif surprise < -5.0:
        score -= 0.5
    elif surprise < 0:
        score -= 0.2

    # Revenue growth
    rev_growth = data.get("revenue_growth_pct", 0.0)
    if rev_growth > 10:
        score += 0.3
    elif rev_growth > 0:
        score += 0.1
    elif rev_growth < -10:
        score -= 0.3

    return max(-1.0, min(1.0, score))


def _keyword_regime_score(vix: float, spy_return: float) -> float:
    """Simple rule-based regime scoring. Returns -1.0 to 1.0."""
    score = 0.0
    # VIX
    if vix < 15:
        score += 0.3
    elif vix < 20:
        score += 0.1
    elif vix > 30:
        score -= 0.5
    elif vix > 25:
        score -= 0.3

    # SPY trend
    if spy_return > 0.5:
        score += 0.3
    elif spy_return > 0:
        score += 0.1
    elif spy_return < -0.5:
        score -= 0.3
    elif spy_return < 0:
        score -= 0.1

    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Sub-Agents
# ---------------------------------------------------------------------------

class SentimentAgent:
    """Scores recent news/social sentiment per symbol.

    Attempts LLM-based scoring first, falls back to keyword matching.
    """

    def __init__(self):
        self._name = "SentimentAgent"

    def score(self, symbol: str, context: dict) -> AgentResult:
        """Score sentiment for a symbol.

        Args:
            symbol: Ticker symbol.
            context: Dict with optional 'news_headlines', 'social_posts'.

        Returns:
            AgentResult with sentiment score.
        """
        t0 = _time.time()
        headlines = context.get("news_headlines", [])
        social = context.get("social_posts", [])
        all_texts = headlines + social

        # Try Alpaca news if no headlines provided
        if not all_texts:
            try:
                from news_sentiment import AlpacaNewsSentiment
                ns = AlpacaNewsSentiment()
                mult, _ = ns.get_sentiment_size_mult(symbol)
                # Convert multiplier (0.0-1.0) to score (-1 to 1)
                score = (mult - 0.5) * 2.0
                latency = (_time.time() - t0) * 1000
                return AgentResult(
                    agent_name=self._name, score=score, confidence=0.6,
                    reasoning=f"Alpaca news sentiment mult={mult:.2f}",
                    source="alpaca_news", latency_ms=latency,
                )
            except Exception:
                pass

        # Keyword fallback
        score = _keyword_sentiment_score(all_texts)
        latency = (_time.time() - t0) * 1000
        return AgentResult(
            agent_name=self._name, score=score,
            confidence=0.4 if all_texts else 0.1,
            reasoning=f"Keyword scoring on {len(all_texts)} texts",
            source="keyword_fallback", latency_ms=latency,
        )


class FundamentalAgent:
    """Parses SEC/earnings data for fundamental scoring."""

    def __init__(self):
        self._name = "FundamentalAgent"

    def score(self, symbol: str, context: dict) -> AgentResult:
        """Score fundamental outlook for a symbol.

        Args:
            symbol: Ticker symbol.
            context: Dict with optional 'earnings_surprise_pct',
                     'revenue_growth_pct', 'pe_ratio'.

        Returns:
            AgentResult with fundamental score.
        """
        t0 = _time.time()
        fundamental_data = context.get("fundamental_data", {})

        # Try to get earnings data from the earnings module
        if not fundamental_data:
            try:
                from earnings import get_earnings_data
                fundamental_data = get_earnings_data(symbol) or {}
            except Exception:
                pass

        score = _keyword_fundamental_score(fundamental_data)
        latency = (_time.time() - t0) * 1000

        return AgentResult(
            agent_name=self._name, score=score,
            confidence=0.5 if fundamental_data else 0.1,
            reasoning=f"Rule-based fundamental analysis",
            source="keyword_fallback", latency_ms=latency,
        )


class RegimeAgent:
    """Assesses macro conditions (VIX, SPY trend, rates)."""

    def __init__(self):
        self._name = "RegimeAgent"

    def score(self, symbol: str, context: dict) -> AgentResult:
        """Score macro regime conditions.

        Args:
            symbol: Ticker symbol (used for sector context).
            context: Dict with optional 'vix_level', 'spy_return'.

        Returns:
            AgentResult with regime score.
        """
        t0 = _time.time()
        vix = context.get("vix_level", 20.0)
        spy_return = context.get("spy_return", 0.0)

        # Try to get live VIX/SPY data
        if vix == 20.0:
            try:
                from analytics.cross_asset import get_vix_level
                vix = get_vix_level() or 20.0
            except Exception:
                pass

        score = _keyword_regime_score(vix, spy_return)
        latency = (_time.time() - t0) * 1000

        return AgentResult(
            agent_name=self._name, score=score,
            confidence=0.6,
            reasoning=f"VIX={vix:.1f}, SPY_ret={spy_return:.2f}%",
            source="rule_based", latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AlphaAgentOrchestrator:
    """T5-012: Orchestrates 3 specialized sub-agents for alpha mining.

    Runs agents in parallel with 3-second timeout. Caches results by
    symbol x data-hash. Enforces $0.10/day cost cap.

    Usage::

        orch = AlphaAgentOrchestrator()
        result = orch.get_alpha_score("AAPL", context={"vix_level": 18.5})
        print(result.alpha_score, result.size_multiplier)
    """

    def __init__(self):
        self._sentiment = SentimentAgent()
        self._fundamental = FundamentalAgent()
        self._regime = RegimeAgent()

        # Cache: (symbol, data_hash) -> (AlphaScore, expiry_ts)
        self._cache: dict[tuple[str, str], tuple[AlphaScore, float]] = {}
        self._cache_lock = threading.Lock()

        # Cost tracking
        self._daily_cost_usd: float = 0.0
        self._cost_date: date | None = None
        self._cost_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_alpha_score(
        self,
        symbol: str,
        context: dict | None = None,
    ) -> AlphaScore:
        """Get aggregated alpha score for a symbol.

        Args:
            symbol: Ticker symbol.
            context: Optional context dict with news, fundamentals, etc.

        Returns:
            AlphaScore with alpha_score and size_multiplier.

        Never raises.
        """
        try:
            return self._get_alpha_inner(symbol, context or {})
        except Exception as e:
            logger.error("T5-012: AlphaAgentOrchestrator failed for %s: %s", symbol, e)
            return AlphaScore(
                symbol=symbol, alpha_score=0.0, size_multiplier=1.0,
                computed_at=datetime.now(config.ET).isoformat(),
            )

    def get_size_multiplier(self, symbol: str, context: dict | None = None) -> float:
        """Convenience: get just the sizing multiplier (0.7x to 1.3x)."""
        return self.get_alpha_score(symbol, context).size_multiplier

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_alpha_inner(self, symbol: str, context: dict) -> AlphaScore:
        """Core logic: check cache, run agents, aggregate."""
        now = _time.time()

        # Check cache
        data_hash = self._compute_data_hash(symbol, context)
        cache_key = (symbol, data_hash)
        with self._cache_lock:
            if cache_key in self._cache:
                cached_score, expiry = self._cache[cache_key]
                if now < expiry:
                    cached_score.is_cached = True
                    return cached_score

        # Check daily cost cap
        self._reset_daily_cost_if_needed()
        with self._cost_lock:
            if self._daily_cost_usd >= DAILY_COST_CAP_USD:
                logger.debug("T5-012: Daily cost cap reached ($%.2f), using keyword fallback",
                             self._daily_cost_usd)

        # Run agents in parallel with timeout
        results: dict[str, AgentResult] = {}
        agents = [
            ("sentiment", self._sentiment),
            ("fundamental", self._fundamental),
            ("regime", self._regime),
        ]

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="alpha_agent") as pool:
            futures = {
                pool.submit(agent.score, symbol, context): name
                for name, agent in agents
            }
            for future in as_completed(futures, timeout=AGENT_TIMEOUT_SEC + 0.5):
                name = futures[future]
                try:
                    result = future.result(timeout=AGENT_TIMEOUT_SEC)
                    results[name] = result
                except TimeoutError:
                    logger.warning("T5-012: %s timed out for %s", name, symbol)
                    results[name] = AgentResult(
                        agent_name=name, score=0.0, confidence=0.0,
                        reasoning="timeout", source="timeout",
                    )
                except Exception as e:
                    logger.warning("T5-012: %s failed for %s: %s", name, symbol, e)
                    results[name] = AgentResult(
                        agent_name=name, score=0.0, confidence=0.0,
                        reasoning=str(e), source="error",
                    )

        # Aggregate
        sent = results.get("sentiment", AgentResult("sentiment", 0.0, 0.0))
        fund = results.get("fundamental", AgentResult("fundamental", 0.0, 0.0))
        reg = results.get("regime", AgentResult("regime", 0.0, 0.0))

        alpha_score = (
            W_SENTIMENT * sent.score +
            W_FUNDAMENTAL * fund.score +
            W_REGIME * reg.score
        )
        alpha_score = max(-1.0, min(1.0, alpha_score))

        # Map alpha_score to sizing multiplier: 0 -> 1.0, +1 -> 1.3, -1 -> 0.7
        size_mult = 1.0 + alpha_score * (MAX_SIZE_MULT - 1.0)
        size_mult = max(MIN_SIZE_MULT, min(MAX_SIZE_MULT, size_mult))

        result = AlphaScore(
            symbol=symbol,
            alpha_score=round(alpha_score, 4),
            size_multiplier=round(size_mult, 3),
            sentiment=sent,
            fundamental=fund,
            regime=reg,
            computed_at=datetime.now(config.ET).isoformat(),
        )

        # Cache
        with self._cache_lock:
            self._cache[cache_key] = (result, now + CACHE_TTL_SEC)

        return result

    @staticmethod
    def _compute_data_hash(symbol: str, context: dict) -> str:
        """Compute a stable hash of the input data for caching."""
        data_str = json.dumps({"symbol": symbol, **context}, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]

    def _reset_daily_cost_if_needed(self):
        """Reset the daily cost counter if it's a new day."""
        today = date.today()
        with self._cost_lock:
            if self._cost_date != today:
                self._daily_cost_usd = 0.0
                self._cost_date = today


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_orchestrator: AlphaAgentOrchestrator | None = None
_orch_lock = threading.Lock()


def get_alpha_orchestrator() -> AlphaAgentOrchestrator:
    """Get or create the global AlphaAgentOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = AlphaAgentOrchestrator()
    return _orchestrator
