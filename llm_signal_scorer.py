"""LLM-based signal scoring using Claude Haiku for trade quality assessment."""

import json
import logging
from dataclasses import dataclass

import anthropic

import config
from strategies.base import Signal

log = logging.getLogger(__name__)


@dataclass
class SignalScore:
    score: float          # 0.0 (reject) to 1.0 (strong conviction)
    confidence: str       # 'HIGH', 'MEDIUM', 'LOW'
    reasoning: str        # Short explanation from LLM
    size_mult: float      # 0.5 to 1.5 — position size multiplier


DEFAULT_SCORE = SignalScore(score=0.7, confidence='LOW', reasoning='llm_error', size_mult=1.0)


class LLMSignalScorer:
    """Scores trade signals via Claude Haiku for quality gating and position sizing."""

    MAX_DAILY_COST_USD = config.LLM_MAX_DAILY_COST_USD

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._call_count: int = 0
        self._daily_cost_usd: float = 0.0

    # --------------------------------------------------------------------- #
    #  Public API                                                             #
    # --------------------------------------------------------------------- #

    def score_signal(self, signal: Signal, context: dict) -> SignalScore:
        """Score a trading signal using Claude Haiku.

        Returns a default pass-through score on any failure (fail-open).
        """
        # Cost guard
        if self._daily_cost_usd >= self.MAX_DAILY_COST_USD:
            log.warning("LLM daily budget exhausted ($%.4f / $%.2f), returning default",
                        self._daily_cost_usd, self.MAX_DAILY_COST_USD)
            return SignalScore(score=0.7, confidence='LOW', reasoning='budget_exceeded', size_mult=1.0)

        try:
            prompt = self._build_prompt(signal, context)

            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                timeout=3.0,
                messages=[{"role": "user", "content": prompt}],
            )

            # Track cost (Haiku 4.5: $0.25/M input, $1.25/M output)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 0.25 / 1_000_000) + (output_tokens * 1.25 / 1_000_000)
            self._daily_cost_usd += cost
            self._call_count += 1

            text = response.content[0].text
            return self._parse_response(text)

        except Exception as exc:
            log.warning("LLM scoring failed for %s: %s", signal.symbol, exc)
            return SignalScore(score=0.7, confidence='LOW', reasoning='llm_error', size_mult=1.0)

    def reset_daily(self):
        """Reset daily cost and call counters (call at start of each trading day)."""
        self._daily_cost_usd = 0.0
        self._call_count = 0

    # --------------------------------------------------------------------- #
    #  Internals                                                              #
    # --------------------------------------------------------------------- #

    def _build_prompt(self, signal: Signal, context: dict) -> str:
        """Build the prompt asking the LLM to rate a trade signal."""
        gain_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100
        loss_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        rr_ratio = gain_pct / loss_pct if loss_pct > 0 else 0.0

        # Context values with safe defaults
        spy_ret = context.get('spy_day_return', 0.0)
        sector_ret = context.get('sector_day_return', 0.0)
        vix = context.get('vix_level', 'N/A')
        z_score = context.get('signal_z_score', 'N/A')
        headlines = context.get('recent_news_headlines', [])
        recent_trades = context.get('recent_trades_symbol', [])

        headlines_str = '\n'.join(f"  - {h}" for h in headlines[:5]) if headlines else '  (none)'
        trades_str = '\n'.join(f"  - {t}" for t in recent_trades[:5]) if recent_trades else '  (none)'

        return f"""You are a quantitative trading signal evaluator. Rate this signal's quality.

SIGNAL:
  Symbol: {signal.symbol}
  Strategy: {signal.strategy}
  Side: {signal.side}
  Entry: ${signal.entry_price:.2f}
  Take Profit: ${signal.take_profit:.2f}
  Stop Loss: ${signal.stop_loss:.2f}
  Expected Gain: {gain_pct:.2f}%
  Risk: {loss_pct:.2f}%
  R/R Ratio: {rr_ratio:.1f}

MARKET CONTEXT:
  SPY Day Return: {spy_ret:+.2%}
  Sector Day Return: {sector_ret:+.2%}
  VIX Level: {vix}
  Signal Z-Score: {z_score}

RECENT NEWS:
{headlines_str}

RECENT TRADES (this symbol):
{trades_str}

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{"score": <float 0.0-1.0>, "confidence": "<HIGH|MEDIUM|LOW>", "reasoning": "<1 sentence>", "size_mult": <float 0.5-1.5>}}

Guidelines:
- score 0.0-0.3: poor setup, likely to fail
- score 0.3-0.6: marginal, proceed with caution
- score 0.6-0.8: solid setup
- score 0.8-1.0: exceptional confluence
- size_mult: scale position size (0.5=half, 1.0=normal, 1.5=conviction)
- Penalize signals fighting the broader trend (e.g. long in a down market)
- Reward high R/R ratios and trend alignment"""

    def _parse_response(self, text: str) -> SignalScore:
        """Parse LLM JSON response into a SignalScore."""
        try:
            # Strip potential markdown code fences
            cleaned = text.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            data = json.loads(cleaned)

            score = float(data['score'])
            score = max(0.0, min(1.0, score))

            confidence = str(data.get('confidence', 'LOW')).upper()
            if confidence not in ('HIGH', 'MEDIUM', 'LOW'):
                confidence = 'LOW'

            reasoning = str(data.get('reasoning', 'no_reasoning'))

            size_mult = float(data.get('size_mult', 1.0))
            size_mult = max(0.5, min(1.5, size_mult))

            return SignalScore(
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                size_mult=size_mult,
            )

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            log.warning("Failed to parse LLM response: %s — raw: %s", exc, text[:200])
            return SignalScore(score=0.7, confidence='LOW', reasoning='llm_error', size_mult=1.0)
