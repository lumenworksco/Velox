"""COMP-002: Unusual options activity detection.

Parses options volume data to detect unusual activity patterns:
- Volume significantly exceeding open interest
- Large block trades (single prints > threshold)
- Unusual call/put skew
- Sweep orders across multiple exchanges

Data source: stub/framework since real-time options data requires paid
feeds (OPRA, CBOE LiveVol, etc.). The framework accepts data from any
source via a standard interface and performs the detection logic.

Fail-open: returns empty results if no data is available.

Usage::

    detector = OptionsFlowDetector()
    detector.ingest_options_data("AAPL", options_chain)
    alerts = detector.detect_unusual_activity("AAPL")
    # [UnusualActivityAlert(symbol="AAPL", alert_type="volume_spike", ...)]
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & thresholds
# ---------------------------------------------------------------------------

# Volume-to-open-interest ratio threshold for "unusual"
VOL_OI_THRESHOLD = 3.0

# Minimum absolute volume to consider (filters noise on illiquid options)
MIN_VOLUME_THRESHOLD = 500

# Block trade size threshold (number of contracts)
BLOCK_TRADE_THRESHOLD = 200

# Call/put ratio extremes (above = bullish, below = bearish)
CALL_PUT_BULLISH_THRESHOLD = 3.0
CALL_PUT_BEARISH_THRESHOLD = 0.33

# Premium threshold for "large" trades (USD)
LARGE_PREMIUM_THRESHOLD = 100_000

# Maximum age of cached data in seconds
CACHE_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class AlertType(str, Enum):
    """Types of unusual options activity."""
    VOLUME_SPIKE = "volume_spike"
    BLOCK_TRADE = "block_trade"
    CALL_SKEW = "call_skew"
    PUT_SKEW = "put_skew"
    LARGE_PREMIUM = "large_premium"
    SWEEP = "sweep"
    OI_DIVERGENCE = "oi_divergence"


class Sentiment(str, Enum):
    """Directional sentiment implied by the activity."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class OptionContract:
    """Standardized representation of a single option contract."""
    symbol: str
    underlying: str
    expiration: str          # ISO date string
    strike: float
    option_type: str         # "call" or "put"
    volume: int = 0
    open_interest: int = 0
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    implied_vol: float = 0.0
    delta: float = 0.0


@dataclass
class OptionsTrade:
    """A single options trade/print."""
    underlying: str
    expiration: str
    strike: float
    option_type: str
    size: int               # number of contracts
    price: float
    timestamp: Optional[datetime] = None
    exchange: str = ""
    condition: str = ""     # "block", "sweep", "single", etc.


@dataclass
class UnusualActivityAlert:
    """An alert for unusual options activity on a symbol."""
    symbol: str
    alert_type: AlertType
    sentiment: Sentiment
    score: float            # 0.0 to 1.0 confidence
    details: str
    contracts: List[OptionContract] = field(default_factory=list)
    total_premium: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def summary(self) -> str:
        return f"[{self.alert_type.value}] {self.symbol}: {self.details} ({self.sentiment.value}, score={self.score:.2f})"


@dataclass
class SymbolFlowSummary:
    """Aggregate options flow summary for one symbol."""
    symbol: str
    total_call_volume: int = 0
    total_put_volume: int = 0
    call_put_ratio: float = 1.0
    total_premium: float = 0.0
    net_sentiment: Sentiment = Sentiment.NEUTRAL
    net_score: float = 0.0
    alerts: List[UnusualActivityAlert] = field(default_factory=list)
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OptionsFlowDetector:
    """Detect unusual options activity from options chain and trade data.

    This is a framework class that accepts data from any source. In
    production, connect it to a real-time options data feed (OPRA, CBOE
    LiveVol, Tradier, etc.) by calling ingest_options_data() and
    ingest_trades().
    """

    def __init__(
        self,
        vol_oi_threshold: float = VOL_OI_THRESHOLD,
        block_threshold: int = BLOCK_TRADE_THRESHOLD,
        min_volume: int = MIN_VOLUME_THRESHOLD,
    ):
        self._vol_oi_threshold = vol_oi_threshold
        self._block_threshold = block_threshold
        self._min_volume = min_volume

        # Data stores
        self._chains: Dict[str, List[OptionContract]] = {}      # symbol -> chain
        self._trades: Dict[str, List[OptionsTrade]] = {}         # symbol -> trades
        self._summaries: Dict[str, SymbolFlowSummary] = {}       # symbol -> summary
        self._last_update: Dict[str, float] = {}                 # symbol -> epoch

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_options_data(
        self,
        symbol: str,
        contracts: List[OptionContract],
    ) -> None:
        """Ingest an options chain snapshot for a symbol.

        Args:
            symbol: Underlying ticker.
            contracts: List of OptionContract objects for all strikes/expirations.
        """
        self._chains[symbol] = contracts
        self._last_update[symbol] = datetime.now().timestamp()
        logger.debug("Ingested %d option contracts for %s", len(contracts), symbol)

    def ingest_trades(
        self,
        symbol: str,
        trades: List[OptionsTrade],
    ) -> None:
        """Ingest options trade prints for a symbol.

        Args:
            symbol: Underlying ticker.
            trades: List of OptionsTrade objects.
        """
        existing = self._trades.get(symbol, [])
        existing.extend(trades)
        # Keep only last 1000 trades per symbol
        self._trades[symbol] = existing[-1000:]
        logger.debug("Ingested %d option trades for %s (total: %d)",
                      len(trades), symbol, len(self._trades[symbol]))

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def detect_unusual_activity(self, symbol: str) -> List[UnusualActivityAlert]:
        """Run all detection algorithms on a symbol.

        Returns a list of UnusualActivityAlert objects, sorted by score
        descending. Returns empty list if no data available.
        """
        alerts: List[UnusualActivityAlert] = []
        chain = self._chains.get(symbol, [])
        trades = self._trades.get(symbol, [])

        if not chain and not trades:
            logger.debug("No options data for %s", symbol)
            return []

        try:
            alerts.extend(self._detect_volume_spikes(symbol, chain))
            alerts.extend(self._detect_call_put_skew(symbol, chain))
            alerts.extend(self._detect_block_trades(symbol, trades))
            alerts.extend(self._detect_large_premium(symbol, chain))
            alerts.extend(self._detect_sweeps(symbol, trades))
        except Exception as exc:
            logger.warning("Unusual activity detection failed for %s: %s", symbol, exc)
            return []

        alerts.sort(key=lambda a: a.score, reverse=True)
        return alerts

    def _detect_volume_spikes(
        self, symbol: str, chain: List[OptionContract]
    ) -> List[UnusualActivityAlert]:
        """Detect contracts with volume >> open interest."""
        alerts = []
        for contract in chain:
            if contract.volume < self._min_volume:
                continue
            if contract.open_interest <= 0:
                continue

            ratio = contract.volume / contract.open_interest
            if ratio >= self._vol_oi_threshold:
                sentiment = Sentiment.BULLISH if contract.option_type == "call" else Sentiment.BEARISH
                score = min(ratio / (self._vol_oi_threshold * 3), 1.0)
                premium = contract.volume * contract.last_price * 100

                alerts.append(UnusualActivityAlert(
                    symbol=symbol,
                    alert_type=AlertType.VOLUME_SPIKE,
                    sentiment=sentiment,
                    score=score,
                    details=(
                        f"{contract.option_type.upper()} {contract.strike} "
                        f"{contract.expiration}: vol={contract.volume}, "
                        f"OI={contract.open_interest}, ratio={ratio:.1f}x"
                    ),
                    contracts=[contract],
                    total_premium=premium,
                ))
        return alerts

    def _detect_call_put_skew(
        self, symbol: str, chain: List[OptionContract]
    ) -> List[UnusualActivityAlert]:
        """Detect extreme call/put volume skew."""
        total_call_vol = sum(c.volume for c in chain if c.option_type == "call")
        total_put_vol = sum(c.volume for c in chain if c.option_type == "put")

        if total_put_vol == 0 and total_call_vol == 0:
            return []

        ratio = total_call_vol / max(total_put_vol, 1)

        if ratio >= CALL_PUT_BULLISH_THRESHOLD:
            return [UnusualActivityAlert(
                symbol=symbol,
                alert_type=AlertType.CALL_SKEW,
                sentiment=Sentiment.BULLISH,
                score=min(ratio / (CALL_PUT_BULLISH_THRESHOLD * 2), 1.0),
                details=f"Call/put ratio {ratio:.1f} (calls={total_call_vol}, puts={total_put_vol})",
            )]
        elif ratio <= CALL_PUT_BEARISH_THRESHOLD:
            inverse_ratio = total_put_vol / max(total_call_vol, 1)
            return [UnusualActivityAlert(
                symbol=symbol,
                alert_type=AlertType.PUT_SKEW,
                sentiment=Sentiment.BEARISH,
                score=min(inverse_ratio / (1 / CALL_PUT_BEARISH_THRESHOLD * 2), 1.0),
                details=f"Put/call ratio {inverse_ratio:.1f} (puts={total_put_vol}, calls={total_call_vol})",
            )]

        return []

    def _detect_block_trades(
        self, symbol: str, trades: List[OptionsTrade]
    ) -> List[UnusualActivityAlert]:
        """Detect large block trades."""
        alerts = []
        for trade in trades:
            if trade.size >= self._block_threshold:
                sentiment = Sentiment.BULLISH if trade.option_type == "call" else Sentiment.BEARISH
                premium = trade.size * trade.price * 100
                score = min(trade.size / (self._block_threshold * 5), 1.0)

                alerts.append(UnusualActivityAlert(
                    symbol=symbol,
                    alert_type=AlertType.BLOCK_TRADE,
                    sentiment=sentiment,
                    score=score,
                    details=(
                        f"Block: {trade.size} {trade.option_type.upper()} "
                        f"{trade.strike} {trade.expiration} @ {trade.price:.2f} "
                        f"(${premium:,.0f} premium)"
                    ),
                    total_premium=premium,
                ))
        return alerts

    def _detect_large_premium(
        self, symbol: str, chain: List[OptionContract]
    ) -> List[UnusualActivityAlert]:
        """Detect contracts with exceptionally large daily premium traded."""
        alerts = []
        for contract in chain:
            premium = contract.volume * contract.last_price * 100
            if premium >= LARGE_PREMIUM_THRESHOLD:
                sentiment = Sentiment.BULLISH if contract.option_type == "call" else Sentiment.BEARISH
                score = min(premium / (LARGE_PREMIUM_THRESHOLD * 10), 1.0)

                alerts.append(UnusualActivityAlert(
                    symbol=symbol,
                    alert_type=AlertType.LARGE_PREMIUM,
                    sentiment=sentiment,
                    score=score,
                    details=(
                        f"${premium:,.0f} premium in {contract.option_type.upper()} "
                        f"{contract.strike} {contract.expiration}"
                    ),
                    contracts=[contract],
                    total_premium=premium,
                ))
        return alerts

    def _detect_sweeps(
        self, symbol: str, trades: List[OptionsTrade]
    ) -> List[UnusualActivityAlert]:
        """Detect sweep orders (same contract filled across multiple exchanges)."""
        # Group trades by (expiration, strike, option_type) within a short window
        from collections import defaultdict
        groups: Dict[Tuple, List[OptionsTrade]] = defaultdict(list)

        for trade in trades:
            key = (trade.expiration, trade.strike, trade.option_type)
            groups[key].append(trade)

        alerts = []
        for key, group_trades in groups.items():
            if len(group_trades) < 3:
                continue

            exchanges = set(t.exchange for t in group_trades if t.exchange)
            if len(exchanges) < 2:
                continue

            total_size = sum(t.size for t in group_trades)
            total_premium = sum(t.size * t.price * 100 for t in group_trades)
            exp, strike, opt_type = key
            sentiment = Sentiment.BULLISH if opt_type == "call" else Sentiment.BEARISH
            score = min(total_size / (self._block_threshold * 3), 1.0)

            alerts.append(UnusualActivityAlert(
                symbol=symbol,
                alert_type=AlertType.SWEEP,
                sentiment=sentiment,
                score=score,
                details=(
                    f"Sweep: {total_size} {opt_type.upper()} {strike} {exp} "
                    f"across {len(exchanges)} exchanges (${total_premium:,.0f})"
                ),
                total_premium=total_premium,
            ))

        return alerts

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_flow_summary(self, symbol: str) -> SymbolFlowSummary:
        """Compute aggregate flow summary for a symbol.

        Combines chain data and alerts into a single summary with
        net sentiment scoring.
        """
        chain = self._chains.get(symbol, [])
        alerts = self.detect_unusual_activity(symbol)

        total_call_vol = sum(c.volume for c in chain if c.option_type == "call")
        total_put_vol = sum(c.volume for c in chain if c.option_type == "put")
        call_put_ratio = total_call_vol / max(total_put_vol, 1)
        total_premium = sum(a.total_premium for a in alerts)

        # Net sentiment: weighted average of alert sentiments
        bullish_score = sum(a.score for a in alerts if a.sentiment == Sentiment.BULLISH)
        bearish_score = sum(a.score for a in alerts if a.sentiment == Sentiment.BEARISH)
        total_score = bullish_score + bearish_score

        if total_score > 0:
            net_score = (bullish_score - bearish_score) / total_score
        else:
            net_score = 0.0

        if net_score > 0.3:
            net_sentiment = Sentiment.BULLISH
        elif net_score < -0.3:
            net_sentiment = Sentiment.BEARISH
        else:
            net_sentiment = Sentiment.NEUTRAL

        summary = SymbolFlowSummary(
            symbol=symbol,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            call_put_ratio=call_put_ratio,
            total_premium=total_premium,
            net_sentiment=net_sentiment,
            net_score=net_score,
            alerts=alerts,
            updated_at=datetime.now(),
        )
        self._summaries[symbol] = summary
        return summary

    def screen_symbols(
        self,
        symbols: List[str],
        min_score: float = 0.3,
    ) -> List[SymbolFlowSummary]:
        """Screen multiple symbols for unusual options flow.

        Returns summaries for symbols with alerts exceeding min_score,
        sorted by absolute net_score descending.
        """
        results: List[SymbolFlowSummary] = []
        for symbol in symbols:
            try:
                summary = self.get_flow_summary(symbol)
                if summary.alerts and abs(summary.net_score) >= min_score:
                    results.append(summary)
            except Exception as exc:
                logger.debug("Options flow screening failed for %s: %s", symbol, exc)

        results.sort(key=lambda s: abs(s.net_score), reverse=True)
        return results

    def clear_data(self, symbol: Optional[str] = None) -> None:
        """Clear cached data for one or all symbols."""
        if symbol:
            self._chains.pop(symbol, None)
            self._trades.pop(symbol, None)
            self._summaries.pop(symbol, None)
        else:
            self._chains.clear()
            self._trades.clear()
            self._summaries.clear()
