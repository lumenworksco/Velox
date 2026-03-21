"""RISK-003: Margin monitoring with progressive risk controls.

Tracks margin usage via Alpaca account data and enforces tiered responses:
    - Alert at 70% margin usage (log warning + optional notification)
    - Halt new short positions at 80%
    - Start unwinding positions at 90% (close most recent, smallest-edge trades)

Fail-open: if Alpaca margin data is unavailable, allows trading to continue
(logs a warning). This prevents a data outage from halting the bot.

Usage:
    monitor = MarginMonitor()
    monitor.update()  # Fetches latest margin data from Alpaca
    if not monitor.can_open_short():
        # Block short entry
    if monitor.should_unwind():
        # Start closing positions
"""

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import config

logger = logging.getLogger(__name__)

# Margin usage thresholds
MARGIN_ALERT_PCT = 0.70       # Alert / log warning
MARGIN_SHORT_HALT_PCT = 0.80  # Block new short positions
MARGIN_UNWIND_PCT = 0.90      # Start unwinding positions
MARGIN_CACHE_TTL_SEC = 30     # Cache Alpaca margin data for 30 seconds


class MarginState(Enum):
    NORMAL = "normal"           # Usage < 70%
    ALERT = "alert"             # 70% <= usage < 80%
    SHORT_HALTED = "short_halted"  # 80% <= usage < 90%
    UNWINDING = "unwinding"     # Usage >= 90%
    DATA_UNAVAILABLE = "data_unavailable"  # Fail-open


@dataclass
class MarginSnapshot:
    """Point-in-time margin account status."""
    equity: float = 0.0
    buying_power: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    margin_usage_pct: float = 0.0      # maintenance_margin / equity
    state: MarginState = MarginState.NORMAL
    fetched_at: datetime = field(default_factory=datetime.now)
    data_available: bool = True


class MarginMonitor:
    """Real-time margin usage monitoring with progressive controls.

    Fetches margin data from Alpaca account API and enforces tiered
    risk responses. Thread-safe.
    """

    def __init__(
        self,
        alert_pct: float = MARGIN_ALERT_PCT,
        short_halt_pct: float = MARGIN_SHORT_HALT_PCT,
        unwind_pct: float = MARGIN_UNWIND_PCT,
        cache_ttl_sec: float = MARGIN_CACHE_TTL_SEC,
    ):
        self._alert_pct = alert_pct
        self._short_halt_pct = short_halt_pct
        self._unwind_pct = unwind_pct
        self._cache_ttl_sec = cache_ttl_sec

        self._last_snapshot: MarginSnapshot = MarginSnapshot(data_available=False)
        self._last_fetch_ts: float = 0.0
        self._lock = threading.Lock()
        self._alert_emitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> MarginSnapshot:
        """Fetch latest margin data from Alpaca and evaluate state.

        Returns the current MarginSnapshot. Caches for MARGIN_CACHE_TTL_SEC.
        Fail-open: returns DATA_UNAVAILABLE state if fetch fails.
        """
        now_ts = _time.time()

        with self._lock:
            if (now_ts - self._last_fetch_ts) < self._cache_ttl_sec:
                return self._last_snapshot

        # Fetch from Alpaca
        snapshot = self._fetch_margin_data()

        with self._lock:
            self._last_snapshot = snapshot
            self._last_fetch_ts = now_ts

            # Emit alert once per transition
            if snapshot.state == MarginState.ALERT and not self._alert_emitted:
                logger.warning(
                    f"RISK-003: Margin ALERT — usage at {snapshot.margin_usage_pct:.1%} "
                    f"(threshold: {self._alert_pct:.0%})"
                )
                self._alert_emitted = True
            elif snapshot.state == MarginState.SHORT_HALTED:
                logger.warning(
                    f"RISK-003: Margin SHORT HALT — usage at {snapshot.margin_usage_pct:.1%} "
                    f"(threshold: {self._short_halt_pct:.0%}). New shorts blocked."
                )
            elif snapshot.state == MarginState.UNWINDING:
                logger.critical(
                    f"RISK-003: Margin UNWIND — usage at {snapshot.margin_usage_pct:.1%} "
                    f"(threshold: {self._unwind_pct:.0%}). Begin position unwinding."
                )
            elif snapshot.state == MarginState.NORMAL:
                self._alert_emitted = False

        return snapshot

    def can_open_short(self) -> bool:
        """Return True if margin usage allows opening new short positions.

        Fail-open: returns True if margin data is unavailable.
        """
        with self._lock:
            snap = self._last_snapshot

        if not snap.data_available:
            logger.debug("RISK-003: Margin data unavailable, fail-open for short check")
            return True

        return snap.state not in (MarginState.SHORT_HALTED, MarginState.UNWINDING)

    def should_unwind(self) -> bool:
        """Return True if margin usage requires unwinding positions.

        Fail-open: returns False if margin data is unavailable.
        """
        with self._lock:
            snap = self._last_snapshot

        if not snap.data_available:
            return False

        return snap.state == MarginState.UNWINDING

    def get_unwind_candidates(
        self,
        positions: dict[str, Any],
    ) -> list[str]:
        """Return symbols to unwind, ordered by least edge (most recent, smallest P&L).

        Args:
            positions: Dict of symbol -> TradeRecord or dict.

        Returns:
            List of symbols to close, in suggested unwind order.
        """
        if not self.should_unwind() or not positions:
            return []

        # Sort by: (1) most recent entry (LIFO), (2) smallest unrealized P&L
        scored: list[tuple[str, float, datetime]] = []
        for symbol, pos in positions.items():
            entry_time = self._get_attr(pos, "entry_time", datetime.min)
            pnl = self._get_attr(pos, "pnl", 0.0)
            scored.append((symbol, pnl, entry_time))

        # Sort: most recent first, then smallest P&L first
        scored.sort(key=lambda x: (-x[2].timestamp() if isinstance(x[2], datetime) else 0, x[1]))

        return [s[0] for s in scored]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_margin_data(self) -> MarginSnapshot:
        """Fetch margin data from Alpaca account API (fail-open)."""
        try:
            from broker.alpaca_client import get_trading_client

            client = get_trading_client()
            account = client.get_account()

            equity = float(account.equity)
            buying_power = float(account.buying_power)
            initial_margin = float(account.initial_margin)
            maintenance_margin = float(account.maintenance_margin)

            # Compute margin usage as maintenance margin / equity
            usage_pct = maintenance_margin / equity if equity > 0 else 0.0

            # Determine state
            if usage_pct >= self._unwind_pct:
                state = MarginState.UNWINDING
            elif usage_pct >= self._short_halt_pct:
                state = MarginState.SHORT_HALTED
            elif usage_pct >= self._alert_pct:
                state = MarginState.ALERT
            else:
                state = MarginState.NORMAL

            return MarginSnapshot(
                equity=equity,
                buying_power=buying_power,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                margin_usage_pct=usage_pct,
                state=state,
                fetched_at=datetime.now(),
                data_available=True,
            )

        except Exception as e:
            logger.warning(f"RISK-003: Failed to fetch margin data (fail-open): {e}")
            return MarginSnapshot(
                state=MarginState.DATA_UNAVAILABLE,
                fetched_at=datetime.now(),
                data_available=False,
            )

    @staticmethod
    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        """Get attribute from object or dict."""
        if hasattr(obj, name):
            return getattr(obj, name, default)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> MarginSnapshot:
        with self._lock:
            return self._last_snapshot

    @property
    def status(self) -> dict:
        with self._lock:
            s = self._last_snapshot
            return {
                "state": s.state.value,
                "margin_usage_pct": round(s.margin_usage_pct, 4),
                "equity": round(s.equity, 2),
                "buying_power": round(s.buying_power, 2),
                "initial_margin": round(s.initial_margin, 2),
                "maintenance_margin": round(s.maintenance_margin, 2),
                "data_available": s.data_available,
                "fetched_at": s.fetched_at.isoformat() if s.fetched_at else None,
                "thresholds": {
                    "alert": self._alert_pct,
                    "short_halt": self._short_halt_pct,
                    "unwind": self._unwind_pct,
                },
            }
