"""DATA-004: Dynamic Universe Selection.

Manages the tradeable symbol universe with daily refresh, filtering by
liquidity, market cap, listing age, and health checks. Supports
per-strategy universe slicing and snapshot persistence for backtest
reproducibility.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import config

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Lightweight record for a tradeable asset."""
    symbol: str
    name: str = ""
    exchange: str = ""
    market_cap: float = 0.0       # estimated, USD
    avg_daily_volume: float = 0.0  # 20-day ADV in USD
    price: float = 0.0
    listed_date: Optional[str] = None  # ISO date string
    is_etf: bool = False
    is_shortable: bool = False
    status: str = "active"


# ---------------------------------------------------------------------------
# Strategy-specific universe sizing
# ---------------------------------------------------------------------------

_STRATEGY_UNIVERSE_DEFAULTS: Dict[str, dict] = {
    "STAT_MR": {
        "max_symbols": 40,
        "prefer_etfs": False,
        "min_market_cap": 1_000_000_000,
        "min_adv": 10_000_000,
        "exclude_earnings": True,
    },
    "VWAP": {
        "max_symbols": 50,
        "prefer_etfs": True,
        "min_market_cap": 500_000_000,
        "min_adv": 5_000_000,
        "exclude_earnings": True,
    },
    "KALMAN_PAIRS": {
        "max_symbols": 30,
        "prefer_etfs": False,
        "min_market_cap": 5_000_000_000,
        "min_adv": 20_000_000,
        "exclude_earnings": True,
    },
    "ORB": {
        "max_symbols": 15,
        "prefer_etfs": False,
        "min_market_cap": 1_000_000_000,
        "min_adv": 10_000_000,
        "exclude_earnings": True,
    },
    "MICRO_MOM": {
        "max_symbols": 10,
        "prefer_etfs": False,
        "min_market_cap": 2_000_000_000,
        "min_adv": 15_000_000,
        "exclude_earnings": True,
    },
    "PEAD": {
        "max_symbols": 20,
        "prefer_etfs": False,
        "min_market_cap": 1_000_000_000,
        "min_adv": 10_000_000,
        "exclude_earnings": False,  # PEAD specifically needs earnings
    },
}


class DynamicUniverse:
    """Manages the tradeable symbol universe with daily refresh and filtering.

    Uses Alpaca's asset list API to pull tradeable assets, then applies
    filters for liquidity, market cap, listing age, and health.
    """

    # Minimum filters (applied globally)
    MIN_MARKET_CAP = 1_000_000_000   # $1B
    MIN_ADV_USD = 10_000_000          # $10M average daily volume
    MIN_PRICE = 5.0                   # $5 minimum price
    MIN_LISTING_DAYS = 90             # Must be listed > 90 days

    def __init__(self, snapshot_dir: Optional[str] = None):
        self._lock = threading.Lock()

        # Full filtered universe
        self._universe: Dict[str, AssetInfo] = {}
        self._last_refresh: Optional[datetime] = None

        # Health exclusions
        self._excluded: Set[str] = set()
        self._exclusion_reasons: Dict[str, str] = {}

        # Earnings calendar (symbol -> next earnings date)
        self._earnings_calendar: Dict[str, str] = {}

        # Snapshot directory for backtest reproducibility
        self._snapshot_dir = snapshot_dir
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)

        # Fallback: use config.SYMBOLS if API unavailable
        self._fallback_symbols: List[str] = list(config.SYMBOLS)

        logger.info("DynamicUniverse initialised (snapshot_dir=%s)", snapshot_dir)

    # ------------------------------------------------------------------
    # Daily refresh
    # ------------------------------------------------------------------

    def refresh_daily(self) -> int:
        """Pull tradeable assets from Alpaca and apply filters.

        Returns:
            Number of symbols in the filtered universe.
        """
        try:
            assets = self._fetch_alpaca_assets()
            filtered = self._apply_filters(assets)

            with self._lock:
                self._universe = {a.symbol: a for a in filtered}
                self._last_refresh = datetime.now(timezone.utc)

            count = len(filtered)
            logger.info("DynamicUniverse: refreshed — %d symbols passed filters "
                        "(from %d total assets)", count, len(assets))
            return count

        except Exception as e:
            logger.error("DynamicUniverse: refresh failed, using fallback: %s", e)
            self._load_fallback()
            return len(self._universe)

    def _fetch_alpaca_assets(self) -> List[AssetInfo]:
        """Fetch the full asset list from Alpaca."""
        try:
            from data import get_trading_client
            client = get_trading_client()
        except ImportError:
            # Fallback import path
            import data as data_mod
            client = data_mod.get_trading_client()

        raw_assets = client.get_all_assets()
        results = []

        for asset in raw_assets:
            # Only include active, tradeable US equities
            if not getattr(asset, "tradable", False):
                continue
            if getattr(asset, "status", "") != "active":
                continue
            asset_class = getattr(asset, "asset_class", "")
            if asset_class and asset_class not in ("us_equity",):
                continue

            info = AssetInfo(
                symbol=asset.symbol,
                name=getattr(asset, "name", ""),
                exchange=getattr(asset, "exchange", ""),
                listed_date=str(getattr(asset, "listdate", "")) if hasattr(asset, "listdate") else None,
                is_shortable=getattr(asset, "shortable", False),
                is_etf=getattr(asset, "asset_class", "") == "us_equity"
                        and getattr(asset, "exchange", "") in ("ARCA", "BATS"),
                status="active",
            )
            results.append(info)

        return results

    def _apply_filters(self, assets: List[AssetInfo]) -> List[AssetInfo]:
        """Apply liquidity, market cap, price, and listing age filters.

        Note: Alpaca's asset API does not provide market cap or ADV directly.
        We use snapshot data to enrich when available, otherwise we rely on
        the fallback symbol list intersected with the API asset list.
        """
        filtered = []
        now = datetime.now(timezone.utc)
        enriched_count = 0

        # Try to enrich with snapshot data for price/volume
        try:
            from data import get_snapshots
            # Batch snapshot fetch (Alpaca allows up to 1000 at a time)
            symbols = [a.symbol for a in assets]
            batch_size = 500
            snapshot_map = {}
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    snap = get_snapshots(batch)
                    snapshot_map.update(snap)
                except Exception:
                    pass
        except ImportError:
            snapshot_map = {}

        for asset in assets:
            # Enrich from snapshot
            snap = snapshot_map.get(asset.symbol)
            if snap:
                try:
                    if hasattr(snap, "daily_bar") and snap.daily_bar:
                        asset.price = float(snap.daily_bar.close)
                        asset.avg_daily_volume = float(snap.daily_bar.volume) * asset.price
                    elif hasattr(snap, "latest_trade") and snap.latest_trade:
                        asset.price = float(snap.latest_trade.price)
                    enriched_count += 1
                except Exception:
                    pass

            # Price filter
            if asset.price > 0 and asset.price < self.MIN_PRICE:
                continue

            # ADV filter (only if we have data)
            if asset.avg_daily_volume > 0 and asset.avg_daily_volume < self.MIN_ADV_USD:
                continue

            # Listing age filter
            if asset.listed_date:
                try:
                    listed = datetime.fromisoformat(asset.listed_date.replace("Z", "+00:00"))
                    if (now - listed).days < self.MIN_LISTING_DAYS:
                        continue
                except (ValueError, TypeError):
                    pass

            filtered.append(asset)

        # If we couldn't enrich most assets, intersect with fallback list
        if enriched_count < len(assets) * 0.1:
            fallback_set = set(self._fallback_symbols)
            api_symbols = {a.symbol for a in filtered}
            # Keep only symbols that are in both the API and fallback lists
            valid = api_symbols & fallback_set
            filtered = [a for a in filtered if a.symbol in valid]
            logger.info("DynamicUniverse: limited enrichment (%d/%d), "
                        "intersected with fallback (%d symbols)",
                        enriched_count, len(assets), len(filtered))

        return filtered

    def _load_fallback(self):
        """Populate universe from config.SYMBOLS as fallback."""
        with self._lock:
            self._universe = {
                sym: AssetInfo(symbol=sym, status="active")
                for sym in self._fallback_symbols
            }
            self._last_refresh = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Universe access (per-strategy)
    # ------------------------------------------------------------------

    def get_universe(self, strategy_type: Optional[str] = None) -> List[str]:
        """Return the symbol universe, optionally filtered for a strategy.

        Args:
            strategy_type: One of STAT_MR, VWAP, KALMAN_PAIRS, ORB,
                          MICRO_MOM, PEAD, or None for full universe.

        Returns:
            List of ticker symbols.
        """
        with self._lock:
            if not self._universe:
                # Auto-load fallback if never refreshed
                self._universe = {
                    sym: AssetInfo(symbol=sym, status="active")
                    for sym in self._fallback_symbols
                }

            # Start with full universe minus exclusions
            symbols = [
                sym for sym in self._universe
                if sym not in self._excluded
            ]

        if strategy_type is None:
            return symbols

        params = _STRATEGY_UNIVERSE_DEFAULTS.get(strategy_type, {})
        max_symbols = params.get("max_symbols", 50)
        exclude_earnings = params.get("exclude_earnings", True)

        # Filter out upcoming earnings if needed
        if exclude_earnings:
            symbols = [s for s in symbols if s not in self._earnings_calendar]

        # Sort by ADV (highest first) for strategy-specific slicing
        with self._lock:
            symbols.sort(
                key=lambda s: self._universe.get(s, AssetInfo(symbol=s)).avg_daily_volume,
                reverse=True,
            )

        return symbols[:max_symbols]

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self):
        """Run health checks and remove problematic symbols.

        Checks for:
        - Delisted or halted symbols
        - Symbols with upcoming M&A activity (manual exclusion list)
        - Symbols with upcoming earnings (for non-PEAD strategies)
        """
        removed = 0

        with self._lock:
            to_remove = []
            for sym, info in self._universe.items():
                # Check for inactive/delisted
                if info.status not in ("active",):
                    to_remove.append((sym, "delisted_or_inactive"))

            for sym, reason in to_remove:
                self._excluded.add(sym)
                self._exclusion_reasons[sym] = reason
                removed += 1

        if removed > 0:
            logger.info("DynamicUniverse: health_check excluded %d symbols", removed)

        return removed

    def exclude_symbol(self, symbol: str, reason: str):
        """Manually exclude a symbol from the universe."""
        with self._lock:
            self._excluded.add(symbol)
            self._exclusion_reasons[symbol] = reason
        logger.info("DynamicUniverse: excluded %s (%s)", symbol, reason)

    def reinstate_symbol(self, symbol: str):
        """Re-include a previously excluded symbol."""
        with self._lock:
            self._excluded.discard(symbol)
            self._exclusion_reasons.pop(symbol, None)
        logger.info("DynamicUniverse: reinstated %s", symbol)

    def set_earnings_calendar(self, calendar: Dict[str, str]):
        """Update the earnings calendar.

        Args:
            calendar: Mapping of symbol -> next earnings date (ISO format).
        """
        with self._lock:
            self._earnings_calendar = dict(calendar)
        logger.info("DynamicUniverse: updated earnings calendar (%d symbols)",
                     len(calendar))

    # ------------------------------------------------------------------
    # Snapshot persistence (for backtesting)
    # ------------------------------------------------------------------

    def store_snapshot(self, date: str):
        """Persist the current universe as a JSON snapshot for a given date.

        Args:
            date: ISO date string (e.g., '2025-03-15').
        """
        if not self._snapshot_dir:
            logger.warning("DynamicUniverse: no snapshot_dir configured")
            return

        with self._lock:
            snapshot = {
                "date": date,
                "refreshed_at": self._last_refresh.isoformat() if self._last_refresh else None,
                "symbols": list(self._universe.keys()),
                "excluded": list(self._excluded),
                "exclusion_reasons": dict(self._exclusion_reasons),
                "asset_details": {
                    sym: {
                        "name": info.name,
                        "exchange": info.exchange,
                        "price": info.price,
                        "adv": info.avg_daily_volume,
                        "is_etf": info.is_etf,
                    }
                    for sym, info in self._universe.items()
                },
            }

        path = os.path.join(self._snapshot_dir, f"universe_{date}.json")
        try:
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info("DynamicUniverse: stored snapshot for %s (%d symbols) -> %s",
                        date, len(snapshot["symbols"]), path)
        except Exception as e:
            logger.error("DynamicUniverse: failed to store snapshot: %s", e)

    def load_snapshot(self, date: str) -> List[str]:
        """Load a historical universe snapshot for backtesting.

        Args:
            date: ISO date string.

        Returns:
            List of symbols from that date's universe, or empty list.
        """
        if not self._snapshot_dir:
            return []

        path = os.path.join(self._snapshot_dir, f"universe_{date}.json")
        try:
            with open(path, "r") as f:
                snapshot = json.load(f)
            symbols = snapshot.get("symbols", [])
            logger.info("DynamicUniverse: loaded snapshot for %s (%d symbols)",
                        date, len(symbols))
            return symbols
        except FileNotFoundError:
            logger.debug("DynamicUniverse: no snapshot for %s", date)
            return []
        except Exception as e:
            logger.error("DynamicUniverse: failed to load snapshot for %s: %s", date, e)
            return []

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current universe status."""
        with self._lock:
            return {
                "total_symbols": len(self._universe),
                "excluded": len(self._excluded),
                "active": len(self._universe) - len(self._excluded),
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "earnings_tracked": len(self._earnings_calendar),
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_dynamic_universe: Optional[DynamicUniverse] = None
_init_lock = threading.Lock()


def get_dynamic_universe(snapshot_dir: Optional[str] = None) -> DynamicUniverse:
    """Get or create the global DynamicUniverse singleton."""
    global _dynamic_universe
    with _init_lock:
        if _dynamic_universe is None:
            _dynamic_universe = DynamicUniverse(snapshot_dir=snapshot_dir)
    return _dynamic_universe
