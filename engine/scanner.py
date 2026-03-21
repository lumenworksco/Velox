"""V10 Engine — Strategy scanning and signal generation.

Extracts the multi-strategy scanning loop from main.py into a reusable module.
Handles scanning all active strategies, ranking signals, and processing them.

PROD-001: Added prefetch_bars() for concurrent bar fetching across all symbols
before strategies run their scans. Strategies can optionally accept a `bars_cache`
parameter to avoid redundant API calls.

ARCH-011: Added StrategyRegistry for data-driven strategy dispatch (replaces
hard-coded if/elif chains).
"""

import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from strategies.base import Signal
from engine.signal_processor import process_signals
from engine.exit_processor import handle_strategy_exits, get_current_prices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ARCH-011: Strategy Registry
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """Registry mapping strategy names to their scan/exit callables.

    Instead of hard-coded if/elif chains in scan_all_strategies(), callers
    register strategy instances here.  The scanner iterates the registry
    to collect signals.

    Usage::

        registry = StrategyRegistry()
        registry.register("STAT_MR", stat_mr, required=True)
        registry.register("ORB", orb_strategy, required=False)
        signals = registry.scan_all(current, regime)
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        strategy,
        required: bool = False,
        scan_kwargs_fn: Optional[Callable] = None,
    ) -> None:
        """Register a strategy instance.

        Args:
            name: Unique strategy name (e.g. "STAT_MR", "VWAP").
            strategy: Strategy instance with a `scan()` method.
            required: If True, log error on scan failure; otherwise warning.
            scan_kwargs_fn: Optional callable returning extra kwargs for scan().
                            Called with (current, regime) and merged into scan() args.
        """
        if strategy is None:
            return
        self._strategies[name] = {
            "instance": strategy,
            "required": required,
            "scan_kwargs_fn": scan_kwargs_fn,
        }

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry."""
        self._strategies.pop(name, None)

    @property
    def names(self) -> List[str]:
        """Return registered strategy names."""
        return list(self._strategies.keys())

    def get(self, name: str):
        """Return the strategy instance by name, or None."""
        entry = self._strategies.get(name)
        return entry["instance"] if entry else None

    def scan_all(
        self,
        current: datetime,
        regime: str,
        signal_ranker=None,
        extra_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Signal]:
        """Scan all registered strategies and return combined signals.

        Each strategy is scanned inside its own try/except (fail-open).
        Optional signal_ranker is applied after collection.

        Args:
            current: Current datetime.
            regime: Current market regime string.
            signal_ranker: Optional ranker with .rank(signals, regime=...).
            extra_kwargs: Per-strategy extra kwargs, keyed by strategy name.

        Returns:
            Combined (optionally ranked) list of Signal objects.
        """
        all_signals: List[Signal] = []
        counts: Dict[str, int] = {}
        extra_kwargs = extra_kwargs or {}

        for name, entry in self._strategies.items():
            strategy = entry["instance"]
            try:
                # Build scan kwargs
                kwargs: Dict[str, Any] = {"regime": regime}
                if entry["scan_kwargs_fn"]:
                    kwargs.update(entry["scan_kwargs_fn"](current, regime))
                if name in extra_kwargs:
                    kwargs.update(extra_kwargs[name])

                sigs = strategy.scan(current, **kwargs)
                all_signals.extend(sigs)
                counts[name] = len(sigs)
            except Exception as e:
                level = logging.ERROR if entry["required"] else logging.WARNING
                logger.log(level, "%s scan failed: %s", name, e)
                counts[name] = 0

        # Rank if available
        if signal_ranker and all_signals:
            try:
                all_signals = signal_ranker.rank(all_signals, regime=regime)
            except Exception:
                pass

        parts = " ".join(f"{k}={v}" for k, v in counts.items())
        logger.info(
            "Registry scan complete: %d signals (%s) regime=%s",
            len(all_signals), parts, regime,
        )
        return all_signals

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies

    def __repr__(self) -> str:
        return f"StrategyRegistry({list(self._strategies.keys())})"

# ---------------------------------------------------------------------------
# PROD-001: Concurrent bar prefetching
# ---------------------------------------------------------------------------

# Module-level cache for prefetched bars (cleared each scan cycle)
_bars_cache: dict[str, pd.DataFrame] = {}
_cache_timestamp: Optional[datetime] = None


def prefetch_bars(
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: Optional[datetime] = None,
    max_workers: int = 8,
    cache_ttl_sec: float = 60.0,
) -> dict[str, pd.DataFrame]:
    """PROD-001: Pre-fetch bars for all symbols concurrently using ThreadPoolExecutor.

    Called at the top of each scan cycle so that individual strategies can
    read from the cache instead of making sequential API calls.

    Args:
        symbols: List of ticker symbols to fetch.
        timeframe: Alpaca TimeFrame (e.g., 1-min, 5-min, daily).
        start: Start datetime for bars.
        end: Optional end datetime.
        max_workers: Max concurrent fetch threads (default 8).
        cache_ttl_sec: How long cached bars remain valid (default 60s).

    Returns:
        Dict mapping symbol -> DataFrame of bars.
    """
    global _bars_cache, _cache_timestamp
    from data.fetcher import get_bars

    # Return cached data if still fresh
    now = datetime.now()
    if (_cache_timestamp and (now - _cache_timestamp).total_seconds() < cache_ttl_sec
            and all(sym in _bars_cache for sym in symbols)):
        logger.debug("PROD-001: Returning cached bars for %d symbols", len(symbols))
        return {sym: _bars_cache.get(sym, pd.DataFrame()) for sym in symbols}

    results: dict[str, pd.DataFrame] = {}

    def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, get_bars(sym, timeframe, start=start, end=end)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym, df = future.result()
                results[sym] = df
            except Exception as e:
                logger.warning("PROD-001: Failed to prefetch bars for %s: %s", sym, e)
                results[sym] = pd.DataFrame()

    # Update module-level cache
    _bars_cache.update(results)
    _cache_timestamp = now
    logger.info(
        "PROD-001: Prefetched bars for %d/%d symbols (max_workers=%d)",
        sum(1 for df in results.values() if not df.empty), len(symbols), max_workers,
    )
    return results


def get_cached_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Retrieve bars from the prefetch cache for a single symbol.

    Returns None if not cached (caller should fall back to direct fetch).
    """
    return _bars_cache.get(symbol)


def clear_bars_cache():
    """Clear the prefetched bars cache (call at end of scan cycle or EOD)."""
    global _bars_cache, _cache_timestamp
    _bars_cache.clear()
    _cache_timestamp = None


def scan_all_strategies(
    current: datetime,
    regime: str,
    stat_mr,
    kalman_pairs,
    micro_mom,
    vwap_strategy,
    orb_strategy=None,
    pead_strategy=None,
    copula_pairs=None,
    cross_sectional_momentum=None,
    sector_momentum=None,
    multi_timeframe=None,
    signal_ranker=None,
    day_pnl_pct: float = 0.0,
) -> list[Signal]:
    """Scan all active strategies and return ranked signals.

    Each strategy scan is individually wrapped in try/except (fail-open).
    Returns a combined, optionally ranked signal list.
    """
    signals: list[Signal] = []
    mr_signals = []
    vwap_signals = []
    pair_signals = []
    orb_signals = []
    micro_signals = []
    pead_signals = []

    # Detect micro momentum events first
    try:
        micro_mom.detect_event(current)
    except Exception as e:
        logger.error(f"Micro event detection failed: {e}")

    # Scan each strategy
    try:
        mr_signals = stat_mr.scan(current, regime)
        signals.extend(mr_signals)
    except Exception as e:
        logger.error(f"StatMR scan failed: {e}")

    try:
        vwap_signals = vwap_strategy.scan(current, regime)
        signals.extend(vwap_signals)
    except Exception as e:
        logger.error(f"VWAP scan failed: {e}")

    try:
        pair_signals = kalman_pairs.scan(current, regime)
        signals.extend(pair_signals)
    except Exception as e:
        logger.error(f"KalmanPairs scan failed: {e}")

    if orb_strategy:
        try:
            orb_signals = orb_strategy.scan(current, regime)
            signals.extend(orb_signals)
        except Exception as e:
            logger.error(f"ORB scan failed: {e}")

    try:
        micro_signals = micro_mom.scan(
            current, day_pnl_pct=day_pnl_pct, regime=regime
        )
        signals.extend(micro_signals)
    except Exception as e:
        logger.error(f"MicroMom scan failed: {e}")

    if pead_strategy:
        try:
            pead_signals = pead_strategy.scan(current)
            signals.extend(pead_signals)
        except Exception as e:
            logger.error(f"PEAD scan failed: {e}")

    # --- New V10 strategies ---
    copula_signals = []
    csm_signals = []
    sector_mom_signals = []
    mtf_signals = []

    if copula_pairs:
        try:
            copula_signals = copula_pairs.scan(current, regime)
            signals.extend(copula_signals)
        except Exception as e:
            logger.error(f"CopulaPairs scan failed: {e}")

    if cross_sectional_momentum:
        try:
            csm_signals = cross_sectional_momentum.scan(current, regime)
            signals.extend(csm_signals)
        except Exception as e:
            logger.error(f"CrossSectionalMomentum scan failed: {e}")

    if sector_momentum:
        try:
            sector_mom_signals = sector_momentum.scan(current, regime)
            signals.extend(sector_mom_signals)
        except Exception as e:
            logger.error(f"SectorMomentum scan failed: {e}")

    if multi_timeframe:
        try:
            mtf_signals = multi_timeframe.scan(current, regime)
            signals.extend(mtf_signals)
        except Exception as e:
            logger.error(f"MultiTimeframe scan failed: {e}")

    # Rank signals by expected value
    if signal_ranker and signals:
        try:
            signals = signal_ranker.rank(signals, regime=regime)
        except Exception:
            pass

    # Log counts
    logger.info(
        f"Scan complete: {len(signals)} signals "
        f"(MR={len(mr_signals)} VWAP={len(vwap_signals)} PAIRS={len(pair_signals)} "
        f"ORB={len(orb_signals)} MICRO={len(micro_signals)} PEAD={len(pead_signals)} "
        f"COPULA={len(copula_signals)} CSM={len(csm_signals)} "
        f"SECTMOM={len(sector_mom_signals)} MTF={len(mtf_signals)}) "
        f"regime={regime}"
    )

    return signals


def check_all_exits(
    current: datetime,
    risk,
    stat_mr,
    kalman_pairs,
    micro_mom,
    orb_strategy=None,
    pead_strategy=None,
    ws_monitor=None,
):
    """Check all strategies for exit signals and process them."""
    for name, strategy in [
        ("StatMR", stat_mr),
        ("KalmanPairs", kalman_pairs),
        ("MicroMom", micro_mom),
        ("ORB", orb_strategy),
        ("PEAD", pead_strategy),
    ]:
        if strategy is None:
            continue
        try:
            exits = strategy.check_exits(risk.open_trades, current)
            if exits:
                handle_strategy_exits(exits, risk, current, ws_monitor)
        except Exception as e:
            logger.error(f"{name} exit check failed: {e}")


def run_beta_neutralization(
    current: datetime,
    risk,
    beta_neutral,
    vol_engine,
    pnl_lock,
    ws_monitor=None,
    regime: str = "UNKNOWN",
):
    """Check and apply beta neutralization if needed."""
    if not beta_neutral.should_check_now(current):
        return

    try:
        prices = get_current_prices(risk.open_trades)
        beta_neutral.compute_portfolio_beta(risk.open_trades, prices)

        if beta_neutral.needs_hedge():
            from data import get_snapshot
            try:
                spy_snap = get_snapshot("SPY")
                spy_price = float(spy_snap.latest_trade.price) if spy_snap and spy_snap.latest_trade else 0
            except Exception:
                spy_price = 0

            if spy_price > 0:
                hedge_signal = beta_neutral.compute_hedge_signal(
                    risk.current_equity, spy_price
                )
                if hedge_signal:
                    process_signals(
                        [hedge_signal], risk, regime, current,
                        vol_engine, pnl_lock, ws_monitor,
                    )
    except Exception as e:
        logger.error(f"Beta neutralization failed: {e}")
