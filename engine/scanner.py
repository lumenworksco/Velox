"""V10 Engine — Strategy scanning and signal generation.

Extracts the multi-strategy scanning loop from main.py into a reusable module.
Handles scanning all active strategies, ranking signals, and processing them.
"""

import logging
from datetime import datetime

from strategies.base import Signal
from engine.signal_processor import process_signals
from engine.exit_processor import handle_strategy_exits, get_current_prices

logger = logging.getLogger(__name__)


def scan_all_strategies(
    current: datetime,
    regime: str,
    stat_mr,
    kalman_pairs,
    micro_mom,
    vwap_strategy,
    orb_strategy=None,
    pead_strategy=None,
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
        f"ORB={len(orb_signals)} MICRO={len(micro_signals)} PEAD={len(pead_signals)}) "
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
