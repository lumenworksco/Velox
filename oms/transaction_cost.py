"""V10 OMS — Pre-trade transaction cost estimation.

Models spread, slippage, and commission to reject negative-EV trades.
This is PROFIT-GAP-001 from the audit — highest-ROI single change.
"""

import logging

import config

logger = logging.getLogger(__name__)

# Default cost parameters (configurable via config)
DEFAULT_COMMISSION_PER_SHARE = 0.0  # Alpaca: zero commission
DEFAULT_SPREAD_BPS = 3.0            # 3 basis points typical spread
DEFAULT_SLIPPAGE_BPS = 2.0          # 2 basis points market impact
MIN_EXPECTED_RETURN_BPS = 10.0      # Minimum 10 bps after costs


def estimate_round_trip_cost(
    entry_price: float,
    qty: int,
    side: str = "buy",
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    commission_per_share: float | None = None,
) -> dict:
    """Estimate round-trip transaction costs for a trade.

    Returns:
        dict with keys:
            spread_cost: Estimated spread cost ($)
            slippage_cost: Estimated slippage cost ($)
            commission_cost: Commission cost ($)
            total_cost: Total round-trip cost ($)
            cost_pct: Total cost as percentage of trade value
            cost_bps: Total cost in basis points
    """
    if spread_bps is None:
        spread_bps = getattr(config, "COST_SPREAD_BPS", DEFAULT_SPREAD_BPS)
    if slippage_bps is None:
        slippage_bps = getattr(config, "COST_SLIPPAGE_BPS", DEFAULT_SLIPPAGE_BPS)
    if commission_per_share is None:
        commission_per_share = getattr(config, "COST_COMMISSION_PER_SHARE", DEFAULT_COMMISSION_PER_SHARE)

    trade_value = entry_price * qty

    # Spread: paid on both entry and exit
    spread_cost = trade_value * (spread_bps / 10_000) * 2

    # Slippage: market impact on both legs
    slippage_cost = trade_value * (slippage_bps / 10_000) * 2

    # Commission: per share on both entry and exit
    commission_cost = commission_per_share * qty * 2

    total_cost = spread_cost + slippage_cost + commission_cost
    cost_pct = total_cost / trade_value if trade_value > 0 else 0
    cost_bps = cost_pct * 10_000

    return {
        "spread_cost": round(spread_cost, 4),
        "slippage_cost": round(slippage_cost, 4),
        "commission_cost": round(commission_cost, 4),
        "total_cost": round(total_cost, 4),
        "cost_pct": round(cost_pct, 6),
        "cost_bps": round(cost_bps, 2),
    }


def _get_strategy_win_rate(strategy: str, default_win_rate: float = 0.55,
                           min_trades: int = 30) -> float:
    """BUG-026: Fetch per-strategy win rate from trade database.

    Falls back to default_win_rate if fewer than min_trades exist for
    the strategy.
    """
    if not strategy:
        return default_win_rate

    try:
        import database
        trades = database.get_recent_trades_by_strategy(strategy, days=90)
        if len(trades) >= min_trades:
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            db_win_rate = wins / len(trades)
            logger.debug(f"BUG-026: Strategy {strategy} win rate from DB: "
                        f"{db_win_rate:.1%} ({wins}/{len(trades)} trades)")
            return db_win_rate
    except Exception as e:
        logger.debug(f"BUG-026: Failed to fetch win rate for {strategy}: {e}")

    return default_win_rate


def is_trade_profitable_after_costs(
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    qty: int,
    side: str = "buy",
    win_rate: float = 0.55,
    strategy: str = "",
) -> tuple[bool, dict]:
    """Check if a trade has positive expected value after transaction costs.

    Args:
        entry_price: Entry price
        take_profit: Take profit target
        stop_loss: Stop loss level
        qty: Position size
        side: "buy" or "sell"
        win_rate: Historical win rate for this strategy (default 55%)
        strategy: BUG-026: Strategy name for per-strategy win rate lookup

    Returns:
        (is_profitable, details_dict)
    """
    # BUG-026: Use strategy-specific win rate if available
    if strategy:
        win_rate = _get_strategy_win_rate(strategy, default_win_rate=win_rate)

    costs = estimate_round_trip_cost(entry_price, qty, side)

    if side == "buy":
        gross_profit = (take_profit - entry_price) * qty
        gross_loss = (entry_price - stop_loss) * qty
    else:
        gross_profit = (entry_price - take_profit) * qty
        gross_loss = (stop_loss - entry_price) * qty

    net_profit = gross_profit - costs["total_cost"]
    net_loss = gross_loss + costs["total_cost"]

    expected_value = (win_rate * net_profit) - ((1 - win_rate) * net_loss)

    min_return_bps = getattr(config, "COST_MIN_EXPECTED_RETURN_BPS", MIN_EXPECTED_RETURN_BPS)
    min_return = entry_price * qty * (min_return_bps / 10_000)

    details = {
        **costs,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit": round(net_profit, 2),
        "net_loss": round(net_loss, 2),
        "expected_value": round(expected_value, 2),
        "win_rate": win_rate,
        "profitable": expected_value > min_return,
    }

    return expected_value > min_return, details
