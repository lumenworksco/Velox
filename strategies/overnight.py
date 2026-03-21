"""Overnight hold management — decide which positions to hold overnight vs close at EOD."""

import logging
from dataclasses import dataclass

import config
from earnings import has_earnings_soon

logger = logging.getLogger(__name__)

# WIRE-008: Gap risk sizing for overnight holds (fail-open)
_gap_risk_manager = None
try:
    from risk.gap_risk import GapRiskManager as _GRM
    _gap_risk_manager = _GRM()
except ImportError:
    _GRM = None


@dataclass
class OvernightDecision:
    symbol: str
    action: str  # "hold" or "close"
    reason: str
    size_reduction: float  # 0.0 (no change) to 1.0 (close all)


class OvernightManager:
    """Decides which positions to hold overnight vs close at EOD.

    Called at 3:45 PM ET to evaluate open positions.  Positions that pass
    all overnight eligibility criteria are marked "hold" with a partial
    size reduction; everything else is marked "close".
    """

    ELIGIBLE_STRATEGIES = set(config.OVERNIGHT_ELIGIBLE_STRATEGIES)

    def __init__(self):
        self._overnight_positions: dict[str, dict] = {}  # symbol -> info

    def select_overnight_holds(
        self,
        open_trades: dict,
        regime: str = "UNKNOWN",
        cross_asset_signals: dict | None = None,
    ) -> list[OvernightDecision]:
        """Run at 3:45 PM ET.  Decide which positions to hold overnight.

        Selection criteria:
        1. Feature must be enabled (OVERNIGHT_HOLD_ENABLED)
        2. Strategy must be in ELIGIBLE_STRATEGIES
        3. Position must be in profit (pnl > MIN_PROFIT_PCT)
        4. No earnings for symbol before next open (has_earnings_soon)
        5. Cross-asset not showing flight-to-safety
        6. Max OVERNIGHT_MAX_POSITIONS positions held overnight
        7. Held positions reduce size by OVERNIGHT_SIZE_REDUCTION

        All closed positions get action="close", reason="eod_close".
        """
        decisions: list[OvernightDecision] = []

        if not config.OVERNIGHT_HOLD_ENABLED:
            # Everything closes
            for symbol in open_trades:
                decisions.append(
                    OvernightDecision(
                        symbol=symbol,
                        action="close",
                        reason="overnight_disabled",
                        size_reduction=1.0,
                    )
                )
            return decisions

        # Check flight-to-safety from cross-asset signals
        flight_to_safety = False
        if cross_asset_signals and cross_asset_signals.get("flight_to_safety"):
            flight_to_safety = True

        # Separate candidates from non-candidates
        candidates: list[tuple[str, object]] = []  # (symbol, trade)
        non_candidates: list[tuple[str, str]] = []  # (symbol, reason)

        for symbol, trade in open_trades.items():
            # 1. Strategy eligibility
            if trade.strategy not in self.ELIGIBLE_STRATEGIES:
                non_candidates.append((symbol, "ineligible_strategy"))
                continue

            # 2. Profit threshold
            if trade.entry_price > 0 and trade.qty > 0:
                position_value = trade.entry_price * trade.qty
                pnl_pct = trade.pnl / position_value if position_value > 0 else 0.0
            else:
                pnl_pct = 0.0

            if pnl_pct < config.OVERNIGHT_MIN_PROFIT_PCT:
                non_candidates.append((symbol, "insufficient_profit"))
                continue

            # 3. Earnings check
            if has_earnings_soon(symbol):
                non_candidates.append((symbol, "earnings_soon"))
                continue

            # 4. Flight-to-safety
            if flight_to_safety:
                non_candidates.append((symbol, "flight_to_safety"))
                continue

            candidates.append((symbol, trade))

        # 5. Max positions limit — take the most profitable candidates
        candidates.sort(
            key=lambda st: st[1].pnl / (st[1].entry_price * st[1].qty)
            if (st[1].entry_price * st[1].qty) > 0
            else 0.0,
            reverse=True,
        )

        held = 0
        for symbol, trade in candidates:
            if held < config.OVERNIGHT_MAX_POSITIONS:
                # WIRE-008: Gap risk sizing adjustment (fail-open)
                gap_size_reduction = config.OVERNIGHT_SIZE_REDUCTION
                try:
                    if _gap_risk_manager is not None:
                        gap_mult = _gap_risk_manager.get_overnight_sizing_multiplier(symbol)
                        if gap_mult < 1.0:
                            # Increase reduction (closer to 1.0 = close more)
                            gap_size_reduction = 1.0 - (1.0 - config.OVERNIGHT_SIZE_REDUCTION) * gap_mult
                            logger.info("WIRE-008: %s gap risk mult=%.2f, adjusted reduction=%.2f",
                                        symbol, gap_mult, gap_size_reduction)
                except Exception as _e:
                    logger.debug("WIRE-008: Gap risk check failed for %s (fail-open): %s", symbol, _e)

                decisions.append(
                    OvernightDecision(
                        symbol=symbol,
                        action="hold",
                        reason="overnight_hold",
                        size_reduction=gap_size_reduction,
                    )
                )
                # Store overnight info
                self._overnight_positions[symbol] = {
                    "entry_price": trade.entry_price,
                    "side": trade.side,
                    "qty": trade.qty,
                    "strategy": trade.strategy,
                    "stop_loss": trade.stop_loss,
                }
                held += 1
            else:
                non_candidates.append((symbol, "max_overnight_reached"))

        # All non-candidates close
        for symbol, reason in non_candidates:
            decisions.append(
                OvernightDecision(
                    symbol=symbol,
                    action="close",
                    reason=reason,
                    size_reduction=1.0,
                )
            )

        logger.info(
            "Overnight decisions: %d hold, %d close",
            sum(1 for d in decisions if d.action == "hold"),
            sum(1 for d in decisions if d.action == "close"),
        )

        return decisions

    def morning_gap_check(
        self,
        overnight_positions: dict,
        current_prices: dict,
    ) -> list[dict]:
        """Run at 9:31 AM ET.  Check gap direction for overnight positions.

        Returns list of exit action dicts for positions that gapped against us:
        - If gap against > OVERNIGHT_GAP_STOP_PCT, return full exit
        - Otherwise, update trailing stop
        """
        actions: list[dict] = []

        for symbol, info in overnight_positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = info["entry_price"]
            side = info["side"]

            # Calculate gap percentage
            gap_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0

            # Determine if gap is adverse
            if side == "buy":
                adverse = gap_pct < 0
                adverse_magnitude = abs(gap_pct)
            else:  # sell / short
                adverse = gap_pct > 0
                adverse_magnitude = abs(gap_pct)

            if adverse and adverse_magnitude > config.OVERNIGHT_GAP_STOP_PCT:
                actions.append({
                    "symbol": symbol,
                    "action": "exit",
                    "reason": "overnight_gap_stop",
                    "gap_pct": gap_pct,
                })
                logger.warning(
                    "Gap stop triggered for %s: gap=%.2f%%, threshold=%.2f%%",
                    symbol,
                    gap_pct * 100,
                    config.OVERNIGHT_GAP_STOP_PCT * 100,
                )
            else:
                # Update trailing stop based on current price
                if side == "buy":
                    new_stop = current_price * (1 - config.OVERNIGHT_GAP_STOP_PCT)
                else:
                    new_stop = current_price * (1 + config.OVERNIGHT_GAP_STOP_PCT)

                actions.append({
                    "symbol": symbol,
                    "action": "update_stop",
                    "reason": "overnight_gap_ok",
                    "new_stop": new_stop,
                    "gap_pct": gap_pct,
                })

        return actions

    def get_overnight_positions(self) -> dict:
        """Return current overnight hold info."""
        return dict(self._overnight_positions)

    def reset_daily(self):
        """Clear overnight state."""
        self._overnight_positions.clear()
