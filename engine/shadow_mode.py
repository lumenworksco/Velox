"""PROD-012: Shadow trading framework — run strategies in parallel without real execution.

Allows new or experimental strategies to run alongside production strategies.
Signals are captured and virtual P&L is tracked, but no real orders are submitted.
Results are stored in the shadow_trades database table for comparison.

Usage:
    from engine.shadow_mode import ShadowTrader

    shadow = ShadowTrader(strategies=["NEW_STRATEGY_V2"])
    shadow.process_signals(signals, current_time)
    shadow.check_exits(current_prices, current_time)
    print(shadow.performance_summary())
"""

import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class ShadowPosition:
    """A virtual position held by the shadow trader."""
    symbol: str
    strategy: str
    side: str  # "buy" or "sell"
    entry_price: float
    qty: float
    entry_time: datetime
    take_profit: float
    stop_loss: float
    time_stop: Optional[datetime] = None
    db_id: Optional[int] = None  # shadow_trades row ID


class ShadowTrader:
    """PROD-012: Shadow trading framework for strategy evaluation.

    Runs in parallel with production trading but never submits real orders.
    Tracks virtual entries/exits and logs results to the shadow_trades table.
    """

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        max_positions: int = 20,
        initial_capital: float = 100_000.0,
        position_size_pct: float = 0.05,
    ):
        """Initialize the shadow trader.

        Args:
            strategies: Strategy names to shadow. If None, shadows all strategies.
            max_positions: Maximum concurrent shadow positions.
            initial_capital: Virtual starting capital for P&L tracking.
            position_size_pct: Default position size as fraction of capital.
        """
        self._strategies: Optional[Set[str]] = set(strategies) if strategies else None
        self._max_positions = max_positions
        self._initial_capital = initial_capital
        self._virtual_capital = initial_capital
        self._position_size_pct = position_size_pct

        # Active shadow positions: symbol -> ShadowPosition
        self._positions: Dict[str, ShadowPosition] = {}

        # Performance tracking
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._trade_log: List[dict] = []

        logger.info(
            "PROD-012: ShadowTrader initialized (strategies=%s, max_positions=%d, capital=$%.0f)",
            strategies or "ALL", max_positions, initial_capital,
        )

    def should_shadow(self, strategy: str) -> bool:
        """Check if a strategy should be shadowed.

        Returns True if no filter is set (shadow all) or strategy is in the filter list.
        """
        if self._strategies is None:
            return True
        return strategy in self._strategies

    def process_signals(self, signals: List[Signal], current_time: datetime):
        """Process signals and create virtual positions (no real orders).

        Args:
            signals: List of trading signals from strategies.
            current_time: Current datetime.
        """
        for signal in signals:
            if not self.should_shadow(signal.strategy):
                continue

            # Skip if already have a position in this symbol
            if signal.symbol in self._positions:
                continue

            # Skip if at max positions
            if len(self._positions) >= self._max_positions:
                logger.debug(
                    "PROD-012: Shadow position limit reached (%d), skipping %s",
                    self._max_positions, signal.symbol,
                )
                continue

            # Calculate virtual position size
            qty = int(
                (self._virtual_capital * self._position_size_pct) / signal.entry_price
            )
            if qty <= 0:
                continue

            position = ShadowPosition(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                entry_price=signal.entry_price,
                qty=qty,
                entry_time=current_time,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                time_stop=getattr(signal, "time_stop", None),
            )

            # Log to database
            try:
                import database
                database.log_shadow_entry(
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    side=signal.side,
                    entry_price=signal.entry_price,
                    qty=qty,
                    entry_time=current_time,
                    take_profit=signal.take_profit,
                    stop_loss=signal.stop_loss,
                    time_stop=position.time_stop,
                )
            except Exception as e:
                logger.warning("PROD-012: Failed to log shadow entry: %s", e)

            self._positions[signal.symbol] = position
            logger.info(
                "PROD-012: Shadow ENTRY %s %s %s @ $%.2f (qty=%d, TP=%.2f, SL=%.2f)",
                signal.strategy, signal.side.upper(), signal.symbol,
                signal.entry_price, qty, signal.take_profit, signal.stop_loss,
            )

    def check_exits(
        self,
        current_prices: Dict[str, float],
        current_time: datetime,
    ):
        """Check shadow positions for exit conditions (TP, SL, time stop).

        Args:
            current_prices: Dict mapping symbol -> current price.
            current_time: Current datetime.
        """
        exits_to_process: List[tuple[str, str, float]] = []  # (symbol, reason, price)

        for symbol, pos in list(self._positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            exit_reason = None

            if pos.side == "buy":
                if price >= pos.take_profit:
                    exit_reason = "take_profit"
                elif price <= pos.stop_loss:
                    exit_reason = "stop_loss"
            else:  # sell/short
                if price <= pos.take_profit:
                    exit_reason = "take_profit"
                elif price >= pos.stop_loss:
                    exit_reason = "stop_loss"

            # Time stop
            if pos.time_stop and current_time >= pos.time_stop:
                exit_reason = "time_stop"

            if exit_reason:
                exits_to_process.append((symbol, exit_reason, price))

        for symbol, reason, price in exits_to_process:
            self._close_position(symbol, price, current_time, reason)

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ):
        """Close a shadow position and record P&L."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return

        # Calculate P&L
        if pos.side == "buy":
            pnl = (exit_price - pos.entry_price) * pos.qty
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) * pos.qty
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        self._total_trades += 1
        self._total_pnl += pnl
        if pnl > 0:
            self._winning_trades += 1
        self._virtual_capital += pnl

        trade_record = {
            "symbol": symbol,
            "strategy": pos.strategy,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "qty": pos.qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, "isoformat") else str(pos.entry_time),
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
        }
        self._trade_log.append(trade_record)

        # Log to database
        try:
            import database
            # Find the shadow trade ID and close it
            open_shadows = database.get_open_shadow_trades()
            for st in open_shadows:
                if st["symbol"] == symbol and st["strategy"] == pos.strategy:
                    database.close_shadow_trade(
                        st["id"], exit_price, exit_time, exit_reason,
                    )
                    break
        except Exception as e:
            logger.warning("PROD-012: Failed to close shadow trade in DB: %s", e)

        logger.info(
            "PROD-012: Shadow EXIT %s %s @ $%.2f -> $%.2f (%s) P&L=$%.2f (%.2f%%)",
            pos.strategy, symbol, pos.entry_price, exit_price,
            exit_reason, pnl, pnl_pct * 100,
        )

    def close_all(self, current_prices: Dict[str, float], current_time: datetime):
        """Close all shadow positions (e.g., at EOD).

        Args:
            current_prices: Current prices for all symbols.
            current_time: Current datetime.
        """
        for symbol in list(self._positions.keys()):
            price = current_prices.get(symbol)
            if price is not None:
                self._close_position(symbol, price, current_time, "market_close")

    def performance_summary(self) -> dict:
        """Return shadow trading performance summary."""
        win_rate = (
            self._winning_trades / self._total_trades
            if self._total_trades > 0
            else 0.0
        )
        return_pct = (
            (self._virtual_capital - self._initial_capital) / self._initial_capital
        )

        return {
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(self._total_pnl, 2),
            "return_pct": round(return_pct, 4),
            "virtual_capital": round(self._virtual_capital, 2),
            "open_positions": len(self._positions),
            "strategies_shadowed": list(self._strategies) if self._strategies else "ALL",
        }

    @property
    def open_positions(self) -> Dict[str, ShadowPosition]:
        """Get currently open shadow positions."""
        return dict(self._positions)

    @property
    def trade_log(self) -> List[dict]:
        """Get the full shadow trade log."""
        return list(self._trade_log)
