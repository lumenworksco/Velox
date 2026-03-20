"""BACKTEST-001: Event-Driven Backtester.

Processes bars one-by-one chronologically with NO look-ahead bias.
Supports multiple strategies simultaneously, realistic fill simulation,
transaction cost modeling, partial fills, and configurable latency.

Usage:
    backtester = EventDrivenBacktester(
        initial_capital=100_000,
        commission_per_share=0.0035,
        slippage_bps=5.0,
    )
    result = backtester.run(
        strategies=[my_strategy_1, my_strategy_2],
        data={"AAPL": df_aapl, "MSFT": df_msft},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 30),
    )
    print(result.sharpe_ratio, result.max_drawdown)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===================================================================== #
#  Strategy protocol — any strategy must implement this interface
# ===================================================================== #

class StrategyProtocol(Protocol):
    """Interface that backtestable strategies must implement."""

    @property
    def name(self) -> str: ...

    def on_bar(
        self,
        symbol: str,
        bar: pd.Series,
        portfolio: "Portfolio",
    ) -> list["Order"]: ...


# ===================================================================== #
#  Data classes
# ===================================================================== #

@dataclass
class Order:
    """Order submitted by a strategy."""
    symbol: str
    side: str                  # "buy" or "sell"
    qty: int
    order_type: str = "market"  # "market" or "limit"
    limit_price: Optional[float] = None
    strategy: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.side = self.side.lower()
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{self.side}'")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")


@dataclass
class Fill:
    """Executed fill from the simulated order book."""
    symbol: str
    side: str
    qty: int
    fill_price: float
    commission: float
    slippage: float
    timestamp: datetime
    strategy: str = ""


@dataclass
class Position:
    """Open position tracked by the portfolio."""
    symbol: str
    side: str
    qty: int
    avg_entry_price: float
    strategy: str
    entry_time: datetime
    unrealized_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.avg_entry_price


@dataclass
class ClosedTrade:
    """Completed round-trip trade."""
    symbol: str
    strategy: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    slippage: float
    hold_minutes: float


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    # Identification
    strategies: list[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # Trades
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_minutes: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # Per-strategy breakdown
    strategy_results: dict[str, dict] = field(default_factory=dict)

    # Time series
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    trades: list[ClosedTrade] = field(default_factory=list)


# ===================================================================== #
#  Portfolio
# ===================================================================== #

class Portfolio:
    """Simulated portfolio with position tracking and P&L computation."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._equity_history: list[tuple[datetime, float]] = []

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + positions at mark)."""
        position_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + sum(p.notional for p in self.positions.values()) + position_value

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_position_qty(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.qty if pos else 0

    def apply_fill(self, fill: Fill) -> Optional[ClosedTrade]:
        """Apply a fill to the portfolio. Returns ClosedTrade if position closed."""
        key = fill.symbol
        self.cash -= fill.commission

        if fill.side == "buy":
            if key in self.positions and self.positions[key].side == "sell":
                # Closing a short
                return self._close_position(key, fill)
            else:
                # Opening or adding to long
                self._open_or_add(key, fill)
                return None
        else:  # sell
            if key in self.positions and self.positions[key].side == "buy":
                # Closing a long
                return self._close_position(key, fill)
            else:
                # Opening or adding to short
                self._open_or_add(key, fill)
                return None

    def _open_or_add(self, key: str, fill: Fill) -> None:
        """Open new position or add to existing."""
        cost = fill.qty * fill.fill_price
        if fill.side == "buy":
            self.cash -= cost
        else:
            self.cash += cost  # short: receive cash

        if key in self.positions:
            pos = self.positions[key]
            total_qty = pos.qty + fill.qty
            pos.avg_entry_price = (
                (pos.avg_entry_price * pos.qty + fill.fill_price * fill.qty) / total_qty
            )
            pos.qty = total_qty
        else:
            self.positions[key] = Position(
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                avg_entry_price=fill.fill_price,
                strategy=fill.strategy,
                entry_time=fill.timestamp,
            )

    def _close_position(self, key: str, fill: Fill) -> ClosedTrade:
        """Close an existing position and record the trade."""
        pos = self.positions[key]
        close_qty = min(fill.qty, pos.qty)

        if pos.side == "buy":
            pnl = (fill.fill_price - pos.avg_entry_price) * close_qty
            self.cash += close_qty * fill.fill_price
        else:
            pnl = (pos.avg_entry_price - fill.fill_price) * close_qty
            self.cash -= close_qty * fill.fill_price

        hold_delta = fill.timestamp - pos.entry_time
        hold_minutes = hold_delta.total_seconds() / 60.0

        trade = ClosedTrade(
            symbol=pos.symbol,
            strategy=pos.strategy,
            side=pos.side,
            qty=close_qty,
            entry_price=pos.avg_entry_price,
            exit_price=fill.fill_price,
            entry_time=pos.entry_time,
            exit_time=fill.timestamp,
            pnl=pnl - fill.commission,
            commission=fill.commission,
            slippage=fill.slippage,
            hold_minutes=hold_minutes,
        )
        self._closed_trades.append(trade)

        # Update or remove position
        remaining = pos.qty - close_qty
        if remaining <= 0:
            del self.positions[key]
        else:
            pos.qty = remaining

        return trade

    def mark_to_market(self, prices: dict[str, float], timestamp: datetime) -> None:
        """Update unrealized P&L for all positions."""
        for sym, pos in self.positions.items():
            if sym in prices:
                if pos.side == "buy":
                    pos.unrealized_pnl = (prices[sym] - pos.avg_entry_price) * pos.qty
                else:
                    pos.unrealized_pnl = (pos.avg_entry_price - prices[sym]) * pos.qty

        self._equity_history.append((timestamp, self.equity))


# ===================================================================== #
#  Fill simulator
# ===================================================================== #

class FillSimulator:
    """Simulates order fills with realistic slippage and partial fills.

    Args:
        commission_per_share: Commission cost per share.
        slippage_bps: Slippage in basis points.
        max_fill_pct: Maximum fraction of bar volume that can be filled.
        latency_bars: Number of bars of delay before fills execute.
    """

    def __init__(
        self,
        commission_per_share: float = 0.0035,
        slippage_bps: float = 5.0,
        max_fill_pct: float = 0.10,
        latency_bars: int = 0,
    ) -> None:
        self._commission = commission_per_share
        self._slippage_bps = slippage_bps
        self._max_fill_pct = max_fill_pct
        self._latency_bars = latency_bars
        self._pending_orders: list[tuple[int, Order]] = []  # (bars_remaining, order)

    def submit_order(self, order: Order) -> None:
        """Submit an order with optional latency delay."""
        self._pending_orders.append((self._latency_bars, order))

    def process_bar(
        self,
        bar: pd.Series,
        symbol: str,
        bar_timestamp: datetime,
    ) -> list[Fill]:
        """Process pending orders against a bar. Returns list of fills."""
        fills = []
        remaining_orders = []

        for bars_left, order in self._pending_orders:
            if order.symbol != symbol:
                remaining_orders.append((bars_left, order))
                continue

            if bars_left > 0:
                remaining_orders.append((bars_left - 1, order))
                continue

            # Ready to fill
            fill = self._try_fill(order, bar, bar_timestamp)
            if fill is not None:
                fills.append(fill)
            # If fill failed (e.g. limit not reached), discard the order

        self._pending_orders = remaining_orders
        return fills

    def _try_fill(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: datetime,
    ) -> Optional[Fill]:
        """Attempt to fill an order against a bar."""
        bar_volume = int(bar.get("volume", 0))
        if bar_volume <= 0:
            return None

        # Partial fill: cap at max_fill_pct of bar volume
        fillable_qty = max(1, int(bar_volume * self._max_fill_pct))
        fill_qty = min(order.qty, fillable_qty)

        # Determine fill price
        if order.order_type == "limit" and order.limit_price is not None:
            if order.side == "buy" and bar["low"] > order.limit_price:
                return None  # Limit not reached
            if order.side == "sell" and bar["high"] < order.limit_price:
                return None
            base_price = order.limit_price
        else:
            # Market order: fill at open of the bar (conservative)
            base_price = float(bar.get("open", bar["close"]))

        # Apply slippage
        slippage_mult = self._slippage_bps / 10_000
        if order.side == "buy":
            fill_price = base_price * (1.0 + slippage_mult)
        else:
            fill_price = base_price * (1.0 - slippage_mult)

        slippage_cost = abs(fill_price - base_price) * fill_qty
        commission = self._commission * fill_qty

        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            timestamp=timestamp,
            strategy=order.strategy,
        )


# ===================================================================== #
#  Event-Driven Backtester
# ===================================================================== #

class EventDrivenBacktester:
    """Event-driven backtester that processes bars chronologically.

    Ensures NO look-ahead bias by feeding bars one at a time to strategies.
    Supports multiple strategies running simultaneously on the same data.

    Args:
        initial_capital: Starting portfolio value.
        commission_per_share: Commission per share for fill simulation.
        slippage_bps: Slippage in basis points.
        max_fill_pct: Max fraction of bar volume fillable per order.
        latency_bars: Simulated execution delay in bars.
        risk_free_rate: Annual risk-free rate for Sharpe computation.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_per_share: float = 0.0035,
        slippage_bps: float = 5.0,
        max_fill_pct: float = 0.10,
        latency_bars: int = 0,
        risk_free_rate: float = 0.045,
    ) -> None:
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate
        self._fill_sim = FillSimulator(
            commission_per_share=commission_per_share,
            slippage_bps=slippage_bps,
            max_fill_pct=max_fill_pct,
            latency_bars=latency_bars,
        )

    def run(
        self,
        strategies: list[Any],
        data: dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Run a full backtest.

        Args:
            strategies: List of strategy objects implementing StrategyProtocol.
            data: Dict of symbol -> DataFrame with OHLCV columns
                  (open, high, low, close, volume). Index must be datetime.
            start_date: Start date filter (inclusive). None = use all data.
            end_date: End date filter (inclusive). None = use all data.

        Returns:
            BacktestResult with comprehensive metrics and trade log.
        """
        if not strategies:
            raise ValueError("At least one strategy is required")
        if not data:
            raise ValueError("Data dictionary cannot be empty")

        # Normalize column names
        normalized_data = {}
        for sym, df in data.items():
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            if start_date is not None:
                df = df[df.index.date >= start_date]
            if end_date is not None:
                df = df[df.index.date <= end_date]
            if len(df) > 0:
                normalized_data[sym] = df

        if not normalized_data:
            logger.warning("No data remaining after date filtering")
            return BacktestResult(
                strategies=[getattr(s, "name", str(s)) for s in strategies],
                start_date=start_date,
                end_date=end_date,
            )

        # Build unified timeline
        all_timestamps = set()
        for df in normalized_data.values():
            all_timestamps.update(df.index)
        timeline = sorted(all_timestamps)

        logger.info(
            "Running backtest: %d strategies, %d symbols, %d bars (%s to %s)",
            len(strategies), len(normalized_data), len(timeline),
            timeline[0].date() if timeline else "N/A",
            timeline[-1].date() if timeline else "N/A",
        )

        portfolio = Portfolio(self._initial_capital)

        # Process each bar chronologically
        for ts in timeline:
            current_prices = {}

            for sym, df in normalized_data.items():
                if ts not in df.index:
                    continue

                bar = df.loc[ts]
                # Handle duplicate index entries
                if isinstance(bar, pd.DataFrame):
                    bar = bar.iloc[0]

                current_prices[sym] = float(bar["close"])

                # Process pending fills
                fills = self._fill_sim.process_bar(bar, sym, ts)
                for fill in fills:
                    portfolio.apply_fill(fill)

                # Let each strategy react to this bar
                for strategy in strategies:
                    try:
                        orders = strategy.on_bar(sym, bar, portfolio)
                        if orders:
                            for order in orders:
                                order.strategy = getattr(strategy, "name", str(strategy))
                                order.timestamp = ts
                                self._fill_sim.submit_order(order)
                    except Exception as e:
                        logger.error(
                            "Strategy %s error on %s at %s: %s",
                            getattr(strategy, "name", "?"), sym, ts, e,
                        )

            # Mark to market at end of bar
            if current_prices:
                portfolio.mark_to_market(current_prices, ts)

        # Compute result
        return self._compute_result(
            portfolio=portfolio,
            strategies=strategies,
            start_date=start_date or (timeline[0].date() if timeline else None),
            end_date=end_date or (timeline[-1].date() if timeline else None),
        )

    def _compute_result(
        self,
        portfolio: Portfolio,
        strategies: list[Any],
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> BacktestResult:
        """Compute comprehensive backtest metrics from portfolio state."""
        trades = portfolio._closed_trades
        equity_hist = portfolio._equity_history
        strategy_names = [getattr(s, "name", str(s)) for s in strategies]

        result = BacktestResult(
            strategies=strategy_names,
            start_date=start_date,
            end_date=end_date,
            total_trades=len(trades),
            trades=trades,
        )

        if not trades:
            return result

        # Equity curve
        if equity_hist:
            timestamps, values = zip(*equity_hist)
            result.equity_curve = pd.Series(values, index=pd.DatetimeIndex(timestamps))

            # Daily returns
            daily_equity = result.equity_curve.resample("D").last().dropna()
            if len(daily_equity) > 1:
                daily_rets = daily_equity.pct_change().dropna()
                result.daily_returns = daily_rets

                # Annualized return
                total_days = (daily_equity.index[-1] - daily_equity.index[0]).days
                total_return = (daily_equity.iloc[-1] / daily_equity.iloc[0]) - 1.0
                result.total_return = total_return
                if total_days > 0:
                    result.annualized_return = (1 + total_return) ** (365.25 / total_days) - 1

                # Volatility
                result.volatility = float(daily_rets.std() * np.sqrt(252))

                # Sharpe ratio
                daily_rf = self._risk_free_rate / 252
                excess = daily_rets - daily_rf
                if len(excess) > 1 and excess.std() > 0:
                    result.sharpe_ratio = float(
                        excess.mean() / excess.std() * np.sqrt(252)
                    )

                # Sortino ratio
                downside = daily_rets[daily_rets < 0]
                if len(downside) > 1:
                    downside_std = float(downside.std())
                    if downside_std > 0:
                        result.sortino_ratio = float(
                            (daily_rets.mean() - daily_rf) / downside_std * np.sqrt(252)
                        )

                # Max drawdown
                cummax = daily_equity.cummax()
                drawdowns = (daily_equity - cummax) / cummax
                result.max_drawdown = float(abs(drawdowns.min()))

                # Max drawdown duration
                in_drawdown = drawdowns < 0
                if in_drawdown.any():
                    dd_groups = (~in_drawdown).cumsum()
                    dd_lengths = in_drawdown.groupby(dd_groups).sum()
                    result.max_drawdown_duration_days = int(dd_lengths.max())

                # Calmar ratio
                if result.max_drawdown > 0:
                    result.calmar_ratio = result.annualized_return / result.max_drawdown

        # Trade statistics
        pnls = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        result.win_rate = len(winners) / len(pnls) if pnls else 0.0
        result.avg_win = float(np.mean(winners)) if winners else 0.0
        result.avg_loss = float(np.mean(losers)) if losers else 0.0
        result.largest_win = max(pnls) if pnls else 0.0
        result.largest_loss = min(pnls) if pnls else 0.0

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.9

        result.avg_hold_minutes = float(np.mean([t.hold_minutes for t in trades]))
        result.total_commission = sum(t.commission for t in trades)
        result.total_slippage = sum(t.slippage for t in trades)

        # Per-strategy breakdown
        strategy_trades: dict[str, list[ClosedTrade]] = defaultdict(list)
        for t in trades:
            strategy_trades[t.strategy].append(t)

        for strat_name, strat_trades in strategy_trades.items():
            strat_pnls = [t.pnl for t in strat_trades]
            strat_winners = [p for p in strat_pnls if p > 0]
            strat_losers = [p for p in strat_pnls if p <= 0]
            strat_gross_profit = sum(strat_winners)
            strat_gross_loss = abs(sum(strat_losers))

            result.strategy_results[strat_name] = {
                "total_trades": len(strat_trades),
                "total_pnl": sum(strat_pnls),
                "win_rate": len(strat_winners) / len(strat_pnls) if strat_pnls else 0.0,
                "profit_factor": (
                    strat_gross_profit / strat_gross_loss if strat_gross_loss > 0 else 999.9
                ),
                "avg_hold_minutes": float(np.mean([t.hold_minutes for t in strat_trades])),
            }

        logger.info(
            "Backtest complete: %d trades, Sharpe=%.2f, Return=%.1f%%, MaxDD=%.1f%%",
            result.total_trades, result.sharpe_ratio,
            result.total_return * 100, result.max_drawdown * 100,
        )

        return result
