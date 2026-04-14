"""Alpaca data fetching — bars, account info, market status."""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

import config

logger = logging.getLogger(__name__)

# --- Clients ---
_trading_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None


def get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
            paper=config.PAPER_MODE,
        )
    return _trading_client


def get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=config.API_KEY or None,
            secret_key=config.API_SECRET or None,
        )
    return _data_client


def get_account():
    """Get current account info."""
    return get_trading_client().get_account()


def get_clock():
    """Get market clock (open/close status, next open/close times)."""
    return get_trading_client().get_clock()


def get_positions():
    """Get all open positions from Alpaca."""
    return get_trading_client().get_all_positions()


def get_open_orders():
    """Get all open orders."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    return get_trading_client().get_orders(request)


def get_bars(symbol: str, timeframe: TimeFrame, start: datetime, end: datetime | None = None, limit: int | None = None) -> pd.DataFrame:
    """Fetch historical bars for a symbol, return as DataFrame."""
    params = {"symbol_or_symbols": symbol, "timeframe": timeframe, "start": start}
    if end:
        params["end"] = end
    if limit:
        params["limit"] = limit

    params["feed"] = DataFeed.IEX
    request = StockBarsRequest(**params)
    barset = get_data_client().get_stock_bars(request)

    # alpaca-py returns a BarSet; access via .data or dict-style
    bar_list = None
    if hasattr(barset, "data") and barset.data:
        bar_list = barset.data.get(symbol)
    elif isinstance(barset, dict):
        bar_list = barset.get(symbol)
    else:
        # Try direct dict access (some alpaca-py versions)
        try:
            bar_list = barset[symbol]
        except (KeyError, TypeError):
            pass

    if not bar_list:
        return pd.DataFrame()

    records = []
    for bar in bar_list:
        records.append({
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.vwap) if bar.vwap else None,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
    return df


def get_daily_bars(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get daily bars for the last N days."""
    start = datetime.now(config.ET) - timedelta(days=days + 5)  # buffer for weekends
    return get_bars(symbol, TimeFrame.Day, start=start, limit=days)


def get_intraday_bars(symbol: str, timeframe: TimeFrame, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """Get intraday bars from start time."""
    return get_bars(symbol, timeframe, start=start, end=end)


def get_filled_exit_info(symbol: str, side: str = "buy",
                         after: datetime | None = None,
                         max_age_minutes: int = 10) -> tuple[float | None, str | None]:
    """Look up the actual fill price and exit reason for a recently closed position.

    BUG-FIX (2026-04-14): See the matching fix in data/fetcher.py. Filter by
    fill time to avoid returning historical fills as the current exit price
    (which caused the INTC -$19,911 phantom P&L on a 2-second hold).
    """
    try:
        client = get_trading_client()
        now_utc = datetime.now(timezone.utc)
        submit_cutoff = after or (now_utc - timedelta(minutes=max_age_minutes * 3))
        if submit_cutoff.tzinfo is None:
            submit_cutoff = submit_cutoff.replace(tzinfo=timezone.utc)
        else:
            submit_cutoff = submit_cutoff.astimezone(timezone.utc)

        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol],
            limit=50,
            after=submit_cutoff,
        )
        orders = client.get_orders(request)

        exit_side = "sell" if side == "buy" else "buy"
        fill_cutoff = now_utc - timedelta(minutes=max_age_minutes)
        if after is not None:
            after_utc = after if after.tzinfo else after.replace(tzinfo=timezone.utc)
            fill_cutoff = max(fill_cutoff, after_utc.astimezone(timezone.utc))

        candidates: list[tuple[datetime, float, str]] = []
        for order in orders:
            if not (order.symbol == symbol
                    and str(order.side).lower().endswith(exit_side)
                    and str(order.status).lower().endswith("filled")
                    and order.filled_avg_price is not None):
                continue
            filled_at = (getattr(order, "filled_at", None)
                         or getattr(order, "updated_at", None)
                         or getattr(order, "submitted_at", None))
            if filled_at is None:
                continue
            if filled_at.tzinfo is None:
                filled_at = filled_at.replace(tzinfo=timezone.utc)
            if filled_at < fill_cutoff:
                continue

            price = float(order.filled_avg_price)
            order_type = str(getattr(order, "order_type", "")).lower()
            if "stop" in order_type:
                reason = "stop_loss"
            elif "limit" in order_type:
                reason = "take_profit"
            elif "market" in order_type:
                reason = "market_close"
            else:
                reason = "broker_exit"
            candidates.append((filled_at, price, reason))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, price, reason = candidates[0]
            return price, reason
    except Exception as e:
        logger.warning(f"Failed to look up fill info for {symbol}: {e}")
    return None, None


def get_filled_exit_price(symbol: str, side: str = "buy") -> float | None:
    """Convenience wrapper — returns just the fill price."""
    price, _ = get_filled_exit_info(symbol, side)
    return price


def get_snapshot(symbol: str):
    """Get latest snapshot for a symbol (latest trade, quote, bar)."""
    client = get_data_client()
    snapshots = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))
    if hasattr(snapshots, "data"):
        return snapshots.data.get(symbol) if snapshots.data else None
    if isinstance(snapshots, dict):
        return snapshots.get(symbol)
    return None


def get_snapshots(symbols: list[str]) -> dict:
    """Get snapshots for multiple symbols at once."""
    client = get_data_client()
    result = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX))
    if hasattr(result, "data") and result.data:
        return dict(result.data)
    if isinstance(result, dict):
        return result
    return {}


def verify_connectivity() -> dict:
    """Verify API connectivity. Returns account info dict or raises."""
    account = get_account()
    clock = get_clock()

    info = {
        "account_id": account.id,
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "paper": config.PAPER_MODE,
        "market_open": clock.is_open,
        "next_open": clock.next_open,
        "next_close": clock.next_close,
    }
    return info


def verify_data_feed(symbol: str = "SPY") -> bool:
    """Verify that we can fetch data for at least one symbol."""
    try:
        df = get_daily_bars(symbol, days=5)
        return len(df) > 0
    except Exception as e:
        logger.error(f"Data feed verification failed for {symbol}: {e}")
        return False
