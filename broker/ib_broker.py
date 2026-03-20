"""V10: Interactive Brokers adapter.

Implements the Broker interface for IB TWS/Gateway via ib_insync.
Requires: pip install ib_insync

Configuration via environment variables:
    IB_HOST: TWS/Gateway host (default: 127.0.0.1)
    IB_PORT: TWS/Gateway port (default: 7497 for paper, 7496 for live)
    IB_CLIENT_ID: Client ID (default: 1)

This is a skeleton implementation — complete the TODO items when ready
to go live with IB.
"""

import os
import logging
from datetime import datetime

import config
from broker.base import Broker, OrderResult, AccountInfo, Position, Snapshot, BrokerError

logger = logging.getLogger(__name__)

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))


class IBBroker(Broker):
    """Interactive Brokers adapter using ib_insync.

    Requires IB TWS or IB Gateway running locally.
    """

    def __init__(self, host: str = None, port: int = None, client_id: int = None):
        raise NotImplementedError(
            "IB broker not yet implemented for production use. "
            "Use the Alpaca broker adapter instead."
        )
        self._host = host or IB_HOST
        self._port = port or IB_PORT
        self._client_id = client_id or IB_CLIENT_ID
        self._ib = None
        self._connected = False

    def connect(self):
        """Connect to IB TWS/Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            self._connected = True
            logger.info(f"Connected to IB at {self._host}:{self._port}")
        except ImportError:
            raise BrokerError("ib_insync not installed. Run: pip install ib_insync")
        except Exception as e:
            raise BrokerError(f"Failed to connect to IB: {e}")

    def disconnect(self):
        """Disconnect from IB."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False

    def _ensure_connected(self):
        if not self._connected:
            raise BrokerError("Not connected to IB. Call connect() first.")

    def submit_order(self, symbol: str, qty: int, side: str,
                     order_type: str = "market",
                     limit_price: float | None = None,
                     take_profit: float | None = None,
                     stop_loss: float | None = None,
                     time_in_force: str = "day") -> OrderResult:
        """Submit an order to IB."""
        self._ensure_connected()

        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, BracketOrder

            contract = Stock(symbol, "SMART", "USD")
            action = "BUY" if side == "buy" else "SELL"

            if take_profit and stop_loss:
                # Bracket order
                bracket = self._ib.bracketOrder(
                    action, qty,
                    limitPrice=limit_price or 0,
                    takeProfitPrice=take_profit,
                    stopLossPrice=stop_loss,
                )
                trades = []
                for order in bracket:
                    trade = self._ib.placeOrder(contract, order)
                    trades.append(trade)

                parent_trade = trades[0]
                return OrderResult(
                    success=True,
                    order_id=str(parent_trade.order.orderId),
                    submitted_at=datetime.now(),
                )
            elif order_type == "limit" and limit_price:
                order = LimitOrder(action, qty, limit_price)
            else:
                order = MarketOrder(action, qty)

            trade = self._ib.placeOrder(contract, order)
            return OrderResult(
                success=True,
                order_id=str(trade.order.orderId),
                submitted_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"IB order failed for {symbol}: {e}")
            return OrderResult(success=False, message=str(e))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an IB order."""
        self._ensure_connected()
        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error(f"IB cancel failed for {order_id}: {e}")
            return False

    def get_positions(self) -> list[Position]:
        """Get all IB positions."""
        self._ensure_connected()
        positions = []
        for pos in self._ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                qty=int(pos.position),
                avg_entry_price=pos.avgCost,
                market_value=pos.position * pos.avgCost,
                unrealized_pl=pos.unrealizedPNL if hasattr(pos, "unrealizedPNL") else 0.0,
                side="long" if pos.position > 0 else "short",
            ))
        return positions

    def get_account(self) -> AccountInfo:
        """Get IB account info."""
        self._ensure_connected()
        summary = self._ib.accountSummary()
        equity = cash = buying_power = 0.0
        for item in summary:
            if item.tag == "NetLiquidation":
                equity = float(item.value)
            elif item.tag == "TotalCashValue":
                cash = float(item.value)
            elif item.tag == "BuyingPower":
                buying_power = float(item.value)
        return AccountInfo(equity=equity, cash=cash, buying_power=buying_power)

    def can_short(self, symbol: str, qty: int, price: float) -> tuple[bool, str]:
        """Check IB short availability."""
        self._ensure_connected()
        if not config.ALLOW_SHORT:
            return False, "shorting_disabled"
        # IB handles short availability internally — assume available
        # A more robust implementation would check short stock availability
        return True, "ok"

    def close_position(self, symbol: str, qty: int | None = None) -> OrderResult:
        """Close an IB position."""
        self._ensure_connected()
        for pos in self._ib.positions():
            if pos.contract.symbol == symbol:
                close_qty = qty or abs(int(pos.position))
                side = "sell" if pos.position > 0 else "buy"
                return self.submit_order(symbol, close_qty, side)
        return OrderResult(success=False, message=f"No IB position in {symbol}")

    def close_all_positions(self) -> list[OrderResult]:
        """Close all IB positions."""
        results = []
        for pos in self._ib.positions():
            results.append(self.close_position(pos.contract.symbol))
        return results

    def get_snapshot(self, symbol: str) -> Snapshot | None:
        """Get a market data snapshot from IB."""
        self._ensure_connected()
        try:
            from ib_insync import Stock
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract, snapshot=True)
            self._ib.sleep(1)  # Wait for data
            return Snapshot(
                symbol=symbol,
                last_price=ticker.last or 0.0,
                bid=ticker.bid or 0.0,
                ask=ticker.ask or 0.0,
                volume=ticker.volume or 0,
            )
        except Exception as e:
            logger.error(f"IB snapshot failed for {symbol}: {e}")
            return None

    @property
    def name(self) -> str:
        return "InteractiveBrokers"
