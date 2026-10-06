"""Thin wrapper around Alpaca's trading API.

The trader talks to this small interface instead of the SDK directly, which
keeps the trading logic readable and lets the tests swap in a fake broker.

This project only ever connects to Alpaca's *paper* environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Protocol, Set


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    avg_entry_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pl: Optional[float] = None


class DuplicateOrderError(Exception):
    """The broker already has an order with this client_order_id."""


class Broker(Protocol):
    def is_market_open(self) -> bool: ...

    def get_account(self) -> Account: ...

    def get_positions(self) -> Dict[str, PositionInfo]: ...

    def open_order_symbols(self) -> Set[str]: ...

    def submit_market_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> str: ...

    def is_fractionable(self, symbol: str) -> bool: ...

    def next_close(self) -> Optional[datetime]: ...


def _f(value) -> Optional[float]:
    return None if value is None else float(value)


class AlpacaBroker:
    def __init__(self, key_id: str, secret_key: str):
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(key_id, secret_key, paper=True)
        self._fractionable: Dict[str, bool] = {}

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def next_close(self) -> Optional[datetime]:
        """When the market next closes (today's close while it's open; early closes included)."""
        close = self._client.get_clock().next_close
        return close if close is None or close.tzinfo else close.replace(tzinfo=timezone.utc)

    def get_account(self) -> Account:
        a = self._client.get_account()
        return Account(equity=float(a.equity), cash=float(a.cash), buying_power=float(a.buying_power))

    def get_positions(self) -> Dict[str, PositionInfo]:
        return {
            p.symbol: PositionInfo(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=_f(p.avg_entry_price),
                market_value=_f(p.market_value),
                unrealized_pl=_f(p.unrealized_pl),
            )
            for p in self._client.get_all_positions()
        }

    def open_order_symbols(self) -> Set[str]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {o.symbol for o in orders}

    def is_fractionable(self, symbol: str) -> bool:
        """Whether Alpaca allows fractional orders for this symbol (cached per process)."""
        if symbol not in self._fractionable:
            try:
                asset = self._client.get_asset(symbol)
                self._fractionable[symbol] = bool(getattr(asset, "fractionable", False))
            except Exception:  # unknown symbol or API hiccup: fall back to whole shares
                return False
        return self._fractionable[symbol]

    def submit_market_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> str:
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,  # fractional orders must be DAY orders
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except APIError as exc:
            if "client_order_id" in str(exc).lower():
                raise DuplicateOrderError(str(exc)) from exc
            raise
        return str(order.id)
