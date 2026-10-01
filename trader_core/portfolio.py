"""Turning target weights into orders. Shared by the backtester and the live trader,
so a strategy is sized and traded identically in both.

Rules
    * Exits: a symbol whose target is 0 and that you hold is sold in full.
    * Entries: a symbol with a target above 0 that you don't hold is bought up
      to ``weight x account value``, limited by the cash available. Never margin.
    * Holds: a symbol you hold with a target above 0 is left alone...
    * ...except on a scheduled rebalance of a strategy that resizes (trend,
      momentum): then a position more than ``band`` away from its target
      (default 25%) is trimmed or topped up. The band stops the bot from
      trading tiny amounts every month.
    * Sells are planned first, so their proceeds can pay for the buys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Tuple

from .strategy import Signal

DEFAULT_BAND = 0.25
MIN_ORDER_DOLLARS = 1.0  # Alpaca's minimum for fractional orders


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    note: str


def share_quantity(budget: float, price: float, fractional: bool) -> float:
    """Shares that ``budget`` buys at ``price``: 4 decimals if fractional, else whole shares."""
    if budget <= 0 or price <= 0:
        return 0.0
    if fractional:
        qty = math.floor(budget / price * 10_000) / 10_000
        return qty if qty * price >= MIN_ORDER_DOLLARS else 0.0
    return float(math.floor(budget / price))


def plan_sells(
    signals: Mapping[str, Signal],
    held: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    rebalance: bool,
    resize: bool,
    fractional: Callable[[str], bool],
    band: float = DEFAULT_BAND,
) -> List[Order]:
    orders = []
    for symbol, sig in signals.items():
        qty, price = held.get(symbol, 0.0), prices.get(symbol)
        if qty <= 0 or sig.weight is None or not price:
            continue
        if sig.weight == 0:
            orders.append(Order(symbol, "sell", qty, "exit"))
        elif rebalance and resize:
            target, value = sig.weight * equity, qty * price
            if value > target * (1 + band):
                trim = min(share_quantity(value - target, price, fractional(symbol)), qty)
                if trim > 0:
                    orders.append(Order(symbol, "sell", trim, "trim back to target weight"))
    return orders


def plan_buys(
    signals: Mapping[str, Signal],
    held: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    cash: float,
    rebalance: bool,
    resize: bool,
    fractional: Callable[[str], bool],
    band: float = DEFAULT_BAND,
) -> Tuple[List[Order], Dict[str, str]]:
    """Buy orders that fit in ``cash``, plus the reason for any wanted buy that didn't fit."""
    wanted = []
    for symbol, sig in signals.items():
        qty, price = held.get(symbol, 0.0), prices.get(symbol)
        if not sig.weight or sig.weight <= 0 or not price:
            continue
        target = sig.weight * equity
        if qty <= 0:
            wanted.append((0, symbol, target, "new position"))
        elif rebalance and resize and qty * price < target * (1 - band):
            wanted.append((1, symbol, target - qty * price, "top up to target weight"))

    orders: List[Order] = []
    skipped: Dict[str, str] = {}
    cash = max(cash, 0.0)
    for _, symbol, amount, note in sorted(wanted, key=lambda w: w[0]):  # new positions before top-ups
        price = prices[symbol]
        qty = share_quantity(min(amount, cash), price, fractional(symbol))
        if qty <= 0:
            skipped[symbol] = "not enough cash" if cash < amount else "target is smaller than one share"
            continue
        orders.append(Order(symbol, "buy", qty, note))
        cash -= qty * price
    return orders, skipped
