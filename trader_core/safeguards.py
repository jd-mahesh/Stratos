"""Safeguards: checks that stop the bot from trading when something looks wrong.

Built to the standard of a real-money account, even while Stratos paper trades.
Every check is a plain function so it can be tested on its own; the live trader
(live_trader/trader.py) decides what to do with the answers.

    Kill switch        TRADING_HALTED=true: record balances, place no orders.
    Circuit breaker    The account fell more than DAILY_LOSS_HALT_PCT since the
                       previous day, or DRAWDOWN_HALT_PCT from its peak: halt and
                       stay halted until someone resumes it on purpose
                       (python -m live_trader --resume).
    Price sanity       A symbol whose live price is missing, whose daily data is
                       stale, or whose live price is wildly away from its last
                       close is left out of this run (it's probably bad data).
    Order limits       Before anything is sent, the planned orders are checked:
                       not too many, no buy bigger than MAX_ORDER_PCT of the
                       account, no buy that leaves a symbol above MAX_POSITION_PCT,
                       and never a sell of more than is held (that would be a short
                       sale). Any problem cancels the whole run.
    Settings check     At start-up, refuse a configuration whose normal targets
                       would break the order limits (e.g. MOMENTUM_TOP=2 wants 50%
                       per stock while MAX_POSITION_PCT is 30).
    Trading budget     With CAPITAL_RESERVE set, every limit above is measured
                       against the budget (account value - reserve), not the whole
                       account, and a budget that's used up halts trading.

The defaults are set beyond anything the strategy does in normal operation, so
they only trip when something is broken.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Dict, Iterable, List, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from . import db
from .portfolio import Order

NY = ZoneInfo("America/New_York")
HALT_KEY = "halt"
EPS = 1e-6


# --- halts -------------------------------------------------------------------

def get_halt(engine) -> Optional[Dict[str, str]]:
    """The recorded circuit-breaker halt ({"reason", "since"}), or None if trading is allowed."""
    raw = db.get_state(engine, HALT_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:  # unreadable record: fail safe and treat it as halted
        return {"reason": raw, "since": "unknown"}


def set_halt(engine, reason: str, now: datetime) -> None:
    db.set_state(engine, HALT_KEY, json.dumps({"reason": reason, "since": now.isoformat()}))


def clear_halt(engine) -> Optional[Dict[str, str]]:
    """Resume trading. Returns the halt that was cleared (None if there wasn't one)."""
    previous = get_halt(engine)
    db.set_state(engine, HALT_KEY, "")
    return previous


# --- circuit breaker ---------------------------------------------------------

def reserve_of(settings) -> float:
    """The CAPITAL_RESERVE in dollars (0 when unset)."""
    return float(getattr(settings, "capital_reserve", 0.0) or 0.0)


def check_budget(equity: float, settings) -> Optional[str]:
    """A reason to halt if a CAPITAL_RESERVE leaves nothing to trade with; None if fine (or no reserve)."""
    reserve = reserve_of(settings)
    if reserve <= 0:
        return None
    budget = equity - reserve
    if not math.isfinite(budget) or budget <= 0:
        return (f"trading budget is used up: account ${equity:,.2f} minus the ${reserve:,.0f} reserve "
                f"leaves ${budget:,.2f}")
    return None


def check_losses(engine, value_now: float, now: datetime, settings) -> Optional[str]:
    """A reason to halt if the money Stratos trades fell too far since yesterday or from its peak.

    ``value_now`` is the trading budget: the account value minus CAPITAL_RESERVE
    (just the account value when there's no reserve). It's compared with earlier
    balance snapshots (equity_history) taken under the same reserve, so changing
    the reserve starts a fresh history instead of looking like a huge loss. The
    first run under a reserve has nothing to compare with and never trips.
    """
    reserve = reserve_of(settings)
    label = "trading budget" if reserve > 0 else "account"
    if not value_now or value_now <= 0 or not math.isfinite(value_now):
        return f"{label} value looks wrong ({value_now!r})"
    t = db.equity_history.c
    with engine.connect() as conn:
        rows = conn.execute(select(t.ts, t.equity, t.reserve)).fetchall()
    history = pd.DataFrame(rows, columns=["ts", "equity", "reserve"])
    history["reserve"] = pd.to_numeric(history["reserve"]).fillna(0.0)
    history = history[(history["reserve"] - reserve).abs() < 0.005]
    if history.empty:
        return None
    history["value"] = history["equity"] - history["reserve"]
    today = now.astimezone(NY).date()
    history["day"] = pd.to_datetime(history["ts"], utc=True).dt.tz_convert(NY).dt.date
    earlier = history[history["day"] < today].sort_values("ts")
    if not earlier.empty:
        previous = float(earlier["value"].iloc[-1])
        change = value_now / previous - 1
        if change <= -settings.daily_loss_halt_pct:
            return (f"{label} fell {change * 100:.1f}% since the previous day "
                    f"(${previous:,.0f} -> ${value_now:,.0f}); the limit is "
                    f"{settings.daily_loss_halt_pct * 100:.0f}%")
    peak = max(float(history["value"].max()), value_now)
    drawdown = value_now / peak - 1
    if drawdown <= -settings.drawdown_halt_pct:
        return (f"{label} is {drawdown * 100:.1f}% below its peak (${peak:,.0f} -> ${value_now:,.0f}); "
                f"the limit is {settings.drawdown_halt_pct * 100:.0f}%")
    return None


# --- price sanity ------------------------------------------------------------

def check_price(bars: Optional[pd.DataFrame], live_price: Optional[float], today: date, settings) -> Optional[str]:
    """Why this symbol's prices can't be trusted right now, or None if they look fine."""
    if live_price is None or not math.isfinite(float(live_price)) or float(live_price) <= 0:
        return f"no valid live price ({live_price!r})"
    if bars is None or bars.empty:
        return "no daily price history"
    completed = bars[bars.index.date < today]
    if completed.empty:
        return "no completed daily bars"
    last_day = completed.index[-1].date()
    age = (today - last_day).days
    if age > settings.max_data_age_days:
        return f"latest daily bar is {age} days old ({last_day})"
    last_close = float(completed["close"].iloc[-1])
    if not math.isfinite(last_close) or last_close <= 0:
        return f"last close looks wrong ({last_close!r})"
    move = float(live_price) / last_close - 1
    if abs(move) > settings.max_price_move_pct:
        return (f"live price {float(live_price):.2f} is {move * 100:+.0f}% from the last close {last_close:.2f}; "
                f"that's beyond the {settings.max_price_move_pct * 100:.0f}% limit and looks like bad data")
    return None


# --- order limits ------------------------------------------------------------

def check_orders(orders: Iterable[Order], held: Mapping[str, float], prices: Mapping[str, float],
                 equity: float, settings) -> List[str]:
    """Problems with a planned set of orders (empty list = all within limits).

    ``held`` is the position before any of these orders. Sells only reduce risk,
    so they're checked only for selling more than is held. ``equity`` is the money
    Stratos trades with: the trading budget when CAPITAL_RESERVE is set, so the
    percent limits apply to the budget.
    """
    orders = list(orders)
    problems: List[str] = []
    if len(orders) > settings.max_orders_per_run:
        problems.append(f"{len(orders)} orders planned; the limit per run is {settings.max_orders_per_run}")
    if equity <= 0:
        return problems + [f"account value is {equity}; refusing to trade"]
    after = dict(held)
    for order in orders:
        price = prices.get(order.symbol)
        if price is None or price <= 0:
            problems.append(f"{order.symbol}: no price to check the order against")
            continue
        value = order.qty * price
        if order.side == "sell":
            if order.qty > held.get(order.symbol, 0.0) + EPS:
                problems.append(f"{order.symbol}: sell of {order.qty:g} is more than the {held.get(order.symbol, 0.0):g} "
                                "held (that would open a short position)")
            after[order.symbol] = after.get(order.symbol, 0.0) - order.qty
            continue
        if value > settings.max_order_pct * equity * (1 + EPS):
            problems.append(f"{order.symbol}: buy of ${value:,.0f} is {value / equity * 100:.0f}% of the account; "
                            f"the limit is {settings.max_order_pct * 100:.0f}%")
        after[order.symbol] = after.get(order.symbol, 0.0) + order.qty
        position = after[order.symbol] * price
        if position > settings.max_position_pct * equity * (1 + EPS):
            problems.append(f"{order.symbol}: position after buying would be {position / equity * 100:.0f}% of the "
                            f"account; the limit is {settings.max_position_pct * 100:.0f}%")
    return problems


# --- settings check ----------------------------------------------------------

def largest_target(strategy, n_symbols: int) -> float:
    """The biggest share of the account the strategy can ask for in one symbol."""
    inner = getattr(strategy, "normal", None)
    if inner is not None:  # CrashSwitch: normal strategy, or the backup in crash mode
        backup = getattr(strategy, "backup", None)
        backup_share = 0.0 if backup is None or getattr(backup, "cash_only", False) else 1.0
        return max(largest_target(inner, n_symbols), backup_share)
    if strategy.name == "momentum":
        return 1.0 / strategy.top
    if strategy.name == "defensive":
        return 0.0 if strategy.cash_only else 1.0
    return 1.0 / max(n_symbols, 1)  # ma_crossover, trend: equal slots


def check_settings(strategy, n_symbols: int, settings) -> Optional[str]:
    """A configuration problem that would make normal trading break the limits, or None."""
    biggest = largest_target(strategy, n_symbols)
    limit = min(settings.max_order_pct, settings.max_position_pct)
    if biggest > limit + EPS:
        return (f"the strategy can put {biggest * 100:.0f}% of the account in one symbol, but MAX_ORDER_PCT / "
                f"MAX_POSITION_PCT allow {limit * 100:.0f}%. Hold more symbols (e.g. MOMENTUM_TOP) or raise "
                "the limits on purpose")
    return None
