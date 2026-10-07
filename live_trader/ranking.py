"""The momentum ranking behind each decision, saved for the dashboard.

Once a day (on the last run before the close), at every rebalance, and on the
first run ever, the live trader ranks the whole universe with the strategy's own
return calculation (Momentum.returns) and saves it to the ``rankings`` table:

    rank, momentum   1 = strongest lookback return
    meter            0-100, the stock's percentile in the universe (100 = rank 1)
    category         top (in the strategy's top N), next (the next few that would
                     swap in), ranked, negative (excluded: return not positive),
                     no_data (not enough history yet)
    held             whether the account held it when the snapshot was taken
    gap              for "next" names: how much more the stock must gain, relative
                     to the weakest pick, to overtake it at the next rebalance

The dashboard only reads the database, so it never needs the trading keys.
A failure here is logged and never affects trading.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from trader_core import db, safeguards
from trader_core.strategy import make_strategy

log = logging.getLogger("live_trader.ranking")
NY = ZoneInfo("America/New_York")
NEXT_UP = 5  # how many names after the top N are shown as "next up"


def momentum_strategy(settings):
    """The momentum strategy the bot trades with (inside a crash switch if there is one), or None."""
    strategy = make_strategy(settings)
    inner = getattr(strategy, "normal", strategy)
    return inner if inner.name == "momentum" else None


def build(strategy, returns: Dict[str, float], missing: Dict[str, str], held: set) -> List[Dict]:
    """Rows for one snapshot from the strategy's returns (pure: no I/O, easy to test)."""
    ranked = sorted(returns, key=returns.get, reverse=True)
    n = len(ranked)
    picks = [s for s in ranked[: strategy.top] if not (strategy.absolute and returns[s] <= 0)]
    cutoff = returns[picks[-1]] if len(picks) == strategy.top else None  # the weakest pick
    rows: List[Dict] = []
    next_left = NEXT_UP
    for rank, symbol in enumerate(ranked, start=1):
        r = returns[symbol]
        row = {"symbol": symbol, "rank": rank, "momentum": r,
               "meter": 100.0 if n == 1 else round(100 * (n - rank) / (n - 1), 1),
               "held": symbol in held, "gap": None, "note": None}
        if symbol in picks:
            row["category"] = "top"
        elif strategy.absolute and r <= 0:
            row["category"] = "negative"
            row["note"] = "return not positive: never bought"
        elif next_left > 0:
            row["category"] = "next"
            next_left -= 1
            if cutoff is not None:
                row["gap"] = (1 + cutoff) / (1 + r) - 1
                row["note"] = f"needs {row['gap'] * 100:+.1f}% vs {picks[-1]} to swap in"
            else:
                row["note"] = "would be bought at the next rebalance"
        else:
            row["category"] = "ranked"
        rows.append(row)
    for symbol, why in sorted(missing.items()):
        rows.append({"symbol": symbol, "rank": None, "momentum": None, "meter": None, "category": "no_data",
                     "held": symbol in held, "gap": None, "note": why})
    return rows


def _held(engine) -> set:
    p = db.positions.c
    with engine.connect() as conn:
        return {s for s, q in conn.execute(select(p.symbol, p.qty)).fetchall() if q and q > 0}


def has_snapshot(engine, since: Optional[datetime] = None, daily_only: bool = False) -> bool:
    """Whether any ranking was saved (since ``since``; with ``daily_only``, end-of-day ones only)."""
    r = db.rankings.c
    query = select(func.count()).select_from(db.rankings)
    if since is not None:
        query = query.where(r.ts >= since)
    if daily_only:
        query = query.where(r.rebalance.is_(False))
    with engine.connect() as conn:
        return bool(conn.execute(query).scalar())


def snapshot(settings, data, engine, now: datetime, rebalance: bool = False) -> int:
    """Rank the universe now and save it. Returns the number of rows saved (0 if not momentum)."""
    from .trader import closes_for_signal  # imported here: trader imports this module

    strategy = momentum_strategy(settings)
    if strategy is None:
        return 0
    symbols = settings.symbols
    today = now.astimezone(NY).date()
    bars = data.daily_bars(symbols, today - timedelta(days=math.ceil(strategy.warmup_bars * 1.6) + 15), today)
    prices = data.latest_prices(symbols)
    history = {}
    for s in symbols:
        if s not in bars:
            continue
        price = prices.get(s)
        if safeguards.check_price(bars.get(s), price, today, settings) is not None:
            price = None  # suspicious live price: rank on completed closes only, like the trader does
        history[s] = closes_for_signal(bars[s], today, price, settings.signal_mode)
    returns, missing = strategy.returns(history)
    for s in symbols:
        if s not in history:
            missing.setdefault(s, "no market data")
    rows = build(strategy, returns, missing, _held(engine))
    for row in rows:
        row.update(ts=now, rebalance=rebalance)
    db.insert_rows(engine, db.rankings, rows)
    return len(rows)


def maybe_snapshot(settings, broker, data, engine, result: Dict, now: datetime) -> int:
    """Take a snapshot if this run calls for one: a rebalance, the day's last run, or none saved yet."""
    from . import status

    rebalanced = bool(result.get("rebalanced"))
    if not rebalanced:
        if result.get("status") == "market_closed" and has_snapshot(engine):
            return 0
        start = datetime.combine(now.astimezone(NY).date(), datetime.min.time(), NY).astimezone(timezone.utc)
        last_run = result.get("status") != "market_closed" and status.is_last_run(broker, now)
        if has_snapshot(engine) and not (last_run and not has_snapshot(engine, since=start, daily_only=True)):
            return 0
    return snapshot(settings, data, engine, now, rebalance=rebalanced)
