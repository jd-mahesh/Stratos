"""One live-trader tick: look at prices, ask the strategy, trade, record.

Each run is nearly stateless: it reads the account, positions and open orders
from Alpaca and prices from the market-data API, and writes every decision to
the database. The one thing it remembers is when a monthly strategy last
rebalanced (the bot_state table), so it acts once per month and not on every
five-minute tick. That makes it safe to run as a Lambda function that may
start cold, run twice, or be retried.

Duplicate-trade protection, in layers:
    1. Targets, not events. If the strategy says "hold 9% in XLK" and the
       account already holds XLK, the action is "hold". A second run can't buy again.
    2. Symbols with an open (unfilled) order are skipped.
    3. Each order carries a client_order_id derived from the scheduled run
       time. If Lambda retries the same invocation, Alpaca rejects the repeat.

Signal modes (SIGNAL_MODE):
    close     The strategy sees completed daily closes only, exactly what the
              backtester's close mode saw. Decisions can only change once a day.
    intraday  The strategy also sees the live price right now as the latest
              point in its history, so it reacts to today's moves. For monthly
              strategies this applies on the rebalance day; for daily ones on
              every run. The backtester's intraday mode simulates this.

Sizing and order planning use trader_core/portfolio.py, the same code as the
backtest: target weight x account value, sells first, never more than the cash
on hand (no margin), fractional shares where the asset allows.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trader_core import db
from trader_core.config import Settings
from trader_core.portfolio import Order, plan_buys, plan_sells, share_quantity  # noqa: F401 (re-exported)
from trader_core.strategy import Strategy, make_strategy

from .broker import Broker, DuplicateOrderError

log = logging.getLogger("live_trader")
NY = ZoneInfo("America/New_York")


def order_bucket(scheduled_time: Optional[str], now: datetime) -> str:
    """Stable id for this run: the scheduled time if given, else now floored to 5 minutes."""
    if scheduled_time:
        ts = pd.Timestamp(scheduled_time)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.strftime("%Y%m%dT%H%M")
    floored = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    floored -= timedelta(minutes=floored.minute % 5)
    return floored.strftime("%Y%m%dT%H%M")


def closes_for_signal(bars: pd.DataFrame, today, latest_price: Optional[float], mode: str) -> np.ndarray:
    """Closing prices the strategy should see, oldest first.

    Today's bar is dropped because it's still forming. In intraday mode the
    live price takes its place as the most recent point.
    """
    completed = bars.loc[bars.index.date < today, "close"].to_numpy(dtype=float)
    if mode == "intraday" and latest_price is not None:
        return np.append(completed, latest_price)
    return completed


def rebalance_key(strategy: Strategy) -> str:
    """bot_state key for a strategy's last rebalance; changing the strategy or its settings starts fresh."""
    return f"last_rebalance:{strategy.name}:{json.dumps(strategy.params(), sort_keys=True)}"


def _next_month(today: date) -> str:
    first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    return first.strftime("%B %Y")


def _snapshot(engine, broker: Broker, now: datetime) -> float:
    after = broker.get_account()
    db.insert_rows(engine, db.equity_history, [{
        "ts": now, "equity": after.equity, "cash": after.cash, "buying_power": after.buying_power,
    }])
    db.replace_positions(engine, [
        {
            "symbol": p.symbol,
            "qty": p.qty,
            "avg_entry_price": p.avg_entry_price,
            "market_value": p.market_value,
            "unrealized_pl": p.unrealized_pl,
            "updated_at": now,
        }
        for p in broker.get_positions().values()
    ])
    return after.equity


def run_tick(
    settings: Settings,
    broker: Broker,
    data,
    engine,
    now: Optional[datetime] = None,
    scheduled_time: Optional[str] = None,
) -> Dict:
    now = now or datetime.now(timezone.utc)
    symbols = settings.symbols

    if not broker.is_market_open() and not settings.force_run:
        log.info("market closed, nothing to do")
        return {"status": "market_closed", "ts": now.isoformat()}

    strategy = make_strategy(settings)
    today = now.astimezone(NY).date()
    month = today.strftime("%Y-%m")
    key = rebalance_key(strategy)
    mode_key = f"mode:{key}"
    has_modes = getattr(strategy, "detector", None) is not None
    done_this_month = strategy.rebalance == "monthly" and db.get_state(engine, key) == month

    def not_due(note: str, mode: Optional[str] = None) -> Dict:
        equity = _snapshot(engine, broker, now)
        log.info(note)
        return {"status": "ok", "rebalanced": False, "note": note, "mode": mode, "ts": now.isoformat(),
                "equity": equity, "dry_run": settings.dry_run, "decisions": []}

    if done_this_month and not has_modes:
        return not_due(f"{strategy.name} rebalances monthly; done for {month}, next in {_next_month(today)}")

    lookback_days = math.ceil(strategy.warmup_bars * 1.6) + 15
    extras = [s for s in strategy.extra_symbols if s not in symbols]
    bars = data.daily_bars(symbols + extras, today - timedelta(days=lookback_days), today)
    prices = data.latest_prices(symbols + extras)
    extra_history = {
        s: closes_for_signal(bars[s], today, prices.get(s), settings.signal_mode)
        for s in strategy.extra_symbols if s in bars
    }

    # Crash mode is checked on every run; a change of mode rebalances right away.
    previous_mode = db.get_state(engine, mode_key) if has_modes else None
    mode_history = extra_history
    detector = getattr(strategy, "detector", None)
    if detector is not None and detector.closes_only and detector.index in bars:
        # a confirming detector looks at finished daily closes only, never the live price
        mode_history = {**extra_history,
                        detector.index: closes_for_signal(bars[detector.index], today, None, "close")}
    regime = strategy.mode(mode_history, previous_mode)
    mode = regime.mode if regime is not None else previous_mode
    mode_changed = regime is not None and previous_mode is not None and mode != previous_mode
    if has_modes and regime is None:
        log.warning("not enough %s prices to check for a crash; staying in %s mode",
                    strategy.detector.index, previous_mode or "normal")
    if done_this_month and not mode_changed:
        if has_modes and previous_mode is None and mode is not None and not settings.dry_run:
            db.set_state(engine, mode_key, mode)
        why = regime.reason if regime is not None else "mode unchanged"
        return not_due(f"{strategy.name} rebalances monthly; done for {month}, next in {_next_month(today)}. "
                       f"{why}", mode)
    if mode_changed:
        log.info("market mode changed from %s to %s: %s", previous_mode, mode, regime.reason)

    account = broker.get_account()
    positions = broker.get_positions()
    pending_orders = broker.open_order_symbols()

    trade_symbols = symbols + [a for a in strategy.tradable_extras if a not in symbols]
    rows: Dict[str, Dict] = {}
    history = {}
    for symbol in trade_symbols:
        rows[symbol] = {
            "ts": now,
            "symbol": symbol,
            "price": prices.get(symbol),
            "fast_ma": None,
            "slow_ma": None,
            "target": None,
            "current_qty": positions[symbol].qty if symbol in positions else 0.0,
            "target_qty": None,
            "action": "skip",
            "order_qty": None,
            "order_id": None,
            "reason": "no market data",
            "dry_run": settings.dry_run,
        }
        if symbol in symbols and symbol in bars and prices.get(symbol):
            history[symbol] = closes_for_signal(bars[symbol], today, prices[symbol], settings.signal_mode)

    # Every symbol with data is ranked/evaluated, even ones we can't act on right now.
    signals = strategy.decide(history, len(symbols), extra_history, mode)
    actionable = {}
    for symbol, sig in signals.items():
        if symbol not in rows or not prices.get(symbol):
            continue
        row = rows[symbol]
        row.update(fast_ma=sig.fast_ma, slow_ma=sig.slow_ma, target=sig.weight, reason=sig.reason)
        if symbol in pending_orders:
            row["reason"] += "; an order for this symbol is still open"
        elif row["current_qty"] < 0:
            row["reason"] += "; short position held, this bot only manages long positions"
        elif sig.weight is not None:
            actionable[symbol] = sig
            row.update(action="hold", target_qty=sig.weight * account.equity / prices[symbol])

    held = {s: rows[s]["current_qty"] for s in actionable if rows[s]["current_qty"] > 0}
    bucket = order_bucket(scheduled_time, now)
    errors = False

    def fractional(symbol: str) -> bool:
        return settings.fractional_shares and broker.is_fractionable(symbol)

    def send(order: Order) -> bool:
        nonlocal errors
        row = rows[order.symbol]
        row.update(action=order.side, order_qty=float(order.qty))
        row["reason"] += f"; {order.note}"
        if settings.dry_run:
            row["reason"] += "; dry run, order not sent"
            return True
        try:
            row["order_id"] = broker.submit_market_order(
                order.symbol, order.qty, order.side, f"tb-{order.symbol}-{order.side}-{bucket}")
            return True
        except DuplicateOrderError:
            row.update(action="skip", reason=row["reason"] + "; duplicate run, order already placed")
        except Exception as exc:  # keep going for the other symbols, but record it
            log.exception("order failed for %s", order.symbol)
            row.update(action="error", reason=f"{row['reason']}; order failed: {exc}")
            errors = True
        return False

    # Sells first; the cash they free up (at today's price) pays for the buys.
    freed = 0.0
    for order in plan_sells(actionable, held, prices, account.equity, True, strategy.resize, fractional):
        if send(order):
            freed += order.qty * prices[order.symbol]
            held[order.symbol] = held.get(order.symbol, 0.0) - order.qty
    held = {s: q for s, q in held.items() if q > 1e-9}

    buys, skipped = plan_buys(actionable, held, prices, account.equity, max(account.cash, 0.0) + freed,
                              True, strategy.resize, fractional)
    for symbol, why in skipped.items():
        rows[symbol].update(action="skip", reason=f"{rows[symbol]['reason']}; {why}")
    for order in buys:
        send(order)

    # Log every symbol in the list, plus crash-mode assets whenever they're involved.
    decisions: List[Dict] = [rows[s] for s in trade_symbols
                             if s in symbols or rows[s]["action"] != "skip" or rows[s]["current_qty"] > 0]
    db.insert_rows(engine, db.decisions, decisions)
    if not settings.dry_run and not errors:
        if strategy.rebalance == "monthly":
            db.set_state(engine, key, month)
        if has_modes and mode is not None:
            db.set_state(engine, mode_key, mode)
    equity = _snapshot(engine, broker, now)

    summary = {
        "status": "ok",
        "rebalanced": True,
        "strategy": strategy.name,
        "signal_mode": settings.signal_mode,
        "mode": mode,
        "mode_note": regime.reason if regime is not None else None,
        "ts": now.isoformat(),
        "equity": equity,
        "dry_run": settings.dry_run,
        "decisions": [{k: d[k] for k in ("symbol", "action", "order_qty", "price", "reason")} for d in decisions],
    }
    for d in summary["decisions"]:
        log.info("%s %s qty=%s price=%s | %s", d["symbol"], d["action"], d["order_qty"], d["price"], d["reason"])
    return summary
