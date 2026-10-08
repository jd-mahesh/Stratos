"""Status emails: what Stratos is doing, not just what went wrong.

With STATUS_EMAILS on (and an alert topic set), each trading day you get:

    Stratos is online             first run after the open: budget, earned overnight, holdings, next rebalance
    Stratos bought ... / sold ... any run that places orders: one email per side, per run
    Stratos is offline for the day  the end of the last run before the close: the day's summary

"Earned Overnight" compares the day's first run with the previous trading day's last
run; "Earned Today" compares the day's last run with its first. Together they make
up the whole change since the previous trading day.

"Offline for the day" is sent by the last run itself, as its final step, so it
arrives when Stratos stops. The broker's clock says when today's market closes,
so early-close days work without any setting. Runs don't happen at the close
itself, so the summary uses prices from that last run, a few minutes before it.

Separately, the watchdog (invoked on its own schedule with {"watchdog": true})
emails you if Stratos stops running during market hours. It's a problem alert, so
it only needs the alert topic, not STATUS_EMAILS. If a run crashes outright, the
CloudWatch alarm in docs/DEPLOY.md covers it.

Nothing in here ever trades, and a failed email never stops a run.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select

from trader_core import db

from . import alerts

log = logging.getLogger("live_trader.status")
NY = ZoneInfo("America/New_York")
RUN_EVERY = timedelta(minutes=5)  # the schedule's interval: a run this close to the close is the last one
STALE_AFTER = timedelta(minutes=15)  # watchdog: no successful run for this long while open = offline
OPEN = time(9, 30)


def _money(v: float) -> str:
    return f"-${-v:,.2f}" if v < 0 else f"${v:,.2f}"


def _next_month(day) -> str:
    first = (day.replace(day=1) + timedelta(days=32)).replace(day=1)
    return first.strftime("%B %Y")


def _value(equity: float, settings) -> Tuple[float, str]:
    """The money Stratos trades (budget when there's a reserve) and what to call it."""
    reserve = float(getattr(settings, "capital_reserve", 0.0) or 0.0)
    if reserve > 0:
        return equity - reserve, "Trading Budget"
    return equity, "Account Value"


def _holdings(engine) -> List[str]:
    p = db.positions.c
    with engine.connect() as conn:
        rows = conn.execute(select(p.symbol, p.qty, p.market_value).order_by(p.symbol)).fetchall()
    return [f"  {s}: {q:g} shares" + (f", {_money(v)}" if v is not None else "") for s, q, v in rows if q]


def _history(engine, settings) -> pd.DataFrame:
    """Snapshots taken with the current reserve: ts, day (New York), value (equity minus reserve)."""
    t = db.equity_history.c
    with engine.connect() as conn:
        rows = conn.execute(select(t.ts, t.equity, t.reserve)).fetchall()
    h = pd.DataFrame(rows, columns=["ts", "equity", "reserve"])
    if h.empty:
        return h.assign(day=[], value=[])
    h["reserve"] = pd.to_numeric(h["reserve"]).fillna(0.0)
    reserve = float(getattr(settings, "capital_reserve", 0.0) or 0.0)
    h = h[(h["reserve"] - reserve).abs() < 0.005].copy()
    h["ts"] = pd.to_datetime(h["ts"], utc=True)
    h["day"] = h["ts"].dt.tz_convert(NY).dt.date
    h["value"] = h["equity"] - h["reserve"]
    return h.sort_values("ts")


def _overnight(engine, value_now: float, settings, now: datetime) -> Optional[float]:
    """Change since the previous trading day's last run (same reserve), or None if there isn't one."""
    h = _history(engine, settings)
    earlier = h[h["day"] < now.astimezone(NY).date()]
    return None if earlier.empty else value_now - float(earlier["value"].iloc[-1])


def _today(engine, value_now: float, settings, now: datetime) -> Optional[float]:
    """Change since today's first run (same reserve), or None if there isn't one."""
    h = _history(engine, settings)
    today = h[h["day"] == now.astimezone(NY).date()]
    return None if today.empty else value_now - float(today["value"].iloc[0])


def _earned(label: str, change: Optional[float], value_now: float) -> List[str]:
    """'Earned ...: +$12.34 (+0.25%)', or nothing when there's no earlier run to compare with."""
    start = None if change is None else value_now - change
    if not start:
        return []
    return [f"{label}: {'+' if change >= 0 else ''}{_money(change)} ({change / start * 100:+.2f}%)"]


def _start_of_day(now: datetime) -> datetime:
    """Midnight New York time today, in UTC (how timestamps are stored)."""
    return datetime.combine(now.astimezone(NY).date(), time(0), NY).astimezone(timezone.utc)


def _trades_today(engine, now: datetime) -> List[str]:
    d = db.decisions.c
    with engine.connect() as conn:
        rows = conn.execute(
            select(d.ts, d.symbol, d.action, d.order_qty, d.price)
            .where(d.ts >= _start_of_day(now), d.action.in_(["buy", "sell"]), d.dry_run.is_(False))
            .order_by(d.ts)
        ).fetchall()
    out = []
    for ts, symbol, action, qty, price in rows:
        when = pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)
        amount = f", about {_money(qty * price)}" if qty and price else ""
        out.append(f"  {when.tz_convert(NY):%H:%M} {action} {qty:g} {symbol} at {_money(price or 0)}{amount}")
    return out


def _problems_today(engine, now: datetime) -> List[str]:
    """Alerts sent today (alerts.alert remembers each key with the date it last went out)."""
    s = db.bot_state.c
    today = now.astimezone(NY).date().isoformat()
    with engine.connect() as conn:
        rows = conn.execute(select(s.key).where(s.key.like("alert:%"), s.value == today)).fetchall()
    return [f"  {k.split(':', 1)[1]}" for (k,) in rows]


def _order_lines(result: Dict, side: str) -> List[str]:
    lines = []
    for d in result.get("decisions", []):
        if d.get("action") == side and d.get("order_qty"):
            qty, price = d["order_qty"], d.get("price") or 0
            lines.append(f"  {d['symbol']}: {qty:g} shares at about {_money(price)} ({_money(qty * price)})")
    return lines


def is_last_run(broker, now: datetime) -> bool:
    """True if this run is the last one before today's close (so Stratos goes offline after it)."""
    close = getattr(broker, "next_close", None)
    close = close() if callable(close) else None
    return close is not None and close - now <= RUN_EVERY


def after_run(engine, settings, broker, result: Dict, now: datetime) -> List[str]:
    """Send whichever status emails this run calls for. Returns the subjects sent (for tests and logs)."""
    if result.get("status") == "market_closed" or not getattr(settings, "status_emails", False) or settings.dry_run:
        return []
    sent: List[str] = []
    day = now.astimezone(NY).date()
    equity = result.get("equity")
    if equity is None:
        equity = broker.get_account().equity
    value, label = _value(equity, settings)
    holdings = _holdings(engine)
    halted = result.get("status") == "halted"

    def send(key: str, subject: str, body: str, once: Optional[str] = None) -> None:
        if alerts.notify(engine, settings, key, subject, body, once=once):
            sent.append(subject)

    # 1. Online: the first run of the trading day
    body = [f"Stratos is running for {day:%A, %B %d}.", "", f"{label}: {_money(value)}",
            *_earned("Earned Overnight", _overnight(engine, value, settings, now), value), "",
            "Holding:" if holdings else "Holding: nothing (all cash)", *holdings, "",
            f"Next scheduled rebalance: first trading day of {_next_month(day)}."]
    if halted:
        body += ["", f"Note: trading is halted. {result.get('note', '')}"]
    send("online", "Stratos is online", "\n".join(body), once=day.isoformat())

    # 2. Trades placed this run
    for side, verb in (("buy", "bought"), ("sell", "sold")):
        lines = _order_lines(result, side)
        if lines:
            symbols = ", ".join(line.split(":")[0].strip() for line in lines)
            send(f"{side}:{result.get('ts')}", f"Stratos {verb} {symbols}",
                 "\n".join([f"Orders placed at {now.astimezone(NY):%H:%M} ET:", *lines, "",
                            "These are market orders sent to Alpaca; check Alpaca's Orders page for the fills."]))

    # 3. Offline: the end of the last run before the close
    if is_last_run(broker, now):
        trades = _trades_today(engine, now)
        problems = _problems_today(engine, now)
        body = [f"Stratos has finished for {day:%A, %B %d} and is offline until the next trading day's open.", "",
                f"{label}: {_money(value)}", *_earned("Earned Today", _today(engine, value, settings, now), value)]
        body += ["", "Trades today:" if trades else "Trades today: none", *trades, "",
                 "Problems today:" if problems else "Problems today: none", *problems, "",
                 "Holding:" if holdings else "Holding: nothing (all cash)", *holdings, "",
                 "Figures are from Stratos's last run, a few minutes before the close."]
        send("offline", "Stratos is offline for the day", "\n".join(body), once=day.isoformat())
    return sent


def watchdog(engine, settings, broker, now: datetime) -> Dict:
    """Email once per outage if Stratos hasn't completed a run recently while the market is open."""
    if not broker.is_market_open():
        return {"status": "market_closed"}
    local = now.astimezone(NY)
    if local.time() < (datetime.combine(local.date(), OPEN) + STALE_AFTER).time():
        return {"status": "too_early", "note": "the first runs of the day may not have happened yet"}
    t = db.equity_history.c
    with engine.connect() as conn:
        last = conn.execute(select(t.ts).order_by(t.ts.desc()).limit(1)).scalar()
    last_ts = None if last is None else pd.Timestamp(last)
    if last_ts is not None and last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    if last_ts is not None and now - last_ts.to_pydatetime() <= STALE_AFTER:
        return {"status": "ok", "last_run": last_ts.isoformat()}
    marker = "none" if last_ts is None else last_ts.isoformat()
    if db.get_state(engine, "watchdog:alerted") == marker:  # already told you about this outage
        return {"status": "offline", "last_run": marker, "alerted": False}
    since = "no run recorded yet" if last_ts is None else f"no successful run since {last_ts.tz_convert(NY):%H:%M} ET"
    message = (f"Stratos is offline: {since}, while the market is open. It normally runs every 5 minutes. "
               "Check the schedule (stratos-every-5-min) and the logs: "
               "aws logs tail /aws/lambda/stratos-live-trader --since 1h")
    log.error("WATCHDOG %s", message)
    topic = getattr(settings, "alert_topic_arn", None)
    sent = False
    if topic:
        try:
            alerts._publish(topic, "Stratos alert: offline", message)
            sent = True
        except Exception:
            log.exception("watchdog could not send its alert")
    if sent:
        db.set_state(engine, "watchdog:alerted", marker)
    return {"status": "offline", "last_run": marker, "alerted": sent}
