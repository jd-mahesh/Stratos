"""AWS Lambda entry point for the live trader.

EventBridge Scheduler invokes this every few minutes. Configure the schedule's
input as ``{"scheduled_time": "<aws.scheduler.scheduled-time>"}`` so a retried
invocation reuses the same order ids (see trader.py).

To clear a circuit-breaker halt after checking what happened, invoke it once
by hand with ``{"resume": true}``; that only clears the halt and doesn't trade.

To check that alert emails work end to end (this function's permissions, the
SNS topic, your subscription), invoke it with ``{"test_alert": true}``: it sends
one test alert through the same path real alerts use and reports whether that
worked. It doesn't touch the database or the broker and never trades.

The watchdog invokes it with ``{"watchdog": true}`` on its own schedule: if the
market is open and no run has completed for 15 minutes, it emails you once per
outage (live_trader/status.py). It never trades.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from trader_core import db, safeguards
from trader_core.config import Settings
from trader_core.data import make_provider

from . import alerts, status
from .broker import AlpacaBroker
from .trader import run_tick

logging.getLogger().setLevel(logging.INFO)

# Module-level objects survive between "warm" Lambda invocations, so the
# database connection pool and API clients are reused instead of rebuilt.
_cache = {}


def _deps(settings: Settings):
    if "engine" not in _cache:
        settings.require_alpaca()
        engine = db.get_engine(settings.database_url)
        db.init_db(engine)
        _cache["engine"] = engine
        _cache["broker"] = AlpacaBroker(settings.alpaca_key_id, settings.alpaca_secret_key)
        _cache["data"] = make_provider(settings, "alpaca")
    return _cache["broker"], _cache["data"], _cache["engine"]


def test_alert(settings: Settings) -> dict:
    """Send one test alert through SNS and report the outcome (never raises)."""
    topic = settings.alert_topic_arn
    if not topic:
        return {"status": "no_topic", "note": "ALERT_TOPIC_ARN isn't set on this function"}
    try:
        alerts._publish(topic, "Stratos: test alert",
                        "Test alert from the live-trader Lambda. If you're reading this, Stratos can email you "
                        "when a safeguard halts trading, cancels a run or an order fails.")
    except Exception as exc:  # report it instead of failing, so the reason is visible to whoever invoked it
        return {"status": "alert_failed", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "alert_sent", "topic": topic}


def handler(event, context):
    event = event or {}
    settings = Settings.from_env()
    if event.get("test_alert") is True:
        return test_alert(settings)
    if event.get("watchdog") is True:
        broker, _data, engine = _deps(settings)
        return status.watchdog(engine, settings, broker, datetime.now(timezone.utc))
    broker, data, engine = _deps(settings)
    if event.get("resume") is True:
        previous = safeguards.clear_halt(engine)
        return {"status": "resumed" if previous else "not_halted", "previous_halt": previous,
                "kill_switch_on": settings.trading_halted}
    return run_tick(settings, broker, data, engine, scheduled_time=event.get("scheduled_time"))
