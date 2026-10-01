"""AWS Lambda entry point for the live trader.

EventBridge Scheduler invokes this every few minutes. Configure the schedule's
input as ``{"scheduled_time": "<aws.scheduler.scheduled-time>"}`` so a retried
invocation reuses the same order ids (see trader.py).

To clear a circuit-breaker halt after checking what happened, invoke it once
by hand with ``{"resume": true}``; that only clears the halt and doesn't trade.
"""
from __future__ import annotations

import logging

from trader_core import db, safeguards
from trader_core.config import Settings
from trader_core.data import make_provider

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


def handler(event, context):
    event = event or {}
    settings = Settings.from_env()
    broker, data, engine = _deps(settings)
    if event.get("resume") is True:
        previous = safeguards.clear_halt(engine)
        return {"status": "resumed" if previous else "not_halted", "previous_halt": previous,
                "kill_switch_on": settings.trading_halted}
    return run_tick(settings, broker, data, engine, scheduled_time=event.get("scheduled_time"))
