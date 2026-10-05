"""AWS Lambda entry point for the backtester.

Invoke with an optional JSON event using the same keys as the CLI, e.g.::

    {"symbols": ["SPY", "QQQ"], "start": "2021-01-01", "fast": 20, "slow": 50}
"""
from __future__ import annotations

import logging

from trader_core.config import Settings

from .main import run

logging.getLogger().setLevel(logging.INFO)

ALLOWED = {"symbols", "start", "end", "fast", "slow", "capital", "provider", "slippage_bps", "save", "fractional",
           "strategy", "window", "lookback", "top", "signal_mode", "cash_rate", "crash_switch",
           "crash_confirm", "stop"}


def handler(event, context):
    event = event or {}
    kwargs = {k: v for k, v in event.items() if k in ALLOWED}
    if isinstance(kwargs.get("symbols"), str):
        kwargs["symbols"] = kwargs["symbols"].split(",")
    return run(Settings.from_env(), **kwargs)
