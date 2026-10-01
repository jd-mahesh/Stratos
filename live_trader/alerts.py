"""Alerts: tell a person when the bot halts, cancels a run, skips data or fails an order.

Every alert is logged. If ALERT_TOPIC_ARN is set (an AWS SNS topic with your
email subscribed, see docs/DEPLOY.md), it's also published there, which emails
you. Without it (e.g. on your laptop) alerts only go to the log.

The bot runs every five minutes, so the same problem would otherwise send an
email every five minutes. Each alert has a key, and a given key is sent at most
once per day (remembered in the bot_state table). Dry runs only log.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from trader_core import db

log = logging.getLogger("live_trader.alerts")
NY = ZoneInfo("America/New_York")


def _publish(topic_arn: str, subject: str, message: str) -> None:
    import boto3  # only needed on AWS

    boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject[:100], Message=message)


def alert(engine, settings, key: str, message: str, now: datetime, level: int = logging.WARNING) -> bool:
    """Log ``message`` and, at most once per day per ``key``, send it to SNS. Returns True if sent."""
    log.log(level, "ALERT [%s] %s", key, message)
    if settings.dry_run or engine is None:
        return False
    today = now.astimezone(NY).date().isoformat()
    state_key = f"alert:{key}"
    if db.get_state(engine, state_key) == today:
        return False  # already sent today
    db.set_state(engine, state_key, today)
    topic: Optional[str] = getattr(settings, "alert_topic_arn", None)
    if not topic:
        return False
    try:
        _publish(topic, f"Stratos: {key}", message)
    except Exception:  # an alert failure must never stop the safety logic around it
        log.exception("could not publish alert %s to SNS", key)
        return False
    return True
