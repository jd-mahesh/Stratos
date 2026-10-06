"""Status emails (online, bought/sold, offline for the day) and the offline watchdog."""
from datetime import datetime, timedelta, timezone

import pytest

from live_trader import alerts, status
from live_trader.trader import run_tick
from trader_core import db
from trader_core.config import Settings

from .test_live_trader import FakeBroker, VolData

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)  # Monday 10:07 ET
TOPIC = "arn:aws:sns:us-east-2:123456789012:stratos-alerts"


def settings(**extra):
    env = {"SYMBOLS": "UP,UP2", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "126", "MOMENTUM_TOP": "2",
           "MAX_ORDER_PCT": "100", "MAX_POSITION_PCT": "100", "STATUS_EMAILS": "true", "ALERT_TOPIC_ARN": TOPIC}
    env.update(extra)
    return Settings.from_env(env)


@pytest.fixture
def mail(monkeypatch):
    """Capture every email that would go out through SNS, as (subject, body)."""
    out = []
    monkeypatch.setattr(alerts, "_publish", lambda topic, subject, message: out.append((subject, message)))
    return out


def subjects(mail):
    return [s for s, _ in mail]


def test_online_once_per_trading_day(engine, mail):
    broker = FakeBroker()
    run_tick(settings(), broker, VolData(), engine, now=NOW)
    run_tick(settings(), broker, VolData(), engine, now=NOW + timedelta(minutes=5))
    assert subjects(mail).count("Stratos is online") == 1
    body = dict(mail)["Stratos is online"]
    assert "Account value" in body and "Next scheduled rebalance: first trading day of July 2024" in body
    run_tick(settings(), broker, VolData(), engine, now=NOW + timedelta(days=1))  # next day: online again
    assert subjects(mail).count("Stratos is online") == 2


def test_bought_and_sold_emails_list_the_orders_of_that_run(engine, mail):
    run_tick(settings(), FakeBroker(), VolData(), engine, now=NOW)
    bought = [m for m in mail if m[0].startswith("Stratos bought")]
    assert len(bought) == 1 and bought[0][0] == "Stratos bought UP, UP2"
    assert "UP:" in bought[0][1] and "shares at about $" in bought[0][1]
    # a later run that doesn't trade sends no trade email
    mail.clear()
    run_tick(settings(), FakeBroker(), VolData(), engine, now=NOW + timedelta(minutes=5))
    assert not [s for s in subjects(mail) if s.startswith(("Stratos bought", "Stratos sold"))]


def test_sold_email(engine, mail):
    broker = FakeBroker(positions={"UP": 100, "UP2": 100, "OLD": 50}, cash=0.0)
    s = settings(SYMBOLS="UP,UP2,OLD", MOMENTUM_TOP="2")
    data = VolData()
    data.bars["OLD"] = data.bars["UP"].iloc[::-1].set_axis(data.bars["UP"].index)  # falling: not a pick
    run_tick(s, broker, data, engine, now=NOW)
    sold = [m for m in mail if m[0].startswith("Stratos sold")]
    assert len(sold) == 1 and "OLD" in sold[0][0]


def test_offline_is_sent_by_the_last_run_before_the_close(engine, mail):
    close = NOW.replace(hour=20, minute=0)  # 4:00 PM ET
    broker = FakeBroker(close=close)
    run_tick(settings(), broker, VolData(), engine, now=NOW)  # 10:07: not the last run
    assert "Stratos is offline for the day" not in subjects(mail)
    run_tick(settings(), broker, VolData(), engine, now=close - timedelta(minutes=4, seconds=40))  # 3:55 run
    assert subjects(mail)[-1] == "Stratos is offline for the day"  # sent last, after everything else
    body = mail[-1][1]
    assert "offline until the next trading day" in body and "Trades today:\n" in body and " buy " in body
    run_tick(settings(), broker, VolData(), engine, now=close - timedelta(minutes=4))  # a repeat: no second email
    assert subjects(mail).count("Stratos is offline for the day") == 1


def test_offline_on_an_early_close_day(engine, mail):
    early = NOW.replace(hour=17, minute=0)  # 1:00 PM ET close
    broker = FakeBroker(close=early)
    run_tick(settings(), broker, VolData(), engine, now=early - timedelta(minutes=10))  # 12:50: not yet
    assert "Stratos is offline for the day" not in subjects(mail)
    run_tick(settings(), broker, VolData(), engine, now=early - timedelta(minutes=4, seconds=30))  # 12:55
    assert "Stratos is offline for the day" in subjects(mail)


def test_offline_reports_the_trading_budget_with_a_reserve(engine, mail):
    close = NOW.replace(hour=20, minute=0)
    run_tick(settings(CAPITAL_RESERVE="90000"), FakeBroker(close=close), VolData(), engine,
             now=close - timedelta(minutes=4))
    body = dict(mail)["Stratos is offline for the day"]
    assert "Trading budget (account $100,000.00 minus the $90,000.00 reserve): $10,000.00" in body


def test_no_status_emails_when_off_in_dry_runs_or_when_closed(engine, mail):
    close = NOW + timedelta(minutes=3)
    run_tick(settings(STATUS_EMAILS="false"), FakeBroker(close=close), VolData(), engine, now=NOW)
    run_tick(settings(DRY_RUN="true"), FakeBroker(close=close), VolData(), engine, now=NOW)
    run_tick(settings(), FakeBroker(market_open=False, close=close), VolData(), engine, now=NOW)
    assert mail == []


def test_a_failing_email_never_breaks_the_run(engine, monkeypatch):
    def broken(topic, subject, message):
        raise RuntimeError("SNS is down")

    monkeypatch.setattr(alerts, "_publish", broken)
    broker = FakeBroker(close=NOW + timedelta(minutes=3))
    result = run_tick(settings(), broker, VolData(), engine, now=NOW)
    assert result["status"] == "ok" and {o[0] for o in broker.orders} == {"UP", "UP2"}


# --- watchdog ------------------------------------------------------------------

def snapshot(engine, ts):
    db.insert_rows(engine, db.equity_history, [{"ts": ts, "equity": 1.0, "cash": 1.0, "buying_power": 1.0}])


def test_watchdog_emails_once_per_outage(engine, mail):
    s = settings(STATUS_EMAILS="false")  # the watchdog is a problem alert: it doesn't need status emails on
    snapshot(engine, NOW - timedelta(minutes=30))
    out = status.watchdog(engine, s, FakeBroker(), NOW)
    assert out["status"] == "offline" and out["alerted"]
    assert subjects(mail) == ["Stratos alert: offline"] and "no successful run since 09:37 ET" in mail[0][1]
    assert not status.watchdog(engine, s, FakeBroker(), NOW + timedelta(minutes=15))["alerted"]  # same outage
    assert len(mail) == 1
    snapshot(engine, NOW + timedelta(minutes=20))  # it came back...
    assert status.watchdog(engine, s, FakeBroker(), NOW + timedelta(minutes=25))["status"] == "ok"
    out = status.watchdog(engine, s, FakeBroker(), NOW + timedelta(minutes=60))  # ...and stopped again
    assert out["alerted"] and len(mail) == 2


def test_watchdog_is_quiet_when_running_closed_or_too_early(engine, mail):
    s = settings()
    snapshot(engine, NOW - timedelta(minutes=4))
    assert status.watchdog(engine, s, FakeBroker(), NOW)["status"] == "ok"
    assert status.watchdog(engine, s, FakeBroker(market_open=False), NOW)["status"] == "market_closed"
    nine_forty = NOW.replace(hour=13, minute=40)  # 9:40 ET: the day's first runs may not have happened yet
    assert status.watchdog(engine, s, FakeBroker(), nine_forty)["status"] == "too_early"
    assert mail == []


def test_watchdog_never_trades(engine, mail):
    broker = FakeBroker()
    status.watchdog(engine, settings(), broker, NOW)
    assert broker.orders == []


def test_lambda_watchdog_event(engine, mail, monkeypatch):
    from live_trader import handler as lambda_handler

    broker = FakeBroker()
    monkeypatch.setattr(lambda_handler, "_cache", {"engine": engine, "broker": broker, "data": VolData()})
    monkeypatch.setenv("ALERT_TOPIC_ARN", TOPIC)
    out = lambda_handler.handler({"watchdog": True}, None)
    assert out["status"] in ("offline", "too_early", "ok", "market_closed")  # depends on the real clock
    assert broker.orders == []
