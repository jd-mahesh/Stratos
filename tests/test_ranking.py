"""The momentum ranking the live trader saves for the dashboard (live_trader/ranking.py)."""
from datetime import timedelta

import pytest
from sqlalchemy import select

from live_trader import ranking
from live_trader.trader import run_tick
from trader_core import db
from trader_core.strategy import Momentum

from .test_live_trader import FakeBroker, VolData
from .test_status import NOW, settings


def test_build_categories_meter_and_gap():
    strategy = Momentum(lookback=126, top=2)
    returns = {"A": 0.50, "B": 0.30, "C": 0.17, "D": 0.04, "E": -0.10}
    rows = {r["symbol"]: r for r in ranking.build(strategy, returns, {"NEW": "need 127 days of prices"}, {"A", "C"})}
    assert [rows[s]["rank"] for s in "ABCDE"] == [1, 2, 3, 4, 5]
    assert [rows[s]["meter"] for s in "ABCDE"] == [100.0, 75.0, 50.0, 25.0, 0.0]
    assert rows["A"]["category"] == rows["B"]["category"] == "top"
    assert rows["C"]["category"] == rows["D"]["category"] == "next"
    assert rows["C"]["gap"] == pytest.approx(1.30 / 1.17 - 1)  # must gain ~11% more than B to overtake it
    assert "needs +11.1% vs B" in rows["C"]["note"]
    assert rows["E"]["category"] == "negative" and rows["E"]["gap"] is None
    assert rows["NEW"]["category"] == "no_data" and rows["NEW"]["rank"] is None
    assert rows["A"]["held"] and rows["C"]["held"] and not rows["B"]["held"]


def test_build_with_fewer_positive_names_than_slots():
    rows = {r["symbol"]: r for r in ranking.build(Momentum(lookback=126, top=3), {"A": 0.2, "B": -0.1}, {}, set())}
    assert rows["A"]["category"] == "top" and rows["B"]["category"] == "negative"


def test_ranking_matches_the_strategys_own_picks(engine):
    s = settings(STATUS_EMAILS="false", MOMENTUM_TOP="1")
    data = VolData()
    result = run_tick(s, FakeBroker(), data, engine, now=NOW)
    bought = {d["symbol"] for d in result["decisions"] if d["action"] == "buy"}
    with engine.connect() as conn:
        top = {r.symbol for r in conn.execute(select(db.rankings).where(db.rankings.c.category == "top"))}
    assert top == bought == {"UP"}


def saved(engine):
    r = db.rankings.c
    with engine.connect() as conn:
        return conn.execute(select(r.ts, r.rebalance).distinct()).fetchall()


def test_snapshots_at_rebalance_first_run_and_last_run_only(engine):
    s = settings(STATUS_EMAILS="false")
    close = NOW.replace(hour=20, minute=0)
    broker = FakeBroker(close=close)
    run_tick(s, broker, VolData(), engine, now=NOW)  # the month's rebalance
    assert [bool(x.rebalance) for x in saved(engine)] == [True]
    run_tick(s, broker, VolData(), engine, now=NOW + timedelta(minutes=5))  # an ordinary run: nothing new
    assert len(saved(engine)) == 1
    run_tick(s, broker, VolData(), engine, now=close - timedelta(minutes=4))  # the day's last run
    assert len(saved(engine)) == 2
    run_tick(s, broker, VolData(), engine, now=close - timedelta(minutes=3))  # once a day
    assert len(saved(engine)) == 2


def test_first_ever_snapshot_even_before_the_open(engine):
    s = settings(STATUS_EMAILS="false")
    result = run_tick(s, FakeBroker(market_open=False), VolData(), engine, now=NOW)
    assert result["status"] == "market_closed" and result.get("ranking_saved")
    run_tick(s, FakeBroker(market_open=False), VolData(), engine, now=NOW + timedelta(minutes=5))
    assert len(saved(engine)) == 1  # only the first time


def test_a_failing_ranking_never_breaks_the_run(engine, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("data feed down")

    monkeypatch.setattr(ranking, "snapshot", broken)
    broker = FakeBroker()
    result = run_tick(settings(STATUS_EMAILS="false"), broker, VolData(), engine, now=NOW)
    assert result["status"] == "ok" and {o[0] for o in broker.orders} == {"UP", "UP2"}


def test_dashboard_state_is_recorded_without_secrets(engine):
    import json

    s = settings(STATUS_EMAILS="false", ALPACA_API_KEY_ID="PKSECRETKEY", ALPACA_API_SECRET_KEY="topsecret",
                 CAPITAL_RESERVE="100000")
    run_tick(s, FakeBroker(close=NOW + timedelta(hours=6)), VolData(), engine, now=NOW)
    clock = json.loads(db.get_state(engine, "dashboard:clock"))
    stored = db.get_state(engine, "dashboard:settings")
    assert clock["is_open"] and clock["next_close"].startswith("2024-06-03T20:07")
    assert json.loads(stored)["capital_reserve"] == 100000 and json.loads(stored)["top"] == 2
    assert "SECRET" not in stored.upper() and "topsecret" not in stored and "DATABASE" not in stored.upper()
