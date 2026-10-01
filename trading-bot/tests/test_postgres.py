"""Runs the database code against real Postgres, since SQLite is more forgiving
(it happily stores numpy floats that psycopg2 rejects, for example).

Skipped unless TEST_DATABASE_URL is set. CI starts a Postgres service for it; locally:

    docker compose up -d db
    TEST_DATABASE_URL=postgresql+psycopg2://trader:trader@localhost:5432/trading pytest tests/test_postgres.py
"""
import os
from datetime import datetime, timezone

import pytest

from backtester.main import run
from live_trader.trader import run_tick
from trader_core import db
from trader_core.config import Settings

from .test_live_trader import FakeBroker, FakeData, settings

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def pg():
    engine = db.get_engine(URL)
    db.metadata.drop_all(engine)
    db.init_db(engine)
    yield engine
    db.metadata.drop_all(engine)


def test_backtest_saves_to_postgres(pg):
    s = Settings.from_env({"DATABASE_URL": URL, "DATA_PROVIDER": "synthetic", "SYMBOLS": "AAA,BBB"})
    summary = run(s, start="2023-01-01", end="2024-06-30")
    with pg.connect() as conn:
        run_row = conn.execute(db.backtest_runs.select()).one()
        n_trades = len(conn.execute(db.backtest_trades.select()).fetchall())
    assert run_row.id == summary["run_id"]
    assert run_row.params == {"fast": 20, "slow": 50}
    assert n_trades == summary["num_trades"]


def test_live_tick_saves_to_postgres(pg):
    broker = FakeBroker(positions={"DOWN": 300})
    run_tick(settings(), broker, FakeData(), pg, now=datetime(2024, 6, 3, 14, 5, tzinfo=timezone.utc))
    with pg.connect() as conn:
        assert len(conn.execute(db.decisions.select()).fetchall()) == 2
        assert len(conn.execute(db.equity_history.select()).fetchall()) == 1
