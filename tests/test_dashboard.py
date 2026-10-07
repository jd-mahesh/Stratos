"""The dashboard's status bar and momentum ranking, rendered with Streamlit's test runner."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from trader_core import db  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def dash(tmp_path, monkeypatch):
    """A fresh database the dashboard reads, and a function that renders the app against it."""
    url = f"sqlite:///{tmp_path / 'dash.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    engine = db.get_engine(url)
    db.init_db(engine)

    def render():
        st.cache_data.clear()
        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception, [e.value for e in at.exception]
        return at, " ".join(m.value for m in at.markdown)

    return engine, render


def now():
    return datetime.now(timezone.utc)


def setup_live(engine, last_run_minutes=3, is_open=True, **settings):
    t = now()
    db.insert_rows(engine, db.equity_history, [{"ts": t - timedelta(minutes=last_run_minutes), "equity": 104_900.0,
                                               "cash": 100_000.0, "buying_power": 0.0, "reserve": 100_000.0}])
    db.set_state(engine, "dashboard:clock", json.dumps({
        "checked_at": t.isoformat(), "is_open": is_open,
        "next_open": (t + timedelta(hours=17)).isoformat(),
        "next_close": (t + timedelta(hours=2)).isoformat() if is_open else (t + timedelta(hours=23)).isoformat(),
    }))
    base = {"strategy": "momentum", "params": {"lookback": 126, "top": 5}, "rebalance": "monthly", "top": 5,
            "symbols": 53, "dry_run": False, "trading_halted": False, "crash_switch": False, "capital_reserve": 100000}
    base.update(settings)
    db.set_state(engine, "dashboard:settings", json.dumps(base))
    db.set_state(engine, "last_rebalance:momentum:{}", t.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m"))


def test_status_pills_when_running_normally(dash):
    engine, render = dash
    setup_live(engine)
    _, text = render()
    for label in ("Market open", "Last run", "min ago", "Crash switch off", "Trading active", "Next rebalance"):
        assert label in text


def test_last_run_turns_amber_after_15_minutes_while_open(dash):
    engine, render = dash
    setup_live(engine, last_run_minutes=25)
    _, text = render()
    assert "should run every 5 min" in text and "#fab219" in text


def test_closed_market_halt_and_dry_run(dash):
    engine, render = dash
    setup_live(engine, is_open=False, dry_run=True)
    _, text = render()
    assert "Market closed" in text and "opens" in text and "Dry run" in text and "should run every" not in text
    db.set_state(engine, "halt", json.dumps({"reason": "account fell 20%", "since": "2026-10-06"}))
    _, text = render()
    assert "Halted" in text and "circuit breaker" in text


def test_ranking_table_marks_held_and_next_up(dash):
    engine, render = dash
    setup_live(engine)
    ts = now()
    rows = [("DELL", 1, 2.2, 100.0, "top", True, None, None), ("AMD", 2, 1.87, 98.0, "top", True, None, None),
            ("OSCR", 6, 1.507, 90.0, "next", False, 0.087, "needs +8.7% vs CRWD to swap in"),
            ("NFLX", 51, -0.32, 0.0, "negative", False, None, "return not positive: never bought"),
            ("UGLD", None, None, None, "no_data", False, None, "need 127 days of prices")]
    db.insert_rows(engine, db.rankings, [
        {"ts": ts, "symbol": s, "rank": r, "momentum": m, "meter": mt, "category": c, "held": h, "gap": g,
         "note": n, "rebalance": False} for s, r, m, mt, c, h, g, n in rows])
    at, _ = render()
    table = at.dataframe[0].value
    status = dict(zip(table["Symbol"], table["Status"]))
    assert status["DELL"] == "Held" and status["OSCR"].startswith("Next up · needs +8.7% vs CRWD")
    assert "NFLX" not in status  # only the top 10 (plus anything held) until "Show every stock" is on
    assert "6-month return" in table.columns and "Momentum" in table.columns
    assert any("Closest to swapping in: OSCR" in c.value for c in at.caption)


def test_no_ranking_yet(dash):
    engine, render = dash
    setup_live(engine)
    at, _ = render()
    assert any("No ranking saved yet" in c.value for c in at.caption)
