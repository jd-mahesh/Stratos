"""Safeguards: kill switch, loss circuit breaker, price sanity, order limits, settings check, alerts."""
import json
import logging
import runpy
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from live_trader import alerts
from live_trader.trader import SafetyConfigError, rebalance_key, run_tick
from trader_core import db, safeguards
from trader_core.config import Settings
from trader_core.portfolio import Order
from trader_core.strategy import make_strategy

from .conftest import make_bars
from .test_live_trader import FakeBroker, VolData

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)  # Monday 10:07 ET
TODAY = NOW.date()


def two_stock(**extra):
    """Momentum on two rising stocks, top 2 (50% each), so the limits are opened up unless a test sets them."""
    env = {"SYMBOLS": "UP,UP2", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "126", "MOMENTUM_TOP": "2",
           "MAX_ORDER_PCT": "100", "MAX_POSITION_PCT": "100"}
    env.update(extra)
    return Settings.from_env(env)


def snapshot(engine, ts, equity):
    db.insert_rows(engine, db.equity_history, [{"ts": ts, "equity": equity, "cash": equity, "buying_power": equity}])


@pytest.fixture
def sent(monkeypatch):
    """Capture alerts that would be emailed through SNS."""
    out = []
    monkeypatch.setattr(alerts, "_publish", lambda topic, subject, message: out.append((subject, message)))
    return out


# --- settings ------------------------------------------------------------------

def test_safety_settings_defaults_and_validation():
    s = Settings.from_env({})
    assert (s.max_order_pct, s.max_position_pct, s.max_orders_per_run) == (0.25, 0.30, 20)
    assert (s.max_price_move_pct, s.max_data_age_days) == (0.40, 5)
    assert (s.daily_loss_halt_pct, s.drawdown_halt_pct) == (0.15, 0.60)
    assert s.trading_halted is False and s.alert_topic_arn is None
    custom = Settings.from_env({"MAX_ORDER_PCT": "10", "TRADING_HALTED": "true", "MAX_ORDERS_PER_RUN": "3"})
    assert custom.max_order_pct == pytest.approx(0.10) and custom.trading_halted and custom.max_orders_per_run == 3
    for name, bad in (("MAX_ORDER_PCT", "0"), ("MAX_POSITION_PCT", "150"), ("DAILY_LOSS_HALT_PCT", "abc"),
                      ("MAX_ORDERS_PER_RUN", "0"), ("MAX_DATA_AGE_DAYS", "x")):
        with pytest.raises(ValueError, match=name):
            Settings.from_env({name: bad})


def test_settings_check_refuses_targets_above_the_limits():
    five = Settings.from_env({"STRATEGY": "momentum", "MOMENTUM_TOP": "5"})
    assert safeguards.check_settings(make_strategy(five), 53, five) is None  # 20% each: fine
    two = Settings.from_env({"STRATEGY": "momentum", "MOMENTUM_TOP": "2"})
    assert "50%" in safeguards.check_settings(make_strategy(two), 53, two)
    crash = Settings.from_env({"STRATEGY": "momentum", "MOMENTUM_TOP": "5", "CRASH_SWITCH": "true"})
    assert "100%" in safeguards.check_settings(make_strategy(crash), 53, crash)  # backup = one safe haven
    cash = Settings.from_env({"STRATEGY": "momentum", "MOMENTUM_TOP": "5", "CRASH_SWITCH": "true",
                              "CRASH_MODE": "cash"})
    assert safeguards.check_settings(make_strategy(cash), 53, cash) is None
    etf4 = Settings.from_env({"STRATEGY": "trend"})  # @etf4: 4 symbols, 25% each
    assert safeguards.check_settings(make_strategy(etf4), len(etf4.symbols), etf4) is None


def test_bot_refuses_to_start_with_unsafe_settings(engine, sent):
    s = two_stock(MAX_ORDER_PCT="25", MAX_POSITION_PCT="30", ALERT_TOPIC_ARN="arn:aws:sns:x")
    broker = FakeBroker()
    with pytest.raises(SafetyConfigError):
        run_tick(s, broker, VolData(), engine, now=NOW)
    assert broker.orders == [] and len(sent) == 1 and "refuses to trade" in sent[0][1]


# --- price sanity --------------------------------------------------------------

def test_price_checks():
    s = Settings.from_env({})
    bars = make_bars([100.0] * 10, start="2024-05-20")  # last bar Fri 2024-05-31
    assert safeguards.check_price(bars, 101.0, TODAY, s) is None
    assert "no valid live price" in safeguards.check_price(bars, None, TODAY, s)
    assert "no valid live price" in safeguards.check_price(bars, 0.0, TODAY, s)
    assert "no valid live price" in safeguards.check_price(bars, float("nan"), TODAY, s)
    assert "+50%" in safeguards.check_price(bars, 150.0, TODAY, s)
    assert "-45%" in safeguards.check_price(bars, 55.0, TODAY, s)
    stale = make_bars([100.0] * 10, start="2024-05-08")  # last bar 2024-05-21: 13 days old
    assert "days old" in safeguards.check_price(stale, 100.0, TODAY, s)
    assert "no daily price history" in safeguards.check_price(None, 100.0, TODAY, s)


# --- order limits --------------------------------------------------------------

def test_order_limits():
    s = Settings.from_env({})
    prices = {"A": 100.0, "B": 50.0}
    ok = [Order("A", "buy", 200, ""), Order("B", "sell", 300, "")]  # $20k buy = 20%
    assert safeguards.check_orders(ok, {"B": 300}, prices, 100_000, s) == []
    big = safeguards.check_orders([Order("A", "buy", 300, "")], {}, prices, 100_000, s)
    assert any("30% of the account; the limit is 25%" in p for p in big)
    stacked = safeguards.check_orders([Order("A", "buy", 200, "")], {"A": 150}, prices, 100_000, s)
    assert any("position after buying would be 35%" in p for p in stacked)
    short = safeguards.check_orders([Order("B", "sell", 400, "")], {"B": 300}, prices, 100_000, s)
    assert any("short position" in p for p in short)
    many = [Order("A", "buy", 1, "")] * 21
    assert any("21 orders planned" in p for p in safeguards.check_orders(many, {}, prices, 100_000, s))
    # selling a big position is fine: it only reduces risk
    assert safeguards.check_orders([Order("A", "sell", 500, "")], {"A": 500}, prices, 100_000, s) == []


# --- circuit breaker -----------------------------------------------------------

def test_loss_checks(engine):
    s = Settings.from_env({})
    assert safeguards.check_losses(engine, 100_000, NOW, s) is None  # nothing to compare with yet
    snapshot(engine, NOW - timedelta(days=3), 100_000)  # Friday
    assert safeguards.check_losses(engine, 90_000, NOW, s) is None  # -10% < 15%
    assert "fell -20.0% since the previous day" in safeguards.check_losses(engine, 80_000, NOW, s)
    snapshot(engine, NOW - timedelta(days=30), 300_000)  # an old peak
    snapshot(engine, NOW - timedelta(days=2), 110_000)
    assert "below its peak" in safeguards.check_losses(engine, 110_000, NOW, s)  # -63% from 300k
    assert "looks wrong" in safeguards.check_losses(engine, 0.0, NOW, s)


def test_circuit_breaker_halts_until_resumed(engine, sent):
    s = two_stock(ALERT_TOPIC_ARN="arn:aws:sns:x")
    snapshot(engine, NOW - timedelta(days=3), 100_000)
    broker = FakeBroker(equity=80_000)  # -20% since Friday
    result = run_tick(s, broker, VolData(), engine, now=NOW)
    assert result["status"] == "halted" and broker.orders == []
    assert "circuit breaker" in result["note"] and safeguards.get_halt(engine)
    assert len(sent) == 1 and "Trading halted" in sent[0][1]

    recovered = FakeBroker(equity=100_000)
    later = run_tick(s, recovered, VolData(), engine, now=NOW.replace(hour=16))
    assert later["status"] == "halted" and recovered.orders == []  # stays halted after recovering

    assert safeguards.clear_halt(engine)["reason"].startswith("account fell")
    resumed = run_tick(s, recovered, VolData(), engine, now=NOW.replace(hour=17))
    assert resumed["status"] == "ok" and recovered.orders


def test_dry_run_reports_but_does_not_record_a_halt(engine, sent):
    snapshot(engine, NOW - timedelta(days=3), 100_000)
    result = run_tick(two_stock(DRY_RUN="true", ALERT_TOPIC_ARN="arn:aws:sns:x"), FakeBroker(equity=80_000),
                      VolData(), engine, now=NOW)
    assert result["status"] == "halted" and "dry run" in result["note"]
    assert safeguards.get_halt(engine) is None and sent == []


def test_kill_switch_records_balances_and_places_nothing(engine):
    broker = FakeBroker()
    result = run_tick(two_stock(TRADING_HALTED="true"), broker, VolData(), engine, now=NOW)
    assert result["status"] == "halted" and "kill switch" in result["note"] and broker.orders == []
    with engine.connect() as conn:
        assert len(conn.execute(db.equity_history.select()).fetchall()) == 1


# --- whole-run behaviour ---------------------------------------------------------

def test_order_limit_breach_cancels_the_whole_run(engine, sent):
    s = two_stock(MAX_ORDERS_PER_RUN="1", ALERT_TOPIC_ARN="arn:aws:sns:x")  # the plan has 2 buys
    broker = FakeBroker()
    result = run_tick(s, broker, VolData(), engine, now=NOW)
    assert result["status"] == "blocked" and result["rebalanced"] is False and broker.orders == []
    actions = {d["symbol"]: d["action"] for d in result["decisions"]}
    assert actions == {"UP": "blocked", "UP2": "blocked"}
    assert db.get_state(engine, rebalance_key(make_strategy(s))) is None  # month not marked done
    assert len(sent) == 1 and "Run cancelled" in sent[0][1]


def test_suspicious_price_skips_that_symbol_and_retries(engine, sent):
    class Glitch(VolData):
        def latest_prices(self, symbols):
            live = super().latest_prices(symbols)
            live["UP2"] *= 3  # +200% in a day: bad data
            return live

    s = two_stock(ALERT_TOPIC_ARN="arn:aws:sns:x")
    broker = FakeBroker()
    result = run_tick(s, broker, Glitch(), engine, now=NOW)
    by = {d["symbol"]: d for d in result["decisions"]}
    assert by["UP2"]["action"] == "skip" and by["UP2"]["reason"].startswith("safety:")
    assert [o[0] for o in broker.orders] == ["UP"]  # the good symbol still trades
    assert "UP2" in result["safety"]["skipped"] and "Skipped for suspicious price data" in sent[0][1]
    assert db.get_state(engine, rebalance_key(make_strategy(s))) is None  # retried next run
    assert run_tick(s, FakeBroker(positions={"UP": 1}), VolData(), engine, now=NOW.replace(hour=15))["rebalanced"]


def test_normal_run_is_untouched_by_the_safeguards(engine):
    s = Settings.from_env({"SYMBOLS": "UP,UP2", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "126",
                           "MOMENTUM_TOP": "5"})  # 20% per stock: within the default limits
    broker = FakeBroker()
    result = run_tick(s, broker, VolData(), engine, now=NOW)
    assert result["status"] == "ok" and result["safety"] == {}
    assert sorted(o[0] for o in broker.orders) == ["UP", "UP2"]


# --- alerts --------------------------------------------------------------------

def test_alerts_are_sent_once_a_day_per_problem(engine, sent, caplog):
    s = Settings.from_env({"ALERT_TOPIC_ARN": "arn:aws:sns:x"})
    with caplog.at_level(logging.WARNING):
        assert alerts.alert(engine, s, "price-check", "first", NOW) is True
        assert alerts.alert(engine, s, "price-check", "again", NOW.replace(hour=18)) is False
        assert alerts.alert(engine, s, "other", "different problem", NOW) is True
        assert alerts.alert(engine, s, "price-check", "next day", NOW + timedelta(days=1)) is True
    assert [m for _, m in sent] == ["first", "different problem", "next day"]
    assert "again" in caplog.text  # still logged every time


def test_alerts_without_a_topic_or_in_dry_run_only_log(engine, sent):
    assert alerts.alert(engine, Settings.from_env({}), "k", "m", NOW) is False
    assert alerts.alert(engine, Settings.from_env({"DRY_RUN": "true", "ALERT_TOPIC_ARN": "a"}), "k2", "m", NOW) is False
    assert sent == []


def test_a_failing_alert_never_breaks_the_run(engine, monkeypatch):
    def boom(*a):
        raise RuntimeError("SNS down")

    monkeypatch.setattr(alerts, "_publish", boom)
    assert alerts.alert(engine, Settings.from_env({"ALERT_TOPIC_ARN": "a"}), "k", "m", NOW) is False


# --- resume --------------------------------------------------------------------

def test_lambda_resume_event_clears_the_halt(engine, monkeypatch):
    from live_trader import handler as lambda_handler

    safeguards.set_halt(engine, "account fell 20%", NOW)
    monkeypatch.setattr(lambda_handler, "_cache", {"engine": engine, "broker": FakeBroker(), "data": VolData()})
    out = lambda_handler.handler({"resume": True}, None)
    assert out["status"] == "resumed" and out["previous_halt"]["reason"] == "account fell 20%"
    assert safeguards.get_halt(engine) is None
    assert lambda_handler.handler({"resume": True}, None)["status"] == "not_halted"


def test_cli_resume(tmp_path, monkeypatch, capsys):
    url = f"sqlite:///{tmp_path / 'r.db'}"
    eng = db.get_engine(url)
    db.init_db(eng)
    safeguards.set_halt(eng, "account is 65% below its peak", NOW)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("TRADING_HALTED", "false")
    monkeypatch.setattr(sys, "argv", ["live_trader", "--resume"])
    with pytest.raises(SystemExit) as done:
        runpy.run_module("live_trader", run_name="__main__")
    assert done.value.code == 0
    assert "65% below its peak" in capsys.readouterr().out and safeguards.get_halt(eng) is None


def test_unreadable_halt_record_counts_as_halted(engine):
    db.set_state(engine, "halt", "not json")
    assert safeguards.get_halt(engine) == {"reason": "not json", "since": "unknown"}
    assert json.loads(json.dumps(safeguards.clear_halt(engine)))  # cleared, returns what was there
    assert safeguards.get_halt(engine) is None


def test_stale_or_bad_index_price_falls_back_to_closes(engine):
    # the crash detector's index with a garbage live price: the run still works on completed closes
    s = Settings.from_env({"SYMBOLS": "UP,UP2", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "126",
                           "MOMENTUM_TOP": "5", "CRASH_SWITCH": "true", "CRASH_MODE": "cash",
                           "SIGNAL_MODE": "intraday", "CRASH_WINDOW": "50"})

    class WithIndex(VolData):
        def __init__(self):
            super().__init__()
            idx = pd.bdate_range(end="2024-05-31", periods=300)[0].date().isoformat()
            self.bars["QQQ"] = make_bars([400.0] * 300, start=idx)

        def latest_prices(self, symbols):
            live = super().latest_prices(symbols)
            if "QQQ" in symbols:
                live["QQQ"] = 4.0  # -99%: would scream "crash" if trusted
            return live

    result = run_tick(s, FakeBroker(), WithIndex(), engine, now=NOW)
    assert result["status"] == "ok" and result["mode"] == "normal"


# --- trading budget (CAPITAL_RESERVE) ----------------------------------------

def test_capital_reserve_setting():
    assert Settings.from_env({}).capital_reserve == 0
    assert Settings.from_env({"CAPITAL_RESERVE": "100000"}).capital_reserve == 100_000
    for bad in ("abc", "-5", "$100,000", "inf"):
        with pytest.raises(ValueError, match="CAPITAL_RESERVE"):
            Settings.from_env({"CAPITAL_RESERVE": bad})


def test_check_budget():
    assert safeguards.check_budget(50.0, two_stock()) is None  # no reserve: never applies
    with_reserve = two_stock(CAPITAL_RESERVE="100000")
    assert safeguards.check_budget(105_000.0, with_reserve) is None
    assert "used up" in safeguards.check_budget(100_000.0, with_reserve)
    assert "used up" in safeguards.check_budget(90_000.0, with_reserve)


def reserve_snapshot(engine, ts, equity, reserve):
    db.insert_rows(engine, db.equity_history, [{"ts": ts, "equity": equity, "cash": equity,
                                                "buying_power": equity, "reserve": reserve}])


def test_setting_a_reserve_does_not_look_like_a_loss(engine):
    # Yesterday's $105k account vs today's $5k budget is a new history, not a 95% drop.
    snapshot(engine, NOW - timedelta(days=1), 105_000)
    assert safeguards.check_losses(engine, 5_000, NOW, two_stock(CAPITAL_RESERVE="100000")) is None


def test_loss_limits_apply_to_the_budget(engine):
    s = two_stock(CAPITAL_RESERVE="100000")
    reserve_snapshot(engine, NOW - timedelta(days=1), 106_000, 100_000)  # budget $6k yesterday
    assert safeguards.check_losses(engine, 5_500, NOW, s) is None  # -8%
    reason = safeguards.check_losses(engine, 5_000, NOW, s)  # -17% of the budget, under 1% of the account
    assert reason and "trading budget fell" in reason
    # without the reserve, the same account history is judged on the whole account
    assert safeguards.check_losses(engine, 105_000, NOW, two_stock()) is None


def test_order_limits_are_measured_against_the_budget():
    s = two_stock(MAX_ORDER_PCT="25", MAX_POSITION_PCT="30", CAPITAL_RESERVE="100000")
    buy = [Order("UP", "buy", 20, "test")]  # $2,000 at $100
    assert safeguards.check_orders(buy, {}, {"UP": 100.0}, 105_000, s) == []  # 2% of the whole account
    assert safeguards.check_orders(buy, {}, {"UP": 100.0}, 5_000, s)  # 40% of a $5k budget: blocked
