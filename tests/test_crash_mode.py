"""Crash detector, defensive backup, the crash switch in backtest and live, and the crash report."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backtester.crash_report import crash_report, find_crashes
from backtester.engine import run_backtest
from backtester.sweep import sweep
from live_trader.trader import rebalance_key, run_tick
from trader_core import db
from trader_core.config import Settings
from trader_core.regime import CRASH, NORMAL, CrashDetector, regime_series
from trader_core.strategy import CrashSwitch, Defensive, Momentum, make_strategy

from .conftest import make_bars
from .test_live_trader import FakeBroker

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)
SMALL = CrashDetector(index="IDX", drawdown=0.10, window=50, high_window=100)


def rise_crash_recover(n_rise=250, n_fall=30, n_back=120, peak=200.0, fall=0.25):
    rise = np.linspace(100, peak, n_rise)
    down = np.linspace(peak, peak * (1 - fall), n_fall)
    back = np.linspace(peak * (1 - fall), peak * 1.1, n_back)
    return np.concatenate([rise, down, back])


# --- detector -----------------------------------------------------------------

def test_detector_needs_both_conditions_to_enter_crash_mode():
    d = CrashDetector(window=50, high_window=100)
    steady = np.linspace(100, 200, 150)
    assert d.check(steady, None).mode == NORMAL
    dip = np.append(steady, 185.0)  # 7.5% off the high: below the average? no, and not 10% down
    assert d.check(dip, None).mode == NORMAL
    crash = np.append(steady, np.linspace(195, 165, 10))  # 17% off the high and below the 50-day average
    r = d.check(crash, None)
    assert r.mode == CRASH and "crash mode" in r.reason


def test_detector_stays_in_crash_mode_until_back_above_the_average():
    d = CrashDetector(window=50, high_window=100)
    closes = np.append(np.linspace(100, 200, 150), np.linspace(195, 165, 10))
    partial = np.append(closes, 170.0)  # recovering a little, still below the average
    assert d.check(partial, CRASH).mode == CRASH
    assert d.check(partial, NORMAL).mode == CRASH  # it would also qualify as a fresh crash
    recovered = np.append(closes, np.full(5, 205.0))
    assert d.check(recovered, CRASH).mode == NORMAL


def test_detector_without_enough_history():
    assert CrashDetector().check(np.linspace(1, 2, 100), None) is None
    assert CrashDetector().check(None, NORMAL) is None


def test_confirmation_needs_the_rule_to_hold_several_closes_in_a_row():
    d = CrashDetector(window=50, high_window=100, confirm_days=5)
    assert d.required_bars == 104 and d.closes_only and not CrashDetector().closes_only
    steady = np.linspace(100, 200, 150)
    four = np.append(steady, [178.0, 176.0, 174.0, 172.0])  # crash rule met 4 closes running
    r = d.check(four, NORMAL)
    assert r.mode == NORMAL and "4 of the last 5" in r.reason
    assert CrashDetector(window=50, high_window=100).check(four, NORMAL).mode == CRASH  # unconfirmed: at once
    five = np.append(four, 170.0)
    assert d.check(five, NORMAL).mode == CRASH


def test_confirmation_ignores_a_one_day_dip():
    steady = np.linspace(100, 200, 150)
    dip = np.append(steady, [170.0, 199.0, 200.0])  # one bad close, then straight back
    one_day = np.append(steady, 170.0)
    assert CrashDetector(window=50, high_window=100).check(one_day, NORMAL).mode == CRASH
    confirmed = CrashDetector(window=50, high_window=100, confirm_days=5)
    modes = regime_series(pd.Series(dip, index=pd.bdate_range("2020-01-01", periods=dip.size)), confirmed)
    assert (modes.dropna() == NORMAL).all()


def test_confirmation_needs_several_closes_back_above_the_average_to_exit():
    d = CrashDetector(window=50, high_window=100, confirm_days=3)
    crashed = np.append(np.linspace(100, 200, 150), np.linspace(195, 150, 15))
    two_up = np.append(crashed, [210.0, 212.0])
    r = d.check(two_up, CRASH)
    assert r.mode == CRASH and "2 of the last 3" in r.reason
    assert CrashDetector(window=50, high_window=100).check(two_up, CRASH).mode == NORMAL
    assert d.check(np.append(two_up, 214.0), CRASH).mode == NORMAL
    assert d.check(np.append(crashed, [210.0, 150.0, 212.0]), CRASH).mode == CRASH  # not in a row


def test_confirm_days_must_be_positive():
    with pytest.raises(ValueError):
        CrashDetector(confirm_days=0)


def test_regime_series_matches_the_rule_day_by_day():
    closes = pd.Series(rise_crash_recover(), index=pd.bdate_range("2020-01-01", periods=400))
    modes = regime_series(closes, SMALL)
    assert modes.iloc[:99].isna().all()  # not enough history yet
    assert (modes == CRASH).any() and modes.iloc[-1] == NORMAL
    first_crash = modes[modes == CRASH].index[0]
    assert first_crash > closes.index[250]  # only after the fall started


# --- backup strategy and switch ---------------------------------------------

def test_defensive_picks_the_best_safe_haven_or_falls_back_to_bills():
    d = Defensive(lookback=63)
    hist = {"BIL": np.linspace(100, 101, 100), "IEF": np.linspace(100, 98, 100),
            "TLT": np.linspace(100, 110, 100), "GLD": np.linspace(100, 104, 100)}
    sig = d.decide({}, extra=hist)
    assert sig["TLT"].weight == 1.0 and sum(s.weight for s in sig.values()) == 1.0
    losing = {a: np.linspace(100, 90, 100) for a in ("IEF", "TLT", "GLD")} | {"BIL": np.linspace(100, 100.5, 100)}
    assert d.decide({}, extra=losing)["BIL"].weight == 1.0
    assert all(s.weight == 0 for s in Defensive(cash_only=True).decide({}, extra=hist).values())
    assert all(s.weight == 0 for s in d.decide({}, extra={}).values())  # no data yet: cash


def test_switch_uses_the_backup_only_in_crash_mode():
    switch = CrashSwitch(Momentum(lookback=20, top=1), Defensive(lookback=20), SMALL)
    stocks = {"A": np.linspace(100, 150, 60), "B": np.linspace(100, 120, 60)}
    havens = {"BIL": np.linspace(100, 101, 60), "IEF": np.linspace(100, 99, 60),
              "TLT": np.linspace(100, 105, 60), "GLD": np.linspace(100, 102, 60)}
    normal = switch.decide(stocks, 2, havens, NORMAL)
    assert normal["A"].weight == 1.0 and normal["TLT"].weight == 0.0
    crash = switch.decide(stocks, 2, havens, CRASH)
    assert crash["A"].weight == 0.0 and crash["B"].weight == 0.0 and crash["TLT"].weight == 1.0


def test_make_strategy_wraps_when_crash_switch_is_on():
    s = Settings.from_env({"STRATEGY": "momentum", "CRASH_SWITCH": "true", "CRASH_DRAWDOWN": "12",
                           "CRASH_ASSETS": "bil,tlt"})
    strat = make_strategy(s)
    assert isinstance(strat, CrashSwitch) and strat.name == "momentum"
    assert strat.detector.drawdown == pytest.approx(0.12) and strat.backup.assets == ("BIL", "TLT")
    assert strat.params()["crash"] == "QQQ -12%/200d -> BIL/TLT"
    assert strat.extra_symbols == ["QQQ", "BIL", "TLT"] and strat.tradable_extras == ["BIL", "TLT"]
    assert not isinstance(make_strategy(s, crash_switch=False), CrashSwitch)
    assert make_strategy(Settings.from_env({"CRASH_SWITCH": "true", "CRASH_MODE": "cash"})).tradable_extras == []


def test_make_strategy_passes_the_confirmation_setting():
    s = Settings.from_env({"STRATEGY": "momentum", "CRASH_SWITCH": "true", "CRASH_CONFIRM_DAYS": "5"})
    assert s.crash_confirm_days == 5
    strat = make_strategy(s)
    assert strat.detector.confirm_days == 5 and strat.params()["crash"].endswith(", confirm 5d")
    assert make_strategy(s, crash_confirm=3).detector.confirm_days == 3
    plain = make_strategy(Settings.from_env({"STRATEGY": "momentum", "CRASH_SWITCH": "true"}))
    assert plain.detector.confirm_days == 1 and "confirm" not in plain.params()["crash"]


# --- backtest -----------------------------------------------------------------

def crash_prices():
    n = 400
    stocks = {"A": make_bars(np.linspace(100, 300, n)), "B": make_bars(np.linspace(100, 150, n))}
    extra = {
        "IDX": make_bars(rise_crash_recover()),
        "BIL": make_bars(np.linspace(100, 104, n)),
        "IEF": make_bars(np.linspace(100, 101, n)),
        "TLT": make_bars(np.linspace(100, 130, n)),
        "GLD": make_bars(np.linspace(100, 90, n)),
    }
    return stocks, extra


@pytest.mark.parametrize("signal_mode", ["close", "intraday"])
def test_backtest_switches_to_the_backup_during_a_crash_and_back(signal_mode):
    stocks, extra = crash_prices()
    switch = CrashSwitch(Momentum(lookback=20, top=1), Defensive(lookback=63), SMALL)
    result = run_backtest(stocks, switch, slippage_bps=0, extra_prices=extra, signal_mode=signal_mode,
                          start=stocks["A"].index[120].date())
    modes = result.equity["mode"]
    crash_days = modes[modes == CRASH].index
    assert len(crash_days) and modes.iloc[-1] == NORMAL
    tlt = [t for t in result.trades if t.symbol == "TLT"]
    assert tlt, "expected crash mode to buy the best safe haven"
    # the switch happens as soon as the crash is detected, not at the next month start
    detected = crash_days[0]
    expected = detected if signal_mode == "intraday" else result.equity.index[result.equity.index.get_loc(detected) + 1]
    assert pd.Timestamp(tlt[0].entry_ts) <= expected
    assert tlt[0].exit_ts is not None  # sold again once back to normal
    assert result.metrics["crash_days"] == pytest.approx((modes == CRASH).mean())
    # exposure only counts the stock universe, so it's ~0 while holding bonds
    assert result.equity.loc[crash_days[-1], "exposure"] < 0.05


def test_confirmed_backtest_switches_at_the_open_after_the_rule_held_on_daily_closes():
    stocks, extra = crash_prices()
    detector = CrashDetector(index="IDX", drawdown=0.10, window=50, high_window=100, confirm_days=5)
    switch = CrashSwitch(Momentum(lookback=20, top=1), Defensive(lookback=63), detector)
    start = stocks["A"].index[120].date()
    result = run_backtest(stocks, switch, slippage_bps=0, extra_prices=extra, signal_mode="intraday", start=start)
    modes = result.equity["mode"]
    daily = regime_series(extra["IDX"]["close"], detector)
    first_by_close = daily[daily == CRASH].index[0]
    first_in_test = modes[modes == CRASH].index[0]
    # decided on a finished close, acted on at the next day's open, never the same day
    assert first_in_test == modes.index[modes.index.get_loc(first_by_close) + 1]
    unconfirmed = run_backtest(stocks, CrashSwitch(Momentum(lookback=20, top=1), Defensive(lookback=63), SMALL),
                               slippage_bps=0, extra_prices=extra, signal_mode="intraday", start=start)
    assert first_in_test > unconfirmed.equity["mode"][unconfirmed.equity["mode"] == CRASH].index[0]


def test_backtest_without_the_switch_rides_the_crash():
    stocks, extra = crash_prices()
    plain = run_backtest(stocks, Momentum(lookback=20, top=1), slippage_bps=0, extra_prices=extra,
                         start=stocks["A"].index[120].date())
    assert {t.symbol for t in plain.trades} <= {"A", "B"} and "mode" in plain.equity
    assert plain.equity["mode"].isna().all()


# --- crash report and normal-only sweep ------------------------------------

def test_find_crashes_marks_peak_trough_and_recovery():
    idx = pd.bdate_range("2020-01-01", periods=12)
    s = pd.Series([100, 110, 95, 90, 99, 111, 112, 100, 95, 96, 97, 98], index=idx, dtype=float)
    first, second = find_crashes(s, threshold=0.15)
    # 110 -> 90 (-18%), back above 110 on day 5
    assert (first.peak, first.trough, first.recovered) == (idx[1], idx[3], idx[5])
    assert first.depth == pytest.approx(90 / 110 - 1)
    # 112 -> 95 (-15.2%), not recovered by the end of the data
    assert (second.peak, second.trough, second.recovered) == (idx[6], idx[8], None)
    assert find_crashes(s, threshold=0.20) == []


def test_crash_report_runs_end_to_end():
    s = Settings.from_env({"DATA_PROVIDER": "synthetic", "SYMBOLS": "@sectors", "STRATEGY": "momentum",
                           "MOMENTUM_LOOKBACK": "126", "MOMENTUM_TOP": "3"})
    r = crash_report(s, start="2012-01-01", end="2020-12-31", threshold=0.10)
    assert r["with"]["crash_days"] is not None and r["without"]["cagr"] is not None
    for c in r["crashes"]:
        assert c["index_fall"] <= -0.10 and {"without_worst", "with_worst", "switched_on"} <= set(c)
        assert c["crash_days"] == sum(x["days"] for x in c["stretches"]) <= c["days"]
        if c["stretches"]:
            assert c["switched_on"] == c["stretches"][0]["on"]
            assert all(x["on"] <= x["off"] for x in c["stretches"])


def test_sweep_can_score_normal_markets_only():
    s = Settings.from_env({"DATA_PROVIDER": "synthetic", "SYMBOLS": "@sectors"})
    out = sweep(s, start="2014-01-01", end="2020-12-31", lookbacks=(63,), tops=(3,), normal_only=True)
    assert out["normal_only"] and 0 < out["normal_share"] <= 1
    assert out["rows"][0]["cagr"] is not None


# --- live trader --------------------------------------------------------------

class CrashingData:
    """300 days of prices ending the business day before NOW; QQQ is in a crash at the end."""

    def __init__(self, qqq_falls=True):
        start = pd.bdate_range(end="2024-05-31", periods=300)[0]
        mk = lambda c: make_bars(c, start=str(start.date()))  # noqa: E731
        qqq = np.concatenate([np.linspace(300, 450, 270), np.linspace(450, 350, 30)]) if qqq_falls \
            else np.linspace(300, 450, 300)
        self.bars = {
            "UP": mk(np.linspace(50, 100, 300)), "DOWN": mk(np.linspace(100, 50, 300)), "QQQ": mk(qqq),
            "BIL": mk(np.linspace(100, 104, 300)), "IEF": mk(np.linspace(100, 101, 300)),
            "TLT": mk(np.linspace(100, 120, 300)), "GLD": mk(np.linspace(100, 95, 300)),
        }

    def daily_bars(self, symbols, start, end=None):
        return {s: self.bars[s] for s in symbols if s in self.bars}

    def latest_prices(self, symbols):
        live = {s: float(self.bars[s]["close"].iloc[-1]) for s in symbols if s in self.bars}
        live.update({s: p for s, p in getattr(self, "live", {}).items() if s in live})
        return live


def crash_settings(**extra):
    env = {"SYMBOLS": "UP,DOWN", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "20", "MOMENTUM_TOP": "1",
           "CRASH_SWITCH": "true", "CRASH_WINDOW": "50", "MAX_ORDER_PCT": "100", "MAX_POSITION_PCT": "100"}
    env.update(extra)
    return Settings.from_env(env)


def test_live_switches_mid_month_when_a_crash_starts(engine):
    s = crash_settings()
    key = rebalance_key(make_strategy(s))
    db.set_state(engine, key, "2024-06")  # already rebalanced this month...
    db.set_state(engine, f"mode:{key}", NORMAL)  # ...in normal mode
    broker = FakeBroker(positions={"UP": 100})
    result = run_tick(s, broker, CrashingData(), engine, now=NOW)
    assert result["rebalanced"] and result["mode"] == CRASH
    assert [o[:3] for o in broker.orders][0] == ("UP", 100, "sell")
    assert broker.orders[1][0] == "TLT" and broker.orders[1][2] == "buy"
    assert db.get_state(engine, f"mode:{key}") == CRASH

    again = run_tick(s, broker, CrashingData(), engine, now=NOW.replace(hour=15))
    assert again["rebalanced"] is False and again["mode"] == CRASH and len(broker.orders) == 2


def test_live_dry_run_does_not_record_the_new_mode(engine):
    s = crash_settings(DRY_RUN="true")
    key = rebalance_key(make_strategy(s))
    db.set_state(engine, key, "2024-06")
    db.set_state(engine, f"mode:{key}", NORMAL)
    result = run_tick(s, FakeBroker(positions={"UP": 100}), CrashingData(), engine, now=NOW)
    assert result["mode"] == CRASH and db.get_state(engine, f"mode:{key}") == NORMAL


def test_live_normal_market_trades_normally_and_records_the_mode(engine):
    s = crash_settings()
    broker = FakeBroker()
    result = run_tick(s, broker, CrashingData(qqq_falls=False), engine, now=NOW)
    assert result["mode"] == NORMAL and [o[:3] for o in broker.orders][0][0] == "UP"
    assert db.get_state(engine, f"mode:{rebalance_key(make_strategy(s))}") == NORMAL


def test_daily_mode_is_saved_with_the_backtest(tmp_path):
    from sqlalchemy import text

    from backtester.main import run

    url = f"sqlite:///{tmp_path / 'm.db'}"
    s = Settings.from_env({"DATABASE_URL": url, "DATA_PROVIDER": "synthetic", "SYMBOLS": "@sectors",
                           "STRATEGY": "momentum", "CRASH_SWITCH": "true"})
    run(s, start="2015-01-01", end="2020-12-31")
    with db.get_engine(url).connect() as conn:
        modes = dict(conn.execute(text("SELECT mode, count(*) FROM backtest_equity GROUP BY mode")).fetchall())
    assert set(modes) <= {"normal", "crash"} and sum(modes.values()) > 1000


def test_live_confirmed_detector_ignores_the_live_price(engine):
    data = CrashingData(qqq_falls=False)
    data.live = {"QQQ": 300.0}  # a crash-like price right now, but every finished close is normal
    unconfirmed = run_tick(crash_settings(SIGNAL_MODE="intraday", DRY_RUN="true"), FakeBroker(), data, engine, now=NOW)
    assert unconfirmed["mode"] == CRASH
    confirmed = run_tick(crash_settings(SIGNAL_MODE="intraday", DRY_RUN="true", CRASH_CONFIRM_DAYS="5"),
                         FakeBroker(), data, engine, now=NOW)
    assert confirmed["mode"] == NORMAL
