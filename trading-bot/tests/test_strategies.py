"""Trend and momentum strategies, the shared order planner, and signal timing."""
from datetime import datetime, timezone

import numpy as np
import pytest

from backtester.engine import run_backtest
from live_trader.trader import run_tick
from trader_core import db
from trader_core.config import Settings
from trader_core.portfolio import plan_buys, plan_sells
from trader_core.strategy import MACrossover, Momentum, Signal, Trend, make_strategy

from .conftest import make_bars
from .test_live_trader import FakeBroker, FakeData

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)
ALWAYS = lambda s: True  # noqa: E731
NEVER = lambda s: False  # noqa: E731


# --- strategies -------------------------------------------------------------

def test_trend_is_in_above_the_average_and_out_below():
    up = np.linspace(50, 100, 250)
    sig = Trend(200).decide({"UP": up, "DOWN": up[::-1]})
    assert sig["UP"].weight == pytest.approx(0.5) and sig["DOWN"].weight == 0.0
    assert "above 200-day average" in sig["UP"].reason
    assert Trend(200).decide({"NEW": up[:100]})["NEW"].weight is None


def test_momentum_holds_the_top_n_with_positive_returns():
    hist = {s: np.linspace(100, 100 * (1 + r), 300) for s, r in
            {"A": 0.50, "B": 0.30, "C": 0.10, "D": 0.05, "E": -0.20}.items()}
    sig = Momentum(lookback=252, top=3).decide(hist)
    assert [s for s in "ABCDE" if sig[s].weight] == ["A", "B", "C"]
    assert sig["A"].weight == pytest.approx(1 / 3)
    assert "rank 1 of 5" in sig["A"].reason and "not in the top 3" in sig["D"].reason


def test_momentum_absolute_filter_goes_to_cash():
    hist = {"A": np.linspace(100, 120, 300), "B": np.linspace(100, 90, 300), "C": np.linspace(100, 80, 300)}
    sig = Momentum(lookback=252, top=2).decide(hist)
    assert sig["A"].weight == pytest.approx(0.5)  # one slot filled
    assert sig["B"].weight == 0.0 and "cash instead" in sig["B"].reason  # top 2, but falling
    assert Momentum(lookback=252, top=2, absolute=False).decide(hist)["B"].weight == pytest.approx(0.5)


def test_make_strategy_uses_settings_and_ignores_irrelevant_overrides():
    s = Settings.from_env({"STRATEGY": "momentum", "MOMENTUM_TOP": "5"})
    strat = make_strategy(s, fast=10, window=100)
    assert isinstance(strat, Momentum) and strat.top == 5 and strat.lookback == 252
    assert make_strategy(s, "trend", window=150) == Trend(150)
    with pytest.raises(ValueError):
        make_strategy(s, "nope")


# --- shared order planner ---------------------------------------------------

def test_planner_exits_trims_and_tops_up_only_outside_the_band():
    signals = {"EXIT": Signal(0.0, ""), "BIG": Signal(0.25, ""), "OK": Signal(0.25, ""), "SMALL": Signal(0.25, "")}
    held = {"EXIT": 10, "BIG": 40, "OK": 26, "SMALL": 10}
    prices = {s: 10.0 for s in signals}  # equity 1000 -> each target $250
    sells = plan_sells(signals, held, prices, 1000, rebalance=True, resize=True, fractional=NEVER)
    assert {(o.symbol, o.qty) for o in sells} == {("EXIT", 10), ("BIG", 15)}  # BIG $400 -> $250
    buys, _ = plan_buys(signals, held, prices, 1000, cash=1000, rebalance=True, resize=True, fractional=NEVER)
    assert [(o.symbol, o.qty) for o in buys] == [("SMALL", 15)]  # OK ($260) is inside the 25% band
    assert plan_sells(signals, held, prices, 1000, rebalance=True, resize=False, fractional=NEVER)[0].symbol == "EXIT"


def test_planner_never_spends_more_than_cash_and_buys_new_positions_first():
    signals = {"NEW": Signal(0.5, ""), "TOPUP": Signal(0.5, "")}
    buys, skipped = plan_buys(signals, {"TOPUP": 1}, {"NEW": 10.0, "TOPUP": 10.0}, 1000, cash=300,
                              rebalance=True, resize=True, fractional=ALWAYS)
    assert [(o.symbol, o.qty) for o in buys] == [("NEW", 30.0)]
    assert skipped == {"TOPUP": "not enough cash"}


# --- backtest timing --------------------------------------------------------

def gap_up_bars():
    """Flat at 100 for 30 days, then the next day opens at 130 and holds there."""
    closes = [100.0] * 30 + [130.0] * 10
    bars = make_bars(closes)
    bars.iloc[30, bars.columns.get_loc("open")] = 130.0
    return bars


def test_close_mode_waits_for_the_close_and_fills_next_open():
    bars = gap_up_bars()
    t = run_backtest({"A": bars}, MACrossover(5, 20), slippage_bps=0, signal_mode="close").trades[0]
    assert t.entry_ts == bars.index[31].date()  # saw the jump at day 30's close, bought day 31's open


def test_intraday_mode_reacts_to_the_live_price_the_same_day():
    bars = gap_up_bars()
    t = run_backtest({"A": bars}, MACrossover(5, 20), slippage_bps=0, signal_mode="intraday").trades[0]
    assert t.entry_ts == bars.index[30].date() and t.entry_price == pytest.approx(130.0)


@pytest.mark.parametrize("mode", ["close", "intraday"])
def test_monthly_strategies_only_trade_on_the_first_trading_day_of_a_month(mode):
    rng = np.random.default_rng(3)
    bars = {s: make_bars(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 600)), start="2022-01-03") for s in "ABCDE"}
    result = run_backtest(bars, Momentum(lookback=126, top=2), slippage_bps=0, signal_mode=mode)
    days = [t.entry_ts for t in result.trades] + [t.exit_ts for t in result.trades if t.exit_ts]
    idx = bars["A"].index
    first_days = {d.date() for d in idx.to_series().groupby(idx.to_period("M")).min()}
    # the initial investment happens at the start (intraday) or the next open (close mode)
    start_day = result.start if mode == "intraday" else idx[idx.get_loc(str(result.start)) + 1].date()
    assert days and all(d in first_days or d == start_day for d in days)
    assert len({d for d in days if d not in first_days}) <= 1


def test_cash_earns_interest():
    falling = make_bars(np.linspace(200, 100, 300))  # never bought: the account stays in cash
    result = run_backtest({"A": falling}, Trend(50), slippage_bps=0, cash_rate=0.05)
    days = len(result.equity) - 1
    assert not result.trades
    assert result.equity["equity"].iloc[-1] == pytest.approx(100_000 * 1.05 ** (days / 252), rel=1e-6)


# --- live trader with monthly strategies and live prices --------------------

def monthly(**extra):
    env = {"SYMBOLS": "UP,DOWN", "STRATEGY": "trend", "TREND_WINDOW": "20"}
    env.update(extra)
    return Settings.from_env(env)


def test_monthly_strategy_rebalances_once_per_month(engine):
    broker = FakeBroker()
    first = run_tick(monthly(), broker, FakeData(), engine, now=NOW)
    assert first["rebalanced"] and [o[:3] for o in broker.orders] == [("UP", 500.0, "buy")]

    again = run_tick(monthly(), broker, FakeData(), engine, now=NOW.replace(hour=15))
    assert again["rebalanced"] is False and len(broker.orders) == 1  # same month: nothing new

    later = run_tick(monthly(), broker, FakeData(), engine, now=NOW.replace(month=7, day=1))
    assert later["rebalanced"] is True


def test_dry_run_and_errors_do_not_mark_the_month_done(engine):
    run_tick(monthly(DRY_RUN="true"), FakeBroker(), FakeData(), engine, now=NOW)
    assert run_tick(monthly(), FakeBroker(), FakeData(), engine, now=NOW)["rebalanced"] is True

    class Failing(FakeBroker):
        def submit_market_order(self, *a):
            raise RuntimeError("rejected")

    other = monthly(TREND_WINDOW="21")  # different settings -> its own rebalance record
    run_tick(other, Failing(), FakeData(), engine, now=NOW)
    assert run_tick(other, FakeBroker(), FakeData(), engine, now=NOW)["rebalanced"] is True


def test_intraday_mode_uses_the_live_price(engine):
    # UP closed at 100 yesterday, above its 20-day average (~94). A live price of 80 is below it.
    class Crashing(FakeData):
        def latest_prices(self, symbols):
            return {"UP": 80.0, "DOWN": 50.0}

    broker = FakeBroker(positions={"UP": 100})
    run_tick(monthly(SIGNAL_MODE="intraday"), broker, Crashing(), engine, now=NOW)
    assert ("UP", 100, "sell") in [o[:3] for o in broker.orders]

    broker = FakeBroker(positions={"UP": 100})
    run_tick(monthly(SIGNAL_MODE="close", TREND_WINDOW="22"), broker, Crashing(), engine, now=NOW)
    assert "sell" not in [o[2] for o in broker.orders]  # close mode only sees yesterday's 100


def test_bot_state_round_trip(engine):
    assert db.get_state(engine, "k") is None
    db.set_state(engine, "k", "2024-06")
    db.set_state(engine, "k", "2024-07")
    assert db.get_state(engine, "k") == "2024-07"


# --- momentum sweep ---------------------------------------------------------

def test_sweep_runs_the_grid_and_checks_both_halves():
    from backtester.sweep import sweep

    s = Settings.from_env({"DATA_PROVIDER": "synthetic", "SYMBOLS": "@sectors", "FRACTIONAL_SHARES": "true"})
    out = sweep(s, start="2016-01-01", end="2024-12-31", lookbacks=(63, 126), tops=(1, 3))
    assert [(r["lookback"], r["top"]) for r in out["rows"]] == [(63, 1), (63, 3), (126, 1), (126, 3)]
    for r in out["rows"]:
        assert r["first_half"] is not None and r["second_half"] is not None
        assert r["passes"] == (r["first_half"] > r["first_half_benchmark"]
                               and r["second_half"] > r["second_half_benchmark"])
    assert out["split"] == "2020-07-01"
