from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from live_trader.broker import Account, DuplicateOrderError, PositionInfo
from live_trader.trader import order_bucket, run_tick
from trader_core import db
from trader_core.config import Settings

from .conftest import make_bars

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)  # Monday 10:07 ET


class FakeBroker:
    def __init__(self, positions=None, open_orders=(), market_open=True, equity=100_000.0, duplicate=False,
                 cash=None, buying_power=None, fractionable=True, close=None):
        self.positions = {s: PositionInfo(s, q) for s, q in (positions or {}).items()}
        self.open_orders = set(open_orders)
        self.market_open = market_open
        self.equity = equity
        self.cash = equity if cash is None else cash
        self.buying_power = self.cash if buying_power is None else buying_power
        self.duplicate = duplicate
        self.fractionable = fractionable
        self.close = close  # today's market close (None = unknown), for the end-of-day status email
        self.orders = []

    def is_market_open(self):
        return self.market_open

    def get_account(self):
        return Account(self.equity, self.cash, self.buying_power)

    def is_fractionable(self, symbol):
        return self.fractionable

    def next_close(self):
        return self.close

    def get_positions(self):
        return dict(self.positions)

    def open_order_symbols(self):
        return self.open_orders

    def submit_market_order(self, symbol, qty, side, client_order_id):
        if self.duplicate:
            raise DuplicateOrderError("client_order_id must be unique")
        self.orders.append((symbol, qty, side, client_order_id))
        return f"order-{len(self.orders)}"


class FakeData:
    """UP trends upward, DOWN trends downward; history ends the business day before NOW."""

    def __init__(self):
        self.bars = {
            "UP": make_bars(np.linspace(50, 100, 80), start="2024-02-12"),
            "DOWN": make_bars(np.linspace(100, 50, 80), start="2024-02-12"),
        }

    def daily_bars(self, symbols, start, end=None):
        return {s: self.bars[s] for s in symbols if s in self.bars}

    def latest_prices(self, symbols):
        return {s: float(self.bars[s]["close"].iloc[-1]) for s in symbols if s in self.bars}


def settings(**extra):
    # two-stock test universe: 50% per stock, so the safety limits are opened up (see test_safeguards.py)
    env = {"SYMBOLS": "UP,DOWN", "FAST_WINDOW": "5", "SLOW_WINDOW": "20", "MAX_ORDER_PCT": "100", "MAX_POSITION_PCT": "100"}
    env.update(extra)
    return Settings.from_env(env)


def by_symbol(result):
    return {d["symbol"]: d for d in result["decisions"]}


def test_buys_uptrend_and_leaves_downtrend_flat(engine):
    broker = FakeBroker()
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW)
    d = by_symbol(result)
    assert d["UP"]["action"] == "buy"
    assert d["DOWN"]["action"] == "hold"
    [(symbol, qty, side, _)] = broker.orders
    assert (symbol, side) == ("UP", "buy")
    assert qty == int(50_000 // 100)  # half the equity, one sleeve per symbol


def test_already_long_means_hold_so_repeat_runs_do_not_stack(engine):
    broker = FakeBroker(positions={"UP": 500})
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert by_symbol(result)["UP"]["action"] == "hold"
    assert broker.orders == []


def test_sells_when_trend_turns(engine):
    broker = FakeBroker(positions={"DOWN": 300})
    run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert ("DOWN", 300, "sell") in [o[:3] for o in broker.orders]


def test_sells_are_sent_before_buys(engine):
    broker = FakeBroker(positions={"DOWN": 300})
    run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert [o[2] for o in broker.orders] == ["sell", "buy"]


def test_skips_symbol_with_open_order(engine):
    broker = FakeBroker(open_orders={"UP"})
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert by_symbol(result)["UP"]["action"] == "skip"
    assert broker.orders == []


def test_market_closed_does_nothing(engine):
    broker = FakeBroker(market_open=False)
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert result["status"] == "market_closed"
    with engine.connect() as conn:
        assert conn.execute(db.decisions.select()).fetchall() == []


def test_dry_run_records_but_sends_nothing(engine):
    broker = FakeBroker()
    result = run_tick(settings(DRY_RUN="true"), broker, FakeData(), engine, now=NOW)
    assert by_symbol(result)["UP"]["action"] == "buy"
    assert broker.orders == []
    with engine.connect() as conn:
        rows = conn.execute(db.decisions.select()).fetchall()
    assert all(r.dry_run for r in rows)


def test_duplicate_invocation_is_skipped_not_crashed(engine):
    broker = FakeBroker(duplicate=True)
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW, scheduled_time="2024-06-03T14:05:00Z")
    assert by_symbol(result)["UP"]["action"] == "skip"
    assert "duplicate" in by_symbol(result)["UP"]["reason"]


def test_order_id_comes_from_scheduled_time(engine):
    broker = FakeBroker()
    run_tick(settings(), broker, FakeData(), engine, now=NOW, scheduled_time="2024-06-03T14:05:00Z")
    assert broker.orders[0][3] == "tb-UP-buy-20240603T1405"


def test_order_bucket_floors_to_five_minutes():
    assert order_bucket(None, NOW) == "20240603T1405"


def test_everything_is_written_to_the_database(engine):
    run_tick(settings(), FakeBroker(positions={"UP": 10}), FakeData(), engine, now=NOW)
    with engine.connect() as conn:
        assert len(conn.execute(db.decisions.select()).fetchall()) == 2
        assert len(conn.execute(db.equity_history.select()).fetchall()) == 1
        assert [r.symbol for r in conn.execute(db.positions.select()).fetchall()] == ["UP"]


@pytest.mark.parametrize("mode,expected_len", [("close", 80), ("intraday", 81)])
def test_signal_modes(mode, expected_len):
    from live_trader.trader import closes_for_signal

    bars = FakeData().bars["UP"]
    closes = closes_for_signal(bars, NOW.date(), 123.0, mode)
    assert len(closes) == expected_len
    if mode == "intraday":
        assert closes[-1] == 123.0


class PricedData(FakeData):
    """Same trends, but UP ends at a price that doesn't divide the sleeve evenly."""

    def __init__(self):
        super().__init__()
        self.bars["UP"] = make_bars(np.linspace(50, 333, 80), start="2024-02-12")


def test_never_borrows_on_margin(engine):
    # Alpaca shows margin-inflated buying power; only the $10k of real cash may be spent.
    broker = FakeBroker(cash=10_000.0, buying_power=300_000.0)
    run_tick(settings(), broker, FakeData(), engine, now=NOW)
    [(symbol, qty, side, _)] = broker.orders
    assert symbol == "UP" and qty * 100 <= 10_000


def test_skips_when_cash_is_used_up(engine):
    broker = FakeBroker(cash=0.0, buying_power=300_000.0)
    result = run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert by_symbol(result)["UP"]["action"] == "skip"
    assert "not enough cash" in by_symbol(result)["UP"]["reason"]
    assert broker.orders == []


def test_fractional_shares(engine):
    broker = FakeBroker()
    run_tick(settings(), broker, PricedData(), engine, now=NOW)
    qty = broker.orders[0][1]
    assert qty == pytest.approx(50_000 / 333, abs=1e-4) and qty != int(qty)


def test_whole_shares_when_setting_is_off_or_asset_not_fractionable(engine):
    for broker, extra in ((FakeBroker(), {"FRACTIONAL_SHARES": "false"}), (FakeBroker(fractionable=False), {})):
        run_tick(settings(**extra), broker, PricedData(), engine, now=NOW)
        assert broker.orders[0][1] == 150  # floor(50_000 / 333)


def test_sells_fractional_positions(engine):
    broker = FakeBroker(positions={"DOWN": 12.3456})
    run_tick(settings(), broker, FakeData(), engine, now=NOW)
    assert ("DOWN", 12.3456, "sell") in [o[:3] for o in broker.orders]


# --- volatility-scaled momentum in the live bot ----------------------------

class VolData:
    """Two rising stocks with 300 days of history ending the business day before NOW.

    Their daily wiggle is 1% until the last 21 days, then `recent` (e.g. 0.04 = four times as wild).
    Records the earliest date the bot asked for, to check it fetches a year of history.
    """

    def __init__(self, recent=0.01):
        vols = np.concatenate([np.full(279, 0.01), np.full(21, recent)])
        signs = np.where(np.arange(300) % 2 == 0, 1.0, -1.0)
        idx_start = pd.bdate_range(end="2024-05-31", periods=300)[0].date().isoformat()
        self.bars = {
            "UP": make_bars(100 * np.cumprod(1 + 0.003 + signs * vols), start=idx_start),
            "UP2": make_bars(100 * np.cumprod(1 + 0.002 + signs * vols * 1.1), start=idx_start),
        }
        self.asked_from = None

    def daily_bars(self, symbols, start, end=None):
        self.asked_from = start
        return {s: self.bars[s] for s in symbols if s in self.bars}

    def latest_prices(self, symbols):
        return {s: float(self.bars[s]["close"].iloc[-1]) for s in symbols if s in self.bars}


def vol_settings(**extra):
    env = {"SYMBOLS": "UP,UP2", "STRATEGY": "momentum", "MOMENTUM_LOOKBACK": "126", "MOMENTUM_TOP": "2",
           "MOMENTUM_VOL_SCALE": "21", "MAX_ORDER_PCT": "100", "MAX_POSITION_PCT": "100"}
    env.update(extra)
    return Settings.from_env(env)


def test_vol_scaling_fetches_a_year_of_history(engine):
    data = VolData()
    run_tick(vol_settings(DRY_RUN="true"), FakeBroker(), data, engine, now=NOW)
    assert (NOW.date() - data.asked_from).days >= 365  # enough for the 1-year "normal" volatility
    off = VolData()
    run_tick(vol_settings(DRY_RUN="true", MOMENTUM_VOL_SCALE="0"), FakeBroker(), off, engine, now=NOW)
    assert (NOW.date() - off.asked_from).days < (NOW.date() - data.asked_from).days


def test_vol_scaling_invests_less_after_a_volatility_spike(engine):
    calm = run_tick(vol_settings(DRY_RUN="true"), FakeBroker(), VolData(0.01), engine, now=NOW)
    wild = run_tick(vol_settings(DRY_RUN="true"), FakeBroker(), VolData(0.04), engine, now=NOW)

    def spent(result):
        return sum(d["order_qty"] * d["price"] for d in result["decisions"] if d["action"] == "buy")

    assert spent(calm) == pytest.approx(100_000, rel=0.05)
    assert 10_000 < spent(wild) < 60_000  # most of the account stays in cash
    assert all("% invested" in d["reason"] for d in wild["decisions"] if d["action"] == "buy")


def test_turning_vol_scaling_on_rebalances_once_then_waits_for_next_month(engine):
    from live_trader.trader import rebalance_key
    from trader_core.strategy import make_strategy

    off, on = vol_settings(MOMENTUM_VOL_SCALE="0"), vol_settings()
    assert rebalance_key(make_strategy(off)) != rebalance_key(make_strategy(on))
    broker = FakeBroker()
    assert run_tick(off, broker, VolData(0.04), engine, now=NOW)["rebalanced"]
    assert not run_tick(off, broker, VolData(0.04), engine, now=NOW.replace(hour=15))["rebalanced"]
    # switching the setting on counts as a new strategy: one immediate rebalance...
    assert run_tick(on, broker, VolData(0.04), engine, now=NOW.replace(hour=16))["rebalanced"]
    # ...and then it waits for next month like before
    assert not run_tick(on, broker, VolData(0.04), engine, now=NOW.replace(hour=17))["rebalanced"]


# --- trading budget (CAPITAL_RESERVE) ----------------------------------------

def budget_settings(**extra):
    """Momentum top 2 on two rising stocks (50% each), volatility scaling off, $100k reserve."""
    return vol_settings(MOMENTUM_VOL_SCALE="0", CAPITAL_RESERVE="100000", **extra)


def last_price(data, symbol):
    return float(data.bars[symbol]["close"].iloc[-1])


def traded(broker, data, side):
    return sum(qty * last_price(data, s) for s, qty, sd, _ in broker.orders if sd == side)


def test_reserve_trims_positions_down_to_the_budget(engine):
    # $105k account fully invested, $100k reserve: Stratos should keep only its $5k budget invested.
    data = VolData()
    held = {s: 52_500 / last_price(data, s) for s in ("UP", "UP2")}
    broker = FakeBroker(positions=held, equity=105_000.0, cash=0.0)
    result = run_tick(budget_settings(), broker, data, engine, now=NOW)
    assert result["rebalanced"] and result["budget"] == pytest.approx(5_000)
    assert {o[2] for o in broker.orders} == {"sell"}
    assert traded(broker, data, "sell") == pytest.approx(100_000, abs=5)  # $50k off each, $2.5k kept in each
    d = by_symbol(result)
    assert d["UP"]["order_qty"] * last_price(data, "UP") == pytest.approx(50_000, abs=2)


def test_reserve_is_never_spent_on_buys(engine):
    data = VolData()
    broker = FakeBroker(equity=105_000.0)  # all cash
    run_tick(budget_settings(), broker, data, engine, now=NOW)
    spent = traded(broker, data, "buy")
    assert 4_990 <= spent <= 5_000  # the $5k budget, never a cent of the reserve


def test_buys_with_a_reserve_use_only_cash_freed_by_sells(engine):
    # Only the reserve is in cash; the budget is all in UP. Selling half of UP pays for UP2, nothing more.
    data = VolData()
    broker = FakeBroker(positions={"UP": 5_000 / last_price(data, "UP")}, equity=105_000.0, cash=100_000.0)
    run_tick(budget_settings(), broker, data, engine, now=NOW)
    sold, bought = traded(broker, data, "sell"), traded(broker, data, "buy")
    assert sold == pytest.approx(2_500, abs=1)
    assert 0 < bought <= sold + 1e-6


def test_used_up_budget_halts_and_places_nothing(engine):
    broker = FakeBroker(equity=99_000.0)  # below the $100k reserve
    result = run_tick(budget_settings(), broker, VolData(), engine, now=NOW)
    assert result["status"] == "halted" and "budget is used up" in result["note"]
    assert broker.orders == []
    from trader_core import safeguards
    assert "budget" in safeguards.get_halt(engine)["reason"]


def test_setting_a_reserve_rebalances_once_then_waits_for_next_month(engine):
    from live_trader.trader import rebalance_key
    from trader_core.strategy import make_strategy

    off, on = vol_settings(MOMENTUM_VOL_SCALE="0"), budget_settings()
    assert rebalance_key(make_strategy(off)) == rebalance_key(make_strategy(off), 0.0)  # no reserve: key unchanged
    broker = FakeBroker(equity=105_000.0)
    assert run_tick(off, broker, VolData(), engine, now=NOW)["rebalanced"]
    assert not run_tick(off, broker, VolData(), engine, now=NOW.replace(hour=15))["rebalanced"]
    assert run_tick(on, broker, VolData(), engine, now=NOW.replace(hour=16))["rebalanced"]
    assert not run_tick(on, broker, VolData(), engine, now=NOW.replace(hour=17))["rebalanced"]


def test_snapshots_record_the_reserve(engine):
    broker = FakeBroker(equity=105_000.0)
    result = run_tick(budget_settings(DRY_RUN="true"), broker, VolData(), engine, now=NOW)
    assert (result["reserve"], result["budget"]) == (100_000, pytest.approx(5_000))
    with engine.connect() as conn:
        rows = conn.execute(db.equity_history.select()).fetchall()
    assert [r.reserve for r in rows] == [100_000]
    no_reserve = run_tick(vol_settings(DRY_RUN="true"), FakeBroker(), VolData(), engine, now=NOW)
    assert "budget" not in no_reserve
