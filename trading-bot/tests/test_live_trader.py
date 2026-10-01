from datetime import datetime, timezone

import numpy as np
import pytest

from live_trader.broker import Account, DuplicateOrderError, PositionInfo
from live_trader.trader import order_bucket, run_tick
from trader_core import db
from trader_core.config import Settings

from .conftest import make_bars

NOW = datetime(2024, 6, 3, 14, 7, tzinfo=timezone.utc)  # Monday 10:07 ET


class FakeBroker:
    def __init__(self, positions=None, open_orders=(), market_open=True, equity=100_000.0, duplicate=False,
                 cash=None, buying_power=None, fractionable=True):
        self.positions = {s: PositionInfo(s, q) for s, q in (positions or {}).items()}
        self.open_orders = set(open_orders)
        self.market_open = market_open
        self.equity = equity
        self.cash = equity if cash is None else cash
        self.buying_power = self.cash if buying_power is None else buying_power
        self.duplicate = duplicate
        self.fractionable = fractionable
        self.orders = []

    def is_market_open(self):
        return self.market_open

    def get_account(self):
        return Account(self.equity, self.cash, self.buying_power)

    def is_fractionable(self, symbol):
        return self.fractionable

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
    env = {"SYMBOLS": "UP,DOWN", "FAST_WINDOW": "5", "SLOW_WINDOW": "20"}
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
