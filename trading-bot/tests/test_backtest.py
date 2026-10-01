import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest
from backtester.main import run
from backtester.metrics import max_drawdown, sharpe_ratio
from trader_core import db
from trader_core.config import Settings
from trader_core.strategy import MACrossover

from .conftest import make_bars


def test_order_fills_at_next_open_not_same_close():
    # 30 flat days, then a steady climb. Opens sit 0.5 above closes so the
    # fill price tells us exactly which bar it came from.
    closes = [100.0] * 30 + list(np.linspace(101, 130, 30))
    bars = make_bars(closes, open_offset=0.5)
    strat = MACrossover(5, 20)
    result = run_backtest({"AAA": bars}, strat, initial_capital=10_000, slippage_bps=0)

    first_trade = result.trades[0]
    # find the first day the strategy says "long", using only data up to that day
    signal_day = next(i for i in range(len(closes)) if strat.evaluate(closes[: i + 1]).target == 1.0)
    fill_day = bars.index[signal_day + 1]
    assert first_trade.entry_ts == fill_day.date()
    assert first_trade.entry_price == pytest.approx(bars.loc[fill_day, "open"])


def test_equity_starts_at_capital_and_benchmark_matches_buy_and_hold():
    bars = make_bars(np.linspace(100, 150, 120))
    result = run_backtest({"AAA": bars}, MACrossover(5, 20), initial_capital=50_000, slippage_bps=0)
    assert result.equity["equity"].iloc[0] == pytest.approx(50_000)
    closes = bars["close"].loc[result.equity.index]
    assert result.metrics["benchmark_return"] == pytest.approx(closes.iloc[-1] / closes.iloc[0] - 1)


def test_round_trip_is_recorded_with_pnl():
    closes = [100.0] * 25 + list(np.linspace(100, 140, 25)) + list(np.linspace(140, 90, 40))
    result = run_backtest({"AAA": make_bars(closes)}, MACrossover(5, 20), slippage_bps=0)
    closed = [t for t in result.trades if not t.is_open]
    assert closed, "expected the downtrend to close the position"
    t = closed[0]
    assert t.pnl == pytest.approx((t.exit_price - t.entry_price) * t.qty)
    # plain Python floats, not numpy: Postgres' driver can't store numpy types
    assert type(t.entry_price) is float and type(t.exit_price) is float


def test_capital_is_split_into_sleeves():
    bars = make_bars(np.linspace(100, 200, 120))
    result = run_backtest({"AAA": bars, "BBB": bars.copy()}, MACrossover(5, 20), initial_capital=10_000)
    for t in result.trades:
        assert t.qty * t.entry_price <= 5_000


def test_too_little_history_raises():
    with pytest.raises(ValueError):
        run_backtest({"AAA": make_bars([100.0] * 10)}, MACrossover(5, 20))


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 65], index=pd.bdate_range("2024-01-01", periods=5), dtype=float)
    assert max_drawdown(eq) == pytest.approx(65 / 130 - 1)


def test_sharpe_undefined_for_flat_equity():
    eq = pd.Series([100.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    assert sharpe_ratio(eq) is None


def test_run_saves_everything(tmp_path):
    url = f"sqlite:///{tmp_path / 'bt.db'}"
    settings = Settings.from_env({"DATABASE_URL": url, "DATA_PROVIDER": "synthetic", "SYMBOLS": "AAA,BBB"})
    summary = run(settings, start="2023-01-01", end="2024-12-31")

    engine = db.get_engine(url)
    with engine.connect() as conn:
        runs = conn.execute(db.backtest_runs.select()).fetchall()
        curve = conn.execute(db.backtest_equity.select()).fetchall()
        trades = conn.execute(db.backtest_trades.select()).fetchall()
    assert len(runs) == 1 and runs[0].id == summary["run_id"]
    assert runs[0].params == {"fast": 20, "slow": 50}
    assert len(curve) > 400
    assert len(trades) == summary["num_trades"]
    # warmup data was fetched before the requested start, so trading starts on time
    assert summary["start"] <= "2023-01-04"


def test_symbols_with_different_histories():
    # OLD has 300 days of history; NEW lists 200 days later. The test must start with OLD,
    # not wait for NEW, and NEW must only trade once it has its own 20-day window.
    old = make_bars(np.linspace(100, 200, 300), start="2023-01-02")
    new = make_bars(np.linspace(10, 40, 100), start=str(old.index[200].date()))
    result = run_backtest({"OLD": old, "NEW": new}, MACrossover(5, 20), slippage_bps=0)
    assert result.start == old.index[19].date()
    new_trades = [t for t in result.trades if t.symbol == "NEW"]
    assert new_trades and new_trades[0].entry_ts > old.index[200 + 19].date()
    assert result.equity["equity"].notna().all()


def test_fractional_backtest_invests_the_whole_sleeve():
    bars = make_bars(np.linspace(900, 1100, 120))
    whole = run_backtest({"AAA": bars}, MACrossover(5, 20), initial_capital=1_500, slippage_bps=0)
    frac = run_backtest({"AAA": bars}, MACrossover(5, 20), initial_capital=1_500, slippage_bps=0, fractional=True)
    assert whole.trades[0].qty == 1
    assert frac.trades[0].qty == pytest.approx(1_500 / frac.trades[0].entry_price, abs=1e-4)


def test_run_skips_symbols_without_data(tmp_path, monkeypatch):
    from trader_core import data as data_mod

    real = data_mod.SyntheticData.daily_bars

    def without_bad(self, symbols, start, end=None):
        return real(self, [s for s in symbols if s != "BAD"], start, end)

    monkeypatch.setattr(data_mod.SyntheticData, "daily_bars", without_bad)
    settings = Settings.from_env({"DATABASE_URL": f"sqlite:///{tmp_path / 'x.db'}",
                                  "DATA_PROVIDER": "synthetic", "SYMBOLS": "AAA,BAD"})
    summary = run(settings, start="2024-01-01", end="2024-12-31", save=False)
    assert summary["symbols"] == ["AAA"] and summary["skipped"] == ["BAD"]
