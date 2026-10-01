import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, text

from backtester.engine import run_backtest
from backtester.main import run
from backtester.metrics import same_exposure, sharpe_ratio, verdict, yearly_returns
from trader_core import db
from trader_core.config import Settings
from trader_core.strategy import MACrossover
from trader_core.universes import UNIVERSES, expand

from .conftest import make_bars


def curve(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_same_exposure_scales_daily_returns():
    bench = curve([100, 110, 99])
    half = same_exposure(bench, 0.5)
    assert half.tolist() == pytest.approx([100, 105, 105 * (1 + 0.5 * (99 / 110 - 1))])


def test_scaling_does_not_change_sharpe():
    rng = np.random.default_rng(0)
    bench = curve(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)))
    assert sharpe_ratio(same_exposure(bench, 0.4)) == pytest.approx(sharpe_ratio(bench), rel=1e-9)


def test_yearly_returns_chain_across_years():
    idx = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-12-31"])
    rows = yearly_returns(pd.DataFrame({"a": [100.0, 110.0, 121.0, 132.0]}, index=idx))
    assert rows[0] == {"year": 2023, "a": pytest.approx(0.10)}
    assert rows[1] == {"year": 2024, "a": pytest.approx(132 / 110 - 1)}  # measured from 2023's last close


def test_engine_reports_exposure_and_matched_curve():
    bars = make_bars(list(np.linspace(100, 150, 80)) + list(np.linspace(150, 100, 80)))
    result = run_backtest({"AAA": bars}, MACrossover(5, 20), slippage_bps=0)
    exp = result.equity["exposure"]
    assert exp.between(0, 1.0001).all() and 0 < result.metrics["avg_exposure"] < 1
    assert result.equity["matched_equity"].iloc[0] == pytest.approx(result.equity["equity"].iloc[0])
    assert {y["year"] for y in result.yearly()} == {2024}


def test_verdict_wording():
    base = {"sharpe": 1.0, "benchmark_sharpe": 0.8}
    good = verdict({**base, "total_return": 0.5, "matched_return": 0.4,
                    "max_drawdown": -0.1, "matched_max_drawdown": -0.2})
    assert good.startswith("Beat same-exposure") and "better return per unit of risk" in good
    bad = verdict({"sharpe": 0.5, "benchmark_sharpe": 0.9, "total_return": 0.3, "matched_return": 0.4,
                   "max_drawdown": -0.3, "matched_max_drawdown": -0.2})
    assert bad.startswith("Lost to simply holding less") and "worse return per unit of risk" in bad


def test_universes_expand_and_dedupe():
    assert expand(["@etf4", "nvda", "SPY"]) == ["SPY", "QQQ", "GLD", "TLT", "NVDA"]
    with pytest.raises(ValueError):
        expand(["@nope"])
    for spec in UNIVERSES.values():
        assert len(spec["symbols"]) == len(set(spec["symbols"]))


def test_universe_flag_is_recorded(tmp_path):
    url = f"sqlite:///{tmp_path / 'u.db'}"
    s = Settings.from_env({"DATABASE_URL": url, "DATA_PROVIDER": "synthetic"})
    summary = run(s, symbols=["@sectors"], start="2024-01-01", end="2024-12-31")
    assert summary["universe"] == "sectors" and len(summary["symbols"]) == 11
    assert summary["yearly"] and summary["verdict"]
    with db.get_engine(url).connect() as conn:
        row = conn.execute(db.backtest_runs.select()).one()
        assert row.universe == "sectors" and row.matched_return is not None
        assert conn.execute(text("SELECT count(*) FROM backtest_equity WHERE matched_equity IS NULL")).scalar() == 0


def test_old_database_is_upgraded_in_place(tmp_path):
    # the schema as it was before the benchmark columns existed, with a row in it
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = db.get_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, created_at DATETIME NOT NULL, strategy VARCHAR(64) NOT NULL,"
            " params JSON NOT NULL, symbols TEXT NOT NULL, data_provider VARCHAR(32) NOT NULL, start_date DATE NOT NULL,"
            " end_date DATE NOT NULL, initial_capital FLOAT NOT NULL, final_equity FLOAT NOT NULL, total_return FLOAT NOT NULL,"
            " cagr FLOAT, max_drawdown FLOAT NOT NULL, sharpe FLOAT, win_rate FLOAT, num_trades INTEGER NOT NULL,"
            " benchmark_return FLOAT NOT NULL)"))
        conn.execute(text(
            "CREATE TABLE backtest_equity (run_id INTEGER NOT NULL, ts DATE NOT NULL, equity FLOAT NOT NULL,"
            " benchmark_equity FLOAT NOT NULL, PRIMARY KEY (run_id, ts))"))
        conn.execute(text(
            "INSERT INTO backtest_runs VALUES (1, '2026-09-30', 'ma_crossover', '{}', 'SPY', 'alpaca', '2021-01-04',"
            " '2026-09-30', 100000, 136876, 0.3688, 0.0563, -0.184, 0.67, 0.49, 58, 0.8573)"))

    s = Settings.from_env({"DATABASE_URL": url, "DATA_PROVIDER": "synthetic", "SYMBOLS": "AAA"})
    summary = run(s, start="2024-01-01", end="2024-06-30")

    cols = {c["name"] for c in inspect(engine).get_columns("backtest_runs")}
    assert {"universe", "avg_exposure", "matched_return", "benchmark_sharpe"} <= cols
    with engine.connect() as conn:
        old, new = conn.execute(text("SELECT id, matched_return FROM backtest_runs ORDER BY id")).fetchall()
    assert old == (1, None) and new[0] == summary["run_id"] and new[1] is not None
