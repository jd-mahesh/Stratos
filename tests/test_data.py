from types import SimpleNamespace

import pandas as pd

from trader_core.config import Settings, read_dotenv
from trader_core.data import AlpacaData, SyntheticData


def test_alpaca_bars_are_split_per_symbol_on_new_york_dates():
    # Alpaca returns daily bars stamped at midnight New York time, expressed in UTC.
    idx = pd.MultiIndex.from_tuples(
        [
            ("SPY", pd.Timestamp("2024-03-08 05:00", tz="UTC")),  # EST
            ("SPY", pd.Timestamp("2024-03-11 04:00", tz="UTC")),  # EDT after the clocks change
            ("QQQ", pd.Timestamp("2024-03-08 05:00", tz="UTC")),
        ],
        names=["symbol", "timestamp"],
    )
    frame = pd.DataFrame(
        {"open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3], "close": [1.0, 2, 3],
         "volume": [10.0, 20, 30], "trade_count": [1, 2, 3], "vwap": [1.0, 2, 3]},
        index=idx,
    )
    provider = AlpacaData.__new__(AlpacaData)  # skip creating a real API client
    provider._feed = "iex"
    provider._client = SimpleNamespace(get_stock_bars=lambda request: SimpleNamespace(df=frame))

    bars = provider.daily_bars(["SPY", "QQQ"], "2024-03-01", "2024-03-12")
    assert list(bars["SPY"].index) == [pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-11")]
    assert list(bars["SPY"].columns) == ["open", "high", "low", "close", "volume"]
    assert bars["QQQ"]["close"].tolist() == [3.0]


def test_synthetic_data_is_deterministic_and_well_formed():
    a = SyntheticData().daily_bars(["SPY"], "2024-01-01", "2024-06-30")["SPY"]
    b = SyntheticData().daily_bars(["SPY"], "2024-01-01", "2024-06-30")["SPY"]
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()


def test_settings_from_env_and_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("# comment\nSYMBOLS=aapl, msft\nDRY_RUN=true\nFAST_WINDOW='10'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SYMBOLS", raising=False)
    monkeypatch.setenv("FAST_WINDOW", "12")  # real env vars beat the .env file
    assert read_dotenv()["FAST_WINDOW"] == "10"
    s = Settings.from_env()
    assert s.symbols == ["AAPL", "MSFT"]
    assert s.dry_run is True
    assert s.fast_window == 12
