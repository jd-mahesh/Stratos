"""Daily price data providers.

Every provider returns ``{symbol: DataFrame}`` where each frame has a
tz-naive DatetimeIndex of trading dates (oldest first) and lowercase
``open, high, low, close, volume`` columns, adjusted for splits and dividends.

Providers:
    alpaca     Alpaca market data (same API keys as trading).
    yfinance   Yahoo Finance via the yfinance package; no key needed, long history.
    synthetic  Deterministic random-walk prices for offline demos and tests.
               These are NOT real market prices.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Settings

COLUMNS = ["open", "high", "low", "close", "volume"]
NY = "America/New_York"

PriceData = Dict[str, pd.DataFrame]


def _to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price data missing columns: {missing}")
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "close"])


class AlpacaData:
    def __init__(self, key_id: str, secret_key: str, feed: str = "iex"):
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(key_id, secret_key)
        self._feed = feed

    def _feed_enum(self):
        from alpaca.data.enums import DataFeed

        return DataFeed(self._feed)

    def daily_bars(self, symbols: List[str], start, end=None) -> PriceData:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start_dt = datetime.combine(_to_date(start), datetime.min.time())
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=datetime.combine(_to_date(end), datetime.max.time()) if end else None,
            adjustment=Adjustment.ALL,
            feed=self._feed_enum(),
        )
        frame = self._client.get_stock_bars(request).df
        out: PriceData = {}
        if frame.empty:
            return out
        for symbol in symbols:
            if symbol not in frame.index.get_level_values(0):
                continue
            df = frame.xs(symbol, level=0).copy()
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                idx = idx.tz_convert(NY).tz_localize(None)
            df.index = idx.normalize()
            out[symbol] = _clean(df)
        return out

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest

        request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=self._feed_enum())
        trades = self._client.get_stock_latest_trade(request)
        return {sym: float(trade.price) for sym, trade in trades.items()}


class YFinanceData:
    def daily_bars(self, symbols: List[str], start, end=None) -> PriceData:
        import yfinance as yf

        end_excl = (_to_date(end) + timedelta(days=1)) if end else None
        # Yahoo writes share classes with a dash (BRK-B); Alpaca and most brokers use a dot (BRK.B).
        yahoo = {s: s.replace(".", "-") for s in symbols}
        raw = yf.download(
            list(yahoo.values()),
            start=_to_date(start).isoformat(),
            end=end_excl.isoformat() if end_excl else None,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=False,
        )
        out: PriceData = {}
        for symbol in symbols:
            try:
                df = raw[yahoo[symbol]] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df.copy()
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df = _clean(df.dropna(how="all"))
            if not df.empty:
                out[symbol] = df
        return out

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        bars = self.daily_bars(symbols, date.today() - timedelta(days=7))
        return {s: float(df["close"].iloc[-1]) for s, df in bars.items() if not df.empty}


class SyntheticData:
    """Seeded random walk with slowly switching trends, so crossovers happen.

    Useful for running the whole pipeline without API keys or network access.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed

    def daily_bars(self, symbols: List[str], start, end=None) -> PriceData:
        dates = pd.bdate_range(_to_date(start), _to_date(end or date.today()))
        out: PriceData = {}
        for i, symbol in enumerate(symbols):
            rng = np.random.default_rng(self.seed + sum(map(ord, symbol)) + i)
            n = len(dates)
            regime_len = 60
            drifts = rng.normal(0.0004, 0.0012, size=n // regime_len + 1).repeat(regime_len)[:n]
            rets = drifts + rng.normal(0, 0.011, size=n)
            close = 100 * np.exp(np.cumsum(rets))
            gap = rng.normal(0, 0.003, size=n)
            open_ = close * np.exp(-rets + gap)  # open near the previous close
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=n)))
            volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
            out[symbol] = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            )
        return out

    def latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        bars = self.daily_bars(symbols, date.today() - timedelta(days=400))
        return {s: float(df["close"].iloc[-1]) for s, df in bars.items()}


def make_provider(settings: Settings, name: Optional[str] = None):
    name = (name or settings.data_provider).lower()
    if name == "alpaca":
        settings.require_alpaca()
        return AlpacaData(settings.alpaca_key_id, settings.alpaca_secret_key, settings.alpaca_data_feed)
    if name == "yfinance":
        return YFinanceData()
    if name == "synthetic":
        return SyntheticData()
    raise ValueError(f"unknown data provider {name!r} (expected alpaca, yfinance or synthetic)")
