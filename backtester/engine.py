"""Bar-by-bar backtest engine.

Walks history one trading day at a time and asks the strategy the same
question the live trader asks: "given prices up to now, what should I hold?"
Orders are sized by trader_core/portfolio.py, the same code the live trader
uses, from one pool of cash (never borrowing).

When decisions happen depends on SIGNAL_MODE, mirroring the live trader:

    close     The strategy sees completed daily closes only. It decides at a
              day's close and the orders fill at the next day's open. This is
              what the live trader does in "close" mode: decide from
              yesterday's close, trade on the first run after the open.

    intraday  The strategy also sees the current price. Daily bars don't show
              every 5-minute tick, so the backtest checks twice a day: at the
              open (the open price as "now") and at the close (the close as
              "now"), filling immediately at that price. That's a close
              approximation of a bot that checks every few minutes all day.

Monthly strategies (trend, momentum) only decide on the first trading day of
each month, plus the first day of the test. In close mode that means deciding
on the last close of the month and trading at the next open; in intraday mode
it means deciding and trading at the open of the month's first trading day.

Either way the strategy never uses a price it couldn't have seen at the time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trader_core.data import PriceData
from trader_core.portfolio import DEFAULT_BAND, plan_buys, plan_sells
from trader_core.strategy import Signal, Strategy

from .metrics import compute_metrics, same_exposure, yearly_returns

EPS = 1e-9


@dataclass
class Trade:
    """One holding period for a symbol: from first buy until the position is back to zero."""

    symbol: str
    entry_ts: date
    cost: float = 0.0  # dollars spent buying (including commissions)
    bought: float = 0.0  # shares bought
    proceeds: float = 0.0  # dollars received selling (after commissions)
    sold: float = 0.0  # shares sold
    exit_ts: Optional[date] = None

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None

    @property
    def qty(self) -> float:
        return float(self.bought)

    @property
    def entry_price(self) -> float:
        return float(self.cost / self.bought)

    @property
    def exit_price(self) -> Optional[float]:
        return None if self.is_open or not self.sold else float(self.proceeds / self.sold)

    @property
    def pnl(self) -> Optional[float]:
        return None if self.is_open else float(self.proceeds - self.cost)

    @property
    def return_pct(self) -> Optional[float]:
        return None if self.is_open else float(self.pnl / self.cost)


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # index: date; columns: equity, benchmark_equity, exposure, mode, matched_equity
    trades: List[Trade]
    metrics: Dict[str, Optional[float]]
    initial_capital: float
    symbols: List[str] = field(default_factory=list)

    @property
    def start(self) -> date:
        return self.equity.index[0].date()

    @property
    def end(self) -> date:
        return self.equity.index[-1].date()

    def yearly(self):
        cols = self.equity[["equity", "benchmark_equity", "matched_equity"]]
        return yearly_returns(cols.rename(columns={"equity": "strategy", "benchmark_equity": "buy_hold",
                                                   "matched_equity": "same_exposure"}))


def calendar(prices: PriceData) -> pd.DatetimeIndex:
    """Every trading date on which at least one symbol has a price."""
    if not prices:
        raise ValueError("no price data")
    dates = None
    for df in prices.values():
        dates = df.index if dates is None else dates.union(df.index)
    return pd.DatetimeIndex(dates).sort_values()


def run_backtest(
    prices: PriceData,
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    start: Optional[date] = None,
    slippage_bps: float = 5.0,
    commission: float = 0.0,
    fractional: bool = False,
    signal_mode: str = "close",
    cash_rate: float = 0.0,
    band: float = DEFAULT_BAND,
    extra_prices: Optional[PriceData] = None,
) -> BacktestResult:
    """Simulate ``strategy`` over ``prices`` ({symbol: daily bars}).

    Symbols don't need matching histories: a stock that listed in 2024 simply
    can't be bought until the strategy has enough of its history.
    ``cash_rate`` is the yearly interest earned on uninvested cash (0.03 = 3%).
    ``extra_prices`` holds prices for the strategy's ``extra_symbols``: an index
    it watches, and backup assets it may buy in crash mode (``tradable_extras``).
    The buy-and-hold benchmark and "exposure" only count the symbols in ``prices``.
    """
    if signal_mode not in ("close", "intraday"):
        raise ValueError(f"signal_mode must be 'close' or 'intraday', not {signal_mode!r}")
    prices = {s: df for s, df in prices.items() if not df.empty}
    extra_prices = {s: df for s, df in (extra_prices or {}).items() if df is not None and not df.empty}
    universe = sorted(prices)  # the symbols the strategy picks from
    dates = calendar(prices)
    n = len(dates)
    # Everything that can be traded: the universe plus any backup assets with data.
    traded = universe + [a for a in strategy.tradable_extras if a not in prices and a in extra_prices]

    # Every symbol on the shared calendar. Before its first bar, prices are NaN
    # (not listed yet). After that, a missing day keeps the last close and has
    # no open (it can't be traded that day).
    opens: Dict[str, np.ndarray] = {}
    closes: Dict[str, np.ndarray] = {}
    listed: Dict[str, int] = {}
    for s in traded:
        df = (prices.get(s) if s in prices else extra_prices[s]).reindex(dates)
        valid = df["close"].first_valid_index()
        listed[s] = int(dates.get_loc(valid)) if valid is not None else n
        closes[s] = df["close"].ffill().to_numpy(dtype=float)
        opens[s] = df["open"].to_numpy(dtype=float)

    # Prices the strategy looks at (index, backup assets) on the same calendar.
    extra_closes: Dict[str, np.ndarray] = {}
    extra_opens: Dict[str, np.ndarray] = {}
    for s in strategy.extra_symbols:
        df = extra_prices.get(s, prices.get(s))
        if df is None or df.empty:
            continue
        df = df.reindex(dates.union(df.index)).sort_index()
        closes_all = df["close"].ffill()
        extra_closes[s] = closes_all.reindex(dates).to_numpy(dtype=float)
        extra_opens[s] = df["open"].reindex(dates).fillna(closes_all.shift(1).reindex(dates)).to_numpy(dtype=float)

    # Start once the longest-listed symbol has enough history for the strategy.
    first = min(listed[s] for s in universe) + strategy.required_bars - 1
    if start is not None:
        first = max(first, int(dates.searchsorted(pd.Timestamp(start))))
    if first >= n - 1:
        raise ValueError(
            f"not enough history: {n} bars, strategy needs {strategy.required_bars} for warmup plus at least 2 to trade"
        )

    slip = slippage_bps / 10_000
    daily_rate = (1 + cash_rate) ** (1 / 252) - 1
    cash = float(initial_capital)
    shares: Dict[str, float] = {s: 0.0 for s in traded}
    episode: Dict[str, Optional[Trade]] = {s: None for s in traded}
    trades: List[Trade] = []

    def frac(_symbol: str) -> bool:
        return fractional

    def execute(i: int, signals: Dict[str, Signal], px: Dict[str, float]) -> None:
        """Trade toward ``signals`` at prices ``px`` (NaN = that symbol can't trade right now)."""
        nonlocal cash
        day = dates[i].date()
        tradable = {s: float(p) for s, p in px.items() if np.isfinite(p) and s in signals}
        live = {s: sig for s, sig in signals.items() if s in tradable}
        marks = {s: tradable.get(s, closes[s][max(i - 1, 0)]) for s in traded}
        equity_now = cash + sum(shares[s] * marks[s] for s in traded if shares[s] > EPS)
        held = {s: shares[s] for s in traded if shares[s] > EPS}

        for order in plan_sells(live, held, tradable, equity_now, True, strategy.resize, frac, band):
            s = order.symbol
            fill = tradable[s] * (1 - slip)
            qty = min(order.qty, shares[s])
            cash += qty * fill - commission
            shares[s] -= qty
            trade = episode[s]
            if trade is not None:
                trade.proceeds += qty * fill - commission
                trade.sold += qty
            if shares[s] <= EPS:
                shares[s] = 0.0
                if trade is not None:
                    trade.exit_ts = day
                episode[s] = None

        held = {s: shares[s] for s in traded if shares[s] > EPS}
        buy_px = {s: p * (1 + slip) for s, p in tradable.items()}
        orders, _ = plan_buys(live, held, buy_px, equity_now, cash - commission, True, strategy.resize, frac, band)
        for order in orders:
            s, fill = order.symbol, buy_px[order.symbol]
            cash -= order.qty * fill + commission
            if episode[s] is None:
                episode[s] = Trade(s, day)
                trades.append(episode[s])
            episode[s].cost += order.qty * fill + commission
            episode[s].bought += order.qty
            shares[s] += order.qty

    def extra_history(i: int, at_open: bool) -> Dict[str, np.ndarray]:
        """Watched prices up to now: through day i's close, or through yesterday plus today's open."""
        if at_open:
            return {s: np.append(c[:i], extra_opens[s][i]) for s, c in extra_closes.items()}
        return {s: c[: i + 1] for s, c in extra_closes.items()}

    def history(i: int, now: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Universe closes up to day i; or, with ``now``, up to the day before plus the current price."""
        if now is None:
            return {s: closes[s][: i + 1] for s in universe}
        return {s: np.append(closes[s][:i], now[s]) for s in universe}

    mode: Optional[str] = None  # current market mode, for strategies that have one
    # A detector that confirms over several days looks at finished daily closes only (see regime.py):
    # in intraday mode it is checked once, at the open, on the closes up to yesterday.
    closes_only = bool(getattr(getattr(strategy, "detector", None), "closes_only", False))

    def check_mode(extra: Dict[str, np.ndarray]) -> bool:
        """Update the mode; True if it just changed (not counting the very first reading)."""
        nonlocal mode
        regime = strategy.mode(extra, mode)
        if regime is None:
            return False
        changed = mode is not None and regime.mode != mode
        mode = regime.mode
        return changed

    def decide(hist, extra) -> Dict[str, Signal]:
        return strategy.decide(hist, len(universe), extra, mode)

    def month_starts(i: int) -> bool:
        return i == first or dates[i].month != dates[i - 1].month

    # Rebalance schedule: daily, on the first trading day of each month, or every N trading days.
    every = int(getattr(strategy, "every", 0) or 0) if strategy.rebalance == "every" else 0

    def trades_today(i: int) -> bool:
        """Is day i a scheduled rebalance day (trading at its open)?"""
        if strategy.rebalance == "daily":
            return True
        if every:
            return (i - first) % every == 0
        return month_starts(i)

    # Buy-and-hold benchmark: each symbol's slot is bought at the start, or on listing day if later.
    slot = initial_capital / len(universe)
    bench_from = {s: max(first, listed[s]) for s in universe}

    rows = []
    pending: Optional[Dict[str, Signal]] = None
    for i in range(first, n):
        if cash > 0 and i > first:  # a day's interest on idle cash
            cash *= 1 + daily_rate
        open_px = {s: opens[s][i] for s in traded}
        # a symbol only has a real close today if it traded today
        close_px = {s: closes[s][i] if np.isfinite(opens[s][i]) else np.nan for s in traded}

        if signal_mode == "close":
            if pending is not None:  # decided at yesterday's close, filled at today's open
                execute(i, pending, open_px)
                pending = None
        else:
            # "now" = the opening price
            at_open = extra_history(i, True)
            changed = check_mode(extra_history(i - 1, False) if closes_only else at_open)
            if trades_today(i) or changed:
                execute(i, decide(history(i, open_px), at_open), open_px)
            # "now" = the closing price
            at_close = extra_history(i, False)
            changed = False if closes_only else check_mode(at_close)
            if strategy.rebalance == "daily" or changed:
                execute(i, decide(history(i), at_close), close_px)

        invested = sum(shares[s] * closes[s][i] for s in traded if shares[s] > EPS)
        in_universe = sum(shares[s] * closes[s][i] for s in universe if shares[s] > EPS)
        equity = cash + invested
        bench = sum(
            slot * closes[s][i] / closes[s][bench_from[s]] if i >= bench_from[s] else slot for s in universe
        )

        if signal_mode == "close" and i < n - 1:
            at_close = extra_history(i, False)
            changed = check_mode(at_close)
            # decide at today's close when tomorrow is a rebalance day (and on the first day)
            if i == first or trades_today(i + 1) or changed:
                pending = decide(history(i), at_close)

        rows.append((dates[i], float(equity), float(bench), float(in_universe / equity) if equity > 0 else 0.0, mode))

    equity_df = pd.DataFrame(rows, columns=["ts", "equity", "benchmark_equity", "exposure", "mode"]).set_index("ts")
    metrics = compute_metrics(equity_df["equity"], equity_df["benchmark_equity"], trades,
                              equity_df["exposure"], cash_rate)
    equity_df["matched_equity"] = same_exposure(equity_df["benchmark_equity"], metrics["avg_exposure"], cash_rate)
    if equity_df["mode"].notna().any():
        metrics["crash_days"] = float((equity_df["mode"] == "crash").mean())
    return BacktestResult(equity_df, trades, metrics, initial_capital, universe)
