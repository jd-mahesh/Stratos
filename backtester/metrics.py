"""Performance statistics, and the benchmarks a strategy has to beat.

Comparing a strategy only against buy-and-hold is misleading: a strategy that
sits in cash half the time will usually make less money *and* have smaller
drawdowns, simply because it holds less stock. So every backtest is also
compared with "same exposure": the buy-and-hold basket scaled down to the
strategy's average fraction invested, with the rest in cash. If the strategy
can't beat that, its timing isn't adding anything; you'd do as well by just
holding less.

Scaling a portfolio by a constant doesn't change its Sharpe ratio, so the
Sharpe column answers the same question from another angle: strategy Sharpe
above buy-and-hold Sharpe means better return per unit of risk.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative fraction (e.g. -0.18)."""
    peaks = equity.cummax()
    return float((equity / peaks - 1).min())


def sharpe_ratio(equity: pd.Series) -> Optional[float]:
    """Annualised Sharpe of daily returns, risk-free rate taken as zero."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return None
    std = rets.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return None
    return float(rets.mean() / std * math.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> Optional[float]:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return None
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def same_exposure(benchmark: pd.Series, exposure: float, cash_rate: float = 0.0) -> pd.Series:
    """Buy-and-hold scaled to a constant ``exposure`` (0–1), the rest in cash earning ``cash_rate`` a year.

    Rebalanced daily: each day's return is ``exposure`` times the basket's return
    plus ``1 - exposure`` times a day's interest.
    """
    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    rets = benchmark.pct_change().fillna(0.0) * exposure
    rets.iloc[1:] += (1 - exposure) * daily_cash
    return benchmark.iloc[0] * (1 + rets).cumprod()


def yearly_returns(curves: pd.DataFrame) -> List[Dict]:
    """Calendar-year return of each column, e.g. [{"year": 2022, "equity": -0.05, ...}]."""
    rows = []
    for year, chunk in curves.groupby(curves.index.year):
        # measure from the previous year's last close, so years chain together
        before = curves[curves.index < chunk.index[0]]
        base = before.iloc[-1] if not before.empty else chunk.iloc[0]
        rows.append({"year": int(year), **{c: float(chunk[c].iloc[-1] / base[c] - 1) for c in curves.columns}})
    return rows


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: Sequence,
    exposure: Optional[pd.Series] = None,
    cash_rate: float = 0.0,
) -> Dict[str, Optional[float]]:
    closed = [t for t in trades if not t.is_open]
    wins = [t for t in closed if (t.pnl or 0) > 0]
    avg_exposure = float(exposure.mean()) if exposure is not None and len(exposure) else None
    matched = same_exposure(benchmark, avg_exposure, cash_rate) if avg_exposure is not None else None
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(equity),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "num_trades": len(trades),
        "avg_exposure": avg_exposure,
        "benchmark_return": total_return(benchmark),
        "benchmark_cagr": cagr(benchmark),
        "benchmark_max_drawdown": max_drawdown(benchmark),
        "benchmark_sharpe": sharpe_ratio(benchmark),
        "matched_return": total_return(matched) if matched is not None else None,
        "matched_cagr": cagr(matched) if matched is not None else None,
        "matched_max_drawdown": max_drawdown(matched) if matched is not None else None,
    }


def verdict(m: Dict) -> str:
    """Plain-English summary of whether the strategy's timing added anything."""
    if m.get("matched_return") is None:
        return ""
    more_return = m["total_return"] > m["matched_return"]
    smaller_dd = m["max_drawdown"] > m["matched_max_drawdown"]  # drawdowns are negative
    if more_return and smaller_dd:
        first = "Beat same-exposure buy & hold on both return and drawdown"
    elif more_return:
        first = "More return than same-exposure buy & hold, but deeper drawdowns"
    elif smaller_dd:
        first = "Less return than same-exposure buy & hold, with smaller drawdowns"
    else:
        first = "Lost to simply holding less stock on both return and drawdown"
    ours, theirs = m.get("sharpe"), m.get("benchmark_sharpe")
    if ours is None or theirs is None:
        return first + "."
    if ours > theirs * 1.05:
        second = f"Sharpe {ours:.2f} vs {theirs:.2f}: better return per unit of risk than buy & hold."
    elif ours < theirs * 0.95:
        second = f"Sharpe {ours:.2f} vs {theirs:.2f}: worse return per unit of risk than buy & hold."
    else:
        second = f"Sharpe {ours:.2f} vs {theirs:.2f}: about the same risk-adjusted return as buy & hold."
    return f"{first}. {second}"
