"""Momentum parameter sweep: try a small, fixed grid of standard settings and
check which ones hold up in *both* halves of history.

    python -m backtester --strategy momentum --sweep --universe sectors --provider yfinance --start 2007-01-01

Why a fixed grid and a split? Trying many settings and keeping the best one is
how backtests lie: with enough tries, something always looks great by luck.
So the grid is short (3, 6 and 12-month lookbacks; hold the top 1, 2, 3 or 5),
and a setting only "passes" if it beats buy & hold in the first half of the
period AND in the second half, which it had no say in choosing.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence

import pandas as pd

from trader_core.config import Settings
from trader_core.data import make_provider
from trader_core.regime import CrashDetector, regime_series
from trader_core.strategy import make_strategy
from trader_core.universes import describe, expand

from .engine import run_backtest

log = logging.getLogger("backtester")

DEFAULT_LOOKBACKS = (63, 126, 252)  # ~3, 6 and 12 months of trading days
DEFAULT_TOPS = (1, 2, 3, 5)


def _cagr(total_return: float, start, end) -> Optional[float]:
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return None if years <= 0 else (1 + total_return) ** (1 / years) - 1


def _normal_cagr(curve: pd.Series, normal: pd.Series, start=None, end=None) -> Optional[float]:
    """Yearly growth counting only days that began in normal mode (crash-mode days are skipped)."""
    rets = curve.pct_change()
    ok = normal.reindex(curve.index).shift(1).eq(True)  # the mode at the previous close (missing = not normal)
    mask = ok & rets.notna()
    if start is not None:
        mask &= curve.index >= pd.Timestamp(start)
    if end is not None:
        mask &= curve.index < pd.Timestamp(end)
    days = int(mask.sum())
    if days < 20:
        return None
    growth = float((1 + rets[mask]).prod())
    return growth ** (252 / days) - 1


def sweep(
    settings: Settings,
    symbols: Optional[List[str]] = None,
    start=None,
    end=None,
    split=None,
    provider: Optional[str] = None,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    tops: Sequence[int] = DEFAULT_TOPS,
    signal_mode: Optional[str] = None,
    cash_rate: Optional[float] = None,
    slippage_bps: float = 5.0,
    normal_only: bool = False,
) -> Dict:
    """Run the grid. With ``normal_only``, every return is measured on normal-market
    days only, as judged by the crash detector (CRASH_INDEX etc.), so crashes don't
    shape the choice of normal-mode settings; crashes are the backup's job."""
    universe = describe(symbols) if symbols else settings.universe
    symbols = expand(symbols) if symbols else settings.symbols
    signal_mode = (signal_mode or settings.signal_mode).lower()
    cash_rate = settings.cash_rate if cash_rate is None else cash_rate
    provider_name = (provider or settings.data_provider).lower()
    end_d = date.fromisoformat(str(end)) if end else date.today()
    start_d = date.fromisoformat(str(start)) if start else end_d - timedelta(days=365 * 10)

    detector = CrashDetector(settings.crash_index, settings.crash_drawdown, settings.crash_window,
                             confirm_days=settings.crash_confirm_days)
    extras = [detector.index] if normal_only and detector.index not in symbols else []

    # One download for the whole grid, with enough warmup for the longest lookback.
    warmup_bars = max(max(lookbacks) + 1, detector.required_bars if normal_only else 0)
    warmup = math.ceil(warmup_bars * 1.6) + 10
    fetched = make_provider(settings, provider_name).daily_bars(symbols + extras, start_d - timedelta(days=warmup), end_d)
    prices = {s: df for s, df in fetched.items() if s in symbols and not df.empty}
    if not prices:
        raise RuntimeError(f"no price data returned for any of {symbols}")
    skipped = sorted(set(symbols) - set(prices))

    normal = None
    normal_share = None
    if normal_only:
        if detector.index not in fetched or fetched[detector.index].empty:
            raise RuntimeError(f"no price data for the crash-mode index {detector.index}")
        modes = regime_series(fetched[detector.index]["close"], detector)
        normal = modes == "normal"

    split_d = date.fromisoformat(str(split)) if split else start_d + (end_d - start_d) / 2
    first_half = {s: df[df.index < pd.Timestamp(split_d)] for s, df in prices.items()}
    first_half = {s: df for s, df in first_half.items() if not df.empty}
    common = dict(slippage_bps=slippage_bps, fractional=settings.fractional_shares,
                  signal_mode=signal_mode, cash_rate=cash_rate)

    rows = []
    for lookback in lookbacks:
        for top in tops:
            strat = make_strategy(settings, "momentum", lookback=lookback, top=top, crash_switch=False)
            full = run_backtest(prices, strat, settings.initial_capital, start=start_d, **common)
            eq, bh = full.equity["equity"], full.equity["benchmark_equity"]
            if normal_only:
                cagr, bench = _normal_cagr(eq, normal), _normal_cagr(bh, normal)
                parts = [(_normal_cagr(eq, normal, end=split_d), _normal_cagr(bh, normal, end=split_d)),
                         (_normal_cagr(eq, normal, start=split_d), _normal_cagr(bh, normal, start=split_d))]
                normal_share = float(normal.reindex(eq.index).eq(True).mean())
            else:
                m = full.metrics
                cagr = _cagr(m["total_return"], full.start, full.end)
                bench = _cagr(m["benchmark_return"], full.start, full.end)
                parts = []
                for data, part_start in ((first_half, start_d), (prices, split_d)):
                    try:
                        r = run_backtest(data, strat, settings.initial_capital, start=part_start, **common)
                    except ValueError:  # not enough history in that half for this lookback
                        parts.append((None, None))
                        continue
                    pm = r.metrics
                    parts.append((_cagr(pm["total_return"], r.start, r.end), _cagr(pm["benchmark_return"], r.start, r.end)))
            m = full.metrics
            rows.append({
                "lookback": lookback,
                "top": top,
                "cagr": cagr,
                "benchmark_cagr": bench,
                "first_half": parts[0][0],
                "first_half_benchmark": parts[0][1],
                "second_half": parts[1][0],
                "second_half_benchmark": parts[1][1],
                "max_drawdown": m["max_drawdown"],
                "sharpe": m["sharpe"],
                "benchmark_sharpe": m["benchmark_sharpe"],
                "trades": m["num_trades"],
                "passes": all(p[0] is not None and p[1] is not None and p[0] > p[1] for p in parts),
                "start": full.start.isoformat(),
                "end": full.end.isoformat(),
            })
    return {
        "universe": universe,
        "symbols": sorted(prices),
        "skipped": skipped,
        "signal_mode": signal_mode,
        "cash_rate": cash_rate,
        "data_provider": provider_name,
        "split": split_d.isoformat(),
        "normal_only": normal_only,
        "normal_share": normal_share,
        "detector": detector.describe(),
        "rows": rows,
    }


def _pct(v) -> str:
    return "  n/a" if v is None else f"{v * 100:+.1f}%"


def print_sweep(s: Dict) -> None:
    rows = s["rows"]
    if not rows:
        print("  nothing to show")
        return
    r0 = rows[0]
    mode = "live price" if s["signal_mode"] == "intraday" else "previous close"
    print(f"  Momentum sweep on {s['universe']} ({len(s['symbols'])} symbols), {r0['start']} to {r0['end']}")
    print(f"  Signals: {mode} · cash earns {s['cash_rate'] * 100:.1f}% · halves split at {s['split']}")
    if s.get("normal_only"):
        share = s.get("normal_share")
        print(f"  Normal markets only: returns count only normal days ({s['detector']})"
              + (f"; {share * 100:.0f}% of days were normal" if share is not None else ""))
    if s.get("skipped"):
        print(f"  Skipped (no data): {', '.join(s['skipped'])}")
    print(f"  Buy & hold: {_pct(r0['benchmark_cagr'])}/yr overall, {_pct(r0['first_half_benchmark'])} first half, "
          f"{_pct(r0['second_half_benchmark'])} second half (per year)")
    print()
    header = ("Lookback", "Top", "Per year", "1st half", "2nd half", "Worst drop", "Sharpe", "Trades", "Beats B&H in both halves")
    table = [header]
    for r in rows:
        table.append((
            f"{round(r['lookback'] / 21)} mo", str(r["top"]), _pct(r["cagr"]), _pct(r["first_half"]),
            _pct(r["second_half"]), _pct(r["max_drawdown"]),
            "n/a" if r["sharpe"] is None else f"{r['sharpe']:.2f}", str(r["trades"]),
            "YES" if r["passes"] else "no",
        ))
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        print("  " + "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))))
    passing = [r for r in rows if r["passes"]]
    print()
    if passing:
        best_sharpe = r0["benchmark_sharpe"]
        print(f"  {len(passing)} of {len(rows)} settings beat buy & hold in both halves "
              f"(buy & hold Sharpe {best_sharpe:.2f}).")
        print("  Prefer a setting whose neighbours also pass: an isolated winner is more likely luck.")
    else:
        print("  No setting beat buy & hold in both halves on this list.")
