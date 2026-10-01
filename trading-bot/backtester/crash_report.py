"""Crash report: how does crash mode do in every market crash in the data?

    python -m backtester --crash-report --universe sectors --provider yfinance --start 1999-06-01

It runs the same strategy twice over the whole period, once without crash mode
and once with it, then finds every crash in the crash index (QQQ by default)
automatically: every fall of at least 15% from a peak (``--crash-threshold``).
Nothing is hand-picked, so the list includes whatever crashes the data holds
(2000–02, 2008, 2011, 2015–16, 2018, 2020, 2022 with enough history).

For each crash it shows how much each version lost, every stretch crash mode
was on (it can switch on and off several times in one crash), and how long
each version took to get back to its pre-crash value. It also
lists false alarms: times crash mode switched on when no crash followed, which
cost return in normal markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from trader_core.config import Settings
from trader_core.strategy import make_strategy
from trader_core.universes import describe, expand

from .engine import run_backtest
from .main import DEFAULT_YEARS, _parse_date, load_prices


@dataclass
class Crash:
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovered: Optional[pd.Timestamp]  # None if the index hasn't regained its peak yet
    depth: float  # index fall from peak to trough, e.g. -0.35


def find_crashes(index: pd.Series, threshold: float = 0.15) -> List[Crash]:
    """Every fall of at least ``threshold`` from a closing high, until the index regains that high."""
    index = index.dropna()
    crashes: List[Crash] = []
    peak_date, peak = index.index[0], float(index.iloc[0])
    trough_date, trough = peak_date, peak
    in_crash = False
    for ts, price in index.items():
        price = float(price)
        if price >= peak:
            if in_crash:
                crashes.append(Crash(peak_date, trough_date, ts, trough / peak - 1))
                in_crash = False
            peak_date, peak = ts, price
            trough_date, trough = ts, price
            continue
        if price < trough:
            trough_date, trough = ts, price
        if not in_crash and price / peak - 1 <= -threshold:
            in_crash = True
    if in_crash:
        crashes.append(Crash(peak_date, trough_date, None, trough / peak - 1))
    return crashes


def _worst(curve: pd.Series, start, end) -> float:
    """Deepest fall of ``curve`` below its value at ``start``, within [start, end]."""
    window = curve[(curve.index >= start) & (curve.index <= end)]
    return float(window.min() / window.iloc[0] - 1)


def _days_to_recover(curve: pd.Series, start, end) -> Optional[int]:
    """Calendar days from ``start`` until ``curve`` is back to its value at ``start``
    after its low point in [start, end]; 0 if it never fell below it; None if not yet."""
    base = float(curve[curve.index >= start].iloc[0])
    window = curve[(curve.index >= start) & (curve.index <= end)]
    if window.min() >= base:
        return 0
    low_date = window.idxmin()
    back = curve[(curve.index > low_date) & (curve >= base)]
    return None if back.empty else int((back.index[0] - start).days)


def _runs(modes: pd.Series, value: str):
    """(start, end) of each stretch where ``modes`` equals ``value``."""
    runs, begin, prev_ts = [], None, None
    for ts, m in modes.items():
        if m == value and begin is None:
            begin = ts
        elif m != value and begin is not None:
            runs.append((begin, prev_ts))
            begin = None
        prev_ts = ts
    if begin is not None:
        runs.append((begin, prev_ts))
    return runs


def crash_report(
    settings: Settings,
    symbols: Optional[List[str]] = None,
    start=None,
    end=None,
    provider: Optional[str] = None,
    strategy: Optional[str] = None,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    window: Optional[int] = None,
    lookback: Optional[int] = None,
    top: Optional[int] = None,
    signal_mode: Optional[str] = None,
    cash_rate: Optional[float] = None,
    threshold: float = 0.15,
    slippage_bps: float = 5.0,
) -> Dict:
    universe = describe(symbols) if symbols else settings.universe
    symbols = expand(symbols) if symbols else settings.symbols
    params = dict(fast=fast, slow=slow, window=window, lookback=lookback, top=top)
    with_switch = make_strategy(settings, strategy, crash_switch=True, **params)
    without = make_strategy(settings, strategy, crash_switch=False, **params)
    signal_mode = (signal_mode or settings.signal_mode).lower()
    cash_rate = settings.cash_rate if cash_rate is None else cash_rate
    provider_name = (provider or settings.data_provider).lower()
    end_d = _parse_date(end) or date.today()
    start_d = _parse_date(start) or end_d - timedelta(days=365 * DEFAULT_YEARS)

    prices, extra_prices, skipped = load_prices(settings, provider_name, symbols, with_switch, start_d, end_d)
    common = dict(start=start_d, slippage_bps=slippage_bps, fractional=settings.fractional_shares,
                  signal_mode=signal_mode, cash_rate=cash_rate, extra_prices=extra_prices)
    base = run_backtest(prices, without, settings.initial_capital, **common)
    switch = run_backtest(prices, with_switch, settings.initial_capital, **common)

    detector = with_switch.detector
    index = extra_prices.get(detector.index, prices.get(detector.index))["close"]
    index = index[(index.index >= base.equity.index[0]) & (index.index <= base.equity.index[-1])]
    eq_base, eq_switch = base.equity["equity"], switch.equity["equity"]
    modes = switch.equity["mode"]

    episodes = []
    for c in find_crashes(index, threshold):
        until = c.recovered or index.index[-1]
        in_window = modes[(modes.index >= c.peak) & (modes.index <= until)]
        # every stretch in crash mode during this crash, not just the first
        stretches = [{"on": b.date().isoformat(), "off": e.date().isoformat(),
                      "days": int(((in_window.index >= b) & (in_window.index <= e)).sum())}
                     for b, e in _runs(in_window, "crash")]
        on = pd.Timestamp(stretches[0]["on"]) if stretches else None
        episodes.append({
            "peak": c.peak.date().isoformat(),
            "trough": c.trough.date().isoformat(),
            "recovered": c.recovered.date().isoformat() if c.recovered is not None else None,
            "index_fall": c.depth,
            "without_worst": _worst(eq_base, c.peak, until),
            "with_worst": _worst(eq_switch, c.peak, until),
            "switched_on": on.date().isoformat() if on is not None else None,
            "index_fall_at_switch": float(index[on] / index[c.peak] - 1) if on is not None else None,
            "stretches": stretches,
            "crash_days": sum(x["days"] for x in stretches),
            "days": int(len(in_window)),
            "without_recovery_days": _days_to_recover(eq_base, c.peak, until),
            "with_recovery_days": _days_to_recover(eq_switch, c.peak, until),
        })

    windows = [(c.peak, c.recovered or index.index[-1]) for c in find_crashes(index, threshold)]
    false_alarms = []
    for begin, finish in _runs(modes, "crash"):
        if any(begin <= w_end and finish >= w_start for w_start, w_end in windows):
            continue
        end_ts = modes.index[min(modes.index.get_loc(finish) + 1, len(modes) - 1)]
        base_ret = float(eq_base[end_ts] / eq_base[begin] - 1)
        switch_ret = float(eq_switch[end_ts] / eq_switch[begin] - 1)
        false_alarms.append({
            "start": begin.date().isoformat(),
            "end": finish.date().isoformat(),
            "without": base_ret,
            "with": switch_ret,
            "cost": switch_ret - base_ret,
        })

    def summary(result):
        m = result.metrics
        return {k: m.get(k) for k in ("total_return", "cagr", "max_drawdown", "sharpe", "crash_days")}

    return {
        "universe": universe,
        "symbols": base.symbols,
        "skipped": skipped,
        "strategy": without.describe(),
        "detector": detector.describe(),
        "backup": with_switch.backup.describe(),
        "signal_mode": signal_mode,
        "threshold": threshold,
        "start": base.start.isoformat(),
        "end": base.end.isoformat(),
        "without": summary(base),
        "with": summary(switch),
        "crashes": episodes,
        "false_alarms": false_alarms,
    }


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:+.1f}%"


def _table(rows, header):
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  " + "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths))))


def print_crash_report(r: Dict) -> None:
    print(f"  Crash report: {r['universe']} ({len(r['symbols'])} symbols), {r['start']} to {r['end']}")
    print(f"  Normal mode:  {r['strategy']}")
    print(f"  Crash rule:   {r['detector']}")
    print(f"  Crash mode:   {r['backup']}")
    print(f"  A crash here = the index falling {r['threshold'] * 100:.0f}% or more from a high.")
    if r.get("skipped"):
        print(f"  Skipped (no data): {', '.join(r['skipped'])}")
    print()
    w, s = r["without"], r["with"]
    _table([
        ["Per year", _pct(w["cagr"]), _pct(s["cagr"])],
        ["Total return", _pct(w["total_return"]), _pct(s["total_return"])],
        ["Worst drop", _pct(w["max_drawdown"]), _pct(s["max_drawdown"])],
        ["Sharpe", f"{w['sharpe']:.2f}" if w["sharpe"] is not None else "n/a",
         f"{s['sharpe']:.2f}" if s["sharpe"] is not None else "n/a"],
        ["Days in crash mode", "-", _pct(s.get("crash_days")).lstrip("+")],
    ], ["Whole period", "Without crash mode", "With crash mode"])
    print()
    if not r["crashes"]:
        print("  No crashes of that size in this period.")
    else:
        rows = []
        for c in r["crashes"]:
            n = len(c["stretches"])
            switch = "never" if c["switched_on"] is None else (
                f"{c['switched_on']} (index {_pct(c['index_fall_at_switch'])}); "
                f"{n} time{'s' if n != 1 else ''}, {c['crash_days']} of {c['days']} days")
            rec = lambda d: "not yet" if d is None else ("no loss" if d == 0 else f"{d} days")  # noqa: E731
            rows.append([f"{c['peak']} to {c['recovered'] or 'now'}", _pct(c["index_fall"]),
                         _pct(c["without_worst"]), _pct(c["with_worst"]),
                         rec(c["without_recovery_days"]), rec(c["with_recovery_days"]), switch])
        _table(rows, ["Crash (peak to recovery)", "Index", "Loss without", "Loss with",
                      "Recovery without", "Recovery with", "Crash mode: first on; stretches, trading days"])
        print()
        print("  Every stretch in crash mode during each crash (on to last day, trading days):")
        for c in r["crashes"]:
            if not c["stretches"]:
                continue
            parts = [f"{x['on']} to {x['off']} ({x['days']}d)" for x in c["stretches"]]
            print(f"  {c['peak']} crash:")
            for k in range(0, len(parts), 3):
                print("    " + ", ".join(parts[k:k + 3]))
    print()
    alarms = r["false_alarms"]
    if not alarms:
        print("  No false alarms: crash mode only switched on during real crashes.")
    else:
        total = sum(a["cost"] for a in alarms)
        print(f"  False alarms: {len(alarms)} times crash mode switched on with no {r['threshold'] * 100:.0f}% crash "
              f"(combined effect on return {total * 100:+.1f} points):")
        _table([[f"{a['start']} to {a['end']}", _pct(a["without"]), _pct(a["with"]), _pct(a["cost"])] for a in alarms],
               ["Period", "Without", "With", "Effect"])
