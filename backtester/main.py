"""Run a backtest and store the results.

Command line (from the repo root):
    python -m backtester --provider synthetic
    python -m backtester --symbols SPY,QQQ --start 2021-01-01 --fast 20 --slow 50
    python -m backtester --universe sectors --start 2021-01-01
    python -m backtester --strategy momentum --universe sectors --signal-mode intraday --cash-rate 3

On Lambda, ``handler.handler`` calls ``run`` with the invocation's JSON event,
which accepts the same keys as the command-line flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional

from trader_core import db
from trader_core.config import Settings
from trader_core.data import make_provider
from trader_core.strategy import STRATEGIES, Strategy, make_strategy
from trader_core.universes import UNIVERSES, describe, expand

from .engine import BacktestResult, run_backtest
from .metrics import verdict

log = logging.getLogger("backtester")

DEFAULT_YEARS = 5


def load_prices(settings: Settings, provider: str, symbols: List[str], strategy: Strategy, start_d: date,
                end_d: date):
    """Daily bars for the symbols and for anything the strategy watches or may buy in crash mode.

    Returns (prices, extra_prices, skipped). Fetches extra history before ``start_d``
    so every average is warmed up on day one.
    """
    warmup_days = math.ceil(strategy.warmup_bars * 1.6) + 10
    source = make_provider(settings, provider)
    log.info("fetching %s bars for %s from %s", provider, symbols, start_d)
    extras = [s for s in strategy.extra_symbols if s not in symbols]
    fetched = source.daily_bars(symbols + extras, start_d - timedelta(days=warmup_days), end_d)
    extra_prices = {s: fetched[s] for s in strategy.extra_symbols if s in fetched and not fetched[s].empty}
    detector = getattr(strategy, "detector", None)
    if detector is not None and detector.index not in extra_prices:
        raise RuntimeError(f"no price data for the crash-mode index {detector.index}")
    missing_extra = sorted(set(strategy.extra_symbols) - set(extra_prices))
    if missing_extra:
        log.warning("no price data for %s; crash mode will do without them", ", ".join(missing_extra))
    prices = {s: df for s, df in fetched.items() if s in symbols and not df.empty}
    skipped = sorted(set(symbols) - set(prices))
    if skipped:
        # A typo or an unsupported ticker shouldn't sink the whole run.
        log.warning("no price data for %s; leaving them out", ", ".join(skipped))
    if not prices:
        raise RuntimeError(f"no price data returned for any of {symbols}")
    return prices, extra_prices, skipped


def _parse_date(value) -> Optional[date]:
    if value in (None, ""):
        return None
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def save_result(engine, result: BacktestResult, strategy: Strategy, provider: str, universe: str = "",
                signal_mode: str = "close", cash_rate: float = 0.0) -> int:
    m = result.metrics
    with engine.begin() as conn:
        run_id = conn.execute(
            db.backtest_runs.insert().values(
                created_at=db.utcnow(),
                strategy=strategy.name,
                params=strategy.params(),
                symbols=",".join(result.symbols),
                data_provider=provider,
                start_date=result.start,
                end_date=result.end,
                initial_capital=result.initial_capital,
                final_equity=m["final_equity"],
                total_return=m["total_return"],
                cagr=m["cagr"],
                max_drawdown=m["max_drawdown"],
                sharpe=m["sharpe"],
                win_rate=m["win_rate"],
                num_trades=m["num_trades"],
                benchmark_return=m["benchmark_return"],
                universe=universe,
                avg_exposure=m["avg_exposure"],
                benchmark_cagr=m["benchmark_cagr"],
                benchmark_max_drawdown=m["benchmark_max_drawdown"],
                benchmark_sharpe=m["benchmark_sharpe"],
                matched_return=m["matched_return"],
                matched_cagr=m["matched_cagr"],
                matched_max_drawdown=m["matched_max_drawdown"],
                signal_mode=signal_mode,
                cash_rate=cash_rate,
            )
        ).inserted_primary_key[0]

        conn.execute(
            db.backtest_equity.insert(),
            [
                {
                    "run_id": run_id,
                    "ts": ts.date(),
                    "equity": float(row.equity),
                    "benchmark_equity": float(row.benchmark_equity),
                    "matched_equity": float(row.matched_equity),
                    "exposure": float(row.exposure),
                    "mode": row["mode"] if isinstance(row["mode"], str) else None,  # row.mode is a pandas method
                }
                for ts, row in result.equity.iterrows()
            ],
        )
        if result.trades:
            conn.execute(
                db.backtest_trades.insert(),
                [
                    {
                        "run_id": run_id,
                        "symbol": t.symbol,
                        "entry_ts": t.entry_ts,
                        "entry_price": t.entry_price,
                        "exit_ts": t.exit_ts,
                        "exit_price": t.exit_price,
                        "qty": t.qty,
                        "pnl": t.pnl,
                        "return_pct": t.return_pct,
                    }
                    for t in result.trades
                ],
            )
    return int(run_id)


def run(
    settings: Settings,
    symbols: Optional[List[str]] = None,
    start=None,
    end=None,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    capital: Optional[float] = None,
    provider: Optional[str] = None,
    slippage_bps: float = 5.0,
    save: bool = True,
    fractional: Optional[bool] = None,
    strategy: Optional[str] = None,
    window: Optional[int] = None,
    lookback: Optional[int] = None,
    top: Optional[int] = None,
    signal_mode: Optional[str] = None,
    cash_rate: Optional[float] = None,
    crash_switch: Optional[bool] = None,
    crash_confirm: Optional[int] = None,
    every: Optional[int] = None,
) -> Dict:
    universe = describe(symbols) if symbols else settings.universe
    symbols = expand(symbols) if symbols else settings.symbols
    strategy = make_strategy(settings, strategy, fast=fast, slow=slow, window=window, lookback=lookback, top=top,
                             crash_switch=crash_switch, crash_confirm=crash_confirm, every=every)
    if every and strategy.rebalance != "every":
        raise ValueError(f"--every only applies to momentum, not {strategy.name}")
    signal_mode = (signal_mode or settings.signal_mode).lower()
    cash_rate = settings.cash_rate if cash_rate is None else cash_rate
    provider_name = (provider or settings.data_provider).lower()
    end_d = _parse_date(end) or date.today()
    start_d = _parse_date(start) or end_d - timedelta(days=365 * DEFAULT_YEARS)

    prices, extra_prices, skipped = load_prices(settings, provider_name, symbols, strategy, start_d, end_d)
    fractional = settings.fractional_shares if fractional is None else fractional

    result = run_backtest(prices, strategy, settings.initial_capital if capital is None else capital,
                          start=start_d, slippage_bps=slippage_bps, fractional=fractional,
                          signal_mode=signal_mode, cash_rate=cash_rate, extra_prices=extra_prices)

    summary = {
        "strategy": strategy.name,
        "params": strategy.params(),
        "strategy_description": strategy.describe(),
        "signal_mode": signal_mode,
        "cash_rate": cash_rate,
        "universe": universe,
        "symbols": result.symbols,
        "skipped": skipped,
        "fractional": fractional,
        "data_provider": provider_name,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in result.metrics.items()},
        "crash_switch": strategy.params().get("crash"),
        "verdict": verdict(result.metrics),
        "yearly": result.yearly(),
    }
    if save:
        engine = db.get_engine(settings.database_url)
        db.init_db(engine)
        summary["run_id"] = save_result(engine, result, strategy, provider_name, universe, signal_mode, cash_rate)
    return summary


def _pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:+.{digits}f}%"


def _num(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def print_summary(s: Dict) -> None:
    years = (date.fromisoformat(s["end"]) - date.fromisoformat(s["start"])).days / 365.25
    syms = s["symbols"]
    universe = s.get("universe", "")
    about = UNIVERSES.get(universe, {}).get("description", "")
    if universe.startswith("custom"):
        about = "your own picks: if chosen recently, results include hindsight (see README)"
    print(f"  Period     {s['start']} to {s['end']} ({years:.1f} years)")
    print(f"  Symbols    {len(syms)} ({universe}{': ' + about if about else ''})")
    if len(syms) <= 12:
        print(f"             {', '.join(syms)}")
    if s.get("skipped"):
        print(f"  Skipped    no price data for {', '.join(s['skipped'])}")
    shares = "fractional" if s.get("fractional") else "whole"
    print(f"  Strategy   {s['strategy']}: {s.get('strategy_description', s['params'])}")
    mode = ("live price: decides with the current price (checked at the open and close)"
            if s.get("signal_mode") == "intraday" else "close: decides from the previous day's close")
    print(f"  Signals    {mode}")
    print(f"  Details    {shares} shares, cash earns {s.get('cash_rate', 0) * 100:.1f}% a year")
    if s.get("crash_switch"):
        days = s.get("crash_days")
        share = f", in crash mode {days * 100:.0f}% of days" if days is not None else ""
        print(f"  Crash mode {s['crash_switch']}{share}")
    print()

    table = [
        ("", "Strategy", "Buy & hold", "Same exposure*"),
        ("Total return", _pct(s["total_return"]), _pct(s["benchmark_return"]), _pct(s.get("matched_return"))),
        ("Per year", _pct(s["cagr"]), _pct(s.get("benchmark_cagr")), _pct(s.get("matched_cagr"))),
        ("Max drawdown", _pct(s["max_drawdown"]), _pct(s["benchmark_max_drawdown"]), _pct(s.get("matched_max_drawdown"))),
        ("Sharpe", _num(s["sharpe"]), _num(s.get("benchmark_sharpe")), _num(s.get("benchmark_sharpe"))),
    ]
    widths = [max(len(r[i]) for r in table) for i in range(4)]
    for r in table:
        print("  " + r[0].ljust(widths[0]) + "".join("   " + c.rjust(w) for c, w in zip(r[1:], widths[1:])))
    exposure = s.get("avg_exposure")
    if exposure is not None:
        print(f"  * buy & hold scaled to the strategy's average {exposure * 100:.0f}% invested, the rest in cash")
    if s.get("verdict"):
        print(f"\n  {s['verdict']}")

    if s.get("yearly"):
        print()
        print("  " + "By year".ljust(widths[0]) + "".join("   " + h.rjust(w) for h, w in zip(table[0][1:], widths[1:])))
        for y in s["yearly"]:
            cells = (_pct(y["strategy"]), _pct(y["buy_hold"]), _pct(y["same_exposure"]))
            print("  " + str(y["year"]).ljust(widths[0]) + "".join("   " + c.rjust(w) for c, w in zip(cells, widths[1:])))

    print()
    win = "n/a" if s["win_rate"] is None else f"{s['win_rate'] * 100:.0f}%"
    print(f"  Final equity ${s['final_equity']:,.2f} · {s['num_trades']} trades · win rate {win}")
    if "run_id" in s:
        print(f"  Saved as run {s['run_id']}")


def _on_off(value: Optional[str]) -> Optional[bool]:
    return None if value is None else value == "on"


def _int_list(value: Optional[str]) -> Optional[List[int]]:
    """'2,5,10' -> [2, 5, 10]; None stays None."""
    if value is None:
        return None
    try:
        out = [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise ValueError(f"expected comma-separated whole numbers, got {value!r}") from None
    if not out:
        raise ValueError(f"expected at least one number, got {value!r}")
    return list(dict.fromkeys(out))


def _settings(args) -> Settings:
    """Settings from the environment, with --crash-confirm applied."""
    settings = Settings.from_env()
    if args.crash_confirm is not None:
        settings.crash_confirm_days = args.crash_confirm
    return settings


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest the moving-average crossover strategy.")
    parser.add_argument("--symbols", help="comma-separated tickers or @universes (default: SYMBOLS env var)")
    parser.add_argument("--universe", choices=sorted(UNIVERSES),
                        help="a preset list chosen without hindsight (see trader_core/universes.py)")
    parser.add_argument("--start", help="first trading date, YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end", help="last date, YYYY-MM-DD (default: today)")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="default: STRATEGY setting (ma_crossover)")
    parser.add_argument("--fast", type=int, help="ma_crossover: fast moving-average window in days")
    parser.add_argument("--slow", type=int, help="ma_crossover: slow moving-average window in days")
    parser.add_argument("--window", type=int, help="trend: moving-average window in days (default 200)")
    parser.add_argument("--lookback", type=int, help="momentum: return lookback in trading days (default 252)")
    parser.add_argument("--top", type=int, help="momentum: how many symbols to hold (default 3)")
    parser.add_argument("--crash-switch", choices=["on", "off"],
                        help="switch to the crash-mode backup while the market is crash-like (default: CRASH_SWITCH)")
    parser.add_argument("--every", metavar="DAYS",
                        help="momentum: rebalance every DAYS trading days instead of monthly (0 = monthly). "
                             "In a sweep, a comma-separated list to compare, e.g. 2,5,10,0")
    parser.add_argument("--lookbacks", metavar="DAYS",
                        help="sweep: comma-separated lookbacks in trading days (default 63,126,252)")
    parser.add_argument("--tops", metavar="N",
                        help="sweep: comma-separated numbers of symbols to hold (default 1,2,3,5)")
    parser.add_argument("--crash-confirm", type=int, metavar="DAYS",
                        help="crash mode switches only after the rule holds this many daily closes in a row "
                             "(default: CRASH_CONFIRM_DAYS or 1 = switch at once)")
    parser.add_argument("--crash-report", action="store_true",
                        help="run with and without crash mode and compare them in every market crash in the data")
    parser.add_argument("--crash-threshold", type=float, default=15.0,
                        help="crash report: index fall (percent) that counts as a crash (default 15)")
    parser.add_argument("--normal-only", action="store_true",
                        help="sweep: score settings on normal-market days only (crash days excluded)")
    parser.add_argument("--signal-mode", choices=["close", "intraday"],
                        help="close = previous day's close; intraday = include the live price (default: SIGNAL_MODE)")
    parser.add_argument("--cash-rate", type=float, help="yearly interest on cash, in percent (default: CASH_RATE or 0)")
    parser.add_argument("--capital", type=float, help="starting capital")
    parser.add_argument("--provider", choices=["alpaca", "yfinance", "synthetic"], help="price data source")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="cost per fill in basis points")
    parser.add_argument("--whole-shares", action="store_true", help="only buy whole shares (default: FRACTIONAL_SHARES setting)")
    parser.add_argument("--no-save", action="store_true", help="print results without writing to the database")
    parser.add_argument("--sweep", action="store_true",
                        help="momentum only: test a grid of lookbacks and top-N and check both halves of the period")
    parser.add_argument("--split", help="sweep: date that splits the period in two (default: the midpoint)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    if args.crash_confirm is not None and args.crash_confirm < 1:
        parser.error("--crash-confirm must be at least 1")
    try:
        everys = _int_list(args.every)
        lookbacks = _int_list(args.lookbacks)
        tops = _int_list(args.tops)
    except ValueError as exc:
        parser.error(str(exc))
    if everys and any(e < 0 for e in everys):
        parser.error("--every must be 0 (monthly) or a positive number of trading days")
    if (lookbacks and any(x < 2 for x in lookbacks)) or (tops and any(x < 1 for x in tops)):
        parser.error("--lookbacks must be at least 2 and --tops at least 1")
    if not args.sweep and (lookbacks or tops):
        parser.error("--lookbacks and --tops are for --sweep; use --lookback and --top for a single backtest")
    if not args.sweep and everys and len(everys) > 1:
        parser.error("--every takes a single value outside --sweep")
    if args.crash_report and everys:
        parser.error("--every isn't supported with --crash-report yet")
    if everys and any(everys) and (args.strategy or Settings.from_env().strategy) != "momentum":
        parser.error("--every only applies to --strategy momentum")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    symbols = (["@" + args.universe] if args.universe else []) + (args.symbols.split(",") if args.symbols else [])
    if args.crash_report:
        from .crash_report import crash_report, print_crash_report

        report = crash_report(
            _settings(args), symbols=symbols or None, start=args.start, end=args.end, provider=args.provider,
            strategy=args.strategy, fast=args.fast, slow=args.slow, window=args.window, lookback=args.lookback,
            top=args.top, signal_mode=args.signal_mode,
            cash_rate=None if args.cash_rate is None else args.cash_rate / 100,
            threshold=args.crash_threshold / 100, slippage_bps=args.slippage_bps,
        )
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print_crash_report(report)
        return
    if args.sweep:
        from .sweep import print_sweep, sweep

        if (args.strategy or Settings.from_env().strategy) != "momentum":
            parser.error("--sweep only supports --strategy momentum")
        result = sweep(_settings(args), symbols=symbols or None, start=args.start, end=args.end,
                       split=args.split, provider=args.provider, signal_mode=args.signal_mode,
                       normal_only=args.normal_only,
                       everys=everys or (0,),
                       **({"lookbacks": lookbacks} if lookbacks else {}),
                       **({"tops": tops} if tops else {}),
                       cash_rate=None if args.cash_rate is None else args.cash_rate / 100,
                       slippage_bps=args.slippage_bps)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_sweep(result)
        return
    summary = run(
        Settings.from_env(),
        symbols=symbols or None,
        start=args.start,
        end=args.end,
        fast=args.fast,
        slow=args.slow,
        capital=args.capital,
        provider=args.provider,
        slippage_bps=args.slippage_bps,
        save=not args.no_save,
        fractional=False if args.whole_shares else None,
        strategy=args.strategy,
        window=args.window,
        lookback=args.lookback,
        top=args.top,
        signal_mode=args.signal_mode,
        cash_rate=None if args.cash_rate is None else args.cash_rate / 100,
        crash_switch=_on_off(args.crash_switch),
        crash_confirm=args.crash_confirm,
        every=everys[0] if everys else None,
    )
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)
