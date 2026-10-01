"""Check every symbol in SYMBOLS against Alpaca before the bot trades it.

    python scripts/check_symbols.py
    python scripts/check_symbols.py --symbols AAPL,NVDA,XYZ

For each symbol it reports whether Alpaca knows it, whether it can be traded and
bought in fractional shares, and how far back its daily price history goes on
your data feed (a stock that listed recently can't be backtested before that).
At the end it prints a SYMBOLS= line containing only the tradable ones.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make trader_core importable

from trader_core.config import Settings  # noqa: E402
from trader_core.data import AlpacaData  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", help="comma-separated tickers (default: SYMBOLS from .env)")
    parser.add_argument("--since", default="2016-01-01", help="how far back to look for price history")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.require_alpaca()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else settings.symbols

    from alpaca.trading.client import TradingClient

    trading = TradingClient(settings.alpaca_key_id, settings.alpaca_secret_key, paper=True)
    data = AlpacaData(settings.alpaca_key_id, settings.alpaca_secret_key, settings.alpaca_data_feed)

    print(f"Checking {len(symbols)} symbols on the {settings.alpaca_data_feed!r} data feed...\n")
    history = data.daily_bars(symbols, date.fromisoformat(args.since), datetime.now().date())

    ok, rows = [], []
    for s in symbols:
        try:
            asset = trading.get_asset(s)
        except Exception:
            rows.append((s, "NOT FOUND", "", "", "", "Alpaca doesn't list this symbol"))
            continue
        tradable = bool(asset.tradable) and str(getattr(asset.status, "value", asset.status)) == "active"
        bars = history.get(s)
        first = bars.index[0].date().isoformat() if bars is not None and not bars.empty else "none"
        n = len(bars) if bars is not None else 0
        note = ""
        if not tradable:
            note = "not tradable"
        elif n < settings.slow_window:
            note = f"only {n} days of prices; needs {settings.slow_window} before the strategy can act"
        if tradable:
            ok.append(s)
        rows.append((s, "yes" if tradable else "NO", "yes" if asset.fractionable else "no",
                     first, str(n), note or (asset.name or "")[:40]))

    header = ("Symbol", "Tradable", "Fractional", "History from", "Days", "Notes")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    print(f"\n{len(ok)} of {len(symbols)} symbols can be traded. For your .env:\n")
    print("SYMBOLS=" + ",".join(ok))


if __name__ == "__main__":
    main()
