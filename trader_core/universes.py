"""Named lists of symbols ("universes") to backtest or trade.

Use one anywhere a symbol list is accepted by prefixing its name with @:

    SYMBOLS=@sectors                         in .env
    python -m backtester --universe sectors  on the command line
    SYMBOLS=@etf4,NVDA,PLTR                  mixed with plain tickers

Why these exist: a watchlist you put together today is full of stocks that
have already done well. Backtesting it over past years gives the strategy (and
buy-and-hold) credit for picks nobody could have made back then. That's called
hindsight or survivorship bias, and it can make almost any strategy look great.
The lists below were chosen with information available at the start of the
test, so they give an honest read on whether the strategy itself works.
"""
from __future__ import annotations

from typing import Dict, List

UNIVERSES: Dict[str, Dict] = {
    "etf4": {
        "description": "The original four: US stocks, Nasdaq-100, gold, long-term Treasuries",
        "symbols": ["SPY", "QQQ", "GLD", "TLT"],
    },
    "sectors": {
        "description": "The 11 S&P 500 sector ETFs: the whole market sliced by industry, no stock picking",
        "symbols": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"],
    },
    "assets": {
        "description": "Broad asset classes: US, international and emerging stocks, bonds, gold, commodities, real estate",
        "symbols": ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
    },
    "tech2020": {
        # Approximately the 25 largest US-listed tech and semiconductor companies at the end of 2020:
        # a tech list you could have chosen before 2021, without knowing which AI names would soar.
        "description": "About the 25 largest US tech and chip companies at the end of 2020, a tech list without hindsight for 2021 onward",
        "symbols": [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "ADBE", "CRM", "NFLX",
            "INTC", "CSCO", "AVGO", "QCOM", "TXN", "ORCL", "AMD", "MU", "AMAT", "LRCX",
            "PYPL", "NOW", "INTU", "IBM", "ADI",
        ],
    },
    "mega2020": {
        # Approximately the 30 largest US-listed companies by market value at the end of 2020,
        # i.e. what a "big stocks" list would have looked like before the test period starts.
        # (Facebook trades as META today.)
        "description": "About the 30 largest US companies at the end of 2020, a stock list without hindsight for 2021 onward",
        "symbols": [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "V", "JNJ", "WMT",
            "JPM", "MA", "PG", "UNH", "NVDA", "DIS", "HD", "PYPL", "BAC", "VZ",
            "CMCSA", "ADBE", "NFLX", "KO", "NKE", "MRK", "INTC", "PFE", "T", "CRM",
        ],
    },
}


def expand(tokens: List[str]) -> List[str]:
    """Replace @name entries with their symbols, keeping order and dropping duplicates."""
    out: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith("@"):
            name = token[1:].lower()
            if name not in UNIVERSES:
                raise ValueError(f"unknown universe {token!r}; choose from: {', '.join('@' + n for n in UNIVERSES)}")
            symbols = UNIVERSES[name]["symbols"]
        else:
            symbols = [token.upper()]
        out.extend(s for s in symbols if s not in out)
    return out


def describe(tokens: List[str]) -> str:
    """Human label for a symbol list, e.g. 'sectors' or 'custom list'."""
    named = [t[1:].lower() for t in tokens if t.strip().startswith("@")]
    if named and len(named) == len([t for t in tokens if t.strip()]):
        return " + ".join(named)
    return "custom list" if not named else "custom list + " + " + ".join(named)
