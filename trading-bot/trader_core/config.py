"""Settings, read from environment variables (and optionally AWS Secrets Manager).

Locally, values come from a ``.env`` file via docker compose. On AWS, set
``SECRET_ID`` to the name of a Secrets Manager secret holding a JSON object
such as::

    {"ALPACA_API_KEY_ID": "...", "ALPACA_API_SECRET_KEY": "...",
     "DATABASE_URL": "postgresql+psycopg2://..."}

Keys in the secret override plain environment variables, so credentials
never need to be baked into an image or typed into the Lambda console.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


def _bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_SYMBOLS = "@etf4"


def _symbols(value: Optional[str]) -> List[str]:
    """Tickers from a comma-separated string; @name entries expand to a preset list (universes.py)."""
    from .universes import expand

    return expand((value or DEFAULT_SYMBOLS).split(","))


def _universe_label(value: Optional[str]) -> str:
    from .universes import describe

    return describe((value or DEFAULT_SYMBOLS).split(","))


def read_dotenv(path: str = ".env") -> Dict[str, str]:
    """Minimal .env reader (KEY=VALUE lines, # comments) for running outside Docker."""
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_secret(secret_id: str) -> Dict[str, str]:
    """Fetch a JSON secret from AWS Secrets Manager."""
    import boto3  # imported lazily: only needed on AWS

    client = boto3.client("secretsmanager")
    payload = client.get_secret_value(SecretId=secret_id)["SecretString"]
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"secret {secret_id} must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


@dataclass
class Settings:
    database_url: str = "sqlite:///trading.db"
    symbols: List[str] = field(default_factory=lambda: _symbols(None))
    universe: str = "etf4"  # label for where the symbol list came from
    strategy: str = "ma_crossover"  # ma_crossover | trend | momentum (see strategy.py)
    fast_window: int = 20  # ma_crossover
    slow_window: int = 50  # ma_crossover
    trend_window: int = 200  # trend
    momentum_lookback: int = 252  # momentum, in trading days (~12 months)
    momentum_top: int = 3  # momentum: how many symbols to hold
    # Crash mode (trader_core/regime.py): switch to a backup strategy while the market is crash-like
    crash_switch: bool = False
    crash_index: str = "QQQ"  # index the detector watches
    crash_drawdown: float = 0.10  # how far below its 1-year high counts as crash-like (fraction)
    crash_window: int = 200  # days in the index's moving average
    crash_confirm_days: int = 1  # closes in a row the rule must hold before switching (1 = switch at once)
    crash_mode: str = "defensive"  # defensive = best safe haven; cash = hold cash
    crash_assets: List[str] = field(default_factory=lambda: ["BIL", "IEF", "TLT", "GLD"])
    cash_rate: float = 0.0  # yearly interest on uninvested cash in backtests, as a fraction (0.03 = 3%)
    initial_capital: float = 100_000.0
    data_provider: str = "alpaca"  # alpaca | yfinance | synthetic
    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_data_feed: str = "iex"  # iex works on the free plan
    signal_mode: str = "close"  # close | intraday (see live_trader/trader.py)
    fractional_shares: bool = True  # buy fractional shares where Alpaca allows it
    dry_run: bool = False
    force_run: bool = False  # run even when the market is closed (testing)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        if env is None:
            # real environment variables win over the .env file
            env = {**read_dotenv(), **{k: v for k, v in os.environ.items() if v != ""}}
        else:
            env = dict(env)
        secret_id = env.get("SECRET_ID")
        if secret_id:
            env.update(load_secret(secret_id))

        return cls(
            database_url=env.get("DATABASE_URL") or cls.database_url,
            symbols=_symbols(env.get("SYMBOLS")),
            universe=_universe_label(env.get("SYMBOLS")),
            strategy=(env.get("STRATEGY") or cls.strategy).lower(),
            fast_window=int(env.get("FAST_WINDOW") or cls.fast_window),
            slow_window=int(env.get("SLOW_WINDOW") or cls.slow_window),
            trend_window=int(env.get("TREND_WINDOW") or cls.trend_window),
            momentum_lookback=int(env.get("MOMENTUM_LOOKBACK") or cls.momentum_lookback),
            momentum_top=int(env.get("MOMENTUM_TOP") or cls.momentum_top),
            crash_switch=_bool(env.get("CRASH_SWITCH")),
            crash_index=(env.get("CRASH_INDEX") or cls.crash_index).strip().upper(),
            crash_drawdown=float(env.get("CRASH_DRAWDOWN") or 10) / 100,  # written as a percent
            crash_window=int(env.get("CRASH_WINDOW") or cls.crash_window),
            crash_confirm_days=int(env.get("CRASH_CONFIRM_DAYS") or cls.crash_confirm_days),
            crash_mode=(env.get("CRASH_MODE") or cls.crash_mode).strip().lower(),
            crash_assets=[a.strip().upper() for a in (env.get("CRASH_ASSETS") or "BIL,IEF,TLT,GLD").split(",")
                          if a.strip()],
            cash_rate=float(env.get("CASH_RATE") or 0) / 100,  # written as a percent, e.g. CASH_RATE=3
            initial_capital=float(env.get("INITIAL_CAPITAL") or cls.initial_capital),
            data_provider=(env.get("DATA_PROVIDER") or cls.data_provider).lower(),
            alpaca_key_id=env.get("ALPACA_API_KEY_ID") or None,
            alpaca_secret_key=env.get("ALPACA_API_SECRET_KEY") or None,
            alpaca_data_feed=(env.get("ALPACA_DATA_FEED") or cls.alpaca_data_feed).lower(),
            signal_mode=(env.get("SIGNAL_MODE") or cls.signal_mode).lower(),
            fractional_shares=_bool(env.get("FRACTIONAL_SHARES"), default=True),
            dry_run=_bool(env.get("DRY_RUN")),
            force_run=_bool(env.get("FORCE_RUN")),
        )

    def require_alpaca(self) -> None:
        if not (self.alpaca_key_id and self.alpaca_secret_key):
            raise RuntimeError(
                "Alpaca credentials missing: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
                "(paper-trading keys from https://app.alpaca.markets)"
            )
