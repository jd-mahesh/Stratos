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
import math
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


def _vol_scale(value: Optional[str]) -> int:
    """MOMENTUM_VOL_SCALE: 0 (off) or 2-251 trading days; anything else is a configuration error."""
    if value is None or value.strip() == "":
        return 0
    try:
        days = int(value)
    except ValueError:
        raise ValueError(f"MOMENTUM_VOL_SCALE must be a whole number of trading days, not {value!r}") from None
    if days != 0 and not 2 <= days <= 251:
        raise ValueError(f"MOMENTUM_VOL_SCALE must be 0 (off) or between 2 and 251, not {days}")
    return days


def _percent(env: Mapping[str, str], name: str, default: float, upper: float = 100) -> float:
    """A safeguard limit written as a percent (e.g. 25), returned as a fraction (0.25).

    Must be above 0 and at most ``upper``; a bad value stops the bot with a clear
    error instead of silently trading without the limit.
    """
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default / 100
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number (a percent, e.g. {default:g}), not {raw!r}") from None
    if not 0 < value <= upper:
        raise ValueError(f"{name} must be above 0 and at most {upper:g} (percent), not {raw}")
    return value / 100


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, not {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, not {value}")
    return value


def _dollars(env: Mapping[str, str], name: str, default: float = 0.0) -> float:
    """A dollar amount that must be 0 or more (e.g. CAPITAL_RESERVE=100000).

    Plain numbers only: a typo stops the bot with a clear error instead of
    trading with a different amount than intended.
    """
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise ValueError(f"{name} must be a plain number of dollars (e.g. 100000), not {raw!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be 0 or more dollars, not {raw}")
    return value


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
    # momentum: invest less when the picks' last N trading days were more volatile than their last year
    # (0 = off; 21 = the version that passed the backtests, see README "Volatility scaling")
    momentum_vol_scale: int = 0
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
    # Safeguards (trader_core/safeguards.py). Percents are stored as fractions (25 -> 0.25).
    trading_halted: bool = False  # kill switch: record balances, place no orders
    max_order_pct: float = 0.25  # no single buy larger than this share of the account
    max_position_pct: float = 0.30  # no buy that leaves one symbol above this share of the account
    max_orders_per_run: int = 20  # cancel the run if it plans more orders than this
    max_price_move_pct: float = 0.40  # skip a symbol whose live price is this far from its last close
    max_data_age_days: int = 5  # skip a symbol whose latest daily bar is older than this (calendar days)
    daily_loss_halt_pct: float = 0.15  # halt if the account falls this much since the previous day
    drawdown_halt_pct: float = 0.60  # halt if the account falls this much from its peak
    alert_topic_arn: Optional[str] = None  # AWS SNS topic for alert emails (none = log only)
    # Live trader only: dollars of the account Stratos must leave alone. It trades with
    # (account value - reserve) as if that were the whole account. 0 = use the whole account.
    capital_reserve: float = 0.0

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
            momentum_vol_scale=_vol_scale(env.get("MOMENTUM_VOL_SCALE")),
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
            trading_halted=_bool(env.get("TRADING_HALTED")),
            max_order_pct=_percent(env, "MAX_ORDER_PCT", 25),
            max_position_pct=_percent(env, "MAX_POSITION_PCT", 30),
            max_orders_per_run=_positive_int(env, "MAX_ORDERS_PER_RUN", 20),
            max_price_move_pct=_percent(env, "MAX_PRICE_MOVE_PCT", 40, upper=1000),
            max_data_age_days=_positive_int(env, "MAX_DATA_AGE_DAYS", 5),
            daily_loss_halt_pct=_percent(env, "DAILY_LOSS_HALT_PCT", 15),
            drawdown_halt_pct=_percent(env, "DRAWDOWN_HALT_PCT", 60),
            alert_topic_arn=(env.get("ALERT_TOPIC_ARN") or "").strip() or None,
            capital_reserve=_dollars(env, "CAPITAL_RESERVE"),
        )

    def require_alpaca(self) -> None:
        if not (self.alpaca_key_id and self.alpaca_secret_key):
            raise RuntimeError(
                "Alpaca credentials missing: set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
                "(paper-trading keys from https://app.alpaca.markets)"
            )
