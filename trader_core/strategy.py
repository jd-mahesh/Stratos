"""Trading strategies.

A strategy looks at price history and answers one question for every symbol:
what fraction of the account should be in it right now? It knows nothing about
brokers, databases or dates, which is what lets the backtester and the live
trader call exactly the same code.

The "history" a strategy sees is each symbol's daily closes, oldest first,
ending with the most recent price. Whether that last price is yesterday's
close or the live price right now is decided by SIGNAL_MODE, not by the
strategy (see live_trader/trader.py and backtester/engine.py).

Answers are *targets*, not events ("hold 9% in XLK", not "buy XLK now").
Asking the same question twice gives the same answer, so a scheduler that
fires every five minutes can't stack duplicate trades.

Strategies
    ma_crossover  Own a symbol while its 20-day average is above its 50-day
                  average. Checked daily. (The original strategy.)
    trend         Own a symbol while its price is above its 200-day average.
                  Checked monthly. A slow trend filter in the spirit of
                  Meb Faber's "A Quantitative Approach to Tactical Asset
                  Allocation" (2007).
    momentum      Each month, rank symbols by their 12-month return and hold
                  the top few, but only those whose return is positive
                  (otherwise cash). Relative strength with an absolute-
                  momentum filter, as in Gary Antonacci's "dual momentum".

The default parameters are the standard ones from that research, not values
tuned on recent data, so they aren't fitted to any particular market.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .regime import CrashDetector


@dataclass(frozen=True)
class Signal:
    weight: Optional[float]  # target fraction of the account; 0 = hold cash; None = not enough data
    reason: str
    fast_ma: Optional[float] = None  # indicator values, recorded in the decision log
    slow_ma: Optional[float] = None

    @property
    def target(self) -> Optional[float]:
        """1.0 = own it, 0.0 = don't, None = no opinion (kept for readability in tests)."""
        if self.weight is None:
            return None
        return 1.0 if self.weight > 0 else 0.0


History = Dict[str, Sequence[float]]


def _prices(closes: Sequence[float], need: int) -> Optional[np.ndarray]:
    """The last ``need`` prices as floats, or None if there aren't enough clean ones."""
    arr = np.asarray(closes, dtype=float)
    if arr.size < need:
        return None
    tail = arr[-need:]
    return tail if np.all(np.isfinite(tail)) else None


class Strategy:
    name = "base"
    rebalance = "daily"  # "daily", "monthly" or "every": how often targets are recomputed and acted on
    every = 0  # with rebalance == "every": rebalance every this many trading days (backtester only for now)
    resize = False  # on a rebalance, trim or top up held positions back to their target weight

    def params(self) -> dict:
        return asdict(self)

    @property
    def extra_symbols(self) -> List[str]:
        """Symbols beyond SYMBOLS whose prices the strategy needs (an index to watch, backup assets)."""
        return []

    @property
    def tradable_extras(self) -> List[str]:
        """The subset of ``extra_symbols`` the strategy may buy (e.g. crash-mode safe havens)."""
        return []

    def mode(self, extra: Optional[History], previous: Optional[str]):
        """Market mode for strategies that switch behaviour (see CrashSwitch); None otherwise."""
        return None

    @property
    def required_bars(self) -> int:
        """Days of history a traded symbol needs before the strategy has an opinion on it."""
        raise NotImplementedError

    @property
    def warmup_bars(self) -> int:
        """Days of history to fetch so every input (including extra symbols) is ready."""
        return self.required_bars

    def describe(self) -> str:
        raise NotImplementedError

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        """Target weight for every symbol in ``history``.

        ``slots`` is how many equal slots the account is split into (defaults to
        the number of symbols). The live trader passes the full symbol count so a
        symbol missing data today doesn't make the others' slots bigger.
        ``extra`` holds prices for ``extra_symbols``, in the same form as ``history``.
        ``mode`` is the market mode from ``mode()`` for strategies that use one.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class MACrossover(Strategy):
    fast: int = 20
    slow: int = 50

    name = "ma_crossover"
    rebalance = "daily"
    resize = False

    def __post_init__(self) -> None:
        if self.fast < 1 or self.slow < 2:
            raise ValueError("windows must be positive")
        if self.fast >= self.slow:
            raise ValueError(f"fast window ({self.fast}) must be shorter than slow window ({self.slow})")

    @property
    def required_bars(self) -> int:
        return self.slow

    def describe(self) -> str:
        return f"own each symbol while its {self.fast}-day average is above its {self.slow}-day average; checked daily"

    def evaluate(self, closes: Sequence[float]) -> Signal:
        """Long (weight 1) or flat (weight 0) for a single symbol."""
        arr = np.asarray(closes, dtype=float)
        if arr.size < self.slow:
            return Signal(None, f"need {self.slow} days of prices, have {arr.size}")
        tail = _prices(arr, self.slow)
        if tail is None:
            return Signal(None, "missing prices in lookback window")
        fast_ma, slow_ma = float(tail[-self.fast:].mean()), float(tail.mean())
        if fast_ma > slow_ma:
            return Signal(1.0, f"fast MA {fast_ma:.2f} > slow MA {slow_ma:.2f}", fast_ma, slow_ma)
        return Signal(0.0, f"fast MA {fast_ma:.2f} <= slow MA {slow_ma:.2f}", fast_ma, slow_ma)

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        slot = 1.0 / (slots or len(history) or 1)
        out = {}
        for symbol, closes in history.items():
            sig = self.evaluate(closes)
            weight = None if sig.weight is None else sig.weight * slot
            out[symbol] = Signal(weight, sig.reason, sig.fast_ma, sig.slow_ma)
        return out


@dataclass(frozen=True)
class Trend(Strategy):
    window: int = 200

    name = "trend"
    rebalance = "monthly"
    resize = True

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")

    @property
    def required_bars(self) -> int:
        return self.window

    def describe(self) -> str:
        return f"own each symbol while its price is above its {self.window}-day average; checked monthly"

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        slot = 1.0 / (slots or len(history) or 1)
        out = {}
        for symbol, closes in history.items():
            tail = _prices(closes, self.window)
            if tail is None:
                out[symbol] = Signal(None, f"need {self.window} days of prices")
                continue
            price, avg = float(tail[-1]), float(tail.mean())
            if price > avg:
                out[symbol] = Signal(slot, f"price {price:.2f} above {self.window}-day average {avg:.2f}", None, avg)
            else:
                out[symbol] = Signal(0.0, f"price {price:.2f} below {self.window}-day average {avg:.2f}", None, avg)
        return out


@dataclass(frozen=True)
class Momentum(Strategy):
    lookback: int = 252  # trading days, about 12 months
    top: int = 3
    absolute: bool = True  # only hold symbols whose own return is positive
    every: int = 0  # 0 = rebalance monthly; N = every N trading days (backtester only, see engine.py)
    skip: int = 0  # ignore the most recent N trading days when measuring the return (backtester only)
    # Volatility scaling (backtester only): invest less when the picks have recently been much more
    # volatile than usual. vol_short = days of "recent" volatility (0 = off), vol_long = days of "normal".
    vol_short: int = 0
    vol_long: int = 252
    # Trailing stop (backtester only, see engine.py): between rebalances, sell a position whose
    # close falls this many percent below its highest close since it was bought (0 = off).
    stop: int = 0
    # Take profits (backtester only): between rebalances, trim a position back to its target value
    # once it has grown this many percent above it (0 = off).
    take: int = 0

    name = "momentum"
    resize = True

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.top < 1:
            raise ValueError("lookback must be at least 2 and top at least 1")
        if self.every < 0:
            raise ValueError("every must be 0 (monthly) or a number of trading days")
        if self.skip < 0 or self.skip >= self.lookback - 1:
            raise ValueError("skip must be 0 or a number of trading days well inside the lookback")
        if self.vol_short < 0 or (self.vol_short and not 2 <= self.vol_short < self.vol_long):
            raise ValueError("vol_short must be 0 (off) or at least 2 days and shorter than vol_long")
        if not 0 <= self.stop < 100:
            raise ValueError("stop must be 0 (off) or a percent between 1 and 99")
        if not 0 <= self.take <= 1000:
            raise ValueError("take must be 0 (off) or a percent between 1 and 1000")

    @property
    def rebalance(self) -> str:  # type: ignore[override]
        return "every" if self.every else "monthly"

    def params(self) -> dict:
        # "every" is left out when monthly, so existing runs and the live trader's
        # rebalance bookkeeping (keyed on these params) are unchanged
        out = asdict(self)
        for key in ("every", "skip", "vol_short", "stop", "take"):  # left out at their defaults (see above)
            if not out[key]:
                out.pop(key)
        if not self.vol_short:
            out.pop("vol_long")
        return out

    @property
    def required_bars(self) -> int:
        return self.lookback + 1

    @property
    def warmup_bars(self) -> int:
        return max(self.required_bars, self.vol_long + 1 if self.vol_short else 0)

    def _exposure(self, picks: Dict[str, np.ndarray]) -> Tuple[float, str]:
        """Share of the target to invest (0-1) and a note, from the picks' recent vs normal volatility.

        Volatility is the standard deviation of the equal-weight basket's daily returns.
        Exposure = normal / recent, capped at 1: twice as volatile as usual -> half invested.
        With less than vol_long days of history it uses what there is (at least 2 x vol_short).
        """
        if not self.vol_short or not picks:
            return 1.0, ""
        n = min(min(arr.size for arr in picks.values()) - 1, self.vol_long)
        if n < 2 * self.vol_short:
            return 1.0, "; volatility scaling: not enough history yet, fully invested"
        window = np.vstack([arr[-n - 1:] for arr in picks.values()])
        if not np.all(np.isfinite(window)) or np.any(window <= 0):
            return 1.0, "; volatility scaling: gaps in the prices, fully invested"
        basket = (window[:, 1:] / window[:, :-1] - 1).mean(axis=0)
        recent, normal = float(basket[-self.vol_short:].std()), float(basket.std())
        if recent <= 0:
            return 1.0, "; volatility scaling: fully invested"
        exposure = min(1.0, normal / recent)
        return exposure, (f"; volatility scaling: recent {recent * np.sqrt(252) * 100:.0f}%/yr vs normal "
                          f"{normal * np.sqrt(252) * 100:.0f}%/yr, so {exposure * 100:.0f}% invested")

    def _period(self) -> str:
        months = round(self.lookback / 21)
        if not self.skip:
            return f"{months}-month"
        return f"{months}-month (skipping the last {self.skip} days)"

    def describe(self) -> str:
        rule = ", only if that return is positive" if self.absolute else ""
        when = f"every {self.every} trading days" if self.every else "each month"
        scaling = (f"; invest less when their last {self.vol_short} days were more volatile than their last "
                   f"{self.vol_long}" if self.vol_short else "")
        stop = (f"; between rebalances, sell a stock that falls {self.stop}% below its high since we bought it"
                if self.stop else "")
        take = (f"; between rebalances, trim a stock back to its target once it's {self.take}% above it"
                if self.take else "")
        return (f"{when}, hold the {self.top} symbols with the best {self._period()} return"
                f"{rule}{scaling}{stop}{take}")

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        returns = {}
        out: Dict[str, Signal] = {}
        for symbol, closes in history.items():
            arr = np.asarray(closes, dtype=float)
            if arr.size < self.required_bars:
                out[symbol] = Signal(None, f"need {self.required_bars} days of prices")
                continue
            # return from `lookback` days ago up to `skip` days ago (skip 0 = up to the latest price)
            end, begin = arr[-1 - self.skip], arr[-1 - self.lookback]
            if not (np.isfinite(arr[-1]) and np.isfinite(end) and np.isfinite(begin)):
                out[symbol] = Signal(None, f"need {self.required_bars} days of prices")
                continue
            returns[symbol] = float(end / begin - 1)

        ranked = sorted(returns, key=returns.get, reverse=True)
        picks = [s for s in ranked[: self.top] if not (self.absolute and returns[s] <= 0)]
        exposure, note = self._exposure({s: np.asarray(history[s], dtype=float) for s in picks})
        weight = exposure / self.top
        period = self._period()
        for rank, symbol in enumerate(ranked, start=1):
            ret = returns[symbol]
            where = f"{period} return {ret * 100:+.1f}%, rank {rank} of {len(ranked)}"
            if rank > self.top:
                out[symbol] = Signal(0.0, f"{where}; not in the top {self.top}")
            elif self.absolute and ret <= 0:
                out[symbol] = Signal(0.0, f"{where}; return not positive, so cash instead")
            else:
                out[symbol] = Signal(weight, f"{where}; in the top {self.top}{note}")
        return out


@dataclass(frozen=True)
class Defensive(Strategy):
    """Crash-mode backup: hold whichever safe asset has held up best lately.

    Different crashes rewarded different havens (long Treasuries in 2008 and
    2020, cash in 2022, gold in stretches of 2011), so rather than betting on
    one, it picks the best of the list over the last ``lookback`` days, and
    only if that beat T-bills (the first asset, a stand-in for cash).
    With ``cash_only`` it simply holds cash.
    """

    assets: Tuple[str, ...] = ("BIL", "IEF", "TLT", "GLD")
    lookback: int = 63  # about 3 months
    cash_only: bool = False

    name = "defensive"
    rebalance = "monthly"
    resize = True

    @property
    def required_bars(self) -> int:
        return self.lookback + 1

    def describe(self) -> str:
        if self.cash_only:
            return "hold cash"
        return (f"hold the best of {', '.join(self.assets)} over the last {round(self.lookback / 21)} months, "
                f"or {self.assets[0]} if none beat it")

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        prices = {**(extra or {}), **history}
        if self.cash_only:
            return {a: Signal(0.0, "crash mode: holding cash") for a in self.assets}
        returns = {}
        for asset in self.assets:
            arr = np.asarray(prices.get(asset, []), dtype=float)
            if arr.size >= self.required_bars and np.isfinite(arr[-1]) and np.isfinite(arr[-1 - self.lookback]):
                returns[asset] = float(arr[-1] / arr[-1 - self.lookback] - 1)
        if not returns:
            return {a: Signal(0.0, "crash mode: no safe-haven prices yet, holding cash") for a in self.assets}
        cash_like = self.assets[0]
        best = max(returns, key=returns.get)
        if cash_like in returns and returns[best] <= returns[cash_like]:
            best = cash_like
        period = f"{round(self.lookback / 21)}-month"
        out = {}
        for asset in self.assets:
            if asset not in returns:
                out[asset] = Signal(0.0, f"crash mode: not enough {asset} prices")
            elif asset == best:
                out[asset] = Signal(1.0, f"crash mode: {asset} is the best safe haven ({period} return "
                                         f"{returns[asset] * 100:+.1f}%)")
            else:
                out[asset] = Signal(0.0, f"crash mode: {asset} {period} return {returns[asset] * 100:+.1f}%, "
                                         f"{best} is better")
        return out


@dataclass(frozen=True)
class CrashSwitch(Strategy):
    """Run ``normal`` in normal markets and ``backup`` while ``detector`` sees a crash.

    The mode is checked on every run; a change of mode triggers an immediate
    rebalance, outside the normal monthly schedule.
    """

    normal: Strategy
    backup: Defensive
    detector: "CrashDetector"

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.normal.name

    @property
    def rebalance(self) -> str:  # type: ignore[override]
        return self.normal.rebalance

    @property
    def resize(self) -> bool:  # type: ignore[override]
        return self.normal.resize

    @property
    def every(self) -> int:  # type: ignore[override]
        return getattr(self.normal, "every", 0)

    def params(self) -> dict:
        crash = (f"{self.detector.index} -{self.detector.drawdown * 100:.0f}%/{self.detector.window}d -> "
                 f"{'cash' if self.backup.cash_only else '/'.join(self.backup.assets)}")
        if self.detector.confirm_days > 1:
            crash += f", confirm {self.detector.confirm_days}d"
        return {**self.normal.params(), "crash": crash}

    @property
    def required_bars(self) -> int:
        return self.normal.required_bars

    @property
    def warmup_bars(self) -> int:
        return max(self.normal.warmup_bars, self.detector.required_bars, self.backup.required_bars)

    @property
    def extra_symbols(self) -> List[str]:
        extras = [self.detector.index] + list(self.backup.assets) + self.normal.extra_symbols
        return list(dict.fromkeys(extras))

    @property
    def tradable_extras(self) -> List[str]:
        return [] if self.backup.cash_only else list(self.backup.assets)

    def describe(self) -> str:
        return f"{self.normal.describe()}; {self.detector.describe()}; in crash mode, {self.backup.describe()}"

    def mode(self, extra: Optional[History], previous: Optional[str]):
        return self.detector.check((extra or {}).get(self.detector.index), previous)

    def decide(self, history: History, slots: Optional[int] = None, extra: Optional[History] = None,
               mode: Optional[str] = None) -> Dict[str, Signal]:
        backup_assets = set(self.backup.assets)
        if mode == "crash":
            signals = self.backup.decide({a: c for a, c in history.items() if a in backup_assets}, None, extra)
            out = {s: Signal(0.0, "crash mode: out of the market") for s in history if s not in backup_assets}
            out.update(signals)
            return out
        signals = self.normal.decide(history, slots, extra)
        for asset in self.tradable_extras:  # leave any crash-mode holdings once things are normal again
            if asset not in signals:
                signals[asset] = Signal(0.0, "normal mode: not holding crash-mode assets")
        return signals


STRATEGIES = {cls.name: cls for cls in (MACrossover, Trend, Momentum, Defensive)}


def make_strategy(settings, name: Optional[str] = None, **overrides) -> Strategy:
    """Build the strategy named in settings (or ``name``).

    ``overrides`` may include parameters of any strategy; only the ones this
    strategy has are used, and None means "use the setting".
    """
    name = (name or getattr(settings, "strategy", "ma_crossover")).lower()
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {', '.join(STRATEGIES)}")
    defaults = {
        "ma_crossover": {"fast": settings.fast_window, "slow": settings.slow_window},
        "trend": {"window": getattr(settings, "trend_window", 200)},
        "momentum": {"lookback": getattr(settings, "momentum_lookback", 252),
                     "top": getattr(settings, "momentum_top", 3),
                     "vol_short": getattr(settings, "momentum_vol_scale", 0)},
        "defensive": {"assets": tuple(getattr(settings, "crash_assets", ("BIL", "IEF", "TLT", "GLD"))),
                      "cash_only": getattr(settings, "crash_mode", "defensive") == "cash"},
    }[name]
    cls = STRATEGIES[name]
    own = set(cls.__dataclass_fields__)
    defaults.update({k: v for k, v in overrides.items() if v is not None and k in own})
    strategy = cls(**defaults)

    crash = overrides.get("crash_switch")
    if crash is None:
        crash = getattr(settings, "crash_switch", False)
    if crash and name != "defensive":
        from .regime import CrashDetector

        detector = CrashDetector(
            index=getattr(settings, "crash_index", "QQQ"),
            drawdown=getattr(settings, "crash_drawdown", 0.10),
            window=getattr(settings, "crash_window", 200),
            confirm_days=int(overrides.get("crash_confirm") or getattr(settings, "crash_confirm_days", 1)),
        )
        backup = make_strategy(settings, "defensive", crash_switch=False)
        strategy = CrashSwitch(strategy, backup, detector)
    return strategy
