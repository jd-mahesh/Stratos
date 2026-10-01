"""Crash detection: is the market in normal conditions or a crash-like one?

The bot runs its normal strategy most of the time. When the detector sees a
crash-like market it switches to a backup strategy (see CrashSwitch in
strategy.py), and switches back once conditions return to normal.

Rule (checked on every run, not just monthly):
    crash mode starts when the index (QQQ by default) is
        * below its 200-day average, AND
        * at least 10% below its highest close of the past year (252 days)
    and ends when the index closes back above its 200-day average.

The thresholds are standard market conventions (a 10% fall from a high is the
textbook definition of a "correction"; the 200-day average is the most common
long-term trend line). They were not tuned to any particular crash.

Using different conditions to enter and exit ("hysteresis") stops the bot from
flipping back and forth every day while the index hovers around one line.

Confirmation (CRASH_CONFIRM_DAYS, default 1): with N above 1, the mode only
changes once the rule has held on N daily closes in a row, e.g. 5 = a full
trading week below the line before switching to crash mode, and a full week
back above it before switching back. The check then also uses finished daily
closes only, never the live price, so a bad morning that recovers by the close
can't trigger a switch. The first crash report showed the unconfirmed rule
flipping in and out within days (and a one-day false alarm), each flip selling
everything and buying it back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

NORMAL = "normal"
CRASH = "crash"


@dataclass(frozen=True)
class Regime:
    mode: str  # "normal" | "crash"
    reason: str


@dataclass(frozen=True)
class CrashDetector:
    index: str = "QQQ"
    drawdown: float = 0.10  # how far below its 1-year high the index must be
    window: int = 200  # days in the moving average
    high_window: int = 252  # days used for the "1-year high"
    confirm_days: int = 1  # closes in a row the rule must hold before the mode changes

    def __post_init__(self) -> None:
        if not 0 < self.drawdown < 1:
            raise ValueError("drawdown must be between 0 and 1 (e.g. 0.10 for 10%)")
        if self.window < 2 or self.high_window < 2:
            raise ValueError("windows must be at least 2 days")
        if self.confirm_days < 1:
            raise ValueError("confirm_days must be at least 1")

    @property
    def closes_only(self) -> bool:
        """True if the check should see finished daily closes only (no live price)."""
        return self.confirm_days > 1

    @property
    def required_bars(self) -> int:
        return max(self.window, self.high_window) + self.confirm_days - 1

    def describe(self) -> str:
        if self.confirm_days == 1:
            return (f"crash mode while {self.index} is below its {self.window}-day average and at least "
                    f"{self.drawdown * 100:.0f}% below its 1-year high, until it closes back above the average")
        n = self.confirm_days
        return (f"crash mode once {self.index} has closed below its {self.window}-day average and at least "
                f"{self.drawdown * 100:.0f}% below its 1-year high {n} days in a row, until it closes back "
                f"above the average {n} days in a row (daily closes only)")

    def _levels(self, arr: np.ndarray, end: int):
        """(price, average, fall from the 1-year high) at the close with position ``end``."""
        upto = arr[: end + 1]
        price = float(upto[-1])
        return price, float(upto[-self.window:].mean()), price / float(upto[-self.high_window:].max()) - 1

    def _streak(self, arr: np.ndarray, test) -> int:
        """How many of the most recent closes (up to confirm_days) in a row pass ``test``."""
        count = 0
        for end in range(arr.size - 1, arr.size - 1 - self.confirm_days, -1):
            if not test(*self._levels(arr, end)):
                break
            count += 1
        return count

    def check(self, closes: Optional[Sequence[float]], previous: Optional[str]) -> Optional[Regime]:
        """The mode given the index's closes (oldest first) and the mode before this check.

        Returns None if there isn't enough index history to judge.
        """
        if closes is None:
            return None
        arr = np.asarray(closes, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < self.required_bars:
            return None
        price, avg, off_high = self._levels(arr, arr.size - 1)
        where = (f"{self.index} {price:.2f}: {off_high * 100:+.1f}% from its 1-year high, "
                 f"{'above' if price > avg else 'below'} its {self.window}-day average {avg:.2f}")
        n = self.confirm_days

        def pending(streak: int, what: str) -> str:
            return f" ({what} {streak} of the last {n} closes in a row, needs {n})" if n > 1 and streak else ""

        if previous == CRASH:
            streak = self._streak(arr, lambda p, a, off: p > a)
            if streak >= n:
                return Regime(NORMAL, f"back to normal: {where}")
            return Regime(CRASH, f"crash mode continues: {where}{pending(streak, 'above the average')}")
        streak = self._streak(arr, lambda p, a, off: p < a and off <= -self.drawdown)
        if streak >= n:
            return Regime(CRASH, f"crash mode: {where}")
        return Regime(NORMAL, f"normal: {where}{pending(streak, 'crash rule met')}")


def regime_series(index_closes: pd.Series, detector: CrashDetector) -> pd.Series:
    """The mode at each day's close, applying the rule day by day (NaN until there's enough history)."""
    closes = index_closes.dropna()
    values = closes.to_numpy(dtype=float)
    modes = []
    previous: Optional[str] = None
    for i in range(len(values)):
        regime = detector.check(values[: i + 1], previous)
        if regime is not None:
            previous = regime.mode
        modes.append(previous)
    return pd.Series(modes, index=closes.index, dtype=object)
