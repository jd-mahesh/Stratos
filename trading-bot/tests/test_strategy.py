import numpy as np
import pytest

from trader_core.strategy import MACrossover


def test_uptrend_is_long():
    signal = MACrossover(5, 20).evaluate(np.arange(1, 31))
    assert signal.target == 1.0
    assert signal.fast_ma > signal.slow_ma


def test_downtrend_is_flat():
    signal = MACrossover(5, 20).evaluate(np.arange(30, 0, -1))
    assert signal.target == 0.0


def test_not_enough_history():
    signal = MACrossover(5, 20).evaluate(np.arange(10))
    assert signal.target is None
    assert "need 20" in signal.reason


def test_only_the_lookback_window_matters():
    strat = MACrossover(5, 20)
    recent = np.arange(1, 21)
    with_old_noise = np.concatenate([np.full(500, 1e6), recent])
    assert strat.evaluate(recent) == strat.evaluate(with_old_noise)


def test_missing_prices_do_not_produce_a_signal():
    closes = np.arange(1.0, 21.0)
    closes[-3] = np.nan
    assert MACrossover(5, 20).evaluate(closes).target is None


@pytest.mark.parametrize("fast,slow", [(20, 20), (50, 20), (0, 10)])
def test_invalid_windows(fast, slow):
    with pytest.raises(ValueError):
        MACrossover(fast, slow)
