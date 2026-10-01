from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader_core import db


def make_bars(closes, start="2024-01-01", open_offset=0.0) -> pd.DataFrame:
    """Daily bars from a list of closes; open = close + open_offset unless overridden."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes + open_offset,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


@pytest.fixture
def engine(tmp_path):
    eng = db.get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db.init_db(eng)
    return eng
