"""Forecast features: only what is known when the forecast is made.

The forecast is issued at the end of block t and is for block t+h. Allowed:
* demand up to and including block t
* demand at the same block 1 and 7 days before the target
* the calendar of the target block, including its tariff band (Load_Type),
  which follows a fixed time-of-use calendar: every day in the file is one of
  three patterns (spring-autumn, winter, and all-light for Sundays and holidays)

Not allowed: anything measured in block t+h itself (see data.CONCURRENT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PER_DAY

BANDS = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}


def calendar(index: pd.DatetimeIndex, load_type: pd.Series) -> pd.DataFrame:
    start = index - pd.Timedelta("15min")
    blk = start.hour * 4 + start.minute // 15
    day_light = load_type.groupby(start.date).transform(lambda s: (s == "Light_Load").all()).to_numpy()
    return pd.DataFrame(
        {
            "blk": blk,
            "blk_sin": np.sin(2 * np.pi * blk / PER_DAY),
            "blk_cos": np.cos(2 * np.pi * blk / PER_DAY),
            "dow": start.dayofweek,
            "weekend": (start.dayofweek >= 5).astype(int),
            "band": load_type.map(BANDS).to_numpy(),
            "off_day": day_light.astype(int),  # Sunday or holiday in the tariff calendar
            "winter": start.month.isin([11, 12, 1, 2]).astype(int),
        },
        index=index,
    )


def build(d: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.Series]:
    """Rows indexed by the TARGET block t+h; features known at block t."""
    kw = d["kw"]
    cal = calendar(d.index, d["load_type"])
    X = pd.DataFrame(index=d.index)
    for lag in range(4):  # last hour, ending at the origin t
        X[f"kw_t-{lag}"] = kw.shift(h + lag)
    X["kw_mean_1h"] = kw.shift(h).rolling(4).mean()
    X["kw_max_1h"] = kw.shift(h).rolling(4).max()
    X["kw_mean_4h"] = kw.shift(h).rolling(16).mean()
    X["kw_day_ago"] = kw.shift(PER_DAY)  # same block yesterday (h <= 96)
    X["kw_week_ago"] = kw.shift(7 * PER_DAY)
    X["kw_same_blk_7d_mean"] = sum(kw.shift(k * PER_DAY) for k in range(1, 8)) / 7
    X = X.join(cal)
    y = kw.rename("y")
    ok = X.notna().all(axis=1)
    return X[ok], y[ok]
