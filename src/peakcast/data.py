"""UCI "Steel Industry Energy Consumption" (dataset 851): loading and cleaning.

One year (2018) of 15-minute readings from a steel plant in South Korea: active
energy per block (kWh), lagging and leading reactive energy, power factors,
CO2, and a load-type label.

Two things the raw file gets wrong, both fixed here and both tested:

* Each day's last block, the one ending at midnight, is stamped with that
  day's date at 00:00 instead of the next day's. Read naively, 365 blocks land
  24 hours early, at the start of their day. The file order and the readings
  agree that they belong at the end of the day.
* Rows are therefore out of time order; they are sorted after the fix.

Timestamps mark the END of each 15-minute block, the way an energy meter
bills them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
BLOCK = pd.Timedelta("15min")
PER_DAY = 96

# Columns measured in the same block as the demand. A forecast made before the
# block ends cannot know them; using them is the leak most notebooks on this
# dataset fall into.
CONCURRENT = [
    "lag_kvarh",
    "lead_kvarh",
    "co2",
    "lag_pf",
    "lead_pf",
]


def load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    ts = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M")
    midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0)
    ts = ts.where(~midnight, ts + pd.Timedelta(days=1))
    d = pd.DataFrame(
        {
            "kwh": raw["Usage_kWh"].to_numpy(),
            "lag_kvarh": raw["Lagging_Current_Reactive.Power_kVarh"].to_numpy(),
            "lead_kvarh": raw["Leading_Current_Reactive_Power_kVarh"].to_numpy(),
            "co2": raw["CO2(tCO2)"].to_numpy(),
            "lag_pf": raw["Lagging_Current_Power_Factor"].to_numpy(),
            "lead_pf": raw["Leading_Current_Power_Factor"].to_numpy(),
            "load_type": raw["Load_Type"].to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="block_end"),
    ).sort_index()
    if d.index.has_duplicates:
        raise ValueError("duplicate block timestamps after the midnight fix")
    if not ((d.index[1:] - d.index[:-1]) == BLOCK).all():  # unit-safe across pandas 2 and 3
        raise ValueError("blocks are not on a regular 15-minute grid")
    d["kw"] = d["kwh"] * 4  # average demand over the block
    return d


def split(d: pd.DataFrame):
    """Chronological: train Jan-Sep, calibrate Oct, test Nov-Dec."""
    t = d.index
    train = t <= pd.Timestamp("2018-10-01")
    cal = (t > pd.Timestamp("2018-10-01")) & (t <= pd.Timestamp("2018-11-01"))
    test = t > pd.Timestamp("2018-11-01")
    return train, cal, test
