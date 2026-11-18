"""The data audit and the shape of the peaks, as numbers.

* where the raw midnight rows belong (jump to file neighbours vs to the
  same date's 00:15)
* CO2 as a rescaled copy of demand
* the three daily tariff patterns
* how long limit crossings last, how far they overshoot, and what demand
  looked like an hour before
* kVA vs kW at the peaks

Usage: python scripts/audit.py   (writes results/audit.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peakcast import alerts, data  # noqa: E402


def midnight_check(raw_path: Path) -> dict:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    ts = pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M")
    x = raw["Usage_kWh"].to_numpy()
    mid = np.flatnonzero((ts.dt.hour == 0) & (ts.dt.minute == 0))
    mid = mid[(mid > 0) & (mid < len(raw) - 1)]
    by_ts = pd.Series(x, index=ts)
    same_date_0015 = by_ts.reindex(ts.iloc[mid] + pd.Timedelta("15min")).to_numpy()
    return {
        "midnight_rows": int(len(mid) + 1),
        "mean_jump_to_file_neighbours_kwh": float(np.mean(np.abs(x[mid] - x[mid - 1]) / 2 + np.abs(x[mid] - x[mid + 1]) / 2)),
        "mean_jump_to_same_date_0015_kwh": float(np.nanmean(np.abs(x[mid] - same_date_0015))),
    }


def main():
    path = ROOT / "data/raw/Steel_industry_data.csv"
    d = data.load(path)
    train, _, _ = data.split(d)
    limit = float(np.percentile(d["kw"][train], 95))
    out = {"midnight": midnight_check(path)}

    busy = d["kwh"] > 20
    out["co2"] = {
        "tco2_per_kwh_median": float((d["co2"][busy] / d["kwh"][busy]).median()),
        "corr_with_kwh": float(np.corrcoef(d["co2"], d["kwh"])[0, 1]),
        "distinct_values": int(d["co2"].nunique()),
    }

    start = d.index - data.BLOCK
    code = d["load_type"].map({"Light_Load": "L", "Medium_Load": "M", "Maximum_Load": "X"})
    day = code.groupby(start.date).agg("".join)
    out["tariff_patterns"] = [
        {"days": int(n), "maximum_band_blocks": p.count("X"), "medium_band_blocks": p.count("M"),
         "months": sorted({pd.Timestamp(x).month for x in day[day == p].index})}
        for p, n in day.value_counts().items()
    ]

    kw = d["kw"].to_numpy()
    over = kw > limit
    ev = alerts.episodes(over)
    lens = np.array([e - s + 1 for s, e in ev])
    excess = np.array([kw[s : e + 1].max() - limit for s, e in ev])
    before = np.array([kw[s - 4] for s, _ in ev if s >= 4])
    kva = np.sqrt((d["kwh"] * 4) ** 2 + (d["lag_kvarh"] * 4) ** 2).to_numpy()
    out["peaks_full_year"] = {
        "limit_kw": limit,
        "episodes": len(ev),
        "share_single_block": float(np.mean(lens == 1)),
        "median_overshoot_kw": float(np.median(excess)),
        "p90_overshoot_kw": float(np.percentile(excess, 90)),
        "share_already_above_90pct_one_hour_before": float(np.mean(before > 0.9 * limit)),
        "kva_over_kw_at_peaks_median": float(np.median(kva[over] / kw[over])),
    }
    (ROOT / "results/audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
