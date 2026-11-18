"""Step 1: the leak, the baselines and gradient boosting.

* leak: predict a block's demand from columns measured in that same block,
  with a random split. This is what most notebooks on this dataset report.
* honest: forecast block t+h from data up to block t, chronological split.
  h = 1 (15 minutes ahead) and h = 4 (one hour ahead).

Usage: python scripts/run_baselines.py   (writes results/baselines.json)
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peakcast import data, features, metrics  # noqa: E402

SEED = 0
HORIZONS = (1, 4)


def load():
    d = data.load(ROOT / "data/raw/Steel_industry_data.csv")
    train, _, _ = data.split(d)
    limit = float(np.percentile(d["kw"][train], 95))
    return d, limit


def gbm():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED)


def leak(d, limit):
    cal = features.calendar(d.index, d["load_type"])
    X = d[data.CONCURRENT].join(cal).to_numpy()
    y = d["kw"].to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED)
    p = gbm().fit(X[tr], y[tr]).predict(X[te])
    return metrics.scores(y[te], p, limit)


def honest(d, limit, h):
    X, y = features.build(d, h)
    train, cal, test = (m[d.index.isin(X.index)] for m in data.split(d))
    out = {
        "persistence": X["kw_t-0"][test],
        "same_block_yesterday": X["kw_day_ago"][test],
        "same_block_last_week": X["kw_week_ago"][test],
    }
    model = gbm().fit(X[train | cal], y[train | cal])  # cal is not used for tuning here
    out["gradient_boosting"] = model.predict(X[test])
    return {k: metrics.scores(y[test], v, limit) for k, v in out.items()}


def main():
    d, limit = load()
    res = {"limit_kw": limit, "leak_same_block_random_split": leak(d, limit)}
    for h in HORIZONS:
        res[f"h{h}"] = honest(d, limit, h)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/baselines.json").write_text(json.dumps(res, indent=2))
    print(f"limit (train p95) {limit:.0f} kW")
    lk = res["leak_same_block_random_split"]
    print(f"LEAK  same-block features, random split: R2 {lk['r2']:.3f}  MAE {lk['mae_kw']:.1f} kW")
    for h in HORIZONS:
        for name, s in res[f"h{h}"].items():
            print(f"h={h} {name:22s} MAE {s['mae_kw']:6.1f}  WAPE {s['wape']:.3f}  R2 {s['r2']:.3f}  "
                  f"peak MAE {s['peak_mae_kw']:6.1f}  peak bias {s['peak_bias_kw']:+6.1f}")


if __name__ == "__main__":
    main()
