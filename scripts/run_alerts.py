"""Step 2: conformal intervals and the one-hour-ahead peak alert.

One gradient-boosting model per horizon (15, 30, 45, 60 minutes), fitted on
January-September. October sets the conformal widths. November-December is
the test. Three alert rules are compared on the same test blocks:

  operator rule   warn when the current block is above 90% of the limit
  point forecast  warn when any forecast for the next hour is above the limit
  upper bound     warn when any 90% upper bound for the next hour is above it

Usage: python scripts/run_alerts.py   (writes results/alerts.json)
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from peakcast import alerts, data, features  # noqa: E402
from run_baselines import gbm, load  # noqa: E402

HORIZONS = (1, 2, 3, 4)
LEVELS = (0.8, 0.9, 0.95)


def forecasts(d: pd.DataFrame):
    """Per horizon: test-period point forecast and band, indexed by ORIGIN block."""
    train, cal, test = data.split(d)
    out = {}
    for h in HORIZONS:
        X, y = features.build(d, h)
        tr, ca, te = (m[d.index.isin(X.index)] for m in (train, cal, test))
        model = gbm().fit(X[tr], y[tr])
        res = np.abs(model.predict(X[ca]) - y[ca].to_numpy())
        p = pd.Series(model.predict(X[te]), index=X.index[te] - h * data.BLOCK)  # re-index by origin
        band = pd.Series(X["band"][te].to_numpy(), index=p.index)
        out[h] = {"p": p, "band": band, "cal_res": res, "cal_band": X["band"][ca].to_numpy(),
                  "y": pd.Series(y[te].to_numpy(), index=p.index)}
    return out


LIMIT_PCTS = (95, 99)  # demand limit = this percentile of Jan-Sep demand; 95 is the main setting


def main():
    d, _ = load()
    train, _, _ = data.split(d)
    fc = forecasts(d)
    out = {}
    for pct in LIMIT_PCTS:
        limit = float(np.percentile(d["kw"][train], pct))
        out[f"p{pct}"] = evaluate(d, fc, limit)
    (ROOT / "results/alerts.json").write_text(json.dumps(out, indent=2))


def evaluate(d, fc, limit):
    origins = fc[HORIZONS[-1]]["p"].index  # origins with all four horizons inside the test period
    for h in HORIZONS:
        origins = origins.intersection(fc[h]["p"].index)
    kw = d["kw"]
    kwv = kw.reindex(origins).to_numpy()
    over = kwv > limit
    month = origins.month.to_numpy()

    def score(a):
        return alerts.score_alerts(a, over, kw=kwv, limit=limit)
    res = {"limit_kw": limit, "test_origins": len(origins), "rules": {}, "coverage": {}}

    # operator rule: current demand above 90% of the limit
    operator = kwv > 0.9 * limit
    res["rules"]["operator_90pct_now"] = score(operator)
    point = np.zeros(len(origins), bool)
    for h in HORIZONS:
        point |= (fc[h]["p"].reindex(origins) > limit).to_numpy()
    res["rules"]["point_forecast"] = score(point)

    chosen = {}
    for level in LEVELS:
        upper = np.zeros(len(origins), bool)
        cover = []
        for h in HORIZONS:
            w = alerts.conformal_widths(fc[h]["cal_res"], fc[h]["cal_band"], level)
            p, band, y = (fc[h][k].reindex(origins) for k in ("p", "band", "y"))
            width = band.map(w).to_numpy()
            upper |= (p.to_numpy() + width) > limit
            cover.append(float((np.abs(y.to_numpy() - p.to_numpy()) <= width).mean()))
        res["rules"][f"upper_bound_{int(level * 100)}"] = score(upper)
        chosen[level] = upper
        res["coverage"][f"{int(level * 100)}"] = {f"h{h}": round(c, 3) for h, c in zip(HORIZONS, cover)}

    res["monthly_peak_if_caught_shaved"] = {
        name: alerts.shaved_monthly_peak(kwv, month, over, alerts.caught_blocks(a, over), limit)
        for name, a in (("operator_90pct_now", operator), ("point_forecast", point),
                        ("upper_bound_80", chosen[0.8]), ("upper_bound_90", chosen[0.9]))
    }
    print(f"limit {limit:.0f} kW, {res['test_origins']} test origins")
    for name, s in res["rules"].items():
        print(f"{name:20s} caught {s['caught']}/{s['exceedance_episodes']}  median lead {s['median_lead_min']:.0f} min  "
              f"false alerts {s['false_alert_episodes']} ({s['false_alerts_per_week']:.1f}/week, {s['false_alerts_within_10pct_of_limit']:.0%} near misses)  in alert {s['alert_share']:.1%}")
    print("monthly peak if caught episodes were held at the limit:", json.dumps(res["monthly_peak_if_caught_shaved"]))
    print("coverage", res["coverage"])
    return res


if __name__ == "__main__":
    main()
