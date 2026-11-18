"""Step 3: LSTM and Transformer vs gradient boosting, same split, same test blocks.

All three are fitted on January-September. The sequence models use October
for early stopping. Gradient boosting is refitted here on January-September
only, so no model sees October demand as training data.

Usage: python scripts/run_seq.py   (writes results/sequence.json)
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from peakcast import data, features, metrics, seq  # noqa: E402
from run_baselines import gbm, load  # noqa: E402

SEEDS = (0, 1, 2)


def origin_positions(d, mask):
    """Positions t whose window and all four targets lie inside the masked period."""
    pos = np.flatnonzero(mask)
    lo, hi = pos.min(), pos.max()
    t = np.arange(max(lo, seq.WINDOW - 1), hi - seq.HORIZON + 1)
    return t


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    d, limit = load()
    train, cal, test = data.split(d)
    mu, sd = float(d["kw"][train].mean()), float(d["kw"][train].std())
    step, tgt, z = seq.arrays(d, mu, sd)
    o_tr, o_ca, o_te = (origin_positions(d, m) for m in (train, cal, test))
    s_tr, s_ca, s_te = (seq.samples(step, tgt, z, o) for o in (o_tr, o_ca, o_te))
    y_te = s_te[2].numpy() * sd + mu  # (n, 4) kW at t+1..t+4

    res = {"limit_kw": limit, "test_origins": int(len(o_te)), "models": {}}

    # gradient boosting, refitted on train only, scored on the same targets
    kw = d["kw"]
    gb = np.zeros_like(y_te)
    for h in range(1, seq.HORIZON + 1):
        X, y = features.build(d, h)
        tr = train[d.index.isin(X.index)]
        model = gbm().fit(X[tr], y[tr])
        targets = d.index[o_te + h]
        gb[:, h - 1] = model.predict(X.loc[targets])
    res["models"]["gradient_boosting"] = {f"h{h}": metrics.scores(y_te[:, h - 1], gb[:, h - 1], limit) for h in (1, 4)}
    res["models"]["persistence"] = {
        f"h{h}": metrics.scores(y_te[:, h - 1], kw.to_numpy()[o_te], limit) for h in (1, 4)
    }

    for name, make in (("lstm", seq.LSTMForecaster), ("transformer", seq.TransformerForecaster)):
        runs = []
        for s in SEEDS:
            torch.manual_seed(s)
            t0 = time.perf_counter()
            m, hist = seq.fit(make(), s_tr, s_ca, seed=s)
            p = seq.predict(m, s_te) * sd + mu
            runs.append({
                "seed": s,
                "epochs": len(hist),
                "train_seconds": round(time.perf_counter() - t0, 1),
                "params": sum(q.numel() for q in m.parameters()),
                **{f"h{h}": metrics.scores(y_te[:, h - 1], p[:, h - 1], limit) for h in (1, 4)},
            })
            print(f"{name} seed {s}: h1 MAE {runs[-1]['h1']['mae_kw']:.1f}  h4 MAE {runs[-1]['h4']['mae_kw']:.1f}  "
                  f"({runs[-1]['epochs']} epochs, {runs[-1]['train_seconds']:.0f} s)")
        res["models"][name] = {"runs": runs}

    (ROOT / "results/sequence.json").write_text(json.dumps(res, indent=2))
    for name in ("persistence", "gradient_boosting"):
        r = res["models"][name]
        print(f"{name:18s} h1 MAE {r['h1']['mae_kw']:.1f} peak MAE {r['h1']['peak_mae_kw']:.1f} | "
              f"h4 MAE {r['h4']['mae_kw']:.1f} peak MAE {r['h4']['peak_mae_kw']:.1f}")


if __name__ == "__main__":
    main()
