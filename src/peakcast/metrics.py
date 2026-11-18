"""Error measures.

MAPE is not used: at night this plant draws 10-20 kW, so a 10 kW miss is a
50-100% error while a 10 kW miss at a 400 kW peak is 2.5%. MAPE would be
dominated by the hours that do not matter for a demand charge. WAPE (total
absolute error over total demand) and MAE on the peak blocks are reported
instead.
"""

from __future__ import annotations

import numpy as np


def scores(y: np.ndarray, p: np.ndarray, limit: float) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    peak = y > limit
    return {
        "mae_kw": float(np.abs(e).mean()),
        "rmse_kw": float(np.sqrt((e**2).mean())),
        "wape": float(np.abs(e).sum() / np.abs(y).sum()),
        "r2": float(1 - (e**2).sum() / ((y - y.mean()) ** 2).sum()),
        "peak_mae_kw": float(np.abs(e[peak]).mean()) if peak.any() else float("nan"),
        "peak_bias_kw": float(e[peak].mean()) if peak.any() else float("nan"),
    }
