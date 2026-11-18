"""Prediction intervals and the peak-demand alert.

Intervals are split-conformal: fit the model on January-September, measure
its absolute errors on October, and widen every later forecast by the 90th
percentile of those errors. Errors at a 400 kW peak are far larger than at
night, so the percentile is taken separately per tariff band (Light, Medium,
Maximum). A band's interval width then matches how wrong the model tends to
be in that band.

The alert: at the end of block t, warn if any of the next four blocks
(the next hour) could exceed the demand limit, i.e. if its upper bound does.
"""

from __future__ import annotations

import numpy as np


def conformal_widths(resid_abs: np.ndarray, band: np.ndarray, level: float = 0.9) -> dict[int, float]:
    """Per band, the finite-sample conformal quantile of |error|."""
    out = {}
    for b in np.unique(band):
        r = np.sort(resid_abs[band == b])
        k = int(np.ceil((len(r) + 1) * level)) - 1
        out[int(b)] = float(r[min(k, len(r) - 1)])
    return out


def episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True as (start, end) index pairs, end inclusive."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def score_alerts(alert: np.ndarray, over: np.ndarray, horizon: int = 4, blocks_per_week: int = 672,
                 kw: np.ndarray | None = None, limit: float | None = None) -> dict:
    """alert[t]: warning issued at the end of block t, about blocks t+1..t+horizon.
    over[t]: block t was above the limit.

    An exceedance episode is caught if a warning was issued in the `horizon`
    blocks before it started. Its lead time is counted from the earliest such
    warning. A warning is false if none of the blocks it covers was above the
    limit; false warnings are counted as episodes, so one stretch of warnings
    counts once. Given kw and limit, it also reports the share of false
    warnings whose next hour still came within 10% of the limit (near misses).
    """
    n = len(over)
    ev = episodes(over)
    caught, lead = 0, []
    for s, _ in ev:
        w = [t for t in range(max(0, s - horizon), s) if alert[t]]
        if w:
            caught += 1
            lead.append((s - w[0]) * 15)
    covered = np.array([over[t + 1 : t + 1 + horizon].any() for t in range(n)])
    false = alert & ~covered
    weeks = n / blocks_per_week
    near = None
    if kw is not None and limit is not None:
        peak_next = np.array([kw[t + 1 : t + 1 + horizon].max() if t + 1 < n else 0.0 for t in range(n)])
        fe = [peak_next[s : e + 1].max() for s, e in episodes(false)]
        near = float(np.mean(np.array(fe) >= 0.9 * limit)) if fe else float("nan")
    return {
        "exceedance_episodes": len(ev),
        "caught": caught,
        "recall": caught / len(ev) if ev else float("nan"),
        "median_lead_min": float(np.median(lead)) if lead else 0.0,
        "false_alert_episodes": len(episodes(false)),
        "false_alerts_per_week": len(episodes(false)) / weeks,
        "alert_share": float(alert.mean()),
        "false_alerts_within_10pct_of_limit": near,
    }


def shaved_monthly_peak(kw: np.ndarray, month: np.ndarray, over: np.ndarray, caught_mask: np.ndarray,
                        limit: float) -> dict:
    """Monthly maximum demand if every caught exceedance were held at the limit
    (load deferred, not removed). Uncaught exceedances stay as they were."""
    shaved = np.where(over & caught_mask, np.minimum(kw, limit), kw)
    out = {}
    for m in np.unique(month):
        k = month == m
        out[str(m)] = {"actual_kw": float(kw[k].max()), "shaved_kw": float(shaved[k].max())}
    return out


def caught_blocks(alert: np.ndarray, over: np.ndarray, horizon: int = 4) -> np.ndarray:
    """True on the blocks of every exceedance episode that was warned about."""
    m = np.zeros(len(over), bool)
    for s, e in episodes(over):
        if alert[max(0, s - horizon) : s].any():
            m[s : e + 1] = True
    return m
