"""One test week as a picture: demand, the one-hour-ahead 90% upper bound, the
limit, and when the alert was on. The week shown is the test week with the
most limit crossings, so the chart is chosen by a rule, not by eye.

Usage: python scripts/plot_week.py   (writes docs/week.png)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from peakcast import alerts, data  # noqa: E402
from run_alerts import HORIZONS, forecasts  # noqa: E402
from run_baselines import load  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES, CRITICAL, SURFACE = "#2a78d6", "#d03b3b", "#fcfcfb"


def main():
    d, limit = load()
    fc = forecasts(d)
    origins = fc[1]["p"].index
    for h in HORIZONS:
        origins = origins.intersection(fc[h]["p"].index)
    # upper bound for the block one hour ahead, and the alert (any of the next four)
    upper_h = {}
    alert = np.zeros(len(origins), bool)
    for h in HORIZONS:
        w = alerts.conformal_widths(fc[h]["cal_res"], fc[h]["cal_band"], 0.9)
        p, band = fc[h]["p"].reindex(origins), fc[h]["band"].reindex(origins)
        u = p.to_numpy() + band.map(w).to_numpy()
        upper_h[h] = u
        alert |= u > limit
    kw = d["kw"].reindex(origins)
    over = kw.to_numpy() > limit

    # the 7-day window (starting at midnight) with the most crossing episodes
    days = np.unique(origins.normalize())
    best, best_n = None, -1
    for start in days[:-7]:
        m = (origins >= start) & (origins < start + np.timedelta64(7, "D"))
        n = len(alerts.episodes(over[m]))
        if n > best_n:
            best, best_n = start, n
    m = (origins >= best) & (origins < best + np.timedelta64(7, "D"))
    t = origins[m]

    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s, e in alerts.episodes(alert[m]):
        ax.axvspan(t[s], t[e] + data.BLOCK, color=CRITICAL, alpha=0.10, lw=0)
    # the bound is plotted at the block it is about (one hour after it was issued)
    ax.plot(t + 4 * data.BLOCK, upper_h[4][m], color=MUTED, lw=1.0, label="90% upper bound, issued 1 h earlier")
    ax.plot(t, kw[m], color=SERIES, lw=1.6, label="Demand (15-min average)")
    ax.axhline(limit, color=INK2, lw=1.0, ls=(0, (4, 3)))
    ax.text(t[0], limit + 8, f"limit {limit:.0f} kW", ha="left", va="bottom", color=INK2, fontsize=9)
    ax.set_ylabel("kW", color=MUTED, fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=CRITICAL, alpha=0.10))
    labels.append("Alert on (a crossing is possible within the hour)")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9, ncol=3, labelcolor=INK2)
    ax.set_ylim(0, max(kw[m].max(), upper_h[4][m].max()) * 1.15)
    ax.set_title(f"Test week with the most limit crossings ({best_n}): {np.datetime_as_string(best, 'D')}",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs/week.png", facecolor=SURFACE)
    print("week starting", best, "crossings", best_n)


if __name__ == "__main__":
    main()
