from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peakcast import alerts, data, features, metrics

RAW = Path(__file__).resolve().parents[1] / "data/raw/Steel_industry_data.csv"
needs_data = pytest.mark.skipif(not RAW.exists(), reason="UCI data not downloaded")


def _synthetic(days=20, seed=0):
    """Regular 15-minute grid with a daily shape, in the loader's format."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01 00:15", periods=days * data.PER_DAY, freq="15min")
    blk = np.arange(len(idx)) % data.PER_DAY
    kw = 50 + 300 * ((blk >= 36) & (blk < 84)) + rng.normal(0, 10, len(idx))
    band = np.where((blk >= 36) & (blk < 84), "Medium_Load", "Light_Load")
    return pd.DataFrame({"kw": kw, "kwh": kw / 4, "load_type": band}, index=idx)


@pytest.mark.parametrize("h", [1, 4])
def test_features_never_see_the_block_being_forecast(h):
    """Changing demand after the forecast origin must not change any feature."""
    d = _synthetic()
    X, _ = features.build(d, h)
    target = X.index[500]
    origin = target - h * data.BLOCK
    d2 = d.copy()
    d2.loc[d2.index > origin, "kw"] += 1000.0
    X2, _ = features.build(d2, h)
    assert np.allclose(X.loc[target].to_numpy(float), X2.loc[target].to_numpy(float))


def test_conformal_width_is_the_finite_sample_quantile():
    res = np.arange(1, 101, dtype=float)  # |errors| 1..100
    w = alerts.conformal_widths(res, np.zeros(100, int), level=0.9)
    assert w[0] == 91.0  # ceil(101 * 0.9) = 91st smallest


def test_alert_counts_a_caught_peak_and_a_false_alarm():
    over = np.zeros(40, bool)
    over[20:23] = True  # one exceedance, blocks 20-22
    alert = np.zeros(40, bool)
    alert[17] = True  # 3 blocks (45 min) before it
    alert[5] = True  # nothing follows within the hour: false
    s = alerts.score_alerts(alert, over)
    assert s["caught"] == 1 and s["median_lead_min"] == 45
    assert s["false_alert_episodes"] == 1


def test_shaving_only_touches_caught_episodes():
    kw = np.array([100, 500, 100, 100, 100, 100, 600, 100], float)
    over = kw > 400
    alert = np.zeros(8, bool)
    alert[0] = True  # covers blocks 1-4: the 500 kW block, not the 600 kW one
    caught = alerts.caught_blocks(alert, over)
    out = alerts.shaved_monthly_peak(kw, np.ones(8, int), over, caught, 400)
    assert caught.tolist() == [False, True] + [False] * 6
    assert out["1"] == {"actual_kw": 600.0, "shaved_kw": 600.0}


def test_peak_scores_only_look_at_blocks_above_the_limit():
    s = metrics.scores(np.array([10, 500.0]), np.array([20, 450.0]), limit=400)
    assert s["peak_mae_kw"] == 50 and s["peak_bias_kw"] == -50


@needs_data
def test_midnight_blocks_are_moved_to_the_end_of_their_day():
    d = data.load(RAW)
    assert d.index[0] == pd.Timestamp("2018-01-01 00:15")
    assert d.index[-1] == pd.Timestamp("2019-01-01 00:00")
    assert len(d) == 365 * data.PER_DAY
    # the raw row "01/01/2018 00:00" (3.42 kWh) now closes 1 January
    assert d.loc["2018-01-02 00:00", "kwh"] == pytest.approx(3.42)


@needs_data
def test_co2_is_a_rescaled_copy_of_the_target():
    d = data.load(RAW)
    busy = d["kwh"] > 20
    ratio = (d["co2"][busy] / d["kwh"][busy]).median()
    assert ratio == pytest.approx(0.00046, rel=0.02)
    assert np.corrcoef(d["co2"], d["kwh"])[0, 1] > 0.98


@needs_data
def test_every_day_follows_one_of_three_tariff_patterns():
    d = data.load(RAW)
    start = d.index - data.BLOCK
    code = d["load_type"].map({"Light_Load": "L", "Medium_Load": "M", "Maximum_Load": "X"})
    day = code.groupby(start.date).agg("".join)
    assert day.nunique() == 3
