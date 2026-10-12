"""Ceiling parsing: the cases that a naive skyc1/skyl1 read gets wrong."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import weather


def _met(rows: list[dict]) -> pd.DataFrame:
    cols = ["skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_ceiling_is_the_lowest_broken_or_worse_layer() -> None:
    met = _met([
        {"skyc1": "FEW", "skyl1": 800, "skyc2": "BKN", "skyl2": 2500},
        {"skyc1": "OVC", "skyl1": 700},
        {"skyc1": "SCT", "skyl1": 1200, "skyc2": "SCT", "skyl2": 3000, "skyc3": "OVC",
         "skyl3": 4500},
        {"skyc1": "VV ", "skyl1": 100},
    ])
    got = weather._ceiling_ft(met).to_numpy()
    # FEW and SCT are not ceilings, however low they are
    assert got.tolist() == [2500.0, 700.0, 4500.0, 100.0]


def test_clear_and_scattered_reports_mean_unlimited_not_missing() -> None:
    met = _met([{"skyc1": "NSC"}, {"skyc1": "FEW", "skyl1": 3000}, {"skyc1": "M"},
                {"skyc1": None}])
    got = weather._ceiling_ft(met).to_numpy()
    assert got[0] == weather.UNLIMITED_FT
    assert got[1] == weather.UNLIMITED_FT
    assert np.isnan(got[2]) and np.isnan(got[3])


def test_flight_category_bands() -> None:
    vis = pd.Series([10.0, 4.0, 10.0, 2.0, 10.0, 0.5, 10.0, np.nan])
    cig = pd.Series([5000.0, 5000.0, 2000.0, 5000.0, 800.0, 5000.0, 300.0, np.nan])
    got = weather._flight_category(vis, cig).to_numpy()
    #            VFR MVFR MVFR IFR IFR LIFR LIFR unknown
    assert got.tolist() == [0, 1, 1, 2, 2, 3, 3, -1]


def test_lvp_uses_the_three_mile_or_thousand_foot_threshold() -> None:
    met = _met([{"skyc1": "NSC"}, {"skyc1": "NSC"}, {"skyc1": "OVC", "skyl1": 500.0},
                {"skyc1": "BKN", "skyl1": 1500.0}])
    vis = pd.Series([10.0, 2.0, 10.0, 5.0])
    ceiling = weather._ceiling_ft(met)
    lvp = ((vis < 3.0) | (ceiling < 1000.0)).astype("int8")
    # a 1500 ft ceiling in 5 miles is not LVP; 2 miles or a 500 ft ceiling is
    assert lvp.tolist() == [0, 1, 1, 0]
