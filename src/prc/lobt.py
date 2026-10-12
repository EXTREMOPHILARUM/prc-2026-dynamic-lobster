"""Second anchor: a residual expert on NM's last planned off-block time.

`MVT - LOBT_flt` bounds the target within +/-3606s on every training row, and on 5% of
rows equals it exactly (the airport clocked off-block at the last plan). A LightGBM fitted
on `y - (MVT - LOBT)` therefore sees a different residual than the AOBT-anchored base, and
blending the two predictions per airport reduces error where the clocks disagree
(elegant-alligator V5/V6: -2.2s local, -2.0s official). Blend weights are fitted on
non-scored months only, shrunk toward the global weight.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import decomposition

ANCHOR_COL = "taxi_proxy_LOBT_flt"
BOUND = 3606.0
PRIOR_ROWS = 1000.0
PARAMS = {**decomposition.BASE_PARAMS, "learning_rate": 0.05, "num_leaves": 255, "seed": 7}
ROUNDS = 400
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def scope(X: pd.DataFrame) -> np.ndarray:
    lobt = X[ANCHOR_COL].to_numpy(dtype="float64")
    aobt = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    return np.isfinite(lobt) & np.isfinite(aobt)


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]


def fit(X: pd.DataFrame, y: np.ndarray, cats: list[str], rounds: int = ROUNDS) -> lgb.Booster:
    rows = scope(X)
    cols = feature_columns(X)
    anchor = X.loc[rows, ANCHOR_COL].to_numpy(dtype="float64")
    target = np.clip(y[rows] - anchor, -BOUND, BOUND)
    return lgb.train(PARAMS, lgb.Dataset(X.loc[rows, cols], target,
                                         categorical_feature=[c for c in cats if c in cols]),
                     num_boost_round=rounds)


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Expert prediction on scope rows, NaN elsewhere."""
    out = np.full(len(X), np.nan)
    rows = scope(X)
    if rows.any():
        anchor = X.loc[rows, ANCHOR_COL].to_numpy(dtype="float64")
        out[rows] = np.clip(anchor + booster.predict(X.loc[rows, booster.feature_name()]), 0, None)
    return out


def fit_alphas(y: np.ndarray, base: np.ndarray, expert: np.ndarray, airport: np.ndarray,
               rows: np.ndarray) -> dict[str, float]:
    """Per-airport blend weight, ridge-shrunk toward the global least-squares weight."""
    d = expert[rows] - base[rows]; r = y[rows] - base[rows]; ap = airport[rows]
    ok = np.isfinite(d)
    g = float(np.clip((d[ok] @ r[ok]) / max(d[ok] @ d[ok], 1e-9), 0, 1))
    mean_d2 = float(np.mean(d[ok] ** 2))
    alphas = {"__global__": g}
    for a in AIRPORTS:
        m = ok & (ap == a)
        alphas[a] = float(np.clip((d[m] @ r[m] + PRIOR_ROWS * mean_d2 * g) /
                                  (d[m] @ d[m] + PRIOR_ROWS * mean_d2), 0, 1)) if m.any() else g
    return alphas


def apply(base: np.ndarray, expert: np.ndarray, airport: np.ndarray, alphas: dict[str, float]) -> np.ndarray:
    out = np.asarray(base, dtype="float64").copy()
    ok = np.isfinite(expert)
    a = np.array([alphas.get(x, alphas["__global__"]) for x in airport])
    out[ok] = base[ok] + a[ok] * (expert[ok] - base[ok])
    return out
