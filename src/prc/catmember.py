"""CatBoost residual member with native identity categoricals, blended on matched rows.

Fitted on the GPU box (scripts/fit_catmember.py); scored on CPU here. Target is the full
residual `y - clip(MVT - AOBT, 0, 3600)` on matched rows; identifiers (flight number,
callsign, airport x stand, stand x runway, airport x operator, runway x type, prefixes) enter
as CatBoost categoricals with ordered target statistics, which is the mechanism LightGBM's
native categorical split lacks. Blended into the final prediction on matched rows:
`pred + w * (member - pred)`, with `w` fitted on the ten non-scored months' OOF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc import decomposition, tencode

INTERACTIONS = {
    "airport_stand": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["STAND_mvt"].astype("string"),
    "stand_runway": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["STAND_mvt"].astype("string") + "_" + r["RUNWAY_mvt"].astype("string"),
    "airport_operator": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["AIRCRAFT_OPERATOR_flt"].astype("string"),
    "runway_aircraft": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["RUNWAY_mvt"].astype("string") + "_" + r["AIRCRAFT_TYPE_mvt"].astype("string"),
    "flight_prefix": lambda r: r["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]+)")[0],
    "callsign_prefix": lambda r: r["CALLSIGN_flt"].astype("string").str[:3],
}
PARAMS = {"loss_function": "RMSE", "iterations": 5000, "depth": 8, "learning_rate": 0.06, "l2_leaf_reg": 5.0,
          "border_count": 128, "one_hot_max_size": 20, "max_ctr_complexity": 1, "random_seed": 42, "verbose": 0}


def frame(raw: pd.DataFrame, X: pd.DataFrame, cats: list[str], enc: dict) -> tuple[pd.DataFrame, list[str]]:
    # positional assignment throughout: `raw` and `X` share row order but not always index
    F = X.copy()
    for name, f in INTERACTIONS.items():
        F[name] = f(raw).fillna("__NA__").astype(str).to_numpy()
    for c in cats:
        if c in F.columns:
            F[c] = F[c].astype("string").fillna("__NA__").astype(str).to_numpy()
    te = tencode.transform(raw, enc); te.index = F.index
    F = F.join(te)
    return F, [c for c in cats if c in F.columns] + list(INTERACTIONS)


CLOCK_COLS = {"lobt": "taxi_proxy_LOBT_flt", "eobt": "taxi_proxy_EOBT_1_flt", "iobt": "taxi_proxy_IOBT_flt"}
CLOCK_RANGE = (-3600.0, 7200.0)


def anchor_and_rows(X: pd.DataFrame, anchor: str = "aobt") -> tuple[np.ndarray, np.ndarray]:
    """Clipped reference clock and the rows the member serves. The alternative clocks (LOBT, EOBT,
    IOBT) give a member whose errors differ from the AOBT member's; each is served on matched rows
    that also carry that clock (experiment-log sections 27-28)."""
    aobt = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(aobt)
    if anchor in CLOCK_COLS:
        clock = X[CLOCK_COLS[anchor]].to_numpy(dtype="float64")
        return np.clip(clock, *CLOCK_RANGE), matched & np.isfinite(clock)
    return np.clip(aobt, 0, decomposition.TAIL_THRESHOLD), matched


def fit(raw: pd.DataFrame, X: pd.DataFrame, cats: list[str], y: np.ndarray, enc: dict, task_type: str = "GPU", anchor: str = "aobt") -> CatBoostRegressor:
    anchor_clipped, matched = anchor_and_rows(X, anchor)
    resid = y - anchor_clipped
    F, cat_cols = frame(raw.loc[matched], X.loc[matched], cats, enc)
    model = CatBoostRegressor(**PARAMS, task_type=task_type)
    model.fit(Pool(F, resid[matched], cat_features=cat_cols))
    return model


def predict(model: CatBoostRegressor, raw: pd.DataFrame, X: pd.DataFrame, cats: list[str], enc: dict, anchor: str = "aobt") -> np.ndarray:
    """Member prediction on the rows it serves, NaN elsewhere."""
    out = np.full(len(X), np.nan)
    anchor_clipped, matched = anchor_and_rows(X, anchor)
    if matched.any():
        F, cat_cols = frame(raw.loc[matched], X.loc[matched], cats, enc)
        out[matched] = np.clip(anchor_clipped[matched] + model.predict(Pool(F, cat_features=cat_cols)), 0, None)
    return out


def apply(pred: np.ndarray, member: np.ndarray, weight: float) -> np.ndarray:
    out = np.asarray(pred, dtype="float64").copy(); ok = np.isfinite(member)
    out[ok] = out[ok] + weight * (member[ok] - out[ok])
    return out
