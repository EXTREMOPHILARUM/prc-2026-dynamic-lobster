"""Second-stage stacker on matched rows.

A small LightGBM fitted on the leave-one-month-out predictions of the chain's components
(pre-member prediction, CatBoost member, blend, ordinary base, gate probability, LOBT expert,
four clock proxies, airport), target = residual of the blended prediction. Applied to matched
rows only, before the LOBT clip. Measured -0.8s on Jan+Jul, 12/12 months, drop-50 robust
(experiment-log section 26, `scripts/experiment_stacker.py --mid`).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PROXIES = ["taxi_proxy_LOBT_flt", "taxi_proxy_EOBT_1_flt", "taxi_proxy_IOBT_flt", "taxi_proxy_SCHED_TIME_UTC_mvt"]
FEATURES = ["ref", "base", "cat", *PROXIES, "apt", "ordinary", "gate", "lobt_x"]
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7}
ROUNDS = 800


def airport_codes(airport: np.ndarray) -> np.ndarray:
    """Integer codes in the fixed AIRPORTS order (matches `astype('category').cat.codes` on the ten airports)."""
    return pd.Categorical(pd.Series(airport).astype("string"), categories=AIRPORTS).codes.astype("int64")


def frame(ref: np.ndarray, base: np.ndarray, cat: np.ndarray, ordinary: np.ndarray, gate: np.ndarray, lobt_x: np.ndarray,
          X: pd.DataFrame) -> pd.DataFrame:
    F = pd.DataFrame({"ref": ref, "base": base, "cat": cat}, index=X.index)
    for c in PROXIES:
        F[c] = X[c].to_numpy(dtype="float64")
    F["apt"] = airport_codes(X["ADEP_mvt"].to_numpy())
    F["ordinary"] = ordinary; F["gate"] = gate; F["lobt_x"] = np.nan_to_num(np.asarray(lobt_x, dtype="float64"), nan=0.0)
    return F[FEATURES]


def fit(F: pd.DataFrame, resid: np.ndarray, rounds: int = ROUNDS, threads: int = 8) -> lgb.Booster:
    return lgb.train({**PARAMS, "num_threads": threads}, lgb.Dataset(F[FEATURES], resid, categorical_feature=["apt"]), num_boost_round=rounds)


def save(model: lgb.Booster, stem: Path, extra: dict | None = None) -> None:
    model.save_model(f"{stem}.txt")
    Path(f"{stem}.json").write_text(json.dumps({"features": FEATURES, "airports": AIRPORTS, "rounds": model.num_trees(), **(extra or {})}, indent=2))


def load(stem: Path) -> tuple[lgb.Booster, dict]:
    return lgb.Booster(model_file=f"{stem}.txt"), json.loads(Path(f"{stem}.json").read_text())


def apply(pred: np.ndarray, model: lgb.Booster, F: pd.DataFrame, matched: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Add `weight` x the stacker's residual correction on matched rows. OOF: weight 0.5 keeps -0.70 of
    the -0.82 at weight 1.0; shipped at 0.5 because the live components are stronger than the OOF ones
    the stacker was fitted on and its live shifts run ~30% wider."""
    out = np.asarray(pred, dtype="float64").copy()
    if matched.any():
        out[matched] = out[matched] + weight * model.predict(F.loc[matched, FEATURES])
    return out
