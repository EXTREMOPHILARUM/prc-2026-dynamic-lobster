"""Mixture of experts for LIRF no-record rows under the band scope.

A calibrated copy classifier P(|y - (MVT - SCHED)| <= 60s) and a copy-free regressor
(y < 12000, non-copy rows), both fitted on every LIRF departure with NM-free features and
the stored target encodings, unmatched rows weighted 8x:

    pred = P * max(MVT - SCHED, 0) + (1 - P) * regressor

Replaces the single NM-free expert (v16) on `lirfbands.short_scope` rows. Leave-one-month-out:
lane 2,828 -> 2,320, Jan+Jul 314.43 -> 310.15, 11/12 months
(scripts/experiment_lirf_unmatched_moe.py). Form from elegant-alligator's missing-row MoE.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from prc import decomposition, lirfbands, mechanism, tencode

NM_DEPENDENT_PREFIXES = ("aobt_minus_", "taxi_proxy_AOBT", "taxi_proxy_EOBT", "taxi_proxy_IOBT", "taxi_proxy_LOBT", "arr_next_20m_aobt",
                         "arr_during_taxi", "q_apt_at_aobt", "q_rwy_at_aobt", "nbr_apt_own", "clk_AOBT", "clk_LOBT", "clk_IOBT", "clk_EOBT",
                         "clk_ARVT", "clk_mvt_minus_ARVT", "clk_lobt", "clk_arvt", "clk_sched_minus_aobt", "clk_eobt", "clk_iobt", "ord_queue_eobt", "ord_later")
NM_DEPENDENT = {"offblock_delay", "nm_time_count", "aobt_missing", "nm_record_missing", "ADES_flt"}
TE_EXCLUDED = ("te_operator", "te_segment", "te_wake", "te_ftype")
COPY_TOL = 60.0
REG_CAP = 12000.0
WEIGHT = 8.0
CAL_MONTHS = (4, 10)
CLF_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8, "verbosity": -1, "seed": 3}
CLF_ROUNDS = 300
REG_ROUNDS = 400


def columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.endswith("_flt") and c not in NM_DEPENDENT
            and not c.startswith(NM_DEPENDENT_PREFIXES) and c not in decomposition.BASE_EXCLUDED]


def frame(raw: pd.DataFrame, X: pd.DataFrame, cols: list[str], enc: dict) -> pd.DataFrame:
    te = tencode.transform(raw, enc); te.index = X.index
    return X[cols].join(te[[c for c in te.columns if not c.startswith(TE_EXCLUDED)]])


def target_rows(X: pd.DataFrame, max_gap_h: float | None) -> np.ndarray:
    """Rows the mixture serves: `lirfbands.short_scope`, or LIRF no-record rows with gap <= max_gap_h."""
    if max_gap_h is None:
        return lirfbands.short_scope(X)
    lirf = (X["ADEP_mvt"].astype("string") == lirfbands.AIRPORT).to_numpy(); missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    return lirf & missing & (X[mechanism.PROXY_COL].to_numpy(dtype="float64") <= max_gap_h * 3600.0)


CAT_PARAMS = {"loss_function": "RMSE", "iterations": 3000, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 5, "border_count": 128,
              "one_hot_max_size": 20, "max_ctr_complexity": 1, "random_seed": 42, "verbose": 0}


def _cat_frame(F: pd.DataFrame, vcats: list[str]) -> pd.DataFrame:
    Fc = F.copy()
    for c in vcats:
        Fc[c] = Fc[c].astype("string").fillna("__NA__").astype(str)
    return Fc


def fit(raw: pd.DataFrame, X: pd.DataFrame, cats: list[str], y: np.ndarray, month: np.ndarray, enc: dict, max_gap_h: float | None = None,
        regressor: str = "lightgbm", task_type: str = "GPU") -> dict:
    lirf = (X["ADEP_mvt"].astype("string") == lirfbands.AIRPORT).to_numpy(); missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); copy = np.abs(y - gap) <= COPY_TOL
    cols = columns(X); vcats = [c for c in cats if c in cols]; F = frame(raw, X, cols, enc); w = np.where(missing, WEIGHT, 1.0)
    fit_rows = lirf & ~np.isin(month, CAL_MONTHS); cal_rows = lirf & np.isin(month, CAL_MONTHS)
    clf = lgb.train(CLF_PARAMS, lgb.Dataset(F.loc[fit_rows], copy[fit_rows].astype("int8"), weight=w[fit_rows], categorical_feature=vcats), num_boost_round=CLF_ROUNDS)
    iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict(F.loc[cal_rows]), copy[cal_rows].astype(float))
    rr = lirf & ~copy & (y < REG_CAP)
    if regressor == "catboost":
        from catboost import CatBoostRegressor, Pool
        reg = CatBoostRegressor(**CAT_PARAMS, task_type=task_type)
        reg.fit(Pool(_cat_frame(F.loc[rr], vcats), y[rr], weight=w[rr], cat_features=vcats))
    else:
        reg = lgb.train({**decomposition.BASE_PARAMS, "num_leaves": 63, "seed": 5}, lgb.Dataset(F.loc[rr], y[rr], weight=w[rr], categorical_feature=vcats), num_boost_round=REG_ROUNDS)
    return {"clf": clf, "reg": reg, "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(), "columns": cols, "max_gap_h": max_gap_h,
            "regressor": regressor, "cats": vcats}


def save(model: dict, stem: Path) -> None:
    model["clf"].save_model(f"{stem}.clf.txt")
    if model.get("regressor") == "catboost":
        model["reg"].save_model(f"{stem}.reg.cbm")
    else:
        model["reg"].save_model(f"{stem}.reg.txt")
    Path(f"{stem}.json").write_text(json.dumps({k: v for k, v in model.items() if k not in ("clf", "reg")}))


def load(stem: Path) -> dict:
    meta = json.loads(Path(f"{stem}.json").read_text())
    meta["clf"] = lgb.Booster(model_file=f"{stem}.clf.txt")
    if meta.get("regressor") == "catboost":
        from catboost import CatBoostRegressor
        reg = CatBoostRegressor(); reg.load_model(f"{stem}.reg.cbm"); meta["reg"] = reg
    else:
        meta["reg"] = lgb.Booster(model_file=f"{stem}.reg.txt")
    return meta


def apply(pred: np.ndarray, model: dict, raw: pd.DataFrame, X: pd.DataFrame, enc: dict) -> np.ndarray:
    out = np.asarray(pred, dtype="float64").copy(); rows = target_rows(X, model.get("max_gap_h"))
    if not rows.any():
        return out
    F = frame(raw.loc[rows] if len(raw) == len(X) else raw, X.loc[rows], model["columns"], enc)
    p = np.interp(model["clf"].predict(F[model["clf"].feature_name()]), model["iso_x"], model["iso_y"])
    if model.get("regressor") == "catboost":
        from catboost import Pool
        reg = np.clip(model["reg"].predict(Pool(_cat_frame(F, model["cats"]), cat_features=model["cats"])), 0, None)
    else:
        reg = np.clip(model["reg"].predict(F[model["reg"].feature_name()]), 0, None)
    gap = np.clip(X.loc[rows, mechanism.PROXY_COL].to_numpy(dtype="float64"), 0, None)
    out[rows] = p * gap + (1 - p) * reg
    return out
