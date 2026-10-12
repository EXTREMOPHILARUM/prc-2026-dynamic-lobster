"""Rome matched regime: a calibrated schedule-identity gate scoped to LIRF.

On LIRF, 21% of departures have BLOCK_TIME within 60s of SCHED_TIME, so the recorded target
equals MVT - SCHED. The global mechanism gate sees this regime diluted across ten airports;
this gate is fitted on LIRF rows only, adds leave-one-month-out fallback-rate encoders
(flight prefix, stand, operator, destination, aircraft type), and is calibrated
isotonically on held-out months so its output is a usable mixture weight:

    pred = p * max(MVT - SCHED, 0) + (1 - p) * pred_ordinary

Applied to LIRF rows with an NM record only; unmatched LIRF rows keep the band rule and
the NM-free expert. Measured -0.9s on Jan+Jul (all in July) against v17.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from prc import mechanism

AIRPORT = "LIRF"
TOLERANCE = 60.0
K = 30.0
CAL_MONTHS = (4, 10)
ROUNDS = 300
PARAMS = {**mechanism.GATE_PARAMS, "seed": 3}
CONTEXT_PREFIXES = ("nbr_apt", "q_apt", "arrctx_", "clk_")


def keys(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    s = lambda c: raw[c].astype("string").fillna("?")
    return {
        "flt_prefix": raw["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]+)")[0].str[:3].fillna("?").to_numpy(),
        "stand": s("STAND_mvt").to_numpy(),
        "stand_first": s("STAND_mvt").str[:1].to_numpy(),
        "operator": s("AIRCRAFT_OPERATOR_flt").to_numpy(),
        "op_stand": (s("AIRCRAFT_OPERATOR_flt") + "|" + s("STAND_mvt")).to_numpy(),
        "dest": s("ADES_mvt").to_numpy(),
        "actype": s("AIRCRAFT_TYPE_mvt").to_numpy(),
    }


def label(y: np.ndarray, gap: np.ndarray) -> np.ndarray:
    return np.abs(y - gap) <= TOLERANCE


def gate_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in mechanism.gate_features(X) if c != "ADEP_mvt"] + [c for c in X.columns if c.startswith(CONTEXT_PREFIXES)]


def scope(X: pd.DataFrame) -> np.ndarray:
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    return lirf & X["taxi_proxy_AOBT_3_flt"].notna().to_numpy()


def _encode(tables: dict[str, dict], kk: dict[str, np.ndarray], base_rate: float) -> pd.DataFrame:
    return pd.DataFrame({f"fbrate_{n}": pd.Series(kk[n]).map(tables[n]).fillna(base_rate).to_numpy() for n in tables})


def fit(raw: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, cats: list[str], month: np.ndarray) -> dict:
    """Fit encoders + gate on LIRF rows outside CAL_MONTHS, calibrate on CAL_MONTHS."""
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    lab = label(y, gap)
    fit_rows = lirf & ~np.isin(month, CAL_MONTHS); cal_rows = lirf & np.isin(month, CAL_MONTHS)
    kk = keys(raw); base_rate = float(lab[fit_rows].mean())
    tables = {}
    for n, key in kk.items():
        s = pd.Series(lab[fit_rows].astype(float)).groupby(key[fit_rows]).agg(["sum", "size"])
        tables[n] = ((s["sum"] + K * base_rate) / (s["size"] + K)).to_dict()
    cols = gate_columns(X); gcats = [c for c in cats if c in cols]
    def frame(rows):
        return pd.concat([X.loc[rows, cols].reset_index(drop=True), _encode(tables, {n: k[rows] for n, k in kk.items()}, base_rate)], axis=1)
    booster = lgb.train(PARAMS, lgb.Dataset(frame(fit_rows), lab[fit_rows].astype("int8"), categorical_feature=gcats), num_boost_round=ROUNDS)
    iso = IsotonicRegression(out_of_bounds="clip").fit(booster.predict(frame(cal_rows)), lab[cal_rows].astype(float))
    # refit encoders on all LIRF rows for scoring (more data), keep the gate and calibration
    all_tables = {}
    for n, key in kk.items():
        s = pd.Series(lab[lirf].astype(float)).groupby(key[lirf]).agg(["sum", "size"])
        all_tables[n] = ((s["sum"] + K * base_rate) / (s["size"] + K)).to_dict()
    return {"booster": booster, "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
            "tables": all_tables, "base_rate": base_rate, "columns": cols}


def save(model: dict, stem: Path) -> None:
    model["booster"].save_model(f"{stem}.gate.txt")
    Path(f"{stem}.json").write_text(json.dumps({k: v for k, v in model.items() if k != "booster"}))


def load(stem: Path) -> dict:
    meta = json.loads(Path(f"{stem}.json").read_text())
    meta["booster"] = lgb.Booster(model_file=f"{stem}.gate.txt")
    return meta


def probability(model: dict, raw: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    rows = scope(X); out = np.zeros(len(X))
    if not rows.any():
        return out
    kk = keys(raw)
    frame = pd.concat([X.loc[rows, model["columns"]].reset_index(drop=True),
                       _encode(model["tables"], {n: k[rows] for n, k in kk.items()}, model["base_rate"])], axis=1)
    p = model["booster"].predict(frame)
    out[rows] = np.interp(p, model["iso_x"], model["iso_y"])
    return out


def apply(pred: np.ndarray, p: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    out = np.asarray(pred, dtype="float64").copy(); rows = scope(X)
    gap = np.clip(X[mechanism.PROXY_COL].to_numpy(dtype="float64"), 0, None)
    out[rows] = p[rows] * gap[rows] + (1 - p[rows]) * out[rows]
    return out
