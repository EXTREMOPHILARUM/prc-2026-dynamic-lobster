"""Rome matched regime: calibrated schedule-identity gate with LIRF-scoped rate encoders.

On LIRF rows the current pipeline blends P(schedule copy) from the global gate. kind-mango
(-9.8 then -2.8 live) fits the gate on LIRF rows only, adds leave-one-month-out fallback-rate
encoders keyed on flight prefix / stand / operator / destination, and calibrates it
isotonically on non-scored months. Here, per fold: encoders and gate fitted on the other
eleven months (calibration on months 4 and 10 of those), mixture
`p * (MVT - SCHED) + (1 - p) * ordinary` replaces the prediction on LIRF matched rows only.
LIRF unmatched rows keep their v15/v16 handling.

    uv run python -u scripts/experiment_lirf_regime.py --baseline v17_oof_predictions.parquet
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from prc import arrivals, config, decomposition, evaluation, lirfbands, matrix, mechanism

K = 30.0
CAL_MONTHS = (4, 10)
GATE_PARAMS = {**mechanism.GATE_PARAMS, "seed": 3}
GATE_ROUNDS = 300
ENC_KEYS = {
    "flt_prefix": lambda r: r["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]+)")[0].str[:3].fillna("?"),
    "stand": lambda r: r["STAND_mvt"].astype("string").fillna("?"),
    "stand_first": lambda r: r["STAND_mvt"].astype("string").str[:1].fillna("?"),
    "operator": lambda r: r["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("?"),
    "op_stand": lambda r: r["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("?") + "|" + r["STAND_mvt"].astype("string").fillna("?"),
    "dest": lambda r: r["ADES_mvt"].astype("string").fillna("?"),
    "actype": lambda r: r["AIRCRAFT_TYPE_mvt"].astype("string").fillna("?"),
}


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def encode(keys: dict[str, np.ndarray], label: np.ndarray, fit_rows: np.ndarray, apply_rows: np.ndarray, base_rate: float):
    out = {}
    for name, key in keys.items():
        s = pd.Series(label[fit_rows].astype(float)).groupby(key[fit_rows]).agg(["sum", "size"])
        rate = (s["sum"] + K * base_rate) / (s["size"] + K)
        out[f"fbrate_{name}"] = rate.reindex(key[apply_rows]).fillna(base_rate).to_numpy()
    return pd.DataFrame(out, index=np.flatnonzero(apply_rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="v17_oof_predictions.parquet")
    ap.add_argument("--ordinary", default=None, help="OOF parquet with an 'ordinary' column; default rebuilds nothing and uses baseline as ordinary")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="lirf_regime")
    ap.add_argument("--disruption", action="store_true", help="add 60-min airport disruption meters (mean schedule gap, null-FLIGHT share) to the gate")
    args = ap.parse_args()
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    lirf = (X["ADEP_mvt"].astype("string") == "LIRF").to_numpy()
    matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
    sd = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    label = (np.abs(y - sd) <= 60.0)
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.baseline), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    # the 'ordinary' component for the mixture: undo nothing, use baseline prediction as the normal-regime estimate
    ordinary = base
    keys = {k: f(raw).to_numpy() for k, f in ENC_KEYS.items()}
    gate_cols = [c for c in mechanism.gate_features(X) if c != "ADEP_mvt"] + [c for c in X.columns if c.startswith(("nbr_apt", "q_apt", "arrctx_", "clk_"))]
    gate_cats = [c for c in cats if c in gate_cols]
    if args.disruption:
        t = (pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True) - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        sdc = np.clip(sd, -1800, 14400); fnull = raw["FLIGHT_ID_mvt"].isna().to_numpy().astype(float)
        dis_sd = np.full(len(y), np.nan); dis_fn = np.full(len(y), np.nan)
        for idx in pd.Series(np.arange(len(y))).groupby(X["ADEP_mvt"].astype("string").to_numpy()).indices.values():
            idx = np.asarray(idx); o = np.argsort(t[idx]); idx = idx[o]; tt = t[idx]
            cs1 = np.concatenate([[0.0], np.cumsum(sdc[idx])]); cs2 = np.concatenate([[0.0], np.cumsum(fnull[idx])])
            hi = np.searchsorted(tt, tt, side="left"); lo = np.searchsorted(tt, tt - 3600.0, side="left"); n = hi - lo
            dis_sd[idx] = np.where(n > 0, (cs1[hi] - cs1[lo]) / np.maximum(n, 1), np.nan); dis_fn[idx] = np.where(n > 0, (cs2[hi] - cs2[lo]) / np.maximum(n, 1), np.nan)
        X = X.assign(dis_sd_mean_60m=dis_sd, dis_fnull_60m=dis_fn); gate_cols = gate_cols + ["dis_sd_mean_60m", "dis_fnull_60m"]
    target_rows = lirf & matched
    p_out = np.full(len(y), np.nan)
    print(f"LIRF rows {lirf.sum():,}, target (matched) {target_rows.sum():,}, base rate {label[lirf].mean():.3f}", flush=True)
    for m in range(1, 13):
        tr = lirf & (month != m); va = lirf & (month == m)
        fit_rows = tr & ~np.isin(month, CAL_MONTHS); cal_rows = tr & np.isin(month, CAL_MONTHS)
        base_rate = float(label[fit_rows].mean())
        enc_fit = encode(keys, label, fit_rows, fit_rows, base_rate)     # in-sample encoders for fit rows (smoothed)
        enc_cal = encode(keys, label, fit_rows, cal_rows, base_rate)
        enc_va = encode(keys, label, tr, va, base_rate)
        Xf = pd.concat([X.loc[fit_rows, gate_cols].reset_index(drop=True), enc_fit.reset_index(drop=True)], axis=1)
        Xc = pd.concat([X.loc[cal_rows, gate_cols].reset_index(drop=True), enc_cal.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X.loc[va, gate_cols].reset_index(drop=True), enc_va.reset_index(drop=True)], axis=1)
        booster = lgb.train({**GATE_PARAMS, "num_threads": args.threads},
                            lgb.Dataset(Xf, label[fit_rows].astype("int8"), categorical_feature=gate_cats), num_boost_round=GATE_ROUNDS)
        iso = IsotonicRegression(out_of_bounds="clip").fit(booster.predict(Xc), label[cal_rows].astype(float))
        p_out[va] = iso.predict(booster.predict(Xv))
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    new = base.copy()
    new[target_rows] = p_out[target_rows] * np.clip(sd[target_rows], 0, None) + (1 - p_out[target_rows]) * ordinary[target_rows]
    new = np.clip(lirfbands.lobt_clip(new, X), 0, None)
    scored = np.isin(month, (1, 7))
    rep = {"tag": args.tag}
    for name, mask in [("scored_jan_jul", scored), ("jan", month == 1), ("jul", month == 7), ("all_months", np.ones(len(y), bool)),
                       ("lirf_matched_scored", target_rows & scored), ("lirf_matched_all", target_rows)]:
        rep[name] = {"base": rmse(y[mask], base[mask]), "new": rmse(y[mask], new[mask])}; rep[name]["delta"] = rep[name]["new"] - rep[name]["base"]
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], base[month == m]) for m in range(1, 13)}
    ys_, bs, ns = y[scored], base[scored], new[scored]; order = np.argsort((ys_ - ns) ** 2 - (ys_ - bs) ** 2)
    rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys_), bool); keep[order[:k]] = False
        rep["robustness_scored"][f"drop_{k}"] = rmse(ys_[keep], ns[keep]) - rmse(ys_[keep], bs[keep])
    from sklearn.metrics import roc_auc_score
    ok = lirf & np.isfinite(p_out); rep["gate_auc_oof"] = float(roc_auc_score(label[ok], p_out[ok]))
    rep["gate_mass"] = {"predicted": float(p_out[ok].sum()), "actual": int(label[ok].sum())}
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": new, "p_identity": p_out}).to_parquet(config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all_months", "lirf_matched_scored", "lirf_matched_all"):
        r = rep[k]; print(f"{k:20s} {r['base']:.3f} -> {r['new']:.3f} ({r['delta']:+.3f})")
    print("gate AUC", round(rep["gate_auc_oof"], 4), "mass", rep["gate_mass"])
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()}); print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


if __name__ == "__main__":
    main()
