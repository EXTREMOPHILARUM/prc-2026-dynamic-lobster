"""Two-stage gate: regress d = BLOCK - SCHED, then classify P(|d| <= 60) using the estimate.

The fallback event is |BLOCK - SCHED| <= 60s. BLOCK - SCHED decomposes as
(AOBT - SCHED), which is observed, plus (BLOCK - AOBT), which is exactly what the
ordinary model already estimates. So a point estimate of d is available, and the gate's
job collapses to "how likely is d to land inside a 120s window around zero given this
estimate". Feeding the estimate in explicitly saves the classifier from rediscovering
that arithmetic separately at every value of AOBT - SCHED.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from prc import config, evaluation, matrix, mechanism


def add_candidates(raw, X):
    """Extra gate features. Every input survives in the ranking set."""
    T = lambda c: pd.to_datetime(raw[c], utc=True, errors="coerce")
    out = {}
    sch = T("SCHED_TIME_UTC_mvt")
    # signed offsets of each NM clock from the scheduled time
    for c in ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]:
        v = (T(c) - sch).dt.total_seconds()
        out[f"sched_gap_{c}"] = v.to_numpy(dtype="float32")
        out[f"abs_sched_gap_{c}"] = v.abs().to_numpy(dtype="float32")
    # timestamp resolution: a minute-grid clock lands exactly on SCHED far more often
    for name, c in [("sched", "SCHED_TIME_UTC_mvt"), ("aobt", "AOBT_3_flt"),
                    ("eobt", "EOBT_1_flt"), ("lobt", "LOBT_flt")]:
        s = T(c).dt.second
        out[f"{name}_sec"] = s.fillna(-1).to_numpy(dtype="float32")
        out[f"{name}_on_minute"] = (s == 0).to_numpy().astype("int8")
    # agreement pattern among the planning clocks
    out["sched_eq_lobt"] = ((T("LOBT_flt") - sch).dt.total_seconds().abs() <= 1).to_numpy().astype("int8")
    out["sched_eq_iobt"] = ((T("IOBT_flt") - sch).dt.total_seconds().abs() <= 1).to_numpy().astype("int8")
    out["sched_eq_eobt"] = ((T("EOBT_1_flt") - sch).dt.total_seconds().abs() <= 1).to_numpy().astype("int8")
    return X.join(pd.DataFrame(out, index=X.index))


#: d is heavy-tailed (block-time artifacts reach 24h); L1 keeps the regressor honest
DELTA_PARAMS = {"objective": "regression_l1", "metric": "l1", "learning_rate": 0.05,
                "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.9,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}
DELTA_ROUNDS = 250

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--tag", default="twostage")
    args = ap.parse_args()

    raw, X, cats = matrix.build()
    X = add_candidates(raw, X)
    label = mechanism.fallback_label(raw).to_numpy()
    month = evaluation.movement_month(raw).to_numpy()
    T = lambda c: pd.to_datetime(raw[c], utc=True, errors="coerce")
    d = (T("BLOCK_TIME_UTC_mvt") - T("SCHED_TIME_UTC_mvt")).dt.total_seconds().to_numpy()

    delta_cols = [c for c in X.columns if c.startswith(("sched_gap", "abs_sched_gap", "sched_eq"))
                  or c.endswith(("_sec", "_on_minute"))]
    gcols = list(mechanism.gate_features(X)) + delta_cols + ["AIRCRAFT_OPERATOR_flt"]
    gcols = [c for c in gcols if c in X.columns]
    # the delta regressor may use everything the gate may use
    rcols = gcols
    gcats = [c for c in cats if c in gcols]
    params = dict(mechanism.GATE_PARAMS) | {"num_leaves": args.leaves, "num_threads": args.threads}
    dparams = dict(DELTA_PARAMS) | {"num_threads": args.threads}

    prob = np.full(len(X), np.nan)
    dhat_all = np.full(len(X), np.nan)
    for m in args.months:
        t0 = time.time()
        valid, train = month == m, month != m
        fit = train & np.isfinite(d)
        reg = lgb.train(dparams, lgb.Dataset(X.loc[fit, rcols], d[fit], categorical_feature=gcats),
                        num_boost_round=DELTA_ROUNDS)
        dhat = np.full(len(X), np.nan)
        dhat[train] = reg.predict(X.loc[train, rcols])   # in-sample for the gate's training rows
        dhat[valid] = reg.predict(X.loc[valid, rcols])
        dhat_all[valid] = dhat[valid]
        Xf = X[gcols].copy()
        Xf["dhat"] = dhat
        Xf["abs_dhat"] = np.abs(dhat)
        g = lgb.train(params, lgb.Dataset(Xf.loc[train], label[train].astype("int8"),
                                          categorical_feature=gcats), num_boost_round=args.rounds)
        prob[valid] = g.predict(Xf.loc[valid])
        print(f"  [{args.tag}] month {m:02d} {time.time()-t0:.0f}s  "
              f"MAE(dhat) {np.nanmean(np.abs(dhat[valid]-d[valid])):.0f}s", flush=True)

    sel = np.isfinite(prob)
    print(f"[{args.tag}] n={sel.sum():,} AUC {roc_auc_score(label[sel], prob[sel]):.4f} "
          f"AP {average_precision_score(label[sel], prob[sel]):.4f} "
          f"logloss {log_loss(label[sel], prob[sel]):.5f} "
          f"Brier {brier_score_loss(label[sel], prob[sel]):.6f} "
          f"mass {prob[sel].sum():,.0f} vs {label[sel].sum():,}", flush=True)
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "probability": prob,
                  "delta_estimate": dhat_all}).to_parquet(
        config.DATA_INTERIM / f"gate_{args.tag}_oof.parquet", index=False)

if __name__ == "__main__":
    main()
