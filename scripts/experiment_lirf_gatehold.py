"""Lane-three model: LIRF unmatched rows with a schedule gap under 4h.

At LIRF the recorded target follows T = (MVT - SCHED) - G where G = BLOCK - SCHED is the
time the aircraft sat past its schedule before the airport clocked it off-block (G = 0 on a
schedule copy, negative on an early pushback). G is learned on every LIRF departure with
NM-free features plus the gap D itself, then T_hat = max(D - G_hat, 0) replaces the v15
prediction on LIRF unmatched rows outside the band scope. Rows in `lirfbands.scope` keep v15.

    uv run python -u scripts/experiment_lirf_gatehold.py
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from experiment_missing_expert import nm_free_columns, rmse

from prc import config, evaluation, lirfbands, matrix

G_PARAMS = {"objective": "regression", "metric": "rmse", "learning_rate": 0.04, "num_leaves": 127,
            "min_data_in_leaf": 50, "feature_fraction": 0.9, "bagging_fraction": 0.8,
            "bagging_freq": 1, "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200,
            "max_cat_threshold": 16, "verbosity": -1, "seed": 42}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--missing-weight", type=float, default=8.0)
    ap.add_argument("--g-cap", type=float, default=172800.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--baseline", default="lirfbands_oof_predictions.parquet")
    ap.add_argument("--tag", default="gatehold")
    args = ap.parse_args()
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True)
    lirf = (X["ADEP_mvt"].astype("string") == "LIRF").to_numpy()
    cols = nm_free_columns(X); vcats = [c for c in cats if c in cols]
    y = raw[config.TARGET].to_numpy(dtype="float64")
    D = X[lirfbands.SCHED_COL].to_numpy(dtype="float64")
    G = np.clip(D - y, -args.g_cap, args.g_cap)
    month = evaluation.movement_month(raw).to_numpy()
    missing = X["nm_record_missing"].to_numpy() == 1
    target_rows = lirf & missing & ~lirfbands.scope(X)
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.baseline),
                                      on=config.ID_COL, how="left")["prediction"].to_numpy()
    weight = np.where(missing, args.missing_weight, 1.0)
    params = {**G_PARAMS, "num_threads": args.threads}
    g_hat = np.full(len(y), np.nan)
    print(f"LIRF rows {lirf.sum():,}, target rows {target_rows.sum():,}, {len(cols)} features", flush=True)
    for m in range(1, 13):
        tr = lirf & (month != m); v = lirf & (month == m)
        booster = lgb.train(params, lgb.Dataset(X.loc[tr, cols], G[tr], weight=weight[tr],
                                                categorical_feature=vcats), num_boost_round=args.rounds)
        g_hat[v] = booster.predict(X.loc[v, cols])
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    t_hat = np.clip(D - g_hat, 0, None)
    new = base.copy(); new[target_rows] = t_hat[target_rows]
    scored = np.isin(month, (1, 7))
    rep = {"tag": args.tag}
    for name, mask in [("scored_jan_jul", scored), ("all_months", np.ones(len(y), bool)),
                       ("target_scored", target_rows & scored), ("target_all", target_rows),
                       ("lirf_matched_all", lirf & ~missing)]:
        p = new if not name.startswith("lirf_matched") else np.where(lirf, t_hat, base)
        rep[name] = {"n": int(mask.sum()), "base": rmse(y[mask], base[mask]), "new": rmse(y[mask], p[mask])}
        rep[name]["delta"] = rep[name]["new"] - rep[name]["base"]
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], base[month == m])
                       for m in range(1, 13)}
    ys, bs, ns = y[scored], base[scored], new[scored]
    order = np.argsort((ys - ns) ** 2 - (ys - bs) ** 2)
    rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys), bool); keep[order[:k]] = False
        rep["robustness_scored"][f"drop_{k}"] = rmse(ys[keep], ns[keep]) - rmse(ys[keep], bs[keep])
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": new, "t_hat": t_hat}).to_parquet(
        config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "all_months", "target_scored", "target_all", "lirf_matched_all"):
        r = rep[k]; print(f"{k:18s} n={r['n']:7d} {r['base']:9.3f} -> {r['new']:9.3f} ({r['delta']:+.3f})")
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()})
    print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


if __name__ == "__main__":
    main()
