"""CatBoost GPU residual member with native identity categoricals, paired 12-fold vs v21.

elegant-alligator's main learners are CatBoost depth 8-10, 4.5-6k iterations, with
FLIGHT_mvt / CALLSIGN_flt / airport x stand / stand x runway / airport x operator as native
categoricals (ordered target statistics), target = residual to the AOBT anchor. Our CPU screen
of a 1,500-iteration depth-8 member was negative; this is the GPU-scale version. Per fold the
member is fitted on matched rows (uncapped residual, like v21's base) and blended into the v21
OOF on matched rows with a weight fitted on the training months' OOF-free proxy: the weight is
selected on the ten non-scored months' out-of-fold predictions after all folds are done.

    uv run python -u scripts/experiment_catboost_gpu.py --iterations 5000 --depth 8
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc import arrivals, config, decomposition, evaluation, matrix, tencode

INTERACTIONS = {
    "airport_stand": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["STAND_mvt"].astype("string"),
    "stand_runway": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["STAND_mvt"].astype("string") + "_" + r["RUNWAY_mvt"].astype("string"),
    "airport_operator": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["AIRCRAFT_OPERATOR_flt"].astype("string"),
    "runway_aircraft": lambda r: r["ADEP_mvt"].astype("string") + "_" + r["RUNWAY_mvt"].astype("string") + "_" + r["AIRCRAFT_TYPE_mvt"].astype("string"),
    "flight_prefix": lambda r: r["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]+)")[0],
    "callsign_prefix": lambda r: r["CALLSIGN_flt"].astype("string").str[:3],
}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iterations", type=int, default=5000); ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.06); ap.add_argument("--l2", type=float, default=5.0); ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--tag", default="catgpu"); ap.add_argument("--target", choices=["aobt", "sched", "lobt", "eobt", "iobt"], default="aobt",
                    help="aobt: residual to clip(MVT-AOBT); sched: schedule delay BLOCK-SCHED, prediction = (MVT-SCHED) - delay_hat; lobt: residual to clip(MVT-LOBT), a different reference clock for a blend member")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--own-features", default=None,
                    help="data/interim parquet of own-flight ADS-B features (own_flight_features.py) to join as numeric inputs (section 53)")
    args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw); anchor = np.clip(anchor_raw, 0, 3600)
    sched_gap = X["taxi_proxy_SCHED_TIME_UTC_mvt"].to_numpy(dtype="float64")
    CLOCKS = {"lobt": "taxi_proxy_LOBT_flt", "eobt": "taxi_proxy_EOBT_1_flt", "iobt": "taxi_proxy_IOBT_flt"}
    if args.target in CLOCKS:
        clock_anchor = np.clip(X[CLOCKS[args.target]].to_numpy(dtype="float64"), -3600, 7200)
        matched = matched & np.isfinite(clock_anchor); anchor = clock_anchor
    resid = (y - anchor) if args.target in ("aobt", *CLOCKS) else np.clip(sched_gap - y, -172800, 172800)   # sched: BLOCK - SCHED
    v21 = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    F = X.copy()
    for name, f in INTERACTIONS.items():
        F[name] = f(raw).fillna("__NA__").astype(str)
    if args.own_features:
        own_cols = ["own_joined", "own_to_minus_mvt", "own_dwell_taxi", "own_dwell_len", "own_first_taxi", "own_gap_seg",
                    "own_stand_exit_taxi", "own_stand_dmin"]
        own = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.own_features), on=config.ID_COL, how="left")
        for c in own_cols:
            F[c] = own[c].to_numpy(dtype="float64")
        F["own_joined"] = F["own_joined"].fillna(0.0)
        for c, src in (("own_minus_anchor", "own_dwell_taxi"), ("own_first_minus_anchor", "own_first_taxi"),
                       ("own_stand_minus_anchor", "own_stand_exit_taxi")):
            F[c] = F[src] - anchor_raw
    cat_cols = list(cats) + list(INTERACTIONS)
    for c in cats:
        F[c] = F[c].astype("string").fillna("__NA__").astype(str)
    print(f"{F.shape[1]} features, {len(cat_cols)} categoricals; iterations {args.iterations} depth {args.depth}", flush=True)
    member = np.full(len(y), np.nan)
    for m in args.months:
        tr = matched & (month != m); va = matched & (month == m)
        enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), month != m)
        Ftr = F.loc[tr].join(tencode.transform(raw.loc[tr], enc)); Fva = F.loc[va].join(tencode.transform(raw.loc[va], enc))
        model = CatBoostRegressor(loss_function="RMSE", iterations=args.iterations, depth=args.depth, learning_rate=args.lr, l2_leaf_reg=args.l2,
                                  border_count=128, one_hot_max_size=20, max_ctr_complexity=1, task_type="GPU", random_seed=args.seed, verbose=0)
        model.fit(Pool(Ftr, resid[tr], cat_features=cat_cols))
        pred_resid = model.predict(Pool(Fva, cat_features=cat_cols))
        member[va] = np.clip(anchor[va] + pred_resid, 0, None) if args.target in ("aobt", *CLOCKS) else np.clip(sched_gap[va] - pred_resid, 0, None)
        print(f"  fold {m:02d} member matched RMSE {rmse(y[va], member[va]):.2f} vs v21 {rmse(y[va], v21[va]):.2f}  t={time.time()-start:.0f}s", flush=True)
    done = np.isin(month, args.months) & matched
    sel = done & ~np.isin(month, (1, 7))
    d = member[sel] - v21[sel]; r = y[sel] - v21[sel]; w = float(np.clip((d @ r) / max(d @ d, 1e-9), 0, 1))
    new = v21.copy(); new[done] = v21[done] + w * (member[done] - v21[done])
    rep = {"weight": w, "iterations": args.iterations, "depth": args.depth}
    for name, mask in [("scored_jan_jul", np.isin(month, (1, 7))), ("jan", month == 1), ("jul", month == 7), ("all_months", np.isin(month, args.months)), ("other_months", np.isin(month, args.months) & ~np.isin(month, (1, 7)))]:
        rep[name] = {"v21": rmse(y[mask], v21[mask]), "blend": rmse(y[mask], new[mask]), "member_matched_only": rmse(y[mask & matched], member[mask & matched])}
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], v21[month == m]) for m in args.months}
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "member": member, "blend": new}).to_parquet(config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    print("weight", round(w, 3)); [print(f"{k:15s} v21 {v['v21']:.3f} -> blend {v['blend']:.3f} (member alone on matched {v['member_matched_only']:.3f})") for k, v in rep.items() if isinstance(v, dict) and "v21" in v]
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()})


if __name__ == "__main__":
    main()
