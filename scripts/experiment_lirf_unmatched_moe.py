"""LIRF unmatched rows under the band scope: mixture of a copy classifier and a copy-free
regressor (elegant-alligator's missing-row MoE) against the shipped single expert.

Per fold, on LIRF rows (matched + unmatched) with NM-free features + encodings:
  classifier  P(|y - gap| <= 60), unmatched rows weighted 8x, isotonic-calibrated on months 4/10
  regressor   y on non-copy rows with y < 12000, unmatched weighted 8x
  mixture     P * max(gap, 0) + (1 - P) * regressor
Applied to LIRF unmatched rows outside `lirfbands.scope`; compared with v23-equivalent OOF
(v21 OOF, which carries the shipped short expert).

    uv run python -u scripts/experiment_lirf_unmatched_moe.py --threads 8
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from prc import (
    arrcopy,
    arrivals,
    config,
    data,
    decomposition,
    evaluation,
    lirfbands,
    matrix,
    mechanism,
    tencode,
)

NM_DEPENDENT_PREFIXES = ("aobt_minus_", "taxi_proxy_AOBT", "taxi_proxy_EOBT", "taxi_proxy_IOBT", "taxi_proxy_LOBT", "arr_next_20m_aobt",
                         "arr_during_taxi", "q_apt_at_aobt", "q_rwy_at_aobt", "nbr_apt_own", "clk_AOBT", "clk_LOBT", "clk_IOBT", "clk_EOBT",
                         "clk_ARVT", "clk_mvt_minus_ARVT", "clk_lobt", "clk_arvt", "clk_sched_minus_aobt", "clk_eobt", "clk_iobt", "ord_queue_eobt", "ord_later")
NM_DEPENDENT = {"offblock_delay", "nm_time_count", "aobt_missing", "nm_record_missing"}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--w", type=float, default=8.0); ap.add_argument("--max-gap-h", type=float, default=None, help="extend the target scope to LIRF no-record rows with gap below this many hours (default: lirfbands.short_scope)")
    ap.add_argument("--tag", default="lirf_unmatched_moe"); ap.add_argument("--regressor", choices=["lightgbm", "catboost"], default="lightgbm")
    ap.add_argument("--arr-copy", action="store_true", help="add prc.arrcopy features (log section 58)"); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    if args.arr_copy:
        extra = arrcopy.add(raw[[config.ID_COL, "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt"]], data.load_training())
        X = X.join(extra[arrcopy.FEATURES].set_axis(X.index))
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    lirf = (X["ADEP_mvt"].astype("string") == "LIRF").to_numpy(); missing = X["nm_record_missing"].to_numpy() == 1
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); copy = np.abs(y - gap) <= 60
    target = lirfbands.short_scope(X)
    if args.max_gap_h is not None:
        target = lirf & missing & (gap <= args.max_gap_h * 3600.0)
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    ref = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    cols = [c for c in X.columns if not c.endswith("_flt") and c not in NM_DEPENDENT and not c.startswith(NM_DEPENDENT_PREFIXES) and c not in decomposition.BASE_EXCLUDED]
    cols = [c for c in cols if c not in ("ADES_flt",)]; vcats = [c for c in cats if c in cols]
    print(f"{len(cols)} NM-free columns; target rows {target.sum():,} (scored {int((target & np.isin(month, (1, 7))).sum())})", flush=True)
    p_out = np.full(len(y), np.nan); reg_out = np.full(len(y), np.nan)
    for m in range(1, 13):
        tr = lirf & (month != m); va = target & (month == m)
        enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), month != m)
        te = tencode.transform(raw, enc)
        te_cols = [c for c in te.columns if not c.startswith(("te_operator", "te_segment", "te_wake", "te_ftype"))]   # NM-derived keys excluded
        F = X[cols].join(te[te_cols])
        w = np.where(missing, args.w, 1.0)
        fit_rows = tr & ~np.isin(month, (4, 10)); cal_rows = tr & np.isin(month, (4, 10))
        clf = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8, "verbosity": -1, "num_threads": args.threads, "seed": 3},
                        lgb.Dataset(F.loc[fit_rows], copy[fit_rows].astype("int8"), weight=w[fit_rows], categorical_feature=vcats), num_boost_round=300)
        iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict(F.loc[cal_rows]), copy[cal_rows].astype(float))
        p_out[va] = iso.predict(clf.predict(F.loc[va]))
        rr = tr & ~copy & (y < 12000)
        if args.regressor == "catboost":
            from catboost import CatBoostRegressor, Pool
            Fc = F.copy()
            for c in vcats:
                Fc[c] = Fc[c].astype("string").fillna("__NA__").astype(str)
            model = CatBoostRegressor(loss_function="RMSE", iterations=3000, depth=8, learning_rate=0.05, l2_leaf_reg=5, border_count=128, one_hot_max_size=20, max_ctr_complexity=1, task_type="GPU", random_seed=42, verbose=0)
            model.fit(Pool(Fc.loc[rr], y[rr], weight=w[rr], cat_features=vcats))
            reg_out[va] = np.clip(model.predict(Pool(Fc.loc[va], cat_features=vcats)), 0, None)
        else:
            reg = lgb.train({**decomposition.BASE_PARAMS, "num_leaves": 63, "num_threads": args.threads, "seed": 5}, lgb.Dataset(F.loc[rr], y[rr], weight=w[rr], categorical_feature=vcats), num_boost_round=400)
            reg_out[va] = np.clip(reg.predict(F.loc[va]), 0, None)
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    mix = p_out * np.clip(gap, 0, None) + (1 - p_out) * reg_out
    new = ref.copy(); new[target] = mix[target]; new = np.clip(lirfbands.lobt_clip(lirfbands.rome_rules(new, raw, X), X), 0, None)
    ref2 = np.clip(lirfbands.lobt_clip(lirfbands.rome_rules(ref, raw, X), X), 0, None)
    sc = np.isin(month, (1, 7)); rep = {}
    for name, mask in [("scored_jan_jul", sc), ("jan", month == 1), ("jul", month == 7), ("all", np.ones(len(y), bool)), ("target_scored", target & sc), ("target_all", target)]:
        rep[name] = {"ref": rmse(y[mask], ref2[mask]), "moe": rmse(y[mask], new[mask])}
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], ref2[month == m]) for m in range(1, 13)}
    from sklearn.metrics import roc_auc_score
    ok = target & np.isfinite(p_out); rep["auc"] = float(roc_auc_score(copy[ok], p_out[ok])); rep["mass"] = {"pred": float(p_out[ok].sum()), "actual": int(copy[ok].sum())}
    (config.DATA_INTERIM / f"{args.tag}_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "p_copy": p_out, "regressor": reg_out, "mixture": mix}).to_parquet(config.DATA_INTERIM / f"{args.tag}_oof.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all", "target_scored", "target_all"):
        print(f"{k:14s} {rep[k]['ref']:.3f} -> {rep[k]['moe']:.3f} ({rep[k]['moe']-rep[k]['ref']:+.3f})")
    print("AUC", round(rep["auc"], 3), "mass", rep["mass"]); print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()})


if __name__ == "__main__":
    main()
