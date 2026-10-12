"""Chunk 5 members: CatBoost residual on the matched base, and EDDF/EGLL/EHAM specialists.

Both are fitted on the matched-clock capped residual `min(y, 3600) - clip(MVT - AOBT)` like
`base_matched`, and blended 50/50 with the fold's baseline prediction on matched rows
(CatBoost: all airports; specialists: their own airport only). Screened on months given
by --months (default the two selection months, then Jan/Jul), paired against --baseline.

    uv run python -u scripts/experiment_members.py --member catboost --months 2 10
    uv run python -u scripts/experiment_members.py --member specialists --months 1 2 7 10
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix

SPECIALIST_AIRPORTS = ["EDDF", "EGLL", "EHAM"]
CAT_PARAMS = {"loss_function": "RMSE", "depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 5,
              "iterations": 1500, "border_count": 128, "one_hot_max_size": 20, "max_ctr_complexity": 1,
              "random_seed": 42, "verbose": 0, "allow_writing_files": False}


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", choices=["catboost", "specialists"], required=True)
    ap.add_argument("--months", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--baseline", default="v17_oof_predictions.parquet")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weight", type=float, default=0.5)
    args = ap.parse_args()
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
    anchor = X[decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD).to_numpy(dtype="float64")
    resid = np.minimum(y, decomposition.TAIL_THRESHOLD) - anchor
    airport = X["ADEP_mvt"].astype("string").to_numpy()
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.baseline), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    tag = f"member_{args.member}"
    rep = {"member": args.member, "months": {}}
    new = base.copy()
    for m in args.months:
        va = matched & (month == m); tr = matched & (month != m)
        if args.member == "catboost":
            from catboost import CatBoostRegressor, Pool
            cols = list(X.columns); cat_cols = [c for c in cats if c in cols]
            def frame(rows, cols=cols, cat_cols=cat_cols):
                f = X.loc[rows, cols].copy()
                for c in cat_cols:
                    f[c] = f[c].astype("string").fillna("__NA__").astype(str)
                return f
            model = CatBoostRegressor(**CAT_PARAMS, thread_count=args.threads)
            model.fit(Pool(frame(tr), resid[tr], cat_features=cat_cols))
            member_pred = anchor[va] + model.predict(Pool(frame(va), cat_features=cat_cols))
            # replace the base's matched component estimate: blend on the final prediction scale
            new[va] = base[va] + args.weight * (np.clip(member_pred, 0, None) - base[va])
        else:
            cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]
            vcats = [c for c in cats if c in cols]
            for a in SPECIALIST_AIRPORTS:
                tr_a = tr & (airport == a); va_a = va & (airport == a)
                booster = lgb.train({**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 11},
                                    lgb.Dataset(X.loc[tr_a, cols], resid[tr_a], categorical_feature=vcats), num_boost_round=400)
                spec = anchor[va_a] + booster.predict(X.loc[va_a, cols])
                new[va_a] = base[va_a] + args.weight * (np.clip(spec, 0, None) - base[va_a])
        r = {"n": int(va.sum()), "base": rmse(y[month == m], base[month == m]), "new": rmse(y[month == m], new[month == m])}
        r["delta"] = r["new"] - r["base"]
        if args.member == "specialists":
            r["by_airport"] = {a: round(rmse(y[va & (airport == a)], new[va & (airport == a)]) - rmse(y[va & (airport == a)], base[va & (airport == a)]), 3) for a in SPECIALIST_AIRPORTS}
        rep["months"][str(m)] = r
        print(f"month {m:02d} {r['base']:.3f} -> {r['new']:.3f} ({r['delta']:+.3f}) t={time.time()-start:.0f}s {r.get('by_airport','')}", flush=True)
    (config.DATA_INTERIM / f"{tag}_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": new}).to_parquet(config.DATA_INTERIM / f"{tag}_oof_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
