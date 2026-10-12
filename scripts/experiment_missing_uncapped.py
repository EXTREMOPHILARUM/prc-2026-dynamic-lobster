"""Uncapped `base_missing` for unmatched non-LIRF rows, paired 12-fold on top of v21.

The other-unmatched lane (4,976 scored rows, RMSE ~1,550, 30% of SSE) is served by a capped
base plus the shrunk tail term. Per fold: refit base_missing on y (no cap) and on min(y,3600)
with the v20 feature set + encodings, rebuild `ordinary` for those rows as base (+ tail term
for the capped variant), re-blend with the stored gate, and compare on all rows.

    uv run python -u scripts/experiment_missing_uncapped.py --threads 8
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import (
    arrivals,
    config,
    decomposition,
    evaluation,
    lirfbands,
    lirfregime,
    lobt,
    matrix,
    mechanism,
    tencode,
)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--rounds", type=int, default=300); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy(); airport = X["ADEP_mvt"].astype("string").to_numpy()
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    lirf = airport == "LIRF"; target_rows = ~matched & ~lirf
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    ordinary = col("v21_oof_predictions.parquet", "ordinary"); gate = np.nan_to_num(col("v21_oof_predictions.parquet", "gate_new"))
    expert = col("lobt_expert_oof.parquet", "expert"); short = col("v16_oof_predictions.parquet", "short_expert"); p_reg = np.nan_to_num(col("lirf_regime_oof_predictions.parquet", "p_identity"))
    alphas = json.loads((config.DATA_INTERIM / "lobt_alphas_v21.json").read_text())["alphas_oof"]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
    params = {**decomposition.BASE_PARAMS, "learning_rate": 0.05, "num_threads": args.threads, "seed": 7}
    new_ord = {"capped": ordinary.copy(), "uncapped": ordinary.copy()}
    for m in range(1, 13):
        tr = ~matched & (month != m); va = target_rows & (month == m)
        enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), month != m)
        Xtr = X.loc[tr, cols].join(tencode.transform(raw.loc[tr], enc)); Xva = X.loc[va, cols].join(tencode.transform(raw.loc[va], enc))
        for name, target in (("capped", np.minimum(y, 3600)), ("uncapped", y)):
            b = lgb.train(params, lgb.Dataset(Xtr, target[tr], categorical_feature=vcats), num_boost_round=args.rounds)
            new_ord[name][va] = np.clip(b.predict(Xva), 0, None)
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    def post(stage):
        out = lobt.apply(mechanism.blend(gate, proxy, stage), expert, airport, alphas); out = lirfregime.apply(out, p_reg, X)
        for m in range(1, 13):
            t = lirfbands.fit(X.loc[month != m], y[month != m]); va = month == m; out[va] = lirfbands.apply(out[va], X.loc[va], t)
        out = lirfbands.rome_rules(out, raw, X); s = lirfbands.short_scope(X); out[s] = short[s]; return np.clip(lirfbands.lobt_clip(out, X), 0, None)
    ref = post(ordinary); sc = np.isin(month, (1, 7)); rep = {}
    for name in ("capped", "uncapped"):
        p = post(new_ord[name])
        rep[name] = {"jan_jul": rmse(y[sc], p[sc]), "jan": rmse(y[month == 1], p[month == 1]), "jul": rmse(y[month == 7], p[month == 7]), "all": rmse(y, p),
                     "lane_scored": rmse(y[target_rows & sc], p[target_rows & sc]), "months_improved": int(sum(rmse(y[month == k], p[month == k]) < rmse(y[month == k], ref[month == k]) for k in range(1, 13)))}
        print(f"{name:9s} Jan+Jul {rmse(y[sc], ref[sc]):.3f} -> {rep[name]['jan_jul']:.3f} | Jan {rep[name]['jan']:.2f} Jul {rep[name]['jul']:.2f} | lane {rmse(y[target_rows & sc], ref[target_rows & sc]):.1f} -> {rep[name]['lane_scored']:.1f} | months {rep[name]['months_improved']}/12")
    rep["ref"] = {"jan_jul": rmse(y[sc], ref[sc]), "lane_scored": rmse(y[target_rows & sc], ref[target_rows & sc])}
    (config.DATA_INTERIM / "missing_uncapped_metrics.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
