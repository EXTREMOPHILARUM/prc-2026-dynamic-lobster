"""Uncapped matched base inside the full v19 pipeline, four holdout months.

Per month: fit one LightGBM on y - anchor (no cap) for matched rows; take the shipped
pipeline's ordinary prediction for unmatched rows from the v19 OOF; then apply the stored
gate (v19 OOF), LOBT expert + OOF alphas, Rome regime, fine bands, Rome rules, short expert,
LOBT clip. Compare with v19 on all rows of the month. Also reports a 50/50 blend of the
uncapped-base pipeline with v19.

    uv run python -u scripts/experiment_uncapped_pipeline.py --months 2 10 1 7 --threads 4
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
)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7]); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1000); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy(); airport = X["ADEP_mvt"].astype("string").to_numpy()
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw); anchor = np.clip(anchor_raw, 0, 3600)
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    v19 = col("v19_oof_predictions.parquet", "prediction"); gate = np.nan_to_num(col("v19_oof_predictions.parquet", "gate_new"))
    expert = col("lobt_expert_oof.parquet", "expert"); short = col("v16_oof_predictions.parquet", "short_expert"); p_reg = np.nan_to_num(col("lirf_regime_oof_predictions.parquet", "p_identity"))
    alphas = json.loads((config.DATA_INTERIM / "lobt_alphas_oof_metrics.json").read_text())["alphas_oof"]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
    params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 42}
    out = {}
    for m in args.months:
        tr = matched & (month != m); va = month == m; vam = va & matched
        b = lgb.train(params, lgb.Dataset(X.loc[tr, cols], (y - anchor)[tr], categorical_feature=vcats), num_boost_round=args.rounds)
        ordinary = v19.copy()          # unmatched rows keep the pipeline's value
        ordinary[vam] = np.clip(anchor[vam] + b.predict(X.loc[vam, cols]), 0, None)
        p = mechanism.blend(gate, proxy, ordinary); p = lobt.apply(p, expert, airport, alphas); p = lirfregime.apply(p, p_reg, X)
        table = lirfbands.fit(X.loc[~va], y[~va]); p[va] = lirfbands.apply(p[va], X.loc[va], table); p = lirfbands.rome_rules(p, raw, X)
        s = lirfbands.short_scope(X); p[s] = short[s]; p = np.clip(lirfbands.lobt_clip(p, X), 0, None)
        blend = 0.5 * p + 0.5 * v19
        r = {"v19": rmse(y[va], v19[va]), "uncapped_pipeline": rmse(y[va], p[va]), "blend50": rmse(y[va], blend[va]),
             "matched_v19": rmse(y[vam], v19[vam]), "matched_uncapped": rmse(y[vam], p[vam]), "matched_blend": rmse(y[vam], blend[vam])}
        out[str(m)] = r
        print(f"month {m:02d} all: v19 {r['v19']:.2f} uncapped {r['uncapped_pipeline']:.2f} blend {r['blend50']:.2f} | matched: {r['matched_v19']:.2f} {r['matched_uncapped']:.2f} {r['matched_blend']:.2f}  t={time.time()-start:.0f}s", flush=True)
    (config.DATA_INTERIM / "uncapped_pipeline_screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
