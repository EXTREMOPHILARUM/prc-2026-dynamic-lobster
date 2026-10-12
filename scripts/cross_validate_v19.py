"""Paired 12-fold of the v19 hygiene bundle against a reconstructed v18 OOF.

Per fold: decomposition refit with `component_rounds_ctx.json` rounds, averaged over
--seeds seeds; schedule-fallback gate refit with context features (compared with the frozen
v7 gate); then the shared post-processing (LOBT expert with OOF alphas, Rome regime, bands,
short expert, LOBT clip) applied identically to the reference and the candidate.

    uv run python -u scripts/cross_validate_v19.py --threads 10 --seeds 3
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


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=10); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="v19"); args = ap.parse_args(); start = time.time()
    for p in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS, decomposition.SEVERITY_PARAMS):
        p["num_threads"] = args.threads
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); ys = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw).to_numpy(); airport = X["ADEP_mvt"].astype("string").to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds_ctx.json").read_text())["rounds"]
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    gate_old = col("mechanism_oof_predictions.parquet", "probability")
    context_only = col("v17_oof_predictions.parquet", "context_only")
    expert = col("lobt_expert_oof.parquet", "expert"); short = col("v16_oof_predictions.parquet", "short_expert")
    p_regime = col("lirf_regime_oof_predictions.parquet", "p_identity")
    alphas = json.loads((config.DATA_INTERIM / "lobt_alphas_oof_metrics.json").read_text())["alphas_oof"]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); fb_label = mechanism.fallback_label(raw).to_numpy()
    gate_cols = mechanism.gate_features(X) + [c for c in X.columns if c.startswith(("nbr_apt", "q_apt", "arrctx_", "clk_"))]
    gate_cats = [c for c in cats if c in gate_cols]
    print(f"rounds {rounds}; seeds {args.seeds}; gate columns {len(gate_cols)}", flush=True)

    ordinary = np.full(len(y), np.nan); gate_new = np.full(len(y), np.nan)
    for m in range(1, 13):
        tr = month != m; va = month == m; preds = []
        for s in range(args.seeds):
            for p in (decomposition.BASE_PARAMS, decomposition.SEVERITY_PARAMS):
                p["seed"] = 100 + s
            models = decomposition.fit_fixed(X.loc[tr], ys.loc[tr], cats, rounds=rounds)
            preds.append(decomposition.predict(models, X.loc[va]))
        ordinary[va] = np.mean(preds, axis=0)
        g = lgb.train({**mechanism.GATE_PARAMS, "num_threads": args.threads, "seed": 3},
                      lgb.Dataset(X.loc[tr, gate_cols], fb_label[tr].astype("int8"), categorical_feature=gate_cats), num_boost_round=200)
        gate_new[va] = g.predict(X.loc[va, gate_cols])
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)

    def post(stage):
        out = lobt.apply(stage, expert, airport, alphas)
        out = lirfregime.apply(out, np.nan_to_num(p_regime), X)
        for m in range(1, 13):
            table = lirfbands.fit(X.loc[month != m], y[month != m]); va = month == m
            out[va] = lirfbands.apply(out[va], X.loc[va], table)
        s = lirfbands.short_scope(X); out[s] = short[s]
        return np.clip(lirfbands.lobt_clip(out, X), 0, None)

    ref = post(context_only)
    cand_oldgate = post(mechanism.blend(gate_old, proxy, ordinary))
    cand_newgate = post(mechanism.blend(gate_new, proxy, ordinary))
    scored = np.isin(month, (1, 7)); rep = {"rounds": rounds, "seeds": args.seeds}
    for name, mask in [("scored_jan_jul", scored), ("jan", month == 1), ("jul", month == 7), ("all_months", np.ones(len(y), bool)), ("other_months", ~scored)]:
        rep[name] = {"v18_ref": rmse(y[mask], ref[mask]), "rounds+seeds": rmse(y[mask], cand_oldgate[mask]), "rounds+seeds+gate": rmse(y[mask], cand_newgate[mask])}
    rep["by_month"] = {str(m): rmse(y[month == m], cand_newgate[month == m]) - rmse(y[month == m], ref[month == m]) for m in range(1, 13)}
    ys_, bs, ns = y[scored], ref[scored], cand_newgate[scored]; order = np.argsort((ys_ - ns) ** 2 - (ys_ - bs) ** 2); rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys_), bool); keep[order[:k]] = False; rep["robustness_scored"][f"drop_{k}"] = rmse(ys_[keep], ns[keep]) - rmse(ys_[keep], bs[keep])
    from sklearn.metrics import roc_auc_score
    rep["gate_auc"] = {"old": float(roc_auc_score(fb_label, np.nan_to_num(gate_old))), "new": float(roc_auc_score(fb_label, gate_new))}
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": cand_newgate, "oldgate": cand_oldgate, "v18_ref": ref, "gate_new": gate_new}).to_parquet(
        config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all_months", "other_months"):
        r = rep[k]; print(f"{k:15s} v18 {r['v18_ref']:.3f} -> rounds+seeds {r['rounds+seeds']:.3f} -> +gate {r['rounds+seeds+gate']:.3f}")
    print("gate AUC", rep["gate_auc"]); print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()}); print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


if __name__ == "__main__":
    main()
