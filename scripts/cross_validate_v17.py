"""Paired 12-fold of the v17 bundle against the v16 OOF.

Per fold (train = other 11 months):
  1. decomposition refit on the v13 feature set + `context.FEATURES`
  2. frozen schedule-fallback gate blend (mechanism OOF probabilities)
  3. LOBT residual expert fitted on the fold, blended with per-airport alphas fitted on the
     fold's *non-scored* training months' OOF is unavailable, so alphas are fitted on the
     training rows' in-sample residuals shrunk toward global (conservative)
  4. lirfbands table (fold-fitted) + stored NM-free short expert + LOBT clip
Reports each stage cumulatively. Use --no-context / --no-lobt to ablate.

    uv run python -u scripts/cross_validate_v17.py --threads 8
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from prc import (
    arrivals,
    config,
    decomposition,
    evaluation,
    lirfbands,
    lobt,
    matrix,
    mechanism,
)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-lobt", action="store_true")
    ap.add_argument("--tag", default="v17")
    args = ap.parse_args()
    for p in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS, decomposition.SEVERITY_PARAMS, lobt.PARAMS):
        p["num_threads"] = args.threads
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True,
                                use_context=not args.no_context)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); ys = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw).to_numpy()
    airport = X["ADEP_mvt"].astype("string").to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    ids = raw[[config.ID_COL]]
    gate_p = ids.merge(pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[[config.ID_COL, "probability"]],
                       on=config.ID_COL, how="left")["probability"].to_numpy()
    ref = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v16_oof_predictions.parquet"), on=config.ID_COL, how="left")
    v16 = ref["prediction"].to_numpy(); short_expert = ref["short_expert"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    print(f"matrix {X.shape}; context={'off' if args.no_context else 'on'} lobt={'off' if args.no_lobt else 'on'}", flush=True)

    stage_base = np.full(len(y), np.nan); stage_lobt = np.full(len(y), np.nan)
    alphas_log = {}
    for m in range(1, 13):
        tr = month != m; va = month == m
        models = decomposition.fit_fixed(X.loc[tr], ys.loc[tr], cats, rounds=rounds)
        ordinary = decomposition.predict(models, X.loc[va])
        blended = mechanism.blend(gate_p[va], proxy[va], ordinary)
        stage_base[va] = blended
        if args.no_lobt:
            stage_lobt[va] = blended
        else:
            booster = lobt.fit(X.loc[tr], y[tr], cats)
            # alphas: fitted on training-month rows using in-sample base/expert (conservative shrink)
            ex_tr = lobt.predict(booster, X.loc[tr]); base_tr = mechanism.blend(
                gate_p[tr], proxy[tr], decomposition.predict(models, X.loc[tr]))
            rows = lobt.scope(X.loc[tr]) & ~np.isin(month[tr], (1, 7))
            alphas = lobt.fit_alphas(y[tr], base_tr, ex_tr, airport[tr], rows)
            alphas_log[str(m)] = alphas
            stage_lobt[va] = lobt.apply(blended, lobt.predict(booster, X.loc[va]), airport[va], alphas)
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)

    def post(pred):
        out = pred.copy()
        for m in range(1, 13):
            table = lirfbands.fit(X.loc[month != m], y[month != m]); va = month == m
            out[va] = lirfbands.apply(out[va], X.loc[va], table)
        s = lirfbands.short_scope(X); out[s] = short_expert[s]
        return np.clip(lirfbands.lobt_clip(out, X), 0, None)

    final_base = post(stage_base); final = post(stage_lobt)
    scored = np.isin(month, (1, 7))
    rep = {"tag": args.tag, "alphas": alphas_log}
    for name, mask in [("scored_jan_jul", scored), ("jan", month == 1), ("jul", month == 7),
                       ("all_months", np.ones(len(y), bool)), ("other_months", ~scored)]:
        rep[name] = {"v16": rmse(y[mask], v16[mask]), "context_only": rmse(y[mask], final_base[mask]),
                     "context+lobt": rmse(y[mask], final[mask])}
    rep["by_month"] = {str(m): rmse(y[month == m], final[month == m]) - rmse(y[month == m], v16[month == m]) for m in range(1, 13)}
    ys_, bs, ns = y[scored], v16[scored], final[scored]; order = np.argsort((ys_ - ns) ** 2 - (ys_ - bs) ** 2)
    rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys_), bool); keep[order[:k]] = False
        rep["robustness_scored"][f"drop_{k}"] = rmse(ys_[keep], ns[keep]) - rmse(ys_[keep], bs[keep])
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": final, "context_only": final_base}).to_parquet(
        config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all_months", "other_months"):
        r = rep[k]; print(f"{k:15s} v16 {r['v16']:.3f} -> context {r['context_only']:.3f} -> +lobt {r['context+lobt']:.3f}")
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()})
    print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})
    print("alphas fold 1", alphas_log.get("1"))


if __name__ == "__main__":
    main()
