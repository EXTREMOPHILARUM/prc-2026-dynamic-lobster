"""Paired 12-fold of v20 (context v2 + in-fold target encodings + long training) against v19.

Per fold: encodings fitted on the eleven training months; decomposition refit with the
selected v20 rounds (single seed for the fold run; production uses three); gate refit with
context; then the v19 post-processing (LOBT expert OOF with OOF alphas, Rome regime, fine
bands, short expert, LOBT clip). Reference = the v19 OOF.

    uv run python -u scripts/cross_validate_v20.py --threads 12
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


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=12); ap.add_argument("--tag", default="v20")
    ap.add_argument("--rounds-file", default="rounds_v20_lr0.02.json")
    ap.add_argument("--uncapped", action="store_true", help="matched base on the full residual, tail term off for matched rows")
    ap.add_argument("--rome-rules", action="store_true"); ap.add_argument("--atfm", action="store_true")
    ap.add_argument("--atfm-lag", type=int, default=0, help="join ATFM daily data from N days before take-off")
    ap.add_argument("--base-params", default=None, help="JSON merged into decomposition.BASE_PARAMS (retune sweep, log section 63)")
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)), help="fold months to run (screens)")
    ap.add_argument("--standrwy", action="store_true", help="add prc.standrwy stand-to-runway distances (log section 63)")
    ap.add_argument("--cellspread", action="store_true", help="tencode spread tables (cell p90 / std, log section 63)")
    ap.add_argument("--rwystate", action="store_true", help="add prc.rwystate runway-state and gate-retention features")
    ap.add_argument("--train-jitter", type=int, default=None, metavar="SEED",
                    help="fit on movement/block times jittered as in the final phase (rehearsal.jitter_times)")
    ap.add_argument("--valid-jitter", type=int, default=None, metavar="SEED",
                    help="predict and score held-out months on jittered times (the final set's condition)")
    args = ap.parse_args(); start = time.time()
    for p in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS, decomposition.SEVERITY_PARAMS):
        p["num_threads"] = args.threads

    def source(jitter_seed):
        r, x, c = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True,
                               use_atfm=args.atfm, atfm_lag_days=args.atfm_lag, jitter_seed=jitter_seed)
        x = x.drop(columns=[k for k in arrivals.FEATURES if "during_taxi" in k], errors="ignore")
        if args.rwystate:
            from prc import rwystate
            x = x.join(rwystate.compute(r).set_index(x.index))
        if args.standrwy:
            from prc import standrwy
            x = x.join(standrwy.compute(r).set_index(x.index))
        return r, x, c
    # the training source (fits, encodings, band tables) and the validation source (prediction, post-processing,
    # scoring) are each clean or jittered; the jittered one is aligned to the clean row order by movement id
    raw, X, cats = source(None)
    def aligned(seed):
        r, x, _ = source(seed)
        pos = pd.Series(np.arange(len(r)), index=r[config.ID_COL].to_numpy())[raw[config.ID_COL].to_numpy()].to_numpy()
        return r.iloc[pos].reset_index(drop=True), x.iloc[pos].set_index(X.index)
    rawT, XT = aligned(args.train_jitter) if args.train_jitter is not None else (raw, X)
    if args.valid_jitter is not None:
        raw, X = (rawT, XT) if args.valid_jitter == args.train_jitter else aligned(args.valid_jitter)
    yT = rawT[config.TARGET].to_numpy(dtype="float64"); ysT = rawT[config.TARGET].astype("float32")
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy(); airport = X["ADEP_mvt"].astype("string").to_numpy()
    sel = json.loads((config.DATA_INTERIM / args.rounds_file).read_text())
    rounds = json.loads((config.DATA_INTERIM / "component_rounds_ctx.json").read_text())["rounds"] | {"base_matched": sel["best_iter"]}
    decomposition.BASE_PARAMS["learning_rate"] = sel["lr"]
    if args.base_params:
        decomposition.BASE_PARAMS.update(json.loads(args.base_params))
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    anchor_T = XT[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    delta_T = np.where(np.isfinite(anchor_T), anchor_T - yT, np.nan)
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    v19 = col("v19_oof_predictions.parquet", "prediction"); expert = col("lobt_expert_oof.parquet", "expert")
    short = col("v16_oof_predictions.parquet", "short_expert"); p_regime = np.nan_to_num(col("lirf_regime_oof_predictions.parquet", "p_identity"))
    alphas = json.loads((config.DATA_INTERIM / "lobt_alphas_oof_metrics.json").read_text())["alphas_oof"]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); fb = mechanism.fallback_label(rawT).to_numpy()
    gate_cols = mechanism.gate_features(X) + [c for c in X.columns if c.startswith(("nbr_apt", "q_apt", "arrctx_", "clk_", "ord_"))]
    gate_cats = [c for c in cats if c in gate_cols]
    print(f"rounds {rounds} lr {sel['lr']}; base features {X.shape[1] + len(tencode.FEATURES)}", flush=True)
    ordinary = np.full(len(y), np.nan); gate_new = np.full(len(y), np.nan)
    for m in args.months:
        tr = month != m; va = month == m
        enc = tencode.fit(rawT, yT, delta_T, tr, spread=args.cellspread)
        XteT = XT.join(tencode.transform(rawT, enc)); Xte = X.join(tencode.transform(raw, enc))
        for p in (decomposition.BASE_PARAMS, decomposition.SEVERITY_PARAMS):
            p["seed"] = 100
        models = decomposition.fit_fixed(XteT.loc[tr], ysT.loc[tr], cats, rounds=rounds, uncapped_matched=args.uncapped)
        if args.uncapped:
            comp = decomposition.predict_components(models, Xte.loc[va], matched_uncapped=True)
            m_va = matched[va]
            # matched rows: uncapped base alone; unmatched rows: capped base + tail term as before
            ordinary[va] = np.where(m_va, comp["base"], comp["prediction"])
        else:
            ordinary[va] = decomposition.predict(models, Xte.loc[va])
        g = lgb.train({**mechanism.GATE_PARAMS, "num_threads": args.threads, "seed": 3},
                      lgb.Dataset(XteT.loc[tr, gate_cols], fb[tr].astype("int8"), categorical_feature=gate_cats), num_boost_round=200)
        gate_new[va] = g.predict(Xte.loc[va, gate_cols])
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    out = lobt.apply(mechanism.blend(gate_new, proxy, ordinary), expert, airport, alphas)
    out = lirfregime.apply(out, p_regime, X)
    for m in range(1, 13):
        table = lirfbands.fit(XT.loc[month != m], yT[month != m]); va = month == m; out[va] = lirfbands.apply(out[va], X.loc[va], table)
    if args.rome_rules:
        out = lirfbands.rome_rules(out, raw, X)
    s = lirfbands.short_scope(X); out[s] = short[s]; new = np.clip(lirfbands.lobt_clip(out, X), 0, None)
    scored = np.isin(month, (1, 7)); rep = {"rounds": rounds, "lr": sel["lr"], "uncapped": args.uncapped, "rome_rules": args.rome_rules}
    for name, mask in [("scored_jan_jul", scored), ("jan", month == 1), ("jul", month == 7), ("all_months", np.ones(len(y), bool)), ("other_months", ~scored),
                       ("matched_scored", matched & scored)]:
        rep[name] = {"v19": rmse(y[mask], v19[mask]), "v20": rmse(y[mask], new[mask])}; rep[name]["delta"] = rep[name]["v20"] - rep[name]["v19"]
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], v19[month == m]) for m in range(1, 13)}
    ys_, bs, ns = y[scored], v19[scored], new[scored]; order = np.argsort((ys_ - ns) ** 2 - (ys_ - bs) ** 2); rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys_), bool); keep[order[:k]] = False; rep["robustness_scored"][f"drop_{k}"] = rmse(ys_[keep], ns[keep]) - rmse(ys_[keep], bs[keep])
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": new, "ordinary": ordinary, "gate_new": gate_new}).to_parquet(config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all_months", "other_months", "matched_scored"):
        r = rep[k]; print(f"{k:15s} v19 {r['v19']:.3f} -> v20 {r['v20']:.3f} ({r['delta']:+.3f})")
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()}); print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


if __name__ == "__main__":
    main()
