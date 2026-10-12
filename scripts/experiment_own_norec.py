"""Own-flight ADS-B expert for no-record departures (handoff 2026-09-29, step 2; log section 59).

`prc.ownexpert` serves matched rows only (it needs the NM anchor). No-record departures outside Rome
are ordinary taxis whose trace, where joined, observes the stand exit and the first taxi-out point:
on 2025's 4,782 joined rows the stand exit alone has RMSE 278 against the chain's 427 on the rows
where it exists. The trace times have large outliers (an earlier stop or tow taken for the
departure), so they are inputs to a learned residual expert rather than predictions.

Scope: `nm_record_missing` rows outside LIRF whose own trace joined (Rome's no-record rows are
mostly schedule copies and day offsets, which the mixture and the band table serve). Base: the v6b
OOF (the v35 chain on the v29 reconstruction; no-record rows are untouched by the matched expert),
with the v36 garbled rule, which is LIRF-only and so disjoint from this scope. Target: the base's
residual clipped to +/-CAP, fixed before the first run so a day-slip row cannot pull a leaf.
Twelve folds by month; configurations are chosen on the ten non-scored months only.

    uv run python -u scripts/experiment_own_norec.py l15 l31 l15_r600_lr025
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)
CAP = ownexpert.NOREC_CAP
BASE_OOF = ("own_expert_bundle_l255_r1200_lr025_v6b_oof.parquet", "own_l255_r1200_lr025")
BASE_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
               "verbose": -1, "seed": 7, "num_threads": 4}
CONFIGS = {
    "l15": {"rounds": 300},
    "l31": {"rounds": 300, "params": {"num_leaves": 31}},
    "l7": {"rounds": 300, "params": {"num_leaves": 7}},
    "l15_r600_lr025": {"rounds": 600, "params": {"learning_rate": 0.025}},
    "l15_md100": {"rounds": 300, "params": {"min_data_in_leaf": 100}},
}


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    ap.add_argument("--cap", type=float, default=CAP, help="residual clip (default fixed before the first run)")
    ap.add_argument("--v7", action="store_true", help="v7 feature file and FEATURES_NOREC_V7 (section 60)")
    ap.add_argument("--v8", action="store_true", help="v8 feature file and FEATURES_NOREC_V8 (section 63)")
    ap.add_argument("--rich", choices=["num", "cat"], help="add the matrix's columns as inputs (log section 63)")
    ap.add_argument("--all-rows", action="store_true", help="scope: every no-record row outside LIRF, joined or not")
    ap.add_argument("--sx", action="store_true", help="add other aircraft's surface congestion sx_* (section 45) as inputs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", help="also fit the first config on all twelve months -> models/SAVE.txt"); args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    ids = raw[[config.ID_COL]]
    y = raw[config.TARGET].to_numpy("float64"); month = evaluation.movement_month(raw).to_numpy()
    v6b = ids.merge(pd.read_parquet(config.DATA_INTERIM / BASE_OOF[0]), on=config.ID_COL, how="left")[BASE_OOF[1]].to_numpy("float64")
    if not np.isfinite(v6b).all():
        raise SystemExit("base OOF did not join onto every row")
    base = np.clip(lirfbands.garbled_copy(v6b, raw, X), 0, None)
    ver = 8 if args.v8 else 7 if args.v7 else 6
    FEATURES = {8: ownexpert.FEATURES_NOREC_V8, 7: ownexpert.FEATURES_NOREC_V7, 6: ownexpert.FEATURES_NOREC}[ver]
    own = ownexpert.load_features([f"data/interim/adsb_own_2025_v{ver}.parquet"], raw[config.ID_COL],
                                  ownexpert.OWN_BY_VERSION[ver])
    f = ownexpert.frame(raw["ADEP_mvt"].to_numpy(), raw["MVT_TIME_UTC_mvt"], base, np.full(len(y), np.nan), own)
    f["gap"] = X[lirfbands.SCHED_COL].to_numpy("float64")
    if args.rich:
        xcols = ownexpert.rich_columns(X, args.rich)
        f = ownexpert.with_rich(f, X, xcols)
        FEATURES = [*FEATURES, *("x_" + c for c in xcols)]
    if args.sx:
        sx = raw[[config.ID_COL]].merge(pd.read_parquet("data/external/adsb/surface_features.parquet"), on=config.ID_COL, how="left")
        sxcols = [c for c in sx.columns if c.startswith("sx_")]
        for c in sxcols:
            f[c] = sx[c].to_numpy("float64")
        FEATURES = [*FEATURES, *sxcols]
    CATS = ["apt"] + [c for c in FEATURES if c[2:] in ownexpert.RICH_CATEGORICALS and c.startswith("x_")]
    if args.all_rows:
        rows = (X[lirfbands.MISSING_COL].to_numpy() == 1) & (f["apt"].astype(str).to_numpy() != ownexpert.NOREC_EXCLUDED_AIRPORT)
    else:
        rows = ownexpert.norec_scope(f, X[lirfbands.MISSING_COL].to_numpy())
    resid = np.clip(y - base, -args.cap, args.cap)
    print(f"scope rows {int(rows.sum()):,} (Jan+Jul {int((rows & np.isin(month, SCORED)).sum()):,}); "
          f"base RMSE on scope {rmse((y - base)[rows]):.1f}", flush=True)
    sc = np.isin(month, SCORED); ns = ~sc
    results = {}
    for name in args.configs:
        cfg = CONFIGS[name]; params = {**BASE_PARAMS, **cfg.get("params", {})}
        cand = base.copy()
        for m in range(1, 13):
            tr = rows & (month != m); va = rows & (month == m)
            if not va.any():
                continue
            b = lgb.train(params, lgb.Dataset(f.loc[tr, FEATURES], resid[tr], categorical_feature=CATS), cfg["rounds"])
            cand[va] = np.clip(base[va] + b.predict(f.loc[va, FEATURES]), 0, None)
        dsse = {m: float(((y - cand) ** 2 - (y - base) ** 2)[month == m].sum()) for m in range(1, 13)}
        results[name] = cand
        print(f"\n== {name}: non-scored dSSE {sum(dsse[m] for m in range(1, 13) if m not in SCORED) / 1e6:+.1f}M, "
              f"non-scored months won {sum(dsse[m] < 0 for m in range(1, 13) if m not in SCORED)}/10")
        for label, msk in (("twelve", np.ones(len(y), bool)), ("Jan+Jul", sc), ("non-scored", ns), ("scope", rows)):
            print(f"   {label:10s} {rmse((y - base)[msk]):8.3f} -> {rmse((y - cand)[msk]):8.3f} ({rmse((y - cand)[msk]) - rmse((y - base)[msk]):+.3f})")
        print("   by month (RMSE delta)", {m: round(rmse((y - cand)[month == m]) - rmse((y - base)[month == m]), 3) for m in range(1, 13)})
        d = ((y - cand) ** 2 - (y - base) ** 2)[sc]; order = np.argsort(d)
        drops = []
        for k in (1, 10, 50):
            keep = np.ones(sc.sum(), bool); keep[order[:k]] = False
            drops.append(rmse((y[sc] - cand[sc])[keep]) - rmse((y[sc] - base[sc])[keep]))
        print("   Jan+Jul drop-1/10/50 " + " / ".join(f"{v:+.3f}" for v in drops))
        pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": cand}).to_parquet(
            config.DATA_INTERIM / f"own_norec_{name}_cap{int(args.cap)}{'_v8' if args.v8 else '_v7' if args.v7 else ''}{args.tag}_oof.parquet", index=False)
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": base}).to_parquet(
        config.DATA_INTERIM / "own_norec_base_oof.parquet", index=False)
    if args.save:
        cfg = CONFIGS[args.configs[0]]
        b = lgb.train({**BASE_PARAMS, **cfg.get("params", {})}, lgb.Dataset(f.loc[rows, FEATURES], resid[rows], categorical_feature=CATS), cfg["rounds"])
        b.save_model(str(config.MODELS / f"{args.save}.txt"))
        print(f"saved models/{args.save}.txt on {int(rows.sum()):,} rows")


if __name__ == "__main__":
    main()
