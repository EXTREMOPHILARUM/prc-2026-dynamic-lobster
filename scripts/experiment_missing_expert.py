"""Lane-two expert: departures with no NM record, trained on every row with NM-free features.

The shipped `base_missing` component is fitted on the ~22k unmatched rows alone. This fits
one LightGBM on all 2.08M departures using only columns that exist when the NM record is
absent, then replaces the v15 prediction on the non-LIRF unmatched lane (and reports the
LIRF unmatched rows with gap under 4h separately, without replacing them).

    uv run python -u scripts/experiment_missing_expert.py --cap 12000 --missing-weight 8
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix

NM_DEPENDENT_PREFIXES = ("aobt_minus_", "taxi_proxy_AOBT", "taxi_proxy_EOBT",
                         "taxi_proxy_IOBT", "taxi_proxy_LOBT", "arr_next_20m_aobt",
                         "arr_during_taxi")
NM_DEPENDENT = {"offblock_delay", "nm_time_count", "aobt_missing", "nm_record_missing"}


def nm_free_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns
            if not c.endswith("_flt") and c not in NM_DEPENDENT
            and not c.startswith(NM_DEPENDENT_PREFIXES)
            and c not in decomposition.BASE_EXCLUDED]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=12000.0)
    ap.add_argument("--missing-weight", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--baseline", default="lirfbands_oof_predictions.parquet")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"missexp_cap{int(args.cap)}_w{int(args.missing_weight)}"
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True)
    cols = nm_free_columns(X); vcats = [c for c in cats if c in cols]
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    missing = X["nm_record_missing"].to_numpy() == 1
    lirf = (X["ADEP_mvt"].astype("string") == "LIRF").to_numpy()
    lane2 = missing & ~lirf
    lirf_short = missing & lirf & ~lirfbands.scope(X)
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.baseline),
                                      on=config.ID_COL, how="left")["prediction"].to_numpy()
    target = np.minimum(y, args.cap)
    weight = np.where(missing, args.missing_weight, 1.0)
    params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 42}
    expert = np.full(len(y), np.nan)
    print(f"{tag}: {len(cols)} NM-free features, lane2 rows {lane2.sum():,}, lirf<4h {lirf_short.sum():,}", flush=True)
    for m in range(1, 13):
        tr = month != m
        booster = lgb.train(params, lgb.Dataset(X.loc[tr, cols], target[tr], weight=weight[tr],
                                                categorical_feature=vcats), num_boost_round=args.rounds)
        v = month == m
        expert[v] = np.clip(booster.predict(X.loc[v, cols]), 0, None)
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    new = base.copy(); new[lane2] = expert[lane2]
    scored = np.isin(month, (1, 7))
    rep = {"tag": tag, "features": cols}
    for name, mask in [("scored_jan_jul", scored), ("all_months", np.ones(len(y), bool)),
                       ("lane2_scored", lane2 & scored), ("lane2_all", lane2),
                       ("lirf_short_scored", lirf_short & scored)]:
        p = new if not name.startswith("lirf") else np.where(lirf_short, expert, base)
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
    (config.DATA_INTERIM / f"{tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": new, "expert": expert}).to_parquet(
        config.DATA_INTERIM / f"{tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "all_months", "lane2_scored", "lane2_all", "lirf_short_scored"):
        r = rep[k]; print(f"{k:18s} n={r['n']:7d} {r['base']:9.3f} -> {r['new']:9.3f} ({r['delta']:+.3f})")
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()})
    print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


if __name__ == "__main__":
    main()
