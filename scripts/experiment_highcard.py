"""Ablation: do high-cardinality native categoricals hurt the matched-clock base?

Holds out one month at a time, fits only the base_matched component with a fixed round
budget, and reports holdout RMSE of the capped target at the stored round count and at
the best iteration on the curve. Variants drop FLIGHT_mvt / CALLSIGN_flt and/or tighten
LightGBM's categorical regularisation.
"""
from __future__ import annotations

import argparse
import json
import sys

import lightgbm as lgb
import numpy as np

from prc import config, decomposition, evaluation, matrix

HIGHCARD = ["FLIGHT_mvt", "CALLSIGN_flt"]
REG = {"cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 500, "max_cat_threshold": 16}

VARIANTS = {
    "current": (HIGHCARD, {}),
    "drop_highcard": ([], {}),
    "catreg": (HIGHCARD, REG),
    "drop_highcard+catreg": ([], REG),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7])
    ap.add_argument("--rounds", type=int, default=342)
    args = ap.parse_args()

    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True)
    y = raw[config.TARGET].astype("float64")
    month = evaluation.movement_month(raw)
    matched = X[decomposition.ANCHOR_COL].notna()
    capped = y.clip(upper=decomposition.TAIL_THRESHOLD)
    anchor = X[decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD)
    resid = capped - anchor
    results: dict[str, dict] = {}
    for m in args.months:
        valid = (month == m) & matched
        train = (month != m) & matched
        for name, (keep_hc, reg) in VARIANTS.items():
            cols = [c for c in X.columns if c in keep_hc or c not in HIGHCARD]
            vcats = [c for c in cats if c in cols]
            params = {**decomposition.BASE_PARAMS, **reg, "seed": 42}
            rec: dict = {}
            booster = lgb.train(
                params,
                lgb.Dataset(X.loc[train, cols], resid[train], categorical_feature=vcats),
                num_boost_round=args.rounds,
                valid_sets=[lgb.Dataset(X.loc[valid, cols], resid[valid], categorical_feature=vcats)],
                valid_names=["v"],
                callbacks=[lgb.record_evaluation(rec)],
            )
            curve = np.array(rec["v"]["rmse"])
            pred = booster.predict(X.loc[valid, cols], num_iteration=args.rounds)
            rmse_fixed = float(np.sqrt(np.mean((resid[valid] - pred) ** 2)))
            best_it = int(curve.argmin()) + 1
            results[f"{m}:{name}"] = {
                "rmse_fixed": rmse_fixed,
                "rmse_best": float(curve.min()),
                "best_iter": best_it,
                "n_valid": int(valid.sum()),
            }
            print(
                f"month {m:02d} {name:22s} fixed {rmse_fixed:8.3f}  best {curve.min():8.3f} @ {best_it}",
                flush=True,
            )
    out = config.DATA_INTERIM / "highcard_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
