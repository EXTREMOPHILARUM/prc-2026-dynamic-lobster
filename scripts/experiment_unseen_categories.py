"""Does the production/CV mismatch in unseen FLIGHT/CALLSIGN levels matter?

In whole-month CV only ~3% of holdout rows carry a flight number unseen in the other
eleven months; in the 2026 ranking set 21-24% do. This fits the matched-clock base with
and without the two high-cardinality identifiers and scores each holdout month both intact
and with those identifiers masked to NaN on a random share of rows matching the ranking
rate, which is how LightGBM sees an unseen level.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from prc import config, decomposition, evaluation, matrix

HIGHCARD = ["FLIGHT_mvt", "CALLSIGN_flt"]
REG = {"cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 500, "max_cat_threshold": 16}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7])
    ap.add_argument("--rounds", type=int, default=342)
    ap.add_argument("--mask-rate", type=float, default=0.22)
    args = ap.parse_args()

    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True)
    y = raw[config.TARGET].astype("float64")
    month = evaluation.movement_month(raw)
    matched = X[decomposition.ANCHOR_COL].notna()
    resid = y.clip(upper=decomposition.TAIL_THRESHOLD) - X[decomposition.ANCHOR_COL].clip(
        0, decomposition.TAIL_THRESHOLD
    )
    rng = np.random.default_rng(0)
    results: dict[str, dict] = {}
    for m in args.months:
        valid = (month == m) & matched
        train = (month != m) & matched
        Xv = X.loc[valid]
        Xv_masked = Xv.copy()
        mask = rng.random(len(Xv)) < args.mask_rate
        for c in HIGHCARD:
            Xv_masked.loc[mask, c] = np.nan
        for name, cols, reg in [
            ("current", list(X.columns), {}),
            ("drop_highcard+catreg", [c for c in X.columns if c not in HIGHCARD], REG),
        ]:
            vcats = [c for c in cats if c in cols]
            booster = lgb.train(
                {**decomposition.BASE_PARAMS, **reg, "seed": 42},
                lgb.Dataset(X.loc[train, cols], resid[train], categorical_feature=vcats),
                num_boost_round=args.rounds,
            )
            intact = rmse(resid[valid], booster.predict(Xv[cols]))
            masked = rmse(resid[valid], booster.predict(Xv_masked[cols]))
            results[f"{m}:{name}"] = {"intact": intact, "masked": masked, "n": int(valid.sum())}
            print(f"month {m:02d} {name:22s} intact {intact:8.3f}  masked {masked:8.3f}", flush=True)
    out = config.DATA_INTERIM / "unseen_categories.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
