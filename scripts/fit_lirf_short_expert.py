"""Fit the NM-free expert for LIRF unmatched rows under the band scope, on all 12 months.

One LightGBM on every 2025 departure using only columns that exist without an NM record,
target capped at 12,000s, unmatched rows weighted 8x. Selected in
`scripts/experiment_missing_expert.py` (Jan+Jul 341.93 -> 329.64 leave-one-month-out).

    uv run python scripts/fit_lirf_short_expert.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from prc import config, decomposition, matrix

NM_DEPENDENT_PREFIXES = ("aobt_minus_", "taxi_proxy_AOBT", "taxi_proxy_EOBT",
                         "taxi_proxy_IOBT", "taxi_proxy_LOBT", "arr_next_20m_aobt",
                         "arr_during_taxi")
NM_DEPENDENT = {"offblock_delay", "nm_time_count", "aobt_missing", "nm_record_missing"}
CAP = 12000.0
MISSING_WEIGHT = 8.0
ROUNDS = 300


def nm_free_columns(columns) -> list[str]:
    return [c for c in columns
            if not c.endswith("_flt") and c not in NM_DEPENDENT
            and not c.startswith(NM_DEPENDENT_PREFIXES)
            and c not in decomposition.BASE_EXCLUDED]


def main() -> None:
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True)
    cols = nm_free_columns(X.columns)
    vcats = [c for c in cats if c in cols]
    y = raw[config.TARGET].to_numpy(dtype="float64")
    missing = X["nm_record_missing"].to_numpy() == 1
    params = {**decomposition.BASE_PARAMS, "seed": 42}
    booster = lgb.train(params, lgb.Dataset(X[cols], np.minimum(y, CAP),
                                            weight=np.where(missing, MISSING_WEIGHT, 1.0),
                                            categorical_feature=vcats), num_boost_round=ROUNDS)
    stem = config.MODELS / "lirf_short_expert"
    booster.save_model(f"{stem}.txt")
    (config.MODELS / "lirf_short_expert.json").write_text(json.dumps({
        "features": cols, "categoricals": vcats, "cap": CAP, "missing_weight": MISSING_WEIGHT,
        "rounds": ROUNDS, "rows": len(raw)}, indent=2))
    print(f"saved {stem}.txt with {len(cols)} features")


if __name__ == "__main__":
    main()
