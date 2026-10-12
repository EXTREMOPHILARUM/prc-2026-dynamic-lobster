"""Refit the ordinary component with the arrival-proximity features and save it.

Ships the four **fixed-window** features only. The two `arr_during_taxi_*` variants are
excluded deliberately: their window length is `MVT - AOBT`, which is the taxi proxy itself,
so they partly re-encode a signal the model already has. Dropping them costs 0.0016 RMSE
and they are marginally worse on the scored months (see docs/agent-reports/arrivals.md).

    uv run python scripts/refit_arrivals.py
"""
from __future__ import annotations

import json
import shutil

from prc import arrivals, config, decomposition, features, matrix, sequencing, weather

#: the six-feature family minus the two taxi-length windows
SHIP = list(arrivals.FIXED_WINDOW_FEATURES)


def main() -> None:
    raw, X, cats = matrix.build(use_weather=True, use_sequencing=True, use_arrivals=True)
    drop = [c for c in arrivals.FEATURES if c not in SHIP]
    X = X.drop(columns=drop, errors="ignore")
    missing = [c for c in SHIP if c not in X.columns]
    if missing:
        raise SystemExit(f"arrival features missing from the matrix: {missing}")

    y = raw[config.TARGET].astype("float32")
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    print(
        f"refitting on {len(raw):,} rows, {len(X.columns)} features "
        f"({len(weather.FEATURES)} METAR, {len(sequencing.FEATURES)} sequencing, "
        f"{len(SHIP)} arrival); dropped {drop}"
    )

    models = decomposition.fit_fixed(X, y, cats, rounds=rounds)
    decomposition.save(
        models,
        {
            "features": list(X.columns),
            "categoricals": cats,
            "category_levels": features.category_levels(X, cats),
            "rows": len(raw),
            "kind": "decomposition",
            "tail_threshold": decomposition.TAIL_THRESHOLD,
            "lirf_tail_scale": decomposition.LIRF_TAIL_SCALE,
            "other_tail_scale": decomposition.OTHER_TAIL_SCALE,
            "use_weather": True,
            "use_sequencing": True,
            "use_arrivals": True,
            "arrival_features": SHIP,
        },
        config.MODELS / "decomposition_arr",
    )

    # the mechanism wrapper is unchanged apart from pointing at the new ordinary model
    mech = json.loads((config.MODELS / "mechanism_wx.json").read_text())
    mech |= {
        "kind": "mechanism",
        "ordinary_stem": "decomposition_arr",
        "use_arrivals": True,
        "arrival_features": SHIP,
        "features": list(X.columns),
        "category_levels": features.category_levels(X, cats),
    }
    (config.MODELS / "mechanism_arr.json").write_text(json.dumps(mech, indent=2))
    shutil.copy(config.MODELS / "mechanism_wx.gate.txt", config.MODELS / "mechanism_arr.gate.txt")
    print("saved models/decomposition_arr.* and models/mechanism_arr.*")


if __name__ == "__main__":
    main()
