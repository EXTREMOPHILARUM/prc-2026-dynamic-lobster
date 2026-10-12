"""Refit the v17 ordinary model (v13 features + context) and the LOBT expert on all 2025.

Writes `decomposition_ctx.*`, `mechanism_ctx.json` (gate reused), `lobt_expert.*`, and the
`mechanism_v17.json` wrapper: bands + short expert + LOBT clip as v16, plus the LOBT expert
blended with per-airport alphas (mean of the twelve fold-fitted sets in v17_oof_metrics.json).

    uv run python scripts/refit_v17.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from prc import (
    arrivals,
    config,
    context,
    decomposition,
    features,
    lirfbands,
    lobt,
    matrix,
    sequencing,
    weather,
)


def main() -> None:
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    ship_arr = list(arrivals.FIXED_WINDOW_FEATURES)
    X = X.drop(columns=[c for c in arrivals.FEATURES if c not in ship_arr], errors="ignore")
    y = raw[config.TARGET].astype("float32")
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    print(f"refitting on {len(raw):,} rows, {len(X.columns)} features "
          f"({len(weather.FEATURES)} METAR, {len(sequencing.FEATURES)} sequencing, {len(ship_arr)} arrival, {len(context.FEATURES)} context)")
    models = decomposition.fit_fixed(X, y, cats, rounds=rounds)
    meta = {
        "features": list(X.columns), "categoricals": cats, "category_levels": features.category_levels(X, cats),
        "rows": len(raw), "kind": "decomposition", "tail_threshold": decomposition.TAIL_THRESHOLD,
        "lirf_tail_scale": decomposition.LIRF_TAIL_SCALE, "other_tail_scale": decomposition.OTHER_TAIL_SCALE,
        "use_weather": True, "use_sequencing": True, "use_arrivals": True, "arrival_features": ship_arr,
        "use_context": True, "base_excluded": decomposition.BASE_EXCLUDED,
    }
    decomposition.save(models, meta, config.MODELS / "decomposition_ctx")
    mech = json.loads((config.MODELS / "mechanism_arr.json").read_text())
    mech |= {"ordinary_stem": "decomposition_ctx", "use_context": True, "features": list(X.columns),
             "category_levels": features.category_levels(X, cats)}
    (config.MODELS / "mechanism_ctx.json").write_text(json.dumps(mech, indent=2))
    shutil.copy(config.MODELS / "mechanism_arr.gate.txt", config.MODELS / "mechanism_ctx.gate.txt")

    booster = lobt.fit(X, y.to_numpy(dtype="float64"), cats)
    booster.save_model(str(config.MODELS / "lobt_expert.txt"))
    fold_alphas = json.loads((config.DATA_INTERIM / "v17_oof_metrics.json").read_text())["alphas"]
    alphas = {k: float(np.mean([fold_alphas[f][k] for f in fold_alphas])) for k in fold_alphas["1"]}
    (config.MODELS / "lobt_expert.json").write_text(json.dumps({"alphas": alphas, "rounds": lobt.ROUNDS, "bound": lobt.BOUND}, indent=2))

    table = lirfbands.fit(X, y.to_numpy(dtype="float64"))
    v16 = json.loads((config.MODELS / "mechanism_v16.json").read_text())
    v16 |= {"inner_stem": "mechanism_ctx", "band_table": table, "lobt_expert_stem": "lobt_expert", "lobt_alphas": alphas,
            "use_context": True}
    (config.MODELS / "mechanism_v17.json").write_text(json.dumps(v16, indent=2))
    print("alphas", {k: round(v, 3) for k, v in alphas.items()})
    print("saved decomposition_ctx.*, mechanism_ctx.*, lobt_expert.*, mechanism_v17.json")


if __name__ == "__main__":
    main()
