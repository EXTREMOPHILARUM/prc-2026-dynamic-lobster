"""Refit the v13 candidate: v12 configuration with identity columns removed from the base.

Same features as the shipped `decomposition_wx` (METAR without the storm codes, plus
sequencing); the change is `decomposition.BASE_EXCLUDED` and the categorical
regularisation in `decomposition.BASE_PARAMS`. Saved under new stems so v12's artifacts
remain reproducible.

    uv run python scripts/refit_noid.py
"""
from __future__ import annotations

import json
import shutil

from prc import config, data, decomposition, features, sequencing, weather

STORM_FEATURES = ["wx_thunder", "wx_shower", "wx_hail", "wx_thunder_hours_3h"]


def main() -> None:
    raw = sequencing.add(weather.attach(
        features.departures_with_target(data.load_training()).reset_index(drop=True)
    ))
    wx = [c for c in weather.FEATURES if c not in STORM_FEATURES]
    extra = wx + list(sequencing.FEATURES)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"]).join(raw[extra])
    y = raw[config.TARGET].astype("float32")
    rounds = json.loads(
        (config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    shipped = json.loads((config.MODELS / "decomposition_wx.json").read_text())
    if list(X.columns) != shipped["features"]:
        raise SystemExit("feature list differs from the shipped decomposition_wx")
    print(f"refitting on {len(raw):,} rows, {len(X.columns)} features, "
          f"base excludes {decomposition.BASE_EXCLUDED}")
    models = decomposition.fit_fixed(X, y, cats, rounds=rounds)
    oof = json.loads((config.DATA_INTERIM / "noid_prod_oof_metrics.json").read_text())
    decomposition.save(models, {
        "features": list(X.columns),
        "categoricals": cats,
        "category_levels": features.category_levels(X, cats),
        "rounds": rounds,
        "rows": len(raw),
        "tail_threshold": decomposition.TAIL_THRESHOLD,
        "lirf_tail_scale": decomposition.LIRF_TAIL_SCALE,
        "other_tail_scale": decomposition.OTHER_TAIL_SCALE,
        "base_excluded": decomposition.BASE_EXCLUDED,
        "rmse_oof": oof["all_months"]["rmse_new"],
    }, config.MODELS / "decomposition_noid")
    mech = json.loads((config.MODELS / "mechanism_wx.json").read_text())
    mech |= {
        "ordinary_stem": "decomposition_noid",
        "features": list(X.columns),
        "category_levels": features.category_levels(X, cats),
        "rmse_oof": oof["all_months"]["rmse_new"],
        "rmse_oof_baseline": oof["all_months"]["rmse_base"],
    }
    (config.MODELS / "mechanism_noid.json").write_text(json.dumps(mech, indent=2))
    shutil.copy(config.MODELS / "mechanism_wx.gate.txt", config.MODELS / "mechanism_noid.gate.txt")
    print("saved models/decomposition_noid.* and models/mechanism_noid.*")


if __name__ == "__main__":
    main()
