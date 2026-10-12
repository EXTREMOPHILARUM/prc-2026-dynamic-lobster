"""Refit the ordinary component with METAR features and save a deployable model."""
from __future__ import annotations

import json

from prc import config, data, decomposition, features, sequencing, weather


def main() -> None:
    raw = sequencing.add(weather.attach(
        features.departures_with_target(data.load_training()).reset_index(drop=True)
    ))
    extra = list(weather.FEATURES) + list(sequencing.FEATURES)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"]).join(raw[extra])
    y = raw[config.TARGET].astype("float32")
    rounds = json.loads(
        (config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    print(f"refitting on {len(raw):,} rows, {len(X.columns)} features "
          f"({len(weather.FEATURES)} from METAR)")

    models = decomposition.fit_fixed(X, y, cats, rounds=rounds)
    oof = json.loads((config.DATA_INTERIM / "sequencing_oof_metrics.json").read_text())
    decomposition.save(models, {
        "features": list(X.columns),
        "categoricals": cats,
        "category_levels": features.category_levels(X, cats),
        "rounds": rounds,
        "rows": len(raw),
        "tail_threshold": decomposition.TAIL_THRESHOLD,
        "lirf_tail_scale": decomposition.LIRF_TAIL_SCALE,
        "other_tail_scale": decomposition.OTHER_TAIL_SCALE,
        "rmse_oof": oof["rmse_with"],
    }, config.MODELS / "decomposition_wx")

    mech = json.loads((config.MODELS / "mechanism.json").read_text())
    mech |= {
        "kind": "mechanism",
        "ordinary_stem": "decomposition_wx",
        "use_weather": True,
        "use_sequencing": True,
        "features": list(X.columns),
        "category_levels": features.category_levels(X, cats),
        "rmse_oof": oof["rmse_with"],
        "rmse_oof_baseline": oof["rmse_without"],
        "external_data": {
            "metar": {
                "source": "Iowa Environmental Mesonet ASOS/METAR archive",
                "url": "https://mesonet.agron.iastate.edu/ASOS/",
                "licence": "public domain",
                "licence_url": "https://mesonet.agron.iastate.edu/disclaimer.php",
                "retrieved_by": "scripts/fetch_metar.py",
            }
        },
    }
    (config.MODELS / "mechanism_wx.json").write_text(json.dumps(mech, indent=2))
    # the gate is unchanged; point the wrapper at the existing booster
    import shutil
    shutil.copy(config.MODELS / "mechanism.gate.txt", config.MODELS / "mechanism_wx.gate.txt")
    print("saved models/decomposition_wx.* and models/mechanism_wx.*")


if __name__ == "__main__":
    main()
