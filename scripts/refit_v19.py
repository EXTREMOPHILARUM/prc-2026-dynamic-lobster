"""Refit v19: three-seed decomposition at the context round counts, gate refit with context
features, finer Rome bands; wrapper inherits v18's regime, OOF alphas and short expert.

    uv run python -u scripts/refit_v19.py
"""
from __future__ import annotations

import json

import lightgbm as lgb

from prc import arrivals, config, decomposition, features, lirfbands, matrix, mechanism

SEEDS = (100, 101, 102)


def main() -> None:
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].astype("float32")
    rounds = json.loads((config.DATA_INTERIM / "component_rounds_ctx.json").read_text())["rounds"]
    ctx = json.loads((config.MODELS / "decomposition_ctx.json").read_text())
    assert ctx["features"] == list(X.columns), "feature list differs from decomposition_ctx"
    stems = []
    for s in SEEDS:
        for p in (decomposition.BASE_PARAMS, decomposition.SEVERITY_PARAMS):
            p["seed"] = s
        models = decomposition.fit_fixed(X, y, cats, rounds=rounds)
        stem = f"decomposition_ctx19_s{s}"
        decomposition.save(models, ctx | {"rounds": rounds, "seed": s}, config.MODELS / stem)
        stems.append(stem); print("saved", stem, flush=True)
    # gate refit with context features
    label = mechanism.fallback_label(raw)
    gate_cols = mechanism.gate_features(X) + [c for c in X.columns if c.startswith(("nbr_apt", "q_apt", "arrctx_", "clk_"))]
    gate = lgb.train({**mechanism.GATE_PARAMS, "seed": 3}, lgb.Dataset(X[gate_cols], label.astype("int8"),
                     categorical_feature=[c for c in cats if c in gate_cols]), num_boost_round=200)
    gate.save_model(str(config.MODELS / "mechanism_ctx19.gate.txt"))
    mech = json.loads((config.MODELS / "mechanism_ctx.json").read_text())
    mech |= {"ordinary_stem": stems[0], "seed_stems": stems, "gate_features": gate_cols, "gate_rounds": 200,
             "features": list(X.columns), "category_levels": features.category_levels(X, cats)}
    (config.MODELS / "mechanism_ctx19.json").write_text(json.dumps(mech, indent=2))
    v18 = json.loads((config.MODELS / "mechanism_v18.json").read_text())
    v18 |= {"inner_stem": "mechanism_ctx19", "band_table": lirfbands.fit(X, y.to_numpy(dtype="float64")),
            "band_edges_h": [e if e != float("inf") else None for e in lirfbands.EDGES_H]}
    (config.MODELS / "mechanism_v19.json").write_text(json.dumps(v18, indent=2))
    print("saved mechanism_ctx19.*, mechanism_v19.json; bands", json.dumps(v18["band_table"]["bands"]))


if __name__ == "__main__":
    main()
