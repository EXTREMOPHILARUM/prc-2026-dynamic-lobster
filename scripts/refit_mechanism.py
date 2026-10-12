"""Fit the schedule-fallback gate on all 12 months and save a deployable mechanism model.

The mechanism model is a thin wrapper: the gate plus a pointer to the ordinary model whose
predictions it blends with the schedule reconstruction.

    uv run python scripts/refit_mechanism.py
"""
from __future__ import annotations

import json

import lightgbm as lgb

from prc import config, data, features, mechanism

GATE_ROUNDS = 200
ORDINARY_STEM = "decomposition"


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    label = mechanism.fallback_label(raw)
    gate_cols = mechanism.gate_features(X)
    gate_cats = [c for c in cats if c in gate_cols]
    print(f"fitting gate on {len(raw):,} rows, {label.sum():,} fallback "
          f"({100 * label.mean():.3f}%), {len(gate_cols)} features")

    gate = lgb.train(
        mechanism.GATE_PARAMS,
        lgb.Dataset(X[gate_cols], label.astype("int8"), categorical_feature=gate_cats),
        num_boost_round=GATE_ROUNDS,
    )
    stem = config.MODELS / "mechanism"
    gate.save_model(f"{stem}.gate.txt")

    oof = json.loads((config.DATA_INTERIM / "mechanism_oof_metrics.json").read_text())
    meta = {
        "kind": "mechanism",
        "ordinary_stem": ORDINARY_STEM,
        "gate_features": gate_cols,
        "gate_rounds": GATE_ROUNDS,
        "proxy_col": mechanism.PROXY_COL,
        "fallback_tolerance": mechanism.FALLBACK_TOLERANCE,
        "features": list(X.columns),
        "categoricals": cats,
        "category_levels": features.category_levels(X, cats),
        "rows": len(raw),
        "fallback_rows": int(label.sum()),
        "rmse_oof": oof["rmse_new"],
        "rmse_oof_baseline": oof["rmse_old"],
        "gate_auc_oof": oof["auc"],
    }
    (config.MODELS / "mechanism.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {stem}.gate.txt and mechanism.json  (OOF {oof['rmse_new']:.2f})")


if __name__ == "__main__":
    main()
