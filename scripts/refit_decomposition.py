"""Refit the selected decomposition on all training months."""
from __future__ import annotations

import json

from prc import config, data, decomposition, features


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    oof_metrics = json.loads(
        (config.DATA_INTERIM / "decomposition_oof_metrics.json").read_text()
    )
    holdout_metrics = json.loads(
        (config.DATA_INTERIM / "decomposition_holdout_metrics.json").read_text()
    )
    print(f"refitting decomposition on all {len(raw):,} rows")
    models = decomposition.fit_fixed(X, y, cats)
    decomposition.save(
        models,
        {
            "features": list(X.columns),
            "categoricals": cats,
            "category_levels": features.category_levels(X, cats),
            "rounds": decomposition.DEFAULT_ROUNDS,
            "rows": len(raw),
            "tail_threshold": decomposition.TAIL_THRESHOLD,
            "lirf_tail_scale": decomposition.LIRF_TAIL_SCALE,
            "other_tail_scale": decomposition.OTHER_TAIL_SCALE,
            "rmse_oof": oof_metrics["all"]["rmse"],
            "rmse_bulk_oof": oof_metrics["bulk"]["rmse"],
            "rmse_holdout": holdout_metrics["all"]["rmse"],
            "rmse_bulk_holdout": holdout_metrics["bulk"]["rmse"],
        },
        config.MODELS / "decomposition",
    )
    print(f"saved {config.MODELS / 'decomposition'}.*")


if __name__ == "__main__":
    main()
