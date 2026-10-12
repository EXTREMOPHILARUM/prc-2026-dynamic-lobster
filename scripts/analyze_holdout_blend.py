"""Compare saved model components on the shared January/July holdout."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import config, data, evaluation, features, twostage


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    _, valid_idx = evaluation.month_disjoint_split(raw)
    valid = raw.loc[valid_idx].copy()
    y = valid[config.TARGET].to_numpy()

    models, meta = twostage.load(config.MODELS / "twostage")
    X, _ = features.build(valid, categories=meta["category_levels"])
    X = X[meta["features"]]
    old_prediction = twostage.predict(models, X)
    probability = models["clf"].predict(X[models["clf"].feature_name()])
    old_tail = models["tail"].predict(X[models["tail"].feature_name()])

    new = pd.read_parquet(config.DATA_INTERIM / "decomposition_holdout_predictions.parquet")
    new = valid[[config.ID_COL]].merge(new, on=config.ID_COL, how="left", validate="one_to_one")
    if new["prediction"].isna().any():
        raise ValueError("decomposition predictions do not cover the holdout")

    candidates = {
        "old_twostage": old_prediction,
        "new_decomposition": new["prediction"].to_numpy(),
        "new_base_old_tail_excess": new["base"].to_numpy()
        + probability * np.maximum(old_tail - 3600, 0),
    }
    for weight in (0.25, 0.5, 0.75):
        candidates[f"blend_new_{weight:.2f}"] = (
            weight * new["prediction"].to_numpy() + (1 - weight) * old_prediction
        )

    results = {
        name: evaluation.regression_metrics(valid, y, prediction)
        for name, prediction in candidates.items()
    }
    for name, metrics in results.items():
        print(
            f"{name:32s} full={metrics['all']['rmse']:7.2f} "
            f"bulk={metrics['bulk']['rmse']:7.2f} LIRF={metrics['lirf']['rmse']:8.2f} "
            f"Jan={metrics['by_month']['1']['rmse']:7.2f} "
            f"Jul={metrics['by_month']['7']['rmse']:7.2f}"
        )

    output = config.DATA_INTERIM / "holdout_blend_metrics.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
