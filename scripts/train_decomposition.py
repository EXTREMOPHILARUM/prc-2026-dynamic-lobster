"""Evaluate the clock-anchored winsorized decomposition on January and July."""
from __future__ import annotations

import json

from prc import config, data, decomposition, evaluation, features


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    train_idx, valid_idx = evaluation.month_disjoint_split(raw)
    print(f"rows {len(raw):,} | train {len(train_idx):,} valid {len(valid_idx):,}")

    models = decomposition.fit(
        X.loc[train_idx], y.loc[train_idx], cats, X.loc[valid_idx], y.loc[valid_idx]
    )
    components = decomposition.predict_components(models, X.loc[valid_idx])
    pred = components["prediction"]
    metrics = evaluation.regression_metrics(raw.loc[valid_idx], y.loc[valid_idx], pred)
    print(
        f"holdout RMSE {metrics['all']['rmse']:.2f} "
        f"bulk {metrics['bulk']['rmse']:.2f} LIRF {metrics['lirf']['rmse']:.2f}"
    )

    output = config.DATA_INTERIM / "decomposition_holdout_metrics.json"
    output.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {output}")

    predictions = raw.loc[valid_idx, [config.ID_COL]].copy()
    for name, values in components.items():
        predictions[name] = values
    pred_output = config.DATA_INTERIM / "decomposition_holdout_predictions.parquet"
    predictions.to_parquet(pred_output, index=False)
    print(f"wrote {pred_output}")


if __name__ == "__main__":
    main()
