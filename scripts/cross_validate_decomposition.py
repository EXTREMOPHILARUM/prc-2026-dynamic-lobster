"""Leave-one-month-out evaluation of the winsorized clock decomposition."""
from __future__ import annotations

import argparse
import json

import numpy as np

from prc import config, data, decomposition, evaluation, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    args = parser.parse_args()

    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw)
    component_names = (
        "base",
        "tail_probability",
        "tail_excess",
        "tail_scale",
        "prediction",
    )
    oof = {name: np.full(len(raw), np.nan, dtype="float64") for name in component_names}

    for valid_month in args.months:
        valid = month == valid_month
        if not valid.any():
            raise ValueError(f"no rows found for month {valid_month}")
        print(f"month {valid_month:02d}: train {(~valid).sum():,} valid {valid.sum():,}")
        models = decomposition.fit_fixed(X.loc[~valid], y.loc[~valid], cats)
        components = decomposition.predict_components(models, X.loc[valid])
        for name, values in components.items():
            oof[name][valid] = values

    selected = month.isin(args.months).to_numpy()
    metrics = evaluation.regression_metrics(
        raw.loc[selected], y.loc[selected], oof["prediction"][selected]
    )
    metrics_output = config.DATA_INTERIM / "decomposition_oof_metrics.json"
    metrics_output.write_text(json.dumps(metrics, indent=2))

    predictions = raw.loc[selected, [config.ID_COL]].copy()
    for name, values in oof.items():
        predictions[name] = values[selected]
    pred_output = config.DATA_INTERIM / "decomposition_oof_predictions.parquet"
    predictions.to_parquet(pred_output, index=False)

    print(f"OOF RMSE {metrics['all']['rmse']:.2f} bulk {metrics['bulk']['rmse']:.2f}")
    print(f"wrote {metrics_output}")
    print(f"wrote {pred_output}")


if __name__ == "__main__":
    main()
