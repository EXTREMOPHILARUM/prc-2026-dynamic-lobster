"""Generate leave-one-month-out predictions and diagnostics for the baseline."""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from prc import config, data, evaluation, features, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    args = parser.parse_args()

    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw)
    oof = np.full(len(raw), np.nan, dtype="float64")

    for valid_month in args.months:
        valid = month == valid_month
        if not valid.any():
            raise ValueError(f"no rows found for month {valid_month}")
        print(f"month {valid_month:02d}: train {(~valid).sum():,} valid {valid.sum():,}")
        model = lgb.train(
            train.PARAMS,
            lgb.Dataset(X.loc[~valid], y.loc[~valid], categorical_feature=cats),
            num_boost_round=args.rounds,
        )
        oof[valid] = model.predict(X.loc[valid])

    selected = month.isin(args.months).to_numpy()
    metrics = evaluation.regression_metrics(raw.loc[selected], y.loc[selected], oof[selected])
    output = config.DATA_INTERIM / "baseline_oof_metrics.json"
    output.write_text(json.dumps(metrics, indent=2))
    print(f"OOF RMSE {metrics['all']['rmse']:.2f} bulk {metrics['bulk']['rmse']:.2f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
