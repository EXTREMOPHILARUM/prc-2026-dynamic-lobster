"""Baseline LightGBM model for taxi-out time.

    uv run python -m prc.train --months 12
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from prc import config, data, evaluation, features

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": 0,
}


#: the ranking set is January + July 2026, so validate on the same two months of 2025
VALID_MONTHS = (1, 7)


def month_disjoint_split(
    df: pd.DataFrame, valid_months: tuple[int, ...] = VALID_MONTHS
) -> tuple[pd.Index, pd.Index]:
    """Hold out whole months, mirroring the ranking set's seasonal position.

    A chronological cut would validate on December alone and tell us nothing about
    January or July; the ranking months sit at both extremes of the year.
    """
    return evaluation.month_disjoint_split(df, valid_months)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=None, help="use only the last N monthly files")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=100)
    args = ap.parse_args()

    raw = data.load_training(months=args.months)
    raw = features.departures_with_target(raw).reset_index(drop=True)
    print(f"training rows: {len(raw):,}")

    X, cats = features.build(raw)
    y = raw[config.TARGET].astype("float32")
    # `month` leaks the split itself: it is constant on the holdout and the model
    # would learn a value it can never see for Jan/Jul at scoring time.
    X = X.drop(columns=["month"])
    tr_idx, va_idx = month_disjoint_split(raw)
    print(f"train {len(tr_idx):,} / valid {len(va_idx):,}  features: {list(X.columns)}")

    dtrain = lgb.Dataset(X.loc[tr_idx], y.loc[tr_idx], categorical_feature=cats)
    dvalid = lgb.Dataset(X.loc[va_idx], y.loc[va_idx], reference=dtrain)

    model = lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping),
            lgb.log_evaluation(100),
        ],
    )

    pred = model.predict(X.loc[va_idx], num_iteration=model.best_iteration)
    rmse = root_mean_squared_error(y.loc[va_idx], pred)
    per_month = (
        pd.DataFrame({"m": pd.to_datetime(raw.loc[va_idx, features.TIME_COL]).dt.month,
                      "y": y.loc[va_idx].to_numpy(), "p": pred})
        .groupby("m")
        .apply(lambda g: root_mean_squared_error(g.y, g.p), include_groups=False)
    )
    print("per-month holdout RMSE:", {int(k): round(v, 2) for k, v in per_month.items()})
    baseline = root_mean_squared_error(y.loc[va_idx], np.full(len(va_idx), y.loc[tr_idx].mean()))
    print(f"holdout RMSE: {rmse:.2f} s   (global-mean baseline {baseline:.2f} s)")

    best_iteration = model.best_iteration
    print(f"refitting {best_iteration} rounds on all {len(raw):,} rows")
    final_model = lgb.train(
        PARAMS,
        lgb.Dataset(X, y, categorical_feature=cats),
        num_boost_round=best_iteration,
    )
    final_model.save_model(str(config.MODELS / "lgbm_baseline.txt"))
    (config.MODELS / "lgbm_baseline.json").write_text(
        json.dumps(
            {
                "rmse": float(rmse),
                "baseline_rmse": float(baseline),
                "best_iteration": best_iteration,
                "validation_months": list(VALID_MONTHS),
                "refit_rows": len(raw),
                "features": list(X.columns),
                "categoricals": cats,
                "category_levels": features.category_levels(X, cats),
                "rows": len(raw),
            },
            indent=2,
        )
    )
    print(f"saved {config.MODELS / 'lgbm_baseline.txt'}")


if __name__ == "__main__":
    main()
