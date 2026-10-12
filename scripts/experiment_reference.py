"""A/B the stand x runway reference encoding on the month-disjoint holdout.

Trains the same model with and without the reference features, so the delta is
attributable. Reports full and bulk-only RMSE - full RMSE swings on tail prevalence
and is not a reliable selection signal on its own (docs/submissions.md).

    uv run python scripts/experiment_reference.py
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from prc import config, data, features, reference, train

TAIL = 3600


def evaluate(X, y, tr_idx, va_idx, cats, label):
    model = lgb.train(
        train.PARAMS,
        lgb.Dataset(X.loc[tr_idx], y.loc[tr_idx], categorical_feature=cats),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(X.loc[va_idx], y.loc[va_idx], categorical_feature=cats)],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict(X.loc[va_idx], num_iteration=model.best_iteration)
    yv = y.loc[va_idx]
    bulk = yv <= TAIL
    full = root_mean_squared_error(yv, pred)
    bulk_rmse = root_mean_squared_error(yv[bulk], pred[bulk])
    print(f"{label:24} full {full:7.2f}   bulk {bulk_rmse:7.2f}   iters {model.best_iteration}")
    return model, full, bulk_rmse


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    month = pd.to_datetime(raw[features.TIME_COL], utc=True).dt.month

    print("building leave-one-month-out reference encoding...")
    raw = reference.add_out_of_fold(raw, month)

    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    # features.build selects a fixed column list, so the reference columns must be
    # attached explicitly - otherwise both arms silently train on identical features.
    X = X.join(raw[reference.FEATURES])
    missing = [c for c in reference.FEATURES if c not in X.columns]
    if missing:
        raise SystemExit(f"reference features missing from X: {missing}")
    y = raw[config.TARGET].astype("float32")
    tr_idx, va_idx = train.month_disjoint_split(raw)
    print(f"rows {len(raw):,} | train {len(tr_idx):,} valid {len(va_idx):,}\n")

    base_cols = [c for c in X.columns if c not in reference.FEATURES]
    _, f0, b0 = evaluate(X[base_cols], y, tr_idx, va_idx, cats, "without reference")
    model, f1, b1 = evaluate(X, y, tr_idx, va_idx, cats, "with reference")

    print(f"\ndelta: full {f1 - f0:+.2f}   bulk {b1 - b0:+.2f}")
    imp = pd.Series(model.feature_importance("gain"), index=X.columns)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\ntop features by gain (%):")
    print(imp.head(10).round(1).to_string())
    print("\nreference features:")
    print(imp[reference.FEATURES].round(1).to_string())


if __name__ == "__main__":
    main()
