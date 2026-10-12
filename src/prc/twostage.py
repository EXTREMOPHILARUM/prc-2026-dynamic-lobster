"""Two-stage taxi-out model.

The target carries a small fraction of corrupted labels (BLOCK_TIME_UTC_mvt is wrong
on ~0.2% of departures, concentrated at LIRF) which dominate RMSE. A single L2
regressor averages that class into the bulk; see docs/outlier-analysis.md.

Instead decompose the conditional mean, which is what RMSE asks for:

    E[y|x] = P(tail|x) * E[y | tail, x] + (1 - P(tail|x)) * E[y | bulk, x]

Stage 1 is a *calibrated* classifier - no class reweighting, since the probability
itself is the quantity being used, not a ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, features

#: a taxi-out above this is a corrupted label rather than a slow taxi
TAIL_THRESHOLD = 3600

BULK_PARAMS = {
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

CLF_PARAMS = {
    "objective": "binary",
    # logloss, not AUC: the probability is multiplied into the prediction, so it must be
    # calibrated. Selecting on AUC stops at iteration 1 - one feature already ranks the
    # tail perfectly - leaving probabilities at the base rate and the tail mass 2x too big.
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "num_threads": 0,
}

TAIL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "num_threads": 0,
}


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    cats: list[str],
    X_va: pd.DataFrame,
    y_va: pd.Series,
    rounds: int = 3000,
    stopping: int = 100,
) -> dict[str, lgb.Booster]:
    is_tail = y > TAIL_THRESHOLD
    is_tail_va = y_va > TAIL_THRESHOLD

    def _train(params, Xt, yt, Xv, yv, name):
        model = lgb.train(
            params,
            lgb.Dataset(Xt, yt, categorical_feature=cats),
            num_boost_round=rounds,
            valid_sets=[lgb.Dataset(Xv, yv, categorical_feature=cats)],
            valid_names=[name],
            callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(200)],
        )
        print(f"  {name}: best_iter={model.best_iteration} score={model.best_score[name]}")
        return model

    print("stage 1/3 classifier P(tail|x)")
    clf = _train(CLF_PARAMS, X, is_tail.astype(int), X_va, is_tail_va.astype(int), "clf")
    print("stage 2/3 bulk regressor E[y|bulk,x]")
    bulk = _train(BULK_PARAMS, X[~is_tail], y[~is_tail], X_va[~is_tail_va], y_va[~is_tail_va], "bulk")
    print(f"stage 3/3 tail regressor E[y|tail,x]  ({is_tail.sum():,} rows)")
    tail = _train(TAIL_PARAMS, X[is_tail], y[is_tail], X_va[is_tail_va], y_va[is_tail_va], "tail")
    return {"clf": clf, "bulk": bulk, "tail": tail}


def predict(models: dict[str, lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    p = models["clf"].predict(X[models["clf"].feature_name()], num_iteration=models["clf"].best_iteration)
    bulk = models["bulk"].predict(X[models["bulk"].feature_name()], num_iteration=models["bulk"].best_iteration)
    tail = models["tail"].predict(X[models["tail"].feature_name()], num_iteration=models["tail"].best_iteration)
    tail = np.maximum(tail, TAIL_THRESHOLD)
    return np.clip(p * tail + (1 - p) * bulk, 0, None)


def save(models: dict[str, lgb.Booster], meta: dict, stem: Path) -> None:
    for name, model in models.items():
        model.save_model(f"{stem}.{name}.txt")
    Path(f"{stem}.json").write_text(json.dumps(meta | {"kind": "twostage"}, indent=2))


def load(stem: Path) -> tuple[dict[str, lgb.Booster], dict]:
    models = {n: lgb.Booster(model_file=f"{stem}.{n}.txt") for n in ("clf", "bulk", "tail")}
    return models, json.loads(Path(f"{stem}.json").read_text())


def build_xy(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])  # constant on the holdout; leaks the split
    return X, raw[config.TARGET].astype("float32"), cats
