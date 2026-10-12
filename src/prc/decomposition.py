"""Winsorized conditional-mean model with explicit clock-availability experts."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

TAIL_THRESHOLD = 3600.0
ANCHOR_COL = "taxi_proxy_AOBT_3_flt"
LIRF_TAIL_SCALE = 1.0
OTHER_TAIL_SCALE = 0.25
TAIL_FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "ADES_FILED_flt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "hour",
    "minute_of_day",
    "dow",
    "doy",
    "is_weekend",
    "tod_sin",
    "tod_cos",
    "adep_movements_hour",
    "offblock_delay",
    "taxi_proxy_AOBT_3_flt",
    "taxi_proxy_EOBT_1_flt",
    "taxi_proxy_IOBT_flt",
    "taxi_proxy_LOBT_flt",
    "taxi_proxy_SCHED_TIME_UTC_mvt",
]

#: Identity columns the base models must not see. ATC callsigns and flight numbers rotate
#: with the season: every whole-month CV holdout has its season-mates in training, so under
#: 2% of holdout rows carry an unseen level, but 36-42% of July 2026 ranking rows do. The
#: two columns held 11% of the matched base's split gain and that gain does not transfer;
#: dropping them wins on every fold tested (scripts/experiment_highcard.py) and removes the
#: arbitrary default-branch routing of unseen levels (scripts/experiment_unseen_categories.py).
BASE_EXCLUDED = ["FLIGHT_mvt", "CALLSIGN_flt"]

BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # stronger categorical regularisation for the remaining identity columns (stand,
    # operator, destination); measured together with BASE_EXCLUDED on four folds
    "cat_smooth": 50,
    "cat_l2": 50,
    "min_data_per_group": 500,
    "max_cat_threshold": 16,
    "verbosity": -1,
    "num_threads": 0,
}


def base_columns(X: pd.DataFrame, cats: list[str]) -> tuple[list[str], list[str]]:
    """Feature and categorical lists for the two base models."""
    cols = [c for c in X.columns if c not in BASE_EXCLUDED]
    return cols, [c for c in cats if c in cols]

GATE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "num_threads": 0,
}

#: The severity model was originally kept tiny (15 leaves, 57 rounds) to avoid
#: overfitting a few thousand tail rows. Measured on a Feb+Oct split - disjoint from the
#: ranking months - that was badly under-powered: capacity improves full RMSE
#: monotonically (7 leaves 298.35, 15 leaves 289.69, 31 leaves 289.16, 63 leaves 281.59).
#: The tail has real structure and a small model cannot represent it.
SEVERITY_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 10,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
}

DEFAULT_ROUNDS = {
    "base_matched": 269,
    "base_missing": 125,
    "gate": 71,
    "severity": 300,
}


def _train(
    params: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cats: list[str],
    name: str,
    rounds: int,
    stopping: int,
) -> lgb.Booster:
    model = lgb.train(
        params,
        lgb.Dataset(X, y, categorical_feature=cats),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X_valid, y_valid, categorical_feature=cats)],
        valid_names=[name],
        callbacks=[lgb.early_stopping(stopping, verbose=False), lgb.log_evaluation(200)],
    )
    print(f"  {name}: best_iter={model.best_iteration} score={model.best_score[name]}")
    return model


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    cats: list[str],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    rounds: int = 3000,
    stopping: int = 100,
) -> dict[str, lgb.Booster]:
    """Fit capped-base, tail-probability, and excess-severity components."""
    matched = X[ANCHOR_COL].notna()
    matched_valid = X_valid[ANCHOR_COL].notna()
    capped = y.clip(upper=TAIL_THRESHOLD)
    capped_valid = y_valid.clip(upper=TAIL_THRESHOLD)
    anchor = X.loc[matched, ANCHOR_COL].clip(0, TAIL_THRESHOLD)
    anchor_valid = X_valid.loc[matched_valid, ANCHOR_COL].clip(0, TAIL_THRESHOLD)
    tail = y > TAIL_THRESHOLD
    tail_valid = y_valid > TAIL_THRESHOLD
    tail_features = [column for column in TAIL_FEATURES if column in X.columns]
    tail_cats = [column for column in cats if column in tail_features]
    base_cols, base_cats = base_columns(X, cats)

    print("stage 1/4 matched-clock capped residual")
    base_matched = _train(
        BASE_PARAMS,
        X.loc[matched, base_cols],
        capped.loc[matched] - anchor,
        X_valid.loc[matched_valid, base_cols],
        capped_valid.loc[matched_valid] - anchor_valid,
        base_cats,
        "base_matched",
        rounds,
        stopping,
    )
    print("stage 2/4 missing-clock capped target")
    base_missing = _train(
        BASE_PARAMS,
        X.loc[~matched, base_cols],
        capped.loc[~matched],
        X_valid.loc[~matched_valid, base_cols],
        capped_valid.loc[~matched_valid],
        base_cats,
        "base_missing",
        rounds,
        stopping,
    )
    print("stage 3/4 tail probability")
    gate = _train(
        GATE_PARAMS,
        X[tail_features],
        tail.astype("int8"),
        X_valid[tail_features],
        tail_valid.astype("int8"),
        tail_cats,
        "gate",
        rounds,
        stopping,
    )
    print("stage 4/4 excess severity")
    severity = _train(
        SEVERITY_PARAMS,
        X.loc[tail, tail_features],
        y.loc[tail] - TAIL_THRESHOLD,
        X_valid.loc[tail_valid, tail_features],
        y_valid.loc[tail_valid] - TAIL_THRESHOLD,
        tail_cats,
        "severity",
        rounds,
        stopping,
    )
    return {
        "base_matched": base_matched,
        "base_missing": base_missing,
        "gate": gate,
        "severity": severity,
    }


def fit_fixed(
    X: pd.DataFrame,
    y: pd.Series,
    cats: list[str],
    rounds: dict[str, int] | None = None,
    uncapped_matched: bool = False,
) -> dict[str, lgb.Booster]:
    """Fit all components without a validation set, for OOF folds or final refits.

    `uncapped_matched` fits the matched base on the full residual `y - anchor` instead of
    the winsorised one. merry-quicksand measured winsorising 1% of the clock delta at -30s;
    here the uncapped base inside the full pipeline is -0.7 to -1.9s per month on matched
    rows (scripts/experiment_uncapped_pipeline.py). The tail components are still fitted so
    `predict` keeps working; callers wanting the uncapped base alone pass tail scale 0.
    """
    rounds = DEFAULT_ROUNDS if rounds is None else rounds
    matched = X[ANCHOR_COL].notna()
    capped = y.clip(upper=TAIL_THRESHOLD)
    matched_target = (y if uncapped_matched else capped)
    anchor = X.loc[matched, ANCHOR_COL].clip(0, TAIL_THRESHOLD)
    tail = y > TAIL_THRESHOLD
    tail_features = [column for column in TAIL_FEATURES if column in X.columns]
    tail_cats = [column for column in cats if column in tail_features]
    base_cols, base_cats = base_columns(X, cats)

    def train_model(
        params: dict[str, object],
        frame: pd.DataFrame,
        target: pd.Series,
        categorical: list[str],
        name: str,
    ) -> lgb.Booster:
        return lgb.train(
            params,
            lgb.Dataset(frame, target, categorical_feature=categorical),
            num_boost_round=rounds[name],
        )

    return {
        "base_matched": train_model(
            BASE_PARAMS,
            X.loc[matched, base_cols],
            matched_target.loc[matched] - anchor,
            base_cats,
            "base_matched",
        ),
        "base_missing": train_model(
            BASE_PARAMS,
            X.loc[~matched, base_cols],
            capped.loc[~matched],
            base_cats,
            "base_missing",
        ),
        "gate": train_model(
            GATE_PARAMS,
            X[tail_features],
            tail.astype("int8"),
            tail_cats,
            "gate",
        ),
        "severity": train_model(
            SEVERITY_PARAMS,
            X.loc[tail, tail_features],
            y.loc[tail] - TAIL_THRESHOLD,
            tail_cats,
            "severity",
        ),
    }


def save(models: dict[str, lgb.Booster], meta: dict[str, object], stem: Path) -> None:
    for name, model in models.items():
        model.save_model(f"{stem}.{name}.txt")
    Path(f"{stem}.json").write_text(json.dumps(meta | {"kind": "decomposition"}, indent=2))


def load(stem: Path) -> tuple[dict[str, lgb.Booster], dict[str, object]]:
    models = {
        name: lgb.Booster(model_file=f"{stem}.{name}.txt") for name in DEFAULT_ROUNDS
    }
    return models, json.loads(Path(f"{stem}.json").read_text())


def predict_components(
    models: dict[str, lgb.Booster],
    X: pd.DataFrame,
    lirf_tail_scale: float = LIRF_TAIL_SCALE,
    other_tail_scale: float = OTHER_TAIL_SCALE,
    matched_uncapped: bool = False,
) -> dict[str, np.ndarray]:
    """Predict each conditional-mean component for diagnostics and blending."""
    matched = X[ANCHOR_COL].notna().to_numpy()
    base = np.empty(len(X), dtype="float64")
    if matched.any():
        model = models["base_matched"]
        residual = model.predict(X.loc[matched, model.feature_name()])
        base[matched] = X.loc[matched, ANCHOR_COL].clip(0, TAIL_THRESHOLD) + residual
    if (~matched).any():
        model = models["base_missing"]
        base[~matched] = model.predict(X.loc[~matched, model.feature_name()])

    gate = models["gate"]
    severity = models["severity"]
    probability = gate.predict(X[gate.feature_name()])
    excess = np.maximum(severity.predict(X[severity.feature_name()]), 0)
    tail_scale = np.where(
        X["ADEP_mvt"].astype("string").eq("LIRF"), lirf_tail_scale, other_tail_scale
    )
    base = np.clip(base, 0, None if matched_uncapped else TAIL_THRESHOLD)
    return {
        "base": base,
        "tail_probability": probability,
        "tail_excess": excess,
        "tail_scale": tail_scale,
        "prediction": np.clip(base + tail_scale * probability * excess, 0, None),
    }


def predict(
    models: dict[str, lgb.Booster],
    X: pd.DataFrame,
    lirf_tail_scale: float = LIRF_TAIL_SCALE,
    other_tail_scale: float = OTHER_TAIL_SCALE,
    matched_uncapped: bool = False,
) -> np.ndarray:
    """Predict the conditional mean on the original target scale."""
    return predict_components(
        models, X, lirf_tail_scale, other_tail_scale, matched_uncapped)["prediction"]
