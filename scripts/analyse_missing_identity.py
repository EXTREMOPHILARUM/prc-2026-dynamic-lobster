"""Measure whether flight identity generalises on the no-NM-record lane.

The test separates two questions that are easy to conflate:

1. Does airline / aircraft / flight identity predict that the NM record is absent?
2. Once absence is directly observed, does identity improve v29's taxi prediction?

For question two, the fixed candidate is a leave-one-month-out, K=50 smoothed
residual encoding for ``airline + numeric flight number + route``, applied only to
rows where the NM record is absent.  K=50 is the existing target-encoding prior;
it is not selected from the scored months.

    uv run python scripts/analyse_missing_identity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, lirfmoe, matrix

PRIOR = 50.0
SCORED_MONTHS = (1, 7)


def identity_keys(raw: pd.DataFrame) -> dict[str, pd.Series]:
    flight = raw["FLIGHT_mvt"].astype("string").str.upper().str.strip()
    airline = flight.str.extract(r"^([A-Z]{2,3})", expand=False).fillna("?")
    number = flight.str.extract(r"^[A-Z]{2,3}(\d+)", expand=False).fillna("?")
    route = raw["ADEP_mvt"].astype("string") + "|" + raw["ADES_mvt"].astype("string")
    return {
        "exact_flight": flight.fillna("?"),
        "airline_number": airline + "|" + number,
        "service_route": airline + "|" + number + "|" + route,
    }


def rmse(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))


def v29_oof(raw: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    ids = raw[[config.ID_COL]]

    def column(filename: str, name: str) -> np.ndarray:
        frame = pd.read_parquet(config.DATA_INTERIM / filename)
        return ids.merge(frame, on=config.ID_COL, how="left")[name].to_numpy(dtype="float64")

    base = column("v21_oof_predictions.parquet", "prediction")
    d8 = column("catgpu_d8_oof_predictions.parquet", "member")
    lobt = column("catgpu_lobt_oof_predictions.parquet", "member")
    mixture = column("lirf_unmatched_moe_oof.parquet", "mixture")
    stack = column("stacker_mid_800_oof.parquet", "stack")
    stack_ref = column("stacker_mid_800_oof.parquet", "ref")
    present = np.isfinite(d8) & np.isfinite(lobt)
    prediction = base.copy()
    prediction[present] += 0.3 * (d8[present] - base[present])
    prediction[present] += 0.3 * (lobt[present] - base[present])
    moe_rows = lirfmoe.target_rows(X, 8.0)
    prediction[moe_rows] = mixture[moe_rows]
    return np.clip(
        lirfbands.lobt_clip(prediction + 0.5 * (stack - stack_ref), X),
        0,
        None,
    )


def main() -> None:
    raw, X, _ = matrix.build(
        "train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True
    )
    ranking = pd.read_parquet(config.DATA_RAW / "ranking.parquet")
    submitting = pd.read_parquet(config.DATA_RAW / "submitting.parquet")
    test = submitting[[config.ID_COL]].merge(ranking, on=config.ID_COL, how="left")

    train_keys = identity_keys(raw)
    test_keys = identity_keys(test)
    train_missing = X["nm_record_missing"].to_numpy() == 1
    test_missing = test["AOBT_3_flt"].isna().to_numpy()
    exact_train = pd.DataFrame(
        {"key": train_keys["exact_flight"], "missing": train_missing}
    ).groupby("key")["missing"].agg(train_n="size", train_missing="sum", train_rate="mean")
    exact_test = pd.DataFrame(
        {"key": test_keys["exact_flight"], "missing": test_missing}
    ).groupby("key")["missing"].agg(test_n="size", test_missing="sum", test_rate="mean")
    repeated = exact_train.join(exact_test, how="inner")
    repeated = repeated[(repeated["train_n"] >= 10) & (repeated["test_n"] >= 2)]
    correlation = float(repeated[["train_rate", "test_rate"]].corr().iloc[0, 1])
    prior_missing = set(exact_train.index[exact_train["train_missing"] > 0])
    test_missing_with_history = float(
        test_keys["exact_flight"][test_missing].isin(prior_missing).mean()
    )

    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    reference = v29_oof(raw, X)
    residual = y - reference
    key = train_keys["service_route"]
    correction = np.zeros(len(raw), dtype="float64")
    for fold in range(1, 13):
        train = month != fold
        valid = month == fold
        global_mean = float(residual[train].mean())
        grouped = pd.DataFrame(
            {"key": key[train].to_numpy(), "residual": residual[train]}
        ).groupby("key")["residual"].agg(["sum", "size"])
        encoding = (
            (grouped["sum"] + PRIOR * global_mean) / (grouped["size"] + PRIOR)
        ).to_dict()
        correction[valid] = key[valid].map(encoding).fillna(global_mean).to_numpy()

    candidate = reference.copy()
    candidate[train_missing] = np.clip(
        candidate[train_missing] + correction[train_missing], 0, None
    )
    scored = np.isin(month, SCORED_MONTHS)
    delta = (y[scored] - candidate[scored]) ** 2 - (y[scored] - reference[scored]) ** 2
    order = np.argsort(delta)

    print("Identity persistence")
    print(f"  exact-flight missing-rate correlation: {correlation:.3f}")
    print(f"  test missing rows with prior exact-flight missing history: {test_missing_with_history:.1%}")
    print("\nFixed service-route residual correction, applied only to no-record rows")
    print(
        f"  scored all: {rmse(y, reference, scored):.3f} -> "
        f"{rmse(y, candidate, scored):.3f} "
        f"({rmse(y, candidate, scored) - rmse(y, reference, scored):+.3f}s)"
    )
    print(
        f"  scored no-record: {rmse(y, reference, scored & train_missing):.3f} -> "
        f"{rmse(y, candidate, scored & train_missing):.3f}"
    )
    for count in (1, 10, 50):
        keep = np.ones(int(scored.sum()), dtype=bool)
        keep[order[:count]] = False
        ys = y[scored][keep]
        new = candidate[scored][keep]
        old = reference[scored][keep]
        print(
            f"  drop {count:2d}: "
            f"{np.sqrt(np.mean((ys - new) ** 2)) - np.sqrt(np.mean((ys - old) ** 2)):+.3f}s"
        )


if __name__ == "__main__":
    main()
