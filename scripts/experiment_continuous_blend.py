"""Screen a continuous schedule-proxy correction on saved 2025 OOF predictions.

Fit correction slopes on the ten months outside January and July. For each of
those ten months, leave that month out when estimating its slope. January and
July are evaluated only; neither month contributes to fitted coefficients.

This is a cheap screening experiment, not final model selection: the saved
first-stage OOF predictions are not nested with the second-stage fit. Confirm
any candidate with nested month folds before using it in a submission.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prc import config, data

SCORED_MONTHS = (1, 7)
GAP_EDGES = (1800.0, 7200.0)


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y[mask] - pred[mask]))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="arrival_review/arrivals_fixed_oof_predictions.parquet",
    )
    args = parser.parse_args()

    cols = [config.ID_COL, config.TARGET, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "PHASE_mvt", "AOBT_3_flt"]
    raw = data.load_training(columns=cols)
    raw = raw[(raw["PHASE_mvt"] == "DEP") & raw[config.TARGET].notna()]
    raw = raw.reset_index(drop=True)
    oof = pd.read_parquet(config.DATA_INTERIM / args.baseline)
    baseline = raw[[config.ID_COL]].merge(
        oof[[config.ID_COL, "prediction"]], on=config.ID_COL,
        how="left", validate="one_to_one",
    )["prediction"].to_numpy(dtype="float64")
    if not np.isfinite(baseline).all():
        raise ValueError("baseline OOF predictions are incomplete")

    y = raw[config.TARGET].to_numpy(dtype="float64")
    proxy = (raw["MVT_TIME_UTC_mvt"] - raw["SCHED_TIME_UTC_mvt"])
    proxy = proxy.dt.total_seconds().to_numpy(dtype="float64")
    d = proxy - baseline
    month = raw["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    present = raw["AOBT_3_flt"].notna().to_numpy()
    scored = np.isin(month, SCORED_MONTHS)
    clean = ~scored
    group = np.digitize(np.abs(d), GAP_EDGES)
    candidate = baseline.copy()

    for valid_month in range(1, 13):
        valid = month == valid_month
        train = clean & ~valid & present & np.isfinite(d)
        for j in range(len(GAP_EDGES) + 1):
            fit = train & (group == j)
            denominator = np.square(d[fit]).sum()
            slope = (d[fit] * (y[fit] - baseline[fit])).sum() / denominator \
                if denominator else 0.0
            apply = valid & present & (group == j) & np.isfinite(d)
            candidate[apply] = np.maximum(0.0, baseline[apply] + slope * d[apply])

    print(f"baseline: {args.baseline}")
    for name, mask in (("ten selection months", clean), ("January + July", scored),
                       ("January + July bulk", scored & (y <= 3600))):
        before, after = rmse(y, baseline, mask), rmse(y, candidate, mask)
        print(f"{name}: {before:.4f} -> {after:.4f} ({after - before:+.4f})")

    gain = np.square(y - baseline) - np.square(y - candidate)
    scored_indices = np.flatnonzero(scored)
    top = scored_indices[np.argsort(gain[scored])[::-1]]
    for n in (1, 10, 50):
        kept = scored.copy()
        kept[top[:n]] = False
        print(f"drop {n} most helped: {rmse(y, candidate, kept) - rmse(y, baseline, kept):+.4f}")


if __name__ == "__main__":
    main()
