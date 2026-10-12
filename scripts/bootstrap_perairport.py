"""How much of the LIRF specialist's scored gain is 397 rows of luck?

The whole January+July gain comes from LIRF's missing-AOBT rows: 397 of them, carrying
30.6% of scored squared error. Three tail estimates in docs/experiment-log.md were
over-claimed on segments of that size, so the gain is resampled rather than trusted.

Each resample recomputes the *whole* January+July RMSE with one group of rows drawn with
replacement and the rest held fixed, which is the quantity a submission would be scored on.

    uv run python -u scripts/bootstrap_perairport.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, features, matrix

SCORED = (1, 7)


def bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    new: np.ndarray,
    resampled: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None = None,
) -> np.ndarray:
    """Return the RMSE delta (new - base) for each resample of `resampled`."""
    fixed = ~resampled
    n = len(y)
    fixed_old = float(((y[fixed] - base[fixed]) ** 2).sum())
    fixed_new = float(((y[fixed] - new[fixed]) ** 2).sum())

    if clusters is None:
        pool_old = ((y[resampled] - base[resampled]) ** 2)
        pool_new = ((y[resampled] - new[resampled]) ** 2)
        size = len(pool_old)
        # drawn in chunks: a (draws, size) index array is far too large for the
        # full scored population, which is where the first version of this died
        chunk = max(1, min(draws, 10_000_000 // max(size, 1)))
        old_parts, new_parts = [], []
        for start in range(0, draws, chunk):
            take = min(chunk, draws - start)
            idx = rng.integers(0, size, size=(take, size))
            old_parts.append(pool_old[idx].sum(axis=1))
            new_parts.append(pool_new[idx].sum(axis=1))
        old = fixed_old + np.concatenate(old_parts)
        new_ = fixed_new + np.concatenate(new_parts)
    else:
        keys = clusters[resampled]
        unique = np.unique(keys)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.searchsorted(sorted_keys, unique, side="left")
        ends = np.searchsorted(sorted_keys, unique, side="right")
        se_old = ((y[resampled] - base[resampled]) ** 2)[order]
        se_new = ((y[resampled] - new[resampled]) ** 2)[order]
        cum_old = np.array([se_old[a:b].sum() for a, b in zip(starts, ends)])
        cum_new = np.array([se_new[a:b].sum() for a, b in zip(starts, ends)])
        counts = ends - starts
        pick = rng.integers(0, len(unique), size=(draws, len(unique)))
        old = fixed_old + cum_old[pick].sum(axis=1)
        new_ = fixed_new + cum_new[pick].sum(axis=1)
        n = fixed.sum() + counts[pick].sum(axis=1)

    return np.sqrt(new_ / n) - np.sqrt(old / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--predictions", default="perairport_lirf_oof_predictions.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    raw, X, _ = matrix.build()
    y_all = raw[config.TARGET].astype("float64").to_numpy()
    month = evaluation.movement_month(raw).to_numpy()
    airport = raw["ADEP_mvt"].astype("string").to_numpy()
    missing = X[decomposition.ANCHOR_COL].isna().to_numpy()
    day = pd.to_datetime(raw[features.TIME_COL], utc=True, errors="coerce").dt.floor("D").astype(
        "int64").to_numpy()

    base_all = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet"),
        on=config.ID_COL, how="left")["prediction"].to_numpy()
    new_all = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.predictions),
        on=config.ID_COL, how="left")["prediction"].to_numpy()

    scored = np.isin(month, SCORED)
    y, base, new = y_all[scored], base_all[scored], new_all[scored]
    lirf, miss, days = airport[scored] == "LIRF", missing[scored], day[scored]
    observed = float(np.sqrt(((y - new) ** 2).mean()) - np.sqrt(((y - base) ** 2).mean()))
    print(f"January+July n={scored.sum()}  observed delta {observed:+.3f} s\n")

    schemes = [
        ("LIRF missing-AOBT only (397 rows)", lirf & miss, None),
        ("all LIRF rows (26,528)", lirf, None),
        ("all scored rows", np.ones(len(y), dtype=bool), None),
        ("LIRF rows, by day (cluster)", lirf, days),
    ]
    rows = []
    for name, mask, clusters in schemes:
        deltas = bootstrap(y, base, new, mask, args.draws, rng, clusters)
        p10, p50, p90 = np.percentile(deltas, [10, 50, 90])
        rows.append({
            "resampled": name,
            "n": int(mask.sum()),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p_gain_negative": float((deltas > 0).mean()),
        })
        print(f"  {name}: done", flush=True)

    table = pd.DataFrame(rows)
    print("\nBootstrap of the January+July RMSE delta "
          f"({args.draws} draws; negative is better)")
    print(table.round(3).to_string(index=False))

    stem = args.predictions.replace("_predictions.parquet", "")
    config.DATA_INTERIM.joinpath(f"{stem}_bootstrap.json").write_text(
        json.dumps({"observed_delta": observed, "draws": args.draws,
                    "seed": args.seed, "schemes": rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
