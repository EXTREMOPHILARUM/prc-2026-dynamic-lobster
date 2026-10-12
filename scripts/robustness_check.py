"""Acceptance check for a candidate OOF prediction set.

Two headline results were withdrawn on 2026-09-16 after passing every test this project
had: both improved 10-12 months out of 12, both had a mechanism story that survived
targeted probing, and both turned out to rest on a single movement. The LIRF specialist
went from -6.744s to +0.445s on removing one row; the two-stage gate went from -6.080s to
+1.412s on removing ten.

RMSE squares the error, so one 24-hour "taxi" carries more weight than a hundred thousand
ordinary departures. A candidate can therefore look strong on every conventional summary
while being one lucky row. Months-improved does not catch this, because the lucky row also
improves its own month.

This script reports the conventional numbers and then removes the k rows the candidate
helped most and recomputes. If the gain does not survive, it was never there.

    uv run python scripts/robustness_check.py CANDIDATE.parquet [--baseline B.parquet]
    uv run python scripts/robustness_check.py --all      # every candidate in data/interim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prc import config, data, evaluation, features

#: rows removed for the concentration test
DROP_COUNTS = (1, 10, 50)

#: months withheld by the organisers; the population that actually gets scored
SCORED_MONTHS = (1, 7)


def _prediction_column(frame: pd.DataFrame) -> str:
    for name in ("prediction", "blended"):
        if name in frame.columns:
            return name
    others = [c for c in frame.columns if c != config.ID_COL]
    if not others:
        raise ValueError("no prediction column found")
    return others[0]


def load_aligned(path: Path, keys: pd.DataFrame) -> np.ndarray:
    """Align an OOF artifact to the training frame's row order by movement id."""
    frame = pd.read_parquet(path)
    column = _prediction_column(frame)
    merged = keys.merge(frame[[config.ID_COL, column]], on=config.ID_COL, how="left")
    return merged[column].to_numpy(dtype="float64")


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y[mask] - pred[mask]) ** 2)))


def evaluate(
    y: np.ndarray,
    month: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    """Conventional metrics plus the concentration test that matters."""
    scored = np.isin(month, SCORED_MONTHS)
    everything = np.ones(len(y), dtype=bool)

    # per-row squared-error change; negative means the candidate is better
    delta = (y - candidate) ** 2 - (y - baseline) ** 2
    within_scored = np.where(scored, delta, 0.0)
    most_helped = np.argsort(within_scored)

    gross_gain = -within_scored[within_scored < 0].sum()
    top_share = (
        -within_scored[most_helped[0]] / gross_gain if gross_gain > 0 else float("nan")
    )

    result = {
        "rmse_12mo": rmse(y, candidate, everything),
        "delta_12mo": rmse(y, candidate, everything) - rmse(y, baseline, everything),
        "rmse_scored": rmse(y, candidate, scored),
        "delta_scored": rmse(y, candidate, scored) - rmse(y, baseline, scored),
        "months_improved": sum(
            rmse(y, candidate, month == m) < rmse(y, baseline, month == m)
            for m in range(1, 13)
        ),
        "single_row_share": top_share,
        "rows_improved_pct": 100.0 * float((delta[scored] < 0).mean()),
    }
    for k in DROP_COUNTS:
        kept = scored.copy()
        kept[most_helped[:k]] = False
        result[f"delta_drop{k}"] = rmse(y, candidate, kept) - rmse(y, baseline, kept)
    return result


def verdict(result: dict) -> str:
    """A candidate is only real if the gain survives losing its luckiest rows.

    Judged at the deepest trim, not the shallowest. An earlier version of this
    function stopped at -10 and labelled a candidate robust that retained only 14%
    of its gain at -50; on the scored months the tail is 49.4% of squared error, so
    fifty rows out of 344,419 can carry most of any delta measured there.
    """
    if result["delta_scored"] >= 0:
        return "no gain"
    deepest = result[f"delta_drop{DROP_COUNTS[-1]}"]
    if result[f"delta_drop{DROP_COUNTS[1]}"] >= 0:
        return "FRAGILE - gain is a handful of rows"
    if deepest >= 0:
        return f"FRAGILE at -{DROP_COUNTS[-1]}"
    retained = deepest / result["delta_scored"]
    if retained < 0.35:
        return f"weak - only {retained:.0%} survives -{DROP_COUNTS[-1]}"
    return f"robust - {retained:.0%} survives -{DROP_COUNTS[-1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", help="OOF parquet files to check")
    parser.add_argument(
        "--baseline",
        default="sequencing_oof_predictions.parquet",
        help="frozen OOF to compare against, relative to data/interim",
    )
    parser.add_argument(
        "--all", action="store_true", help="check every *_predictions.parquet found"
    )
    args = parser.parse_args()

    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    keys = raw[[config.ID_COL]]
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()

    baseline = load_aligned(config.DATA_INTERIM / args.baseline, keys)
    scored = np.isin(month, SCORED_MONTHS)
    print(
        f"baseline {args.baseline}: 12-month {rmse(y, baseline, np.ones(len(y), bool)):.4f}, "
        f"scored months {rmse(y, baseline, scored):.4f} (n={int(scored.sum()):,})\n"
    )

    paths = [Path(c) for c in args.candidates]
    if args.all:
        paths = sorted(config.DATA_INTERIM.glob("*_predictions.parquet"))
    paths = [p if p.is_absolute() else config.DATA_INTERIM / p.name for p in paths]

    header = (
        f"{'candidate':44s} {'scored':>9s} {'Δscored':>8s} {'mo':>5s} "
        + " ".join(f"{'-' + str(k):>8s}" for k in DROP_COUNTS)
        + f" {'top1':>6s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for path in paths:
        if path.name == args.baseline:
            continue
        try:
            candidate = load_aligned(path, keys)
        except Exception as exc:  # noqa: BLE001 - report and continue over a mixed directory
            print(f"{path.name:44s}  unreadable: {exc}")
            continue
        if not np.isfinite(candidate).all():
            missing = int((~np.isfinite(candidate)).sum())
            print(f"{path.name:44s}  {missing:,} non-finite predictions - skipped")
            continue
        r = evaluate(y, month, baseline, candidate)
        drops = " ".join(f"{r[f'delta_drop{k}']:+8.3f}" for k in DROP_COUNTS)
        print(
            f"{path.name:44s} {r['rmse_scored']:9.3f} {r['delta_scored']:+8.3f} "
            f"{r['months_improved']:4d}/12 {drops} {r['single_row_share']:6.1%}  {verdict(r)}"
        )

    print(
        "\nColumns -1/-10/-50 are the scored-month delta after removing the rows the\n"
        "candidate helped most. 'top1' is the single best row's share of the gross gain.\n"
        "A candidate whose gain does not survive -10 is one row's luck, not a model."
    )


if __name__ == "__main__":
    main()
