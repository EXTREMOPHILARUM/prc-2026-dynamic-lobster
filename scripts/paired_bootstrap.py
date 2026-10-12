"""Paired bootstrap interval for an OOF candidate versus its baseline.

Rows are resampled *together* from the two aligned OOF prediction files, so each
replicate preserves the paired squared-error difference.  By default the organiser
months (January and July) are sampled separately, preserving their observed mix.
Use this alongside, rather than instead of, ``robustness_check.py``: an interval that
excludes zero is necessary evidence for a gain, not evidence that it is broad.

    uv run python scripts/paired_bootstrap.py CANDIDATE.parquet \
        --baseline BASELINE.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prc import config, data, evaluation, features


def load_aligned(path: Path, keys: pd.DataFrame, requested: str | None = None) -> np.ndarray:
    """Load the conventional prediction column and align it to training row order."""
    frame = pd.read_parquet(path)
    if requested is not None:
        if requested not in frame:
            raise ValueError(f"{path} has no {requested!r} column")
        column = requested
    else:
        for name in ("prediction", "blended"):
            if name in frame:
                column = name
                break
        else:
            candidates = [c for c in frame.columns if c != config.ID_COL]
            if not candidates:
                raise ValueError(f"{path} has no prediction column")
            column = candidates[0]
    return keys.merge(frame[[config.ID_COL, column]], on=config.ID_COL, how="left")[column].to_numpy(dtype="float64")


def paired_interval(
    baseline_se: np.ndarray,
    candidate_se: np.ndarray,
    *,
    strata: np.ndarray | None,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return percentile CI for candidate-minus-baseline RMSE, plus its point delta."""
    n = len(baseline_se)
    if n == 0:
        raise ValueError("cannot bootstrap an empty lane")
    point = float(np.sqrt(candidate_se.mean()) - np.sqrt(baseline_se.mean()))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype="float64")
    # Keep allocations bounded; a full (reps, n) index matrix is unnecessarily large.
    # Resample each held-out month separately so the January/July mix is fixed.
    groups = [np.flatnonzero(strata == group) for group in np.unique(strata)] if strata is not None else [np.arange(n)]
    batch = max(1, min(25, reps))
    for start in range(0, reps, batch):
        stop = min(start + batch, reps)
        base_sum = np.zeros(stop - start, dtype="float64")
        candidate_sum = np.zeros(stop - start, dtype="float64")
        for group in groups:
            index = rng.integers(0, len(group), size=(stop - start, len(group)))
            base_sum += baseline_se[group][index].sum(axis=1)
            candidate_sum += candidate_se[group][index].sum(axis=1)
        draws[start:stop] = np.sqrt(candidate_sum / n) - np.sqrt(base_sum / n)
    low, high = np.quantile(draws, (0.025, 0.975))
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="candidate OOF parquet, relative to data/interim")
    parser.add_argument("--baseline", required=True, help="baseline OOF parquet, relative to data/interim")
    parser.add_argument("--candidate-column", help="prediction column in candidate (default: prediction, blended, then first)")
    parser.add_argument("--baseline-column", help="prediction column in baseline (default: prediction, blended, then first)")
    parser.add_argument("--months", type=int, nargs="+", default=[1, 7], help="holdout months to assess")
    parser.add_argument("--reps", type=int, default=2_000, help="bootstrap replicates (default: 2,000)")
    parser.add_argument("--seed", type=int, default=20260920, help="fixed RNG seed for reproducibility")
    args = parser.parse_args()
    if args.reps < 100:
        raise ValueError("use at least 100 bootstrap replicates")

    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    keys = raw[[config.ID_COL]]
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    root = config.DATA_INTERIM
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline_path = baseline_path if baseline_path.is_absolute() else root / baseline_path.name
    candidate_path = candidate_path if candidate_path.is_absolute() else root / candidate_path.name
    baseline = load_aligned(baseline_path, keys, args.baseline_column)
    candidate = load_aligned(candidate_path, keys, args.candidate_column)
    if not (np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("both OOF files must contain finite aligned predictions")

    se_base = (y - baseline) ** 2
    se_candidate = (y - candidate) ** 2
    masks = [("pooled", np.isin(month, args.months))] + [(str(m), month == m) for m in args.months]
    print(f"paired row bootstrap: {candidate_path.name} vs {baseline_path.name}; {args.reps:,} replicates")
    print("negative delta is a candidate gain; a 95% interval must exclude 0 to call it measurable\n")
    print(f"{'lane':10s} {'n':>9s} {'base':>9s} {'candidate':>10s} {'delta':>9s} {'95% CI':>22s}  verdict")
    print("-" * 91)
    for i, (name, mask) in enumerate(masks):
        b = se_base[mask]
        c = se_candidate[mask]
        point, low, high = paired_interval(b, c, strata=month[mask], reps=args.reps, seed=args.seed + i)
        verdict = "MEASURABLE GAIN" if high < 0 else "MEASURABLE LOSS" if low > 0 else "not resolved"
        print(
            f"{name:10s} {len(b):9,} {np.sqrt(b.mean()):9.3f} {np.sqrt(c.mean()):10.3f} "
            f"{point:+9.3f} [{low:+8.3f}, {high:+8.3f}]  {verdict}"
        )


if __name__ == "__main__":
    main()
