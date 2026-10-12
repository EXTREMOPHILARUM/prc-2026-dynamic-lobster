"""Local end-to-end rehearsal of a real submission.

The 12-fold OOF scores predictions made in memory. It never exercises the path a real
submission takes: feature construction on a frame whose BLOCK_TIME is blanked, the saved
category levels, the template merge, the int32 cast. Those have all broken before.

This builds a stand-in ranking set from 2025 January + July - the same two months the
organisers score, blanked exactly the way they blank them - runs the genuine scoring
path over it, and checks the result against truth we hold back.

It catches two things the OOF cannot:

1. **Leakage.** Any feature that quietly depends on BLOCK_TIME_UTC_mvt looks fine in OOF
   and silently degrades a real submission. Here the column is absent, so a leaking
   feature changes value and the rehearsal score diverges from the OOF score.
2. **Pipeline faults.** Wrong dtype, misaligned ids, a feature missing at scoring time,
   category levels that do not round-trip.

The organisers blank, for PHASE_mvt == "DEP" rows only: BLOCK_TIME_UTC_mvt and
TAXITIME_SEC_mvt. Arrival rows are left intact. See docs/submissions.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config, data, features

REHEARSAL_MONTHS = (1, 7)
BLANKED = ["BLOCK_TIME_UTC_mvt", config.TARGET]


#: the final phase (2026-10-08) shifts every movement and block time by a uniform integer number of seconds in
#: [-JITTER_S, +JITTER_S], independently per row (measured on the shared Jan/Jul rows of final_ranking.parquet)
JITTER_S = 12


def jitter_times(movements: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The final phase's perturbation: MVT_TIME and BLOCK_TIME each shifted by an independent uniform integer number of
    seconds in [-JITTER_S, JITTER_S]; TAXITIME recomputed from the shifted times where it was present (departures:
    MVT - BLOCK; arrivals: BLOCK - MVT)."""
    out = movements.copy()
    rng = np.random.default_rng(seed)
    for column in (features.TIME_COL, "BLOCK_TIME_UTC_mvt"):
        shift = rng.integers(-JITTER_S, JITTER_S + 1, len(out))
        out[column] = pd.to_datetime(out[column], utc=True) + pd.to_timedelta(shift, unit="s")
    dep = (out["PHASE_mvt"] == "DEP").to_numpy()
    taxi = (pd.to_datetime(out[features.TIME_COL], utc=True)
            - pd.to_datetime(out["BLOCK_TIME_UTC_mvt"], utc=True)).dt.total_seconds()
    present = out[config.TARGET].notna()
    out[config.TARGET] = np.where(present, np.where(dep, taxi, -taxi), np.nan)
    return out


def build_mock_ranking(months: tuple[int, ...] = REHEARSAL_MONTHS, jitter_seed: int | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ranking-like frame, submitting-like template, truth).

    With `jitter_seed`, MVT_TIME and BLOCK_TIME are shifted as in the final phase before anything is blanked, and the
    held-back target is recomputed from the shifted times, as it would be in the organisers' truth."""
    train = data.load_training()
    when = pd.to_datetime(train[features.TIME_COL], utc=True)
    subset = train[when.dt.month.isin(months)].copy().reset_index(drop=True)
    if jitter_seed is not None:
        subset = jitter_times(subset, jitter_seed)

    departures = subset["PHASE_mvt"] == "DEP"
    # the block time rides along (unused by `score`) so the leakage probe can restore exactly what was blanked
    truth = subset.loc[departures, [config.ID_COL, config.TARGET, "BLOCK_TIME_UTC_mvt", features.TIME_COL]].reset_index(drop=True)

    mock = subset.copy()
    for column in BLANKED:
        mock.loc[departures, column] = pd.NA
    # match the real file: the blanked target column stays numeric
    mock[config.TARGET] = pd.to_numeric(mock[config.TARGET], errors="coerce")

    template = truth[[config.ID_COL]].copy()
    template[config.TARGET] = np.int32(0)
    return mock, template, truth


def score(submission: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int | str]:
    """Reproduce the organisers' checks, then the metric."""
    ids = submission[config.ID_COL]
    if ids.duplicated().any():
        return {"status": "Rejected", "error": "duplicate MVT_ID_mvt"}
    missing = set(truth[config.ID_COL]) - set(ids)
    extra = set(ids) - set(truth[config.ID_COL])
    if missing:
        return {"status": "Rejected", "error": f"{len(missing)} ids missing"}
    if extra:
        return {"status": "Rejected", "error": f"{len(extra)} extra ids"}

    merged = truth.merge(
        submission.rename(columns={config.TARGET: "_pred"}), on=config.ID_COL, how="left"
    )
    if not np.isfinite(merged["_pred"]).all():
        return {"status": "Rejected", "error": "non-finite predictions"}
    error = merged["_pred"].to_numpy(dtype="float64") - merged[config.TARGET].to_numpy(dtype="float64")
    return {
        "status": "Succeeded",
        "used_pairs": len(merged),
        "score": float(np.sqrt(np.mean(error**2))),
    }
