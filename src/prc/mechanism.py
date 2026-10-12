"""Schedule-fallback mechanism model.

When the airport has no proper off-block record it appears to fall back to writing the
*scheduled* time into BLOCK_TIME_UTC_mvt. The recorded taxi-out then equals
`MVT_TIME - SCHED_TIME`, which is not a taxi duration at all - it is the flight's lateness.

Those rows dominate RMSE (see docs/outlier-analysis.md), and on them the target is
reconstructable almost exactly: measured over 2025, `MVT - SCHED` matches the recorded
target with RMSE 36.6s and 100% of rows within a minute. The difficulty is therefore not
estimating the value but *identifying the regime*, because BLOCK is withheld at scoring time.

    E[Y|X] = P(fallback|X) * (MVT - SCHED) + (1 - P(fallback|X)) * ordinary(X)

The gate must be calibrated, not merely well-ranked: its probability is a mixture weight.
It is trained without class reweighting for that reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: |BLOCK - SCHED| at or below this counts as a schedule copy. 10s and 60s give almost the
#: same population; 60s is used because small timestamp jitter is still the same mechanism.
FALLBACK_TOLERANCE = 60.0

#: MVT - SCHED, the reconstruction used when the gate fires. Present in the ranking set.
PROXY_COL = "taxi_proxy_SCHED_TIME_UTC_mvt"

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

#: deliberately low-cardinality: callsign/operator identity overfit the tail badly
GATE_FEATURES = [
    "ADEP_mvt",
    "AIRCRAFT_TYPE_mvt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_mvt",
    "WK_TBL_CAT_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "hour",
    "dow",
    "doy",
    "is_weekend",
    "adep_movements_hour",
    "aobt_missing",
    "nm_record_missing",
    "taxi_proxy_SCHED_TIME_UTC_mvt",
    "taxi_proxy_AOBT_3_flt",
    "taxi_proxy_EOBT_1_flt",
    "taxi_proxy_IOBT_flt",
    "taxi_proxy_LOBT_flt",
    "offblock_delay",
]


def fallback_label(df: pd.DataFrame, tolerance: float = FALLBACK_TOLERANCE) -> pd.Series:
    """True where BLOCK was copied from SCHED. Training-only: BLOCK is withheld at scoring."""
    block = pd.to_datetime(df["BLOCK_TIME_UTC_mvt"], utc=True, errors="coerce")
    sched = pd.to_datetime(df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce")
    return ((block - sched).dt.total_seconds().abs() <= tolerance).fillna(False)


def gate_features(X: pd.DataFrame) -> list[str]:
    return [c for c in GATE_FEATURES if c in X.columns]


def blend(
    probability: np.ndarray,
    proxy: np.ndarray,
    ordinary: np.ndarray,
    max_probability: float = 1.0,
) -> np.ndarray:
    """Mix the schedule reconstruction with the ordinary prediction by gate confidence.

    A soft mixture rather than a hard switch: a wrong call costs the difference times the
    probability, instead of the full difference.
    """
    p = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, max_probability)
    reconstruction = np.nan_to_num(proxy, nan=np.nan)
    # where the proxy is unavailable the gate cannot be acted on
    usable = np.isfinite(reconstruction)
    p = np.where(usable, p, 0.0)
    reconstruction = np.where(usable, reconstruction, 0.0)
    return np.clip(p * reconstruction + (1.0 - p) * ordinary, 0, None)
