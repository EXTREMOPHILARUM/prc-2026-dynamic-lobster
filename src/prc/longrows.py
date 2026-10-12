"""The unexplained long-taxi regime.

Departures split into four groups once the target exceeds an hour:

  * BLOCK copied from SCHED          1,751 rows, 26.1% of variance  -> mechanism.py
  * ~24h day offset                     16 rows, 19.5% of variance  -> dayoffset.py
  * long, under 24h, BLOCK != SCHED  2,362 rows,  7.8% of variance  -> here
  * everything else                      normal taxiing

This third group is the largest unmodelled block. Described on the challenge Discord by
another team: the airport reports over an hour while the Network Manager sees an ordinary
taxi, with BLOCK sitting a median of 47 minutes before AOBT. The airport clock is early
rather than absent.

They are visible at scoring time - MVT-SCHED runs 5,464s against 1,398s for normal rows,
and AOBT is missing on 16.2% against 1.0% - and a gate on scoring-time features reaches
AUC 0.923 out of fold. The magnitude is not a constant (y minus the NM proxy has median
2,824s but a standard deviation of 1,801s), so severity needs its own model rather than
a fixed offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import mechanism

DAY = 86400.0
TAIL_THRESHOLD = 3600.0

GATE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
}

SEVERITY_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
}


def label(raw: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Long taxis that are neither a schedule copy nor a day offset."""
    fallback = mechanism.fallback_label(raw).to_numpy()
    near_day = ((y % DAY) < 7200) & (y > DAY - 7200)
    return (y > TAIL_THRESHOLD) & ~fallback & ~near_day & (y <= DAY + 7200)


def gate_features(X: pd.DataFrame) -> list[str]:
    return [c for c in mechanism.GATE_FEATURES if c in X.columns]


def blend(
    ordinary: np.ndarray, probability: np.ndarray, severity: np.ndarray
) -> np.ndarray:
    """Mix the ordinary prediction with the regime's conditional mean.

    Soft, as elsewhere: the gate tops out around 22% purity, so a hard rule would be
    wrong four times in five.
    """
    p = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    magnitude = np.maximum(np.nan_to_num(severity, nan=TAIL_THRESHOLD), TAIL_THRESHOLD)
    return np.clip(p * magnitude + (1.0 - p) * ordinary, 0, None)
