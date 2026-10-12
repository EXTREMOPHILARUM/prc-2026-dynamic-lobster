"""Day-offset regime: BLOCK recorded roughly 24 hours before the movement.

A handful of departures carry a taxi-out of ~24h plus an ordinary taxi, because the
airport's block time refers to the previous day. In 2025 there are 16 such rows out of
2,085,047 - and they carry ~19% of the total variance, because RMSE squares a 24-hour
error. On the ranking set 11 rows sit in the at-risk band.

The regime is reconstructable rather than predictable in magnitude: when it fires the
target is `86400 + an ordinary taxi`, so the conditional mean is simply

    E[Y|X] = ordinary(X) + P(day offset | X) * 86400

With 16 positives a learned classifier would be noise. Instead the probability is an
empirical rate over a few disjoint, observable bands, estimated out of fold with
Laplace smoothing so an unseen band cannot assert certainty.

Every band input is present in the ranking set: AOBT availability, the reporting
airport, and MVT_TIME - SCHED_TIME.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400.0

#: a target at or above this is the day-offset regime rather than a long taxi
DAY_THRESHOLD = 79200.0

#: MVT - SCHED band edges, in hours. Below the first edge the regime is not credible.
GAP_EDGES_H = (6.0, 12.0, 20.0)

#: Pseudo-counts. With 16 positives in the whole of 2025 the raw band rates are not
#: trustworthy - LIRF/20h+ reads 5 of 5, which as a 100% claim would bet a full day on
#: five observations. Heavy shrinkage turns that into a hedge of roughly 14%, which is
#: what L2 wants when the regime is real but the rate is barely measured.
#: Chosen with out-of-fold results visible, so the reported OOF gain is optimistic.
PRIOR_STRENGTH = 30.0
PRIOR_RATE = 0.0

#: only bands with direct evidence are gated; the "other" airports show 0 of 145
GATED_BANDS = ("LIRF/12-20h", "LIRF/20h+")

PROXY_COL = "taxi_proxy_SCHED_TIME_UTC_mvt"


def day_offset_label(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype="float64") >= DAY_THRESHOLD


def band(df: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Disjoint at-risk bands. Everything outside them gets probability zero."""
    gap_h = X[PROXY_COL].to_numpy(dtype="float64") / 3600.0
    missing = df["AOBT_3_flt"].isna().to_numpy()
    lirf = (df["ADEP_mvt"].astype("string") == "LIRF").to_numpy()

    labels = np.full(len(df), "none", dtype=object)
    for lo, hi, name in zip(
        GAP_EDGES_H, (*GAP_EDGES_H[1:], np.inf), ("6-12h", "12-20h", "20h+"), strict=True
    ):
        sel = missing & (gap_h >= lo) & (gap_h < hi)
        labels[sel & lirf] = f"LIRF/{name}"
        labels[sel & ~lirf] = f"other/{name}"
    return pd.Series(labels, index=df.index)


def fit_rates(bands: pd.Series, label: np.ndarray) -> dict[str, float]:
    """Smoothed positive rate per band, from training rows only."""
    out = {}
    frame = pd.DataFrame({"band": bands.to_numpy(), "y": label})
    for name, g in frame.groupby("band"):
        if name not in GATED_BANDS:
            continue
        out[name] = float(
            (g.y.sum() + PRIOR_STRENGTH * PRIOR_RATE) / (len(g) + PRIOR_STRENGTH)
        )
    return out


def probability(bands: pd.Series, rates: dict[str, float]) -> np.ndarray:
    return bands.map(lambda b: rates.get(b, 0.0)).to_numpy(dtype="float64")


def apply(ordinary: np.ndarray, probability_: np.ndarray) -> np.ndarray:
    """Add the expected day offset. The magnitude needs no model: it is one day."""
    return np.clip(ordinary + probability_ * DAY, 0, None)
