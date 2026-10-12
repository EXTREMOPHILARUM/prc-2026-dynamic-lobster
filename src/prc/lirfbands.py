"""Broken-clock bands at LIRF, and the LOBT bound.

At Rome, a departure with no Network Manager record and a takeoff many hours after its
scheduled time almost never has a real off-block time: the recorded block time is either
the schedule itself (target = MVT - SCHED) or the previous day (target = 86400 + a normal
taxi). In 2025 the 8-12h band is 21 of 21 schedule copies and the 20h+ band is 5 of 5 day
offsets. RMSE squares the error, so hedging toward a normal taxi on these rows costs more
than every other modelling decision combined; three public leaders predict them outright.

The rule is a band table fitted on training rows only:

    E[Y | band] = p_copy * (MVT - SCHED) + p_day * (86400 + mean extra) + p_normal * mean normal

with rates shrunk toward the pooled scope rate by `PRIOR_STRENGTH` pseudo-rows.

Separately, whenever NM's last off-block time exists, the target never departs from
MVT - LOBT by more than `LOBT_BOUND` seconds (0 violations in 2,062,577 training rows), so
predictions on those rows are clipped to that interval.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DAY = 86400.0
COPY_TOLERANCE = 120.0
DAY_TOLERANCE = 7200.0
#: hours of MVT - SCHED; below the first edge the scope does not apply. Edges at 40k/50k/
#: 60k/70k/90k seconds follow the 2025 mixture: 40-50k is 9/9 schedule copies, 50-60k is
#: 6 copies to 3 day offsets, 60-70k 2 to 4, 70-90k all day offsets. The coarser
#: (4,8,12,20) edges mixed the 40-50k copies with day offsets; the finer edges are worth
#: -1.9s on Jan+Jul leave-one-month-out (329.64 -> 327.78) and improve every month group.
EDGES_H = (4.0, 8.0, 11.1, 13.9, 16.7, 19.4, 25.0, np.inf)
PRIOR_STRENGTH = 3.0
LOBT_BOUND = 3606.0
AIRPORT = "LIRF"

SCHED_COL = "taxi_proxy_SCHED_TIME_UTC_mvt"
LOBT_COL = "taxi_proxy_LOBT_flt"
MISSING_COL = "nm_record_missing"


def scope(X: pd.DataFrame) -> np.ndarray:
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    missing = X[MISSING_COL].to_numpy() == 1
    gap = X[SCHED_COL].to_numpy(dtype="float64")
    return lirf & missing & (gap > EDGES_H[0] * 3600.0)


def band_index(X: pd.DataFrame, edges_h=EDGES_H) -> np.ndarray:
    gap_h = X[SCHED_COL].to_numpy(dtype="float64") / 3600.0
    return np.searchsorted(np.asarray(edges_h[1:], dtype="float64"), gap_h, side="left")


def _edges(table: dict):
    """Edges stored with a table (None stands for +inf), so old tables score consistently."""
    return tuple(np.inf if e is None else float(e) for e in table.get("edges_h", EDGES_H))


def fit(X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict:
    """Band table from labelled rows. Keys are strings so it serialises to JSON."""
    y = np.asarray(y, dtype="float64")
    in_scope = scope(X)
    gap = X[SCHED_COL].to_numpy(dtype="float64")[in_scope]
    ys = y[in_scope]
    copy = np.abs(ys - gap) <= COPY_TOLERANCE
    day = ~copy & (np.abs(ys - DAY) < DAY_TOLERANCE)
    normal = ~copy & ~day
    pooled = {"p_copy": float(copy.mean()), "p_day": float(day.mean())}
    extra = float(np.mean(ys[day] - DAY)) if day.any() else 1150.0
    normal_mean = float(np.mean(ys[normal])) if normal.any() else 1000.0
    bands = band_index(X)[in_scope]
    table: dict[str, dict] = {}
    k = PRIOR_STRENGTH
    for b in range(len(EDGES_H) - 1):
        m = bands == b
        n = int(m.sum())
        table[str(b)] = {
            "n": n,
            "p_copy": float((copy[m].sum() + k * pooled["p_copy"]) / (n + k)),
            "p_day": float((day[m].sum() + k * pooled["p_day"]) / (n + k)),
        }
    return {
        "edges_h": [e if np.isfinite(e) else None for e in EDGES_H],
        "prior_strength": k,
        "day_extra": extra,
        "normal_mean": normal_mean,
        "pooled": pooled,
        "bands": table,
        "rows": int(in_scope.sum()),
    }


def apply(pred: np.ndarray, X: pd.DataFrame, table: dict) -> np.ndarray:
    """Replace in-scope predictions with the band expectation."""
    out = np.asarray(pred, dtype="float64").copy()
    in_scope = scope(X)
    if not in_scope.any():
        return out
    gap = X[SCHED_COL].to_numpy(dtype="float64")[in_scope]
    bands = band_index(X, _edges(table))[in_scope]
    p_copy = np.array([table["bands"][str(b)]["p_copy"] for b in bands])
    p_day = np.array([table["bands"][str(b)]["p_day"] for b in bands])
    p_normal = np.clip(1.0 - p_copy - p_day, 0.0, 1.0)
    out[in_scope] = (
        p_copy * gap + p_day * (DAY + table["day_extra"]) + p_normal * table["normal_mean"]
    )
    return out


def lobt_clip(pred: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """Clip to MVT - LOBT +/- LOBT_BOUND where LOBT is present."""
    out = np.asarray(pred, dtype="float64").copy()
    ml = X[LOBT_COL].to_numpy(dtype="float64")
    has = np.isfinite(ml)
    out[has] = np.clip(out[has], np.maximum(ml[has] - LOBT_BOUND, 0.0), ml[has] + LOBT_BOUND)
    return out


def short_scope(X: pd.DataFrame) -> np.ndarray:
    """LIRF departures with no NM record whose gap is below the band scope."""
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    missing = X[MISSING_COL].to_numpy() == 1
    return lirf & missing & ~scope(X)


def apply_short_expert(pred: np.ndarray, X: pd.DataFrame, booster) -> np.ndarray:
    """Replace predictions on `short_scope` rows with the NM-free expert's."""
    out = np.asarray(pred, dtype="float64").copy()
    rows = short_scope(X)
    if rows.any():
        out[rows] = np.clip(booster.predict(X.loc[rows, booster.feature_name()]), 0, None)
    return out


RULE_LO = 24000.0
RULE_HI = 86400.0
ROME_TZ = "Europe/Rome"


def rome_rules(pred: np.ndarray, raw: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """merry-quicksand's two deterministic LIRF rules for no-record rows with a gap in
    [24000, 86400): floor the prediction at the schedule gap (a copy can never be below it),
    and predict the gap exactly when take-off and schedule fall on the same Rome calendar
    day (14 of 14 such 2025 rows are schedule copies). -0.9s Jan+Jul on the v19 OOF."""
    out = np.asarray(pred, dtype="float64").copy()
    gap = X[SCHED_COL].to_numpy(dtype="float64")
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    missing = X[MISSING_COL].to_numpy() == 1
    band = lirf & missing & (gap >= RULE_LO) & (gap < RULE_HI)
    if not band.any():
        return out
    out[band] = np.maximum(out[band], np.rint(gap[band]))
    mvt = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.tz_convert(ROME_TZ).dt.date.to_numpy()
    sched = pd.to_datetime(raw["SCHED_TIME_UTC_mvt"], utc=True).dt.tz_convert(ROME_TZ).dt.date.to_numpy()
    same_day = band & (mvt == sched)
    out[same_day] = np.rint(gap[same_day])
    return out


GARBLED_LO = EDGES_H[0] * 3600.0
GARBLED_HI = EDGES_H[1] * 3600.0
_GARBLED = re.compile(r"^([A-Z]{3})([A-Z]+)")


def garbled_flight(flight: pd.Series) -> np.ndarray:
    """Flight strings whose operator prefix is typed twice over (`ITYTY672`, `WMTT5232`,
    `RYRR90BN`, `QFAQFA6`): the letters after the three-letter prefix repeat its tail or the
    whole prefix. Ordinary strings (`DLH8LA`, `RYR14WC`, `EJU42AY`) have a digit there."""
    def one(s) -> bool:
        m = _GARBLED.match(s) if isinstance(s, str) else None
        if m is None:
            return False
        p, rest = m.group(1), m.group(2)
        return rest.startswith((p[1:], p[2:], p))
    return np.array([one(s) for s in flight.astype(object)], dtype=bool)


def garbled_copy(pred: np.ndarray, raw: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """LIRF no-record departures with a garbled flight string and a schedule gap in (4h, 8h]
    are schedule copies: predict MVT - SCHED exactly. 20 of 20 in 2025 (17 in the ten
    non-scored months); the edges are the band table's first edge and the mixture's limit,
    not tuned (log section 57)."""
    out = np.asarray(pred, dtype="float64").copy()
    gap = X[SCHED_COL].to_numpy(dtype="float64")
    lirf = (X["ADEP_mvt"].astype("string") == AIRPORT).to_numpy()
    missing = X[MISSING_COL].to_numpy() == 1
    rows = lirf & missing & (gap > GARBLED_LO) & (gap <= GARBLED_HI)
    if rows.any():
        rows &= garbled_flight(raw["FLIGHT_mvt"])
        out[rows] = np.rint(gap[rows])
    return out
