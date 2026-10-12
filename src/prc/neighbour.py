"""Neighbour-observation features: what recent movements actually experienced.

Two families, both DEPARTURE-side and both computed from the challenge's own data.

1. **Neighbour departure taxi proxy.** Every departure that has already taken off carries
   an observable `MVT_TIME - AOBT_3_flt` — the same noisy reading of taxi-out the model
   uses for the row itself. Aggregating it over departures that took off shortly before
   this flight pushed back says how long taxiing is *actually taking right now*, which a
   count of nearby traffic cannot express.

2. **Arrival taxi-in aggregates.** Arrival rows survive the ranking set intact — landing
   time, in-block time, runway and `TAXITIME_SEC_mvt` are all present at 100%, because
   only departures are blanked. The taxi-in time of arrivals that reached their stand just
   before this departure pushed back measures surface conditions on the same tarmac.

Both are anchored on the departure's **off-block** time (`AOBT_3_flt`, falling back to
`EOBT_1_flt` then take-off), not on take-off: the review of the arrival-landing screen
showed that anchoring on take-off measures a window that is largely over by the time the
taxi has finished, and materially weakens the signal.

Windows are strictly backward-looking from the anchor and every input is present at
scoring time, so nothing here can leak `BLOCK_TIME_UTC_mvt` or the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEIGHBOUR_FEATURES = [
    "nbr_taxi_med_30m",
    "nbr_taxi_med_60m",
    "nbr_taxi_n_30m",
    "nbr_taxi_med_30m_runway",
    "nbr_taxi_last",
]

ARRIVAL_FEATURES = [
    "arrin_med_30m",
    "arrin_med_60m",
    "arrin_n_30m",
    "arrin_med_30m_runway",
    "arrin_last",
]

FEATURES = NEIGHBOUR_FEATURES + ARRIVAL_FEATURES


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def anchor_time(df: pd.DataFrame) -> pd.Series:
    """Off-block time of each departure: AOBT, else EOBT, else take-off."""
    anchor = _utc(df["AOBT_3_flt"])
    if "EOBT_1_flt" in df.columns:
        anchor = anchor.fillna(_utc(df["EOBT_1_flt"]))
    return anchor.fillna(_utc(df["MVT_TIME_UTC_mvt"]))


#: cap on elements materialised at once when taking windowed medians
_BLOCK_BUDGET = 4_000_000


def _aggregate(
    obs_time: np.ndarray,
    obs_value: np.ndarray,
    obs_group: np.ndarray,
    anchor_time_values: np.ndarray,
    anchor_group: np.ndarray,
    n_anchors: int,
) -> dict[str, np.ndarray]:
    """Per-group backward-window median/count/last of observations before each anchor.

    Times are float seconds. Observations at exactly the anchor instant are included.
    """
    med30 = np.full(n_anchors, np.nan)
    med60 = np.full(n_anchors, np.nan)
    cnt30 = np.zeros(n_anchors)
    last = np.full(n_anchors, np.nan)

    obs_index = pd.Series(np.arange(len(obs_group))).groupby(obs_group).indices
    anchor_index = pd.Series(np.arange(n_anchors)).groupby(anchor_group).indices

    for key, a_idx in anchor_index.items():
        o_idx = obs_index.get(key)
        a_idx = np.asarray(a_idx)
        a_t = anchor_time_values[a_idx]
        finite = np.isfinite(a_t)
        if o_idx is None or not finite.any():
            continue
        o_idx = np.asarray(o_idx)
        order = np.argsort(obs_time[o_idx], kind="stable")
        o_idx = o_idx[order]
        t = obs_time[o_idx]
        v = obs_value[o_idx]

        hi = np.searchsorted(t, a_t, side="right")
        lo30 = np.searchsorted(t, a_t - 1800.0, side="left")
        lo60 = np.searchsorted(t, a_t - 3600.0, side="left")
        cnt30[a_idx] = np.where(finite, hi - lo30, 0)
        has_last = finite & (hi > 0)
        last[a_idx[has_last]] = v[hi[has_last] - 1]

        for lo, sink in ((lo30, med30), (lo60, med60)):
            width = np.where(finite, hi - lo, 0)
            for width_value in np.unique(width[width > 0]):
                sel = np.flatnonzero(width == width_value)
                # materialise the window blocks in slices: one group-window at a time can
                # otherwise reach hundreds of millions of elements
                step = max(1, _BLOCK_BUDGET // int(width_value))
                offsets = np.arange(width_value)
                for start in range(0, len(sel), step):
                    part = sel[start:start + step]
                    block = v[lo[part][:, None] + offsets[None, :]]
                    sink[a_idx[part]] = np.median(block, axis=1)
    return {"med30": med30, "med60": med60, "cnt30": cnt30, "last": last}


def physical_runway(series: pd.Series) -> np.ndarray:
    """Canonical runway identity, folding reciprocal labels (25L and 07R -> 07)."""
    digits = series.astype("string").str.extract(r"(\d{1,2})", expand=False)
    number = pd.to_numeric(digits, errors="coerce")
    folded = np.minimum(number, (number + 18) % 36)
    return folded.fillna(-1).astype("int16").astype(str).to_numpy()


def _seconds(series: pd.Series) -> np.ndarray:
    values = _utc(series)
    return (values - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def add(df: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Attach both feature families to a departure frame.

    `movements` is the unfiltered movement frame for the same period (departures *and*
    arrivals); `df` is the departure frame being modelled.
    """
    out = df.copy()
    anchor = _seconds(anchor_time(out))
    airport = out["ADEP_mvt"].astype("string").fillna("?").to_numpy()
    runway = airport + "|" + out["RUNWAY_mvt"].astype("string").fillna("?").to_numpy()
    # arrivals label the same strip by its reciprocal, so match on physical runway
    phys = airport + "|" + physical_runway(out["RUNWAY_mvt"])

    deps = movements[movements["PHASE_mvt"] == "DEP"]
    dep_t = _seconds(deps["MVT_TIME_UTC_mvt"])
    dep_v = (
        _utc(deps["MVT_TIME_UTC_mvt"]) - _utc(deps["AOBT_3_flt"])
    ).dt.total_seconds().to_numpy()
    dep_ok = np.isfinite(dep_t) & np.isfinite(dep_v)
    dep_airport = deps["ADEP_mvt"].astype("string").fillna("?").to_numpy()
    dep_runway = dep_airport + "|" + deps["RUNWAY_mvt"].astype("string").fillna("?").to_numpy()

    arrs = movements[movements["PHASE_mvt"] == "ARR"]
    arr_t = _seconds(arrs["BLOCK_TIME_UTC_mvt"])
    arr_v = arrs["TAXITIME_SEC_mvt"].to_numpy(dtype="float64")
    arr_ok = np.isfinite(arr_t) & np.isfinite(arr_v)
    arr_airport = arrs["ADES_mvt"].astype("string").fillna("?").to_numpy()
    arr_runway = arr_airport + "|" + physical_runway(arrs["RUNWAY_mvt"])

    plans = [
        ("nbr_taxi", dep_t[dep_ok], dep_v[dep_ok], dep_airport[dep_ok], dep_runway[dep_ok]),
        ("arrin", arr_t[arr_ok], arr_v[arr_ok], arr_airport[arr_ok], arr_runway[arr_ok]),
    ]
    for prefix, t, v, grp_airport, grp_runway in plans:
        anchor_runway = runway if prefix == "nbr_taxi" else phys
        wide = _aggregate(t, v, grp_airport, anchor, airport, len(out))
        narrow = _aggregate(t, v, grp_runway, anchor, anchor_runway, len(out))
        out[f"{prefix}_med_30m"] = wide["med30"]
        out[f"{prefix}_med_60m"] = wide["med60"]
        out[f"{prefix}_n_30m"] = wide["cnt30"]
        out[f"{prefix}_last"] = wide["last"]
        out[f"{prefix}_med_30m_runway"] = narrow["med30"]
    return out
