"""Departure sequencing features: who took off ahead of you, and how heavy they were.

Wake turbulence separation is the physical reason departures cannot be packed
arbitrarily close: a light aircraft behind a heavy one must wait substantially longer
than one behind a medium. The dataset gives every movement a wake category
(`WK_TBL_CAT_flt`) and a take-off time, so the sequence ahead of each departure is
reconstructable.

All inputs come from the challenge's own data - no external source, so nothing here
affects the external-data declaration.

Everything is computed from movements whose take-off precedes this one, and take-off
times are present for every row of the ranking set, so each feature is available at
scoring time. Note the lesson from the queue-count experiments: raw counts of nearby
departures explained essentially none of the model's residual, because `MVT - AOBT`
already encodes congestion. These features are not counts - they describe the *mix* of
traffic ahead, which separation rules depend on and a count cannot express.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: seq_prev_gap_s was measured and dropped: +0.003 against the model residual, the same
#: null result as the raw queue counts. The gap to the aircraft ahead is already implied
#: by MVT - AOBT; what the model lacks is the *composition* of the traffic ahead.
FEATURES = [
    "seq_prev_wake_heavy",
    "seq_heavy_ahead_15m",
    "seq_heavy_share_30m",
    "seq_departures_30m_runway",
    "seq_median_gap_30m",
]

#: categories that impose extra separation on a following departure
HEAVY_CATEGORIES = ("H", "J", "S")


def _seconds(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series, utc=True)
    unit = values.dt.unit if hasattr(values.dt, "unit") else "ns"
    divisor = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}[unit]
    return values.astype("int64").to_numpy() // divisor


def add(df: pd.DataFrame, time_col: str = "MVT_TIME_UTC_mvt") -> pd.DataFrame:
    """Attach sequencing features, computed per airport and runway."""
    out = df.copy()
    takeoff = _seconds(out[time_col])
    heavy = out["WK_TBL_CAT_flt"].astype("string").isin(HEAVY_CATEGORIES).to_numpy()

    for name in FEATURES:
        out[name] = np.nan

    keys = (out["ADEP_mvt"].astype(str) + "|" + out["RUNWAY_mvt"].astype(str))
    for idx in keys.groupby(keys).indices.values():
        idx = np.asarray(idx)
        order = np.argsort(takeoff[idx], kind="stable")
        idx = idx[order]
        t = takeoff[idx]
        h = heavy[idx].astype("float64")

        # gap to the departure immediately ahead on this runway
        gap = np.full(len(idx), np.nan)
        gap[1:] = np.diff(t)
        out.loc[idx, "seq_prev_gap_s"] = gap

        prev_heavy = np.full(len(idx), np.nan)
        prev_heavy[1:] = h[:-1]
        out.loc[idx, "seq_prev_wake_heavy"] = prev_heavy

        # traffic mix in the windows before this take-off
        lo15 = np.searchsorted(t, t - 900)
        lo30 = np.searchsorted(t, t - 1800)
        position = np.arange(len(idx))
        cumulative_heavy = np.concatenate([[0.0], np.cumsum(h)])
        n30 = position - lo30
        out.loc[idx, "seq_heavy_ahead_15m"] = cumulative_heavy[position] - cumulative_heavy[lo15]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[idx, "seq_heavy_share_30m"] = np.where(
                n30 > 0, (cumulative_heavy[position] - cumulative_heavy[lo30]) / n30, np.nan)
            out.loc[idx, "seq_departures_30m_runway"] = n30
            out.loc[idx, "seq_median_gap_30m"] = np.where(
                n30 > 1, (t - t[lo30]) / np.maximum(n30, 1), np.nan)
    return out
