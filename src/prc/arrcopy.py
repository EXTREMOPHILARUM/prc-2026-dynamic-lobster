"""Arrival schedule copies: how often the airport's arrivals that day have a recorded in-block
equal to their schedule.

At Rome, a departure with no NM record is either an ordinary taxi or a schedule copy (recorded
block = schedule, so the target is MVT - SCHED; `prc.lirfmoe`). Copies look like a reporting
fault, and arrivals show the same fault in their own block times, which the ranking set
publishes in full (only departure block times are withheld). On days when more of the
airport's arrivals copy their schedule, more of its no-record departures do (2025, LIRF lane
<= 8h: 40% in the lowest quarter of days, 62% in the highest; experiment log section 58).

**Not served.** As copy-classifier inputs these raise the lane AUC 0.927 -> 0.932 but worsen the
twelve-fold (+0.33s twelve, +1.30s Jan+Jul, 4/10 non-scored months won; section 58). Kept for
`scripts/experiment_lirf_unmatched_moe.py --arr-copy`.

Features, per departure, from the arrival rows of `movements` only:

    arrcopy_day_rate   share of arrivals at the airport on the departure's UTC day with
                       |BLOCK - SCHED| <= COPY_TOL
    arrcopy_day_n      arrivals behind that share
    arrcopy_op_rate    the same share over arrivals of the same operator prefix that day
                       (NaN when there are none)
    arrcopy_3h_rate    the same share over arrivals within WINDOW_S of the departure's MVT

Nothing reads a departure's own block time or target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COPY_TOL = 60.0
WINDOW_S = 3 * 3600.0
FEATURES = ["arrcopy_day_rate", "arrcopy_day_n", "arrcopy_op_rate", "arrcopy_3h_rate"]


def _sec(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _prefix(s: pd.Series) -> pd.Series:
    return s.astype("string").str.extract(r"^([A-Z]{3})")[0].fillna("?")


def add(df: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    arr = movements[movements["PHASE_mvt"] == "ARR"]
    a_t = _sec(arr["MVT_TIME_UTC_mvt"])
    a_copy = (np.abs(_sec(arr["BLOCK_TIME_UTC_mvt"]) - _sec(arr["SCHED_TIME_UTC_mvt"])) <= COPY_TOL).astype("float64")
    A = pd.DataFrame({"apt": arr["ADES_mvt"].astype("string").to_numpy(), "day": np.floor(a_t / 86400.0),
                      "op": _prefix(arr["FLIGHT_mvt"]).to_numpy(), "t": a_t, "cp": a_copy})
    d_t = _sec(df["MVT_TIME_UTC_mvt"])
    D = pd.DataFrame({"apt": df["ADEP_mvt"].astype("string").to_numpy(), "day": np.floor(d_t / 86400.0),
                      "op": _prefix(df["FLIGHT_mvt"]).to_numpy(), "t": d_t})
    day = A.groupby(["apt", "day"]).cp.agg(arrcopy_day_rate="mean", arrcopy_day_n="size")
    op = A.groupby(["apt", "day", "op"]).cp.mean().rename("arrcopy_op_rate")
    out = D.join(day, on=["apt", "day"]).join(op, on=["apt", "day", "op"])
    rate3h = np.full(len(D), np.nan)
    for apt, idx in D.groupby("apt").indices.items():
        g = A[A.apt == apt].sort_values("t")
        if g.empty:
            continue
        t = g.t.to_numpy(); c = np.concatenate([[0.0], np.cumsum(g.cp.to_numpy())])
        q = D.t.to_numpy()[idx]
        lo = np.searchsorted(t, q - WINDOW_S, side="left"); hi = np.searchsorted(t, q + WINDOW_S, side="right")
        n = hi - lo
        rate3h[idx] = np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)
    df = df.copy()
    df["arrcopy_day_rate"] = out.arrcopy_day_rate.to_numpy()
    df["arrcopy_day_n"] = out.arrcopy_day_n.fillna(0).to_numpy(dtype="float64")
    df["arrcopy_op_rate"] = out.arrcopy_op_rate.to_numpy()
    df["arrcopy_3h_rate"] = rate3h
    return df
