"""Arrival-landing pressure: how many aircraft land around your push-back.

A departure waiting at the holding point competes with arrivals for the same strip of
concrete. The crossing traffic that matters is the traffic landing *while the aircraft
taxis*, which means the windows must be anchored on off-block time (`AOBT_3_flt`), not
on take-off. An earlier screen anchored the forward windows on take-off and concluded
the signal was absent; that measured arrivals landing *after* the taxi had already
finished. See docs/experiment-log.md.

Two spatial buckets per window:

* **same physical runway** - the departure's runway, with reciprocal labels collapsed:
  `25L` and `07R` are the same strip of asphalt and an aircraft landing on either one
  occupies it.
* **other physical runway** - everything else landing at the airport, which is airfield
  loading rather than direct runway contention.

Arrivals are keyed by `ADES_mvt` (where they land) and departures by `ADEP_mvt` (where
they leave from). Both columns, the arrival runway, and arrival movement times are
complete in the ranking set, so every feature here is available at scoring time.

Nothing external is used - this is the competition's own movement frame - so the
external-data declaration is unaffected.

`arr_during_taxi_*` uses a window whose *length* is `MVT - AOBT`, the taxi proxy the
model already carries, so it risks being a nonlinear re-encoding of a signal already
present. `FIXED_WINDOW_FEATURES` is the control that omits those two. Measured as a
paired 12-fold, scored on January+July pooled - the months the organisers actually
score, and a harsher test than the 12-month average because the tail carries 49.4% of
the squared error there against 36.1% over the year:

    Jan+Jul pooled (n=344,419)      12-month (n=2,085,047)
    baseline v11    369.4615        285.6515
    six features    368.9109        285.0229   11/12 months
    fixed windows   368.8987        285.0245   11/12 months

The two are indistinguishable, and the control is marginally *ahead* on the scored
months. The taxi-length windows therefore contribute nothing the AOBT-anchored fixed
windows do not already carry, so `FIXED_WINDOW_FEATURES` is preferred: four features
rather than six, and it does not re-encode the anchor.

The gain is entirely in the bulk, AOBT-present rows (Jan+Jul bulk 263.13 -> 262.30);
the tail and the missing-AOBT rows are flat to fractionally worse.

February appears to regress against the stored `sequencing_oof_predictions` baseline,
but that is an artifact of the comparison, not a property of the features. Refitting
February *paired* - same script, same run, only the arrival columns differing - has the
features improving it under both column arrangements and both sampling rates
(`scripts/experiment_arrivals_february.py`):

    ff=1.0   no arrivals 285.7894  ->  with arrivals 285.6993   -0.0631% SSE
    ff=0.9   no arrivals 285.4390  ->  with arrivals 285.4163   -0.0159% SSE

That same probe shows the stored baseline is not a sound per-month reference: refitting
the baseline reproduces February at 285.4390 against the stored 285.1868, a 0.25s gap
larger than the effect being measured. Trust the pooled figures, and do not read the
per-month win/loss count as a statistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: the six-feature family confirmed on 12/12 months
FEATURES = [
    "arr_prev_15m_same_physical",
    "arr_prev_20m_same_physical",
    "arr_during_taxi_same_physical",
    "arr_next_20m_aobt_same_physical",
    "arr_during_taxi_other_physical",
    "arr_next_20m_aobt_other_physical",
]

#: control: the AOBT-anchored fixed-length windows only, no taxi-length window
FIXED_WINDOW_FEATURES = [name for name in FEATURES if "during_taxi" not in name]

TIME_COL = "MVT_TIME_UTC_mvt"
AOBT_COL = "AOBT_3_flt"
RUNWAY_COL = "RUNWAY_mvt"

_UNKNOWN_RUNWAY = "?"
_RECIPROCAL_SUFFIX = {"L": "R", "R": "L", "C": "C", "": ""}

#: (name, window low, window high); lows/highs are resolved against takeoff/aobt below
_WINDOWS = {
    "arr_prev_15m": ("takeoff", -900, "takeoff", 0),
    "arr_prev_20m": ("takeoff", -1200, "takeoff", 0),
    "arr_during_taxi": ("aobt", 0, "takeoff", 0),
    "arr_next_20m_aobt": ("aobt", 0, "aobt", 1200),
}


def physical_runway(value: object) -> str:
    """Canonical name for the strip of asphalt, collapsing reciprocal designators.

    ``25L`` and ``07R`` are the two ends of one runway, so both map to ``07R/25L``.
    Anything that is not a two-digit designator maps to ``"?"``, which groups the
    unparseable labels together rather than silently matching a real runway.
    """
    if not isinstance(value, str):
        return _UNKNOWN_RUNWAY
    text = value.strip().upper()
    if len(text) < 2 or not text[:2].isdigit():
        return _UNKNOWN_RUNWAY
    number = int(text[:2])
    if not 1 <= number <= 36:
        return _UNKNOWN_RUNWAY
    suffix = text[2:3]
    opposite = ((number + 17) % 36) + 1
    opposite_suffix = _RECIPROCAL_SUFFIX.get(suffix, suffix)
    return "/".join(sorted((f"{number:02d}{suffix}", f"{opposite:02d}{opposite_suffix}")))


def _seconds(series: pd.Series) -> np.ndarray:
    """Epoch seconds as float, NaN where the timestamp is missing."""
    values = pd.to_datetime(series, utc=True, errors="coerce")
    return (values - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(
        dtype="float64"
    )


def _window_count(times: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Number of sorted `times` inside each [lo, hi]; NaN where the window is undefined.

    A row with no AOBT has an undefined window and stays NaN rather than becoming a
    zero count, which would be a lie the trees would happily split on.
    """
    result = np.full(len(lo), np.nan, dtype="float32")
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    if valid.any():
        result[valid] = (
            np.searchsorted(times, hi[valid], side="right")
            - np.searchsorted(times, lo[valid], side="left")
        )
    return result


def add(
    departures: pd.DataFrame,
    movements: pd.DataFrame | None = None,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Attach arrival-pressure features to `departures`.

    `movements` is the *full* movement frame the arrivals are drawn from; the departure
    rows in it are ignored. When omitted, `departures` is used, which yields no arrivals
    and therefore all-NaN features - callers that mean to use these features must pass
    the full frame.
    """
    out = departures.copy()
    source = departures if movements is None else movements

    takeoff = _seconds(out[time_col])
    aobt = _seconds(out[AOBT_COL]) if AOBT_COL in out.columns else np.full(len(out), np.nan)
    anchors = {"takeoff": takeoff, "aobt": aobt}

    dep_airport = out["ADEP_mvt"].astype("string").to_numpy()
    dep_physical = np.array(
        [physical_runway(v) for v in out[RUNWAY_COL].astype(object).to_numpy()]
    )

    arrivals = source[source["PHASE_mvt"].astype("string") == "ARR"]
    arr_time = _seconds(arrivals[time_col])
    arr_airport = arrivals["ADES_mvt"].astype("string").to_numpy()
    arr_physical = np.array(
        [physical_runway(v) for v in arrivals[RUNWAY_COL].astype(object).to_numpy()]
    )
    finite = np.isfinite(arr_time)
    arr_time, arr_airport, arr_physical = arr_time[finite], arr_airport[finite], arr_physical[finite]

    same = {name: np.full(len(out), np.nan, dtype="float32") for name in _WINDOWS}
    total = {name: np.full(len(out), np.nan, dtype="float32") for name in _WINDOWS}

    by_airport: dict[str, np.ndarray] = {
        airport: np.asarray(idx)
        for airport, idx in pd.Series(arr_airport).groupby(arr_airport).indices.items()
    }
    dep_index = pd.Series(dep_airport)
    for airport, dep_idx in dep_index.groupby(dep_airport).indices.items():
        dep_idx = np.asarray(dep_idx)
        arr_idx = by_airport.get(airport, np.empty(0, dtype="int64"))
        arr_idx = arr_idx[np.argsort(arr_time[arr_idx], kind="stable")]
        times = arr_time[arr_idx]
        physical = arr_physical[arr_idx]

        for name, (lo_anchor, lo_offset, hi_anchor, hi_offset) in _WINDOWS.items():
            lo = anchors[lo_anchor][dep_idx] + lo_offset
            hi = anchors[hi_anchor][dep_idx] + hi_offset
            total[name][dep_idx] = _window_count(times, lo, hi)

        for runway in pd.unique(dep_physical[dep_idx]):
            runway_idx = dep_idx[dep_physical[dep_idx] == runway]
            runway_times = times[physical == runway]
            for name, (lo_anchor, lo_offset, hi_anchor, hi_offset) in _WINDOWS.items():
                lo = anchors[lo_anchor][runway_idx] + lo_offset
                hi = anchors[hi_anchor][runway_idx] + hi_offset
                same[name][runway_idx] = _window_count(runway_times, lo, hi)

    for name in _WINDOWS:
        out[f"{name}_same_physical"] = same[name]
        out[f"{name}_other_physical"] = (total[name] - same[name]).astype("float32")
    return out
