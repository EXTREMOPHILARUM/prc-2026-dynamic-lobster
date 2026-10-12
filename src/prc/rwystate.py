"""Runway state and gate retention around each departure (log section 63; idea from team outgoing-boat's public repo).

Computed from the departure frame alone: other departures' movement times, runways and NM clocks. None of it reads
BLOCK_TIME (the target), so it is available on the ranking set exactly as in training.

Runway state, at the departure's movement time t:
    rws_cadence20       mean interval of the airport's last 20 departures before t
    rws_dom_share       share of the airport's departures within t +/- 60 min that use this runway
    rws_n_rwys_60       distinct departure runways used in the 60 min before t
    rws_cfg_change      1 if that runway set differs from the one of the hour before

Gate retention, at the departure's NM off-block clock a = AOBT_3 (NaN without one):
    ret_held            departures at the airport with EOBT in [a - 2h, a) that had not pushed by a (AOBT_3 > a)
    ret_delay15         mean AOBT_3 - SCHED of the departures that pushed in [a - 15 min, a)
    ret_pushed15        how many pushed in that window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["rws_cadence20", "rws_dom_share", "rws_n_rwys_60", "rws_cfg_change", "ret_held", "ret_delay15", "ret_pushed15"]
H = 3600.0


def _sec(s: pd.Series) -> np.ndarray:
    """Seconds since the epoch, whatever the datetime unit (pandas 3 stores microseconds)."""
    v = pd.to_datetime(s, utc=True)
    return ((v - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64", na_value=np.nan)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """FEATURES for every row of a departure frame (ADEP_mvt, MVT_TIME_UTC_mvt, RUNWAY_mvt, AOBT_3_flt, EOBT_1_flt,
    SCHED_TIME_UTC_mvt), aligned to its index."""
    apt = df["ADEP_mvt"].astype(str).to_numpy()
    rwy = df["RUNWAY_mvt"].astype("string").fillna("?").to_numpy()
    t = _sec(df["MVT_TIME_UTC_mvt"]); a = _sec(df["AOBT_3_flt"]); e = _sec(df["EOBT_1_flt"]); s = _sec(df["SCHED_TIME_UTC_mvt"])
    out = {c: np.full(len(df), np.nan) for c in FEATURES}
    for ap in np.unique(apt):
        idx = np.flatnonzero(apt == ap)
        ti = t[idx]; o = np.argsort(ti); ts = ti[o]
        # cadence of the last 20 departures (strictly before t)
        k = np.searchsorted(ts, ti, side="left")
        ok = k >= 20
        cad = np.full(len(idx), np.nan)
        cad[ok] = (ti[ok] - ts[k[ok] - 20]) / 20.0
        out["rws_cadence20"][idx] = cad
        n_all = np.searchsorted(ts, ti + H, side="right") - np.searchsorted(ts, ti - H, side="left")
        runways = np.unique(rwy[idx])
        cnt_now = np.zeros((len(runways), len(idx))); cnt_prev = np.zeros_like(cnt_now)
        same = np.zeros(len(idx))
        for r_i, r in enumerate(runways):
            tr = np.sort(ti[rwy[idx] == r])
            cnt_now[r_i] = np.searchsorted(tr, ti, side="left") - np.searchsorted(tr, ti - H, side="left")
            cnt_prev[r_i] = np.searchsorted(tr, ti - H, side="left") - np.searchsorted(tr, ti - 2 * H, side="left")
            m = rwy[idx] == r
            same[m] = (np.searchsorted(tr, ti[m] + H, side="right") - np.searchsorted(tr, ti[m] - H, side="left"))
        out["rws_dom_share"][idx] = same / np.maximum(n_all, 1)
        out["rws_n_rwys_60"][idx] = (cnt_now > 0).sum(0)
        out["rws_cfg_change"][idx] = np.any((cnt_now > 0) != (cnt_prev > 0), axis=0).astype(float)
        # gate retention at the NM off-block clock
        ai = a[idx]; ei = e[idx]; si = s[idx]
        has_a = np.isfinite(ai)
        oa = np.argsort(np.where(has_a, ai, np.inf)); as_ = ai[oa]; dly = (ai - si)[oa]
        fin = np.isfinite(as_)
        as_f = as_[fin]; dly_f = np.nan_to_num(dly[fin]); cs = np.concatenate([[0.0], np.cumsum(dly_f)])
        lo = np.searchsorted(as_f, ai - 900, side="left"); hi = np.searchsorted(as_f, ai, side="left")
        n15 = (hi - lo).astype(float)
        out["ret_pushed15"][idx] = np.where(has_a, n15, np.nan)
        out["ret_delay15"][idx] = np.where(has_a & (n15 > 0), (cs[hi] - cs[lo]) / np.maximum(n15, 1), np.nan)
        # held: EOBT in [a-2h, a) and AOBT > a. Candidates sorted by EOBT, so each departure only compares against
        # the slice of departures whose EOBT falls in its two-hour window
        held = np.full(len(idx), np.nan)
        ev = np.nan_to_num(ei, nan=np.inf); av = np.nan_to_num(ai, nan=np.inf)
        oe = np.argsort(ev); es = ev[oe]; a_by_e = av[oe]
        lo_e = np.searchsorted(es, ai - 2 * H, side="left"); hi_e = np.searchsorted(es, ai, side="left")
        for i in np.flatnonzero(has_a):
            held[i] = np.count_nonzero(a_by_e[lo_e[i]:hi_e[i]] > ai[i])
        out["ret_held"][idx] = held
    return pd.DataFrame(out, index=df.index)
