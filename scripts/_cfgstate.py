"""Airport configuration-state features (rolling active runway mix and changes to it).

Built over the FULL movement frame (arrivals and departures), keyed on MVT_ID.
Arrivals key on ADES_mvt, departures on ADEP_mvt. Reciprocal runway labels are
canonicalised so 25L and 07R are one physical strip.

All inputs (RUNWAY_mvt, MVT_TIME_UTC_mvt, AOBT_3_flt) survive in the ranking set.
Timestamps: .dt.total_seconds() only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2000-01-01", tz="UTC")

FEATURES = [
    "cfg_n_dep_rwy_30m",
    "cfg_n_dep_rwy_60m",
    "cfg_n_arr_rwy_30m",
    "cfg_n_rwy_30m",
    "cfg_dep_mode_share_30m",
    "cfg_this_rwy_share_30m",
    "cfg_this_differs_from_mode",
    "cfg_dep_changed_30m",
    "cfg_arr_changed_30m",
    "cfg_time_since_dep_change_s",
    "cfg_time_since_arr_change_s",
    "cfg_this_rwy_share_30m_aobt",
    "cfg_dep_changed_30m_aobt",
    "cfg_time_since_dep_change_s_aobt",
    "cfg_dep_set_jaccard_30m",
]

_SWAP = {"L": "R", "R": "L", "C": "C"}


def physical_runway(values: pd.Series) -> pd.Series:
    """Canonicalise reciprocal labels: 25L and 07R -> '07R/25L'-style single key."""
    s = values.astype("string").str.strip().str.upper()
    head = s.str.extract(r"^(\d{1,2})([LRC]?)$")
    num = pd.to_numeric(head[0], errors="coerce")
    side = head[1].fillna("")
    recip_num = (num + 18) % 36
    recip_num = recip_num.where(recip_num != 0, 36)
    recip_side = side.map(lambda x: _SWAP.get(x, ""))
    a = (num.map(lambda v: "" if pd.isna(v) else f"{int(v):02d}") + side).fillna("")
    b = (recip_num.map(lambda v: "" if pd.isna(v) else f"{int(v):02d}") + recip_side).fillna("")
    canon = np.where(num.isna().to_numpy(), s.fillna("NA").to_numpy(),
                     np.where((num.fillna(0) <= recip_num.fillna(0)).to_numpy(), (a + "|" + b).to_numpy(),
                              (b + "|" + a).to_numpy()))
    return pd.Series(canon, index=values.index, dtype="object")


def _secs(series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(series, utc=True, errors="coerce")
    return (t - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def _window_counts(event_t: np.ndarray, codes: np.ndarray, n_codes: int,
                   query: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """counts[i, r] = events on runway r with query[i]-lo <= t < query[i]-hi."""
    out = np.zeros((len(query), n_codes), dtype="float32")
    for r in range(n_codes):
        tr = np.sort(event_t[codes == r])
        if tr.size == 0:
            continue
        out[:, r] = (np.searchsorted(tr, query - hi, side="left")
                     - np.searchsorted(tr, query - lo, side="left"))
    return out


def _mode_stats(counts: np.ndarray):
    total = counts.sum(axis=1)
    mode = np.where(total > 0, counts.argmax(axis=1), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, counts.max(axis=1) / np.maximum(total, 1), np.nan)
    return total, mode, share


def _time_since_change(query: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """Time since the modal runway last changed, over query times sorted ascending."""
    order = np.argsort(query, kind="stable")
    q, m = query[order], mode[order]
    last = np.empty(len(q))
    cur = np.nan
    prev_mode = -2
    for i in range(len(q)):
        if m[i] != prev_mode:
            if prev_mode != -2:
                cur = q[i]
            prev_mode = m[i]
        last[i] = cur
    out = np.empty(len(q))
    out[:] = q - last
    res = np.empty(len(query))
    res[order] = out
    return res


def add(movements: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of MVT_ID + config features for the DEPARTURE rows of `movements`."""
    mv = movements
    is_dep = (mv["PHASE_mvt"] == "DEP").to_numpy()
    airport = np.where(is_dep, mv["ADEP_mvt"].astype(str).to_numpy(),
                       mv["ADES_mvt"].astype(str).to_numpy())
    phys = physical_runway(mv["RUNWAY_mvt"]).to_numpy()
    t = _secs(mv["MVT_TIME_UTC_mvt"])
    aobt = _secs(mv["AOBT_3_flt"]) if "AOBT_3_flt" in mv.columns else np.full(len(mv), np.nan)

    dep_idx = np.flatnonzero(is_dep)
    res = {name: np.full(len(dep_idx), np.nan, dtype="float64") for name in FEATURES}
    dep_pos = {v: i for i, v in enumerate(dep_idx)}

    ap_series = pd.Series(airport)
    for idx in ap_series.groupby(ap_series).indices.values():
        idx = np.asarray(idx)
        sub_dep = idx[is_dep[idx]]
        if sub_dep.size == 0:
            continue
        codes_all, uniq = pd.factorize(pd.Series(phys[idx]))
        n_codes = len(uniq)
        t_ap = t[idx]
        d_mask = is_dep[idx]
        dep_t = t[sub_dep]
        dep_codes = codes_all[d_mask]
        dep_aobt = aobt[sub_dep]
        anchor_aobt = np.where(np.isnan(dep_aobt), dep_t, dep_aobt)

        ev_dep_t, ev_dep_c = t_ap[d_mask], codes_all[d_mask]
        ev_arr_t, ev_arr_c = t_ap[~d_mask], codes_all[~d_mask]

        c_dep30 = _window_counts(ev_dep_t, ev_dep_c, n_codes, dep_t, 1800, 0)
        c_dep60 = _window_counts(ev_dep_t, ev_dep_c, n_codes, dep_t, 3600, 0)
        c_dep_prev = _window_counts(ev_dep_t, ev_dep_c, n_codes, dep_t, 3600, 1800)
        c_arr30 = _window_counts(ev_arr_t, ev_arr_c, n_codes, dep_t, 1800, 0)
        c_arr_prev = _window_counts(ev_arr_t, ev_arr_c, n_codes, dep_t, 3600, 1800)
        c_dep30_a = _window_counts(ev_dep_t, ev_dep_c, n_codes, anchor_aobt, 1800, 0)

        tot30, mode30, share30 = _mode_stats(c_dep30)
        _, mode_prev, _ = _mode_stats(c_dep_prev)
        _, amode30, _ = _mode_stats(c_arr30)
        _, amode_prev, _ = _mode_stats(c_arr_prev)
        tot30a, mode30a, _ = _mode_stats(c_dep30_a)

        rows = np.arange(len(sub_dep))
        own = c_dep30[rows, dep_codes]
        own_a = c_dep30_a[rows, dep_codes]

        set30 = c_dep30 > 0
        setp = c_dep_prev > 0
        inter = (set30 & setp).sum(axis=1)
        union = (set30 | setp).sum(axis=1)

        vals = {
            "cfg_n_dep_rwy_30m": set30.sum(axis=1).astype("float64"),
            "cfg_n_dep_rwy_60m": (c_dep60 > 0).sum(axis=1).astype("float64"),
            "cfg_n_arr_rwy_30m": (c_arr30 > 0).sum(axis=1).astype("float64"),
            "cfg_n_rwy_30m": ((c_dep30 + c_arr30) > 0).sum(axis=1).astype("float64"),
            "cfg_dep_mode_share_30m": share30,
            "cfg_this_rwy_share_30m": np.where(tot30 > 0, own / np.maximum(tot30, 1), np.nan),
            "cfg_this_differs_from_mode": np.where(mode30 >= 0, (dep_codes != mode30).astype(float), np.nan),
            "cfg_dep_changed_30m": np.where((mode30 >= 0) & (mode_prev >= 0),
                                            (mode30 != mode_prev).astype(float), np.nan),
            "cfg_arr_changed_30m": np.where((amode30 >= 0) & (amode_prev >= 0),
                                            (amode30 != amode_prev).astype(float), np.nan),
            "cfg_time_since_dep_change_s": _time_since_change(dep_t, mode30),
            "cfg_time_since_arr_change_s": _time_since_change(dep_t, amode30),
            "cfg_this_rwy_share_30m_aobt": np.where(tot30a > 0, own_a / np.maximum(tot30a, 1), np.nan),
            "cfg_dep_changed_30m_aobt": np.where((mode30a >= 0) & (mode30 >= 0),
                                                 (mode30a != mode30).astype(float), np.nan),
            "cfg_time_since_dep_change_s_aobt": _time_since_change(anchor_aobt, mode30a),
            "cfg_dep_set_jaccard_30m": np.where(union > 0, inter / np.maximum(union, 1), np.nan),
        }
        pos = np.array([dep_pos[i] for i in sub_dep])
        for k, v in vals.items():
            res[k][pos] = v

    out = pd.DataFrame(res)
    out.insert(0, "MVT_ID_mvt", mv["MVT_ID_mvt"].to_numpy()[dep_idx])
    return out
