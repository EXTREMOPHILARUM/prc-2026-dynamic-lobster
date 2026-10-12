"""Direction-aware airport configuration features: genuine runway reversals.

_cfgstate canonicalises reciprocals to a physical strip, which is right for "which strip"
but destroys the direction. A wind-shift reversal is exactly a change of direction on the
same strip (25 -> 07), so these features keep the heading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _cfgstate import _secs, _window_counts, physical_runway

FEATURES = [
    "cfg_reversal_60m",
    "cfg_time_since_reversal_s",
    "cfg_this_dir_differs_from_mode",
    "cfg_dir_mode_share_30m",
    "cfg_new_strip_30m",
    "cfg_this_strip_new",
    "cfg_arr_reversal_60m",
    "cfg_dep_arr_opposed",
]


def _heading(values: pd.Series) -> pd.Series:
    n = pd.to_numeric(values.astype("string").str.extract(r"^(\d{1,2})")[0], errors="coerce")
    return n * 10.0


def _angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def add(movements: pd.DataFrame) -> pd.DataFrame:
    mv = movements
    is_dep = (mv["PHASE_mvt"] == "DEP").to_numpy()
    airport = np.where(is_dep, mv["ADEP_mvt"].astype(str).to_numpy(),
                       mv["ADES_mvt"].astype(str).to_numpy())
    head = _heading(mv["RUNWAY_mvt"]).to_numpy()
    strip = physical_runway(mv["RUNWAY_mvt"]).to_numpy()
    t = _secs(mv["MVT_TIME_UTC_mvt"])

    dep_idx = np.flatnonzero(is_dep)
    res = {name: np.full(len(dep_idx), np.nan) for name in FEATURES}
    dep_pos = {v: i for i, v in enumerate(dep_idx)}
    ap_series = pd.Series(airport)

    for idx in ap_series.groupby(ap_series).indices.values():
        idx = np.asarray(idx)
        sub_dep = idx[is_dep[idx]]
        if sub_dep.size == 0:
            continue
        d_mask = is_dep[idx]
        t_ap = t[idx]
        dep_t = t[sub_dep]

        hcodes, huniq = pd.factorize(pd.Series(head[idx]))
        hvals = np.asarray(huniq, dtype="float64")
        scodes, suniq = pd.factorize(pd.Series(strip[idx]))

        def mode_dir(codes, mask, lo, hi, _t=t_ap, _h=hvals, _q=dep_t):
            c = _window_counts(_t[mask], codes[mask], len(_h), _q, lo, hi)
            tot = c.sum(axis=1)
            m = np.where(tot > 0, c.argmax(axis=1), -1)
            share = np.where(tot > 0, c.max(axis=1) / np.maximum(tot, 1), np.nan)
            return m, share

        dmode, dshare = mode_dir(hcodes, d_mask, 1800, 0)
        dmode_prev, _ = mode_dir(hcodes, d_mask, 5400, 1800)
        amode, _ = mode_dir(hcodes, ~d_mask, 1800, 0)
        amode_prev, _ = mode_dir(hcodes, ~d_mask, 5400, 1800)

        ok = (dmode >= 0) & (dmode_prev >= 0)
        rev = np.where(ok, (_angdiff(hvals[np.maximum(dmode, 0)],
                                     hvals[np.maximum(dmode_prev, 0)]) > 90).astype(float), np.nan)
        aok = (amode >= 0) & (amode_prev >= 0)
        arev = np.where(aok, (_angdiff(hvals[np.maximum(amode, 0)],
                                       hvals[np.maximum(amode_prev, 0)]) > 90).astype(float), np.nan)
        opp = np.where((dmode >= 0) & (amode >= 0),
                       (_angdiff(hvals[np.maximum(dmode, 0)],
                                 hvals[np.maximum(amode, 0)]) > 90).astype(float), np.nan)

        own_h = head[sub_dep]
        this_diff = np.where(dmode >= 0,
                             (_angdiff(own_h, hvals[np.maximum(dmode, 0)]) > 90).astype(float), np.nan)

        s30 = _window_counts(t_ap[d_mask], scodes[d_mask], len(suniq), dep_t, 1800, 0) > 0
        sprev = _window_counts(t_ap[d_mask], scodes[d_mask], len(suniq), dep_t, 5400, 1800) > 0
        new_strip = (s30 & ~sprev).sum(axis=1).astype("float64")
        rows = np.arange(len(sub_dep))
        own_s = scodes[d_mask]
        this_new = (~sprev[rows, own_s]).astype(float)

        # time since the last reversal event, over departures ordered in time
        order = np.argsort(dep_t, kind="stable")
        q, r = dep_t[order], rev[order]
        last = np.full(len(q), np.nan)
        cur = np.nan
        prev_flag = np.nan
        for i in range(len(q)):
            if r[i] == 1.0 and prev_flag != 1.0:
                cur = q[i]
            prev_flag = r[i]
            last[i] = cur
        tsr = np.empty(len(q))
        tsr[:] = q - last
        since = np.empty(len(dep_t))
        since[order] = tsr

        vals = {"cfg_reversal_60m": rev, "cfg_time_since_reversal_s": since,
                "cfg_this_dir_differs_from_mode": this_diff, "cfg_dir_mode_share_30m": dshare,
                "cfg_new_strip_30m": new_strip, "cfg_this_strip_new": this_new,
                "cfg_arr_reversal_60m": arev, "cfg_dep_arr_opposed": opp}
        pos = np.array([dep_pos[i] for i in sub_dep])
        for k, v in vals.items():
            res[k][pos] = v

    out = pd.DataFrame(res)
    out.insert(0, "MVT_ID_mvt", mv["MVT_ID_mvt"].to_numpy()[dep_idx])
    return out
