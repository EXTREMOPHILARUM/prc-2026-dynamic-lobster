"""Ranking-month context features, all label-free and present at scoring time.

Everything here reads other rows of the same movement frame - the neighbouring departures'
NM clocks, the arrivals' taxi-in (unblanked in the ranking set), and the departure's own
NM timestamps - so the ranking months describe themselves. Nothing reads
`BLOCK_TIME_UTC_mvt` or the target of a departure.

Families (naming follows the public entries they were measured in):

* ``nbr_*``   neighbour NM taxi proxy `MVT - AOBT_3` of other departures, past `[t-w, t)`
              and future `(t, t+w]` of this take-off `t`, by airport / runway / stand
* ``q_*``     aircraft taxiing (between their AOBT and take-off) at this row's AOBT and
              at its take-off, by airport and runway
* ``arrctx_*`` arrival taxi-in aggregates for the same (airport, month, stand) and
              (airport, month, runway), and the previous arrival at the same stand
* ``clk_*``   minute and second of every NM timestamp, and the remaining pairwise deltas
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"15m": 900.0, "60m": 3600.0}
PROXY_MAX = 7200.0
QUEUE_MAX = 21600.0
ARR_TAXI_CAP = 7200.0
PREV_ARR_MAX_AGE = 86400.0
DELTA_CLIP = 172800.0
EPOCH = pd.Timestamp("2000-01-01", tz="UTC")

CLOCK_COLS = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
              "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt"]

NBR_FEATURES = [
    f"nbr_{g}_{d}_{w}_{s}"
    for g in ("apt", "rwy", "stand") for d in ("past", "next") for w in WINDOWS
    for s in ("mean", "n")
] + ["nbr_apt_own_minus_past_15m", "nbr_apt_last_proxy", "nbr_apt_next_proxy"]
QUEUE_FEATURES = ["q_apt_at_aobt", "q_apt_at_takeoff", "q_rwy_at_aobt", "q_rwy_at_takeoff"]
ARR_FEATURES = ["arrctx_stand_mean", "arrctx_stand_median", "arrctx_stand_n",
                "arrctx_rwy_mean", "arrctx_rwy_median", "arrctx_rwy_n",
                "arrctx_prev_age", "arrctx_prev_taxi", "arrctx_prev_type_match"]
CLOCK_FEATURES = [f"clk_{c}_{u}" for c in CLOCK_COLS for u in ("minute", "second")] + [
    "clk_mvt_minus_ARVT_1", "clk_mvt_minus_ARVT_3", "clk_lobt_minus_iobt",
    "clk_arvt3_minus_arvt1", "clk_sched_minus_aobt",
    "clk_lobt_minus_sched", "clk_eobt_minus_sched", "clk_eobt_minus_lobt", "clk_iobt_minus_sched"]
ORDER_FEATURES = ["ord_later_eobt_30m", "ord_total_30m", "ord_queue_eobt_mvt"]
FEATURES = NBR_FEATURES + QUEUE_FEATURES + ARR_FEATURES + CLOCK_FEATURES + ORDER_FEATURES


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _sec(s: pd.Series) -> np.ndarray:
    return (_utc(s) - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def _group_keys(df: pd.DataFrame, airport_col: str) -> dict[str, np.ndarray]:
    apt = df[airport_col].astype("string").fillna("?").to_numpy()
    rwy = df["RUNWAY_mvt"].astype("string").fillna("?").to_numpy()
    stand = df["STAND_mvt"].astype("string").fillna("?").to_numpy()
    return {"apt": apt, "rwy": np.char.add(np.char.add(apt.astype(str), "|"), rwy.astype(str)),
            "stand": np.char.add(np.char.add(apt.astype(str), "|"), stand.astype(str))}


def _window_stats(obs_t, obs_v, obs_g, q_t, q_g, n_q):
    """Per-group past/next window mean and count of observations around each query."""
    out = {f"{d}_{w}_{s}": np.full(n_q, np.nan) for d in ("past", "next") for w in WINDOWS for s in ("mean", "n")}
    last = np.full(n_q, np.nan); nxt = np.full(n_q, np.nan)
    obs_idx = pd.Series(np.arange(len(obs_g))).groupby(obs_g).indices
    q_idx = pd.Series(np.arange(n_q)).groupby(q_g).indices
    for key, qi in q_idx.items():
        qi = np.asarray(qi); qt = q_t[qi]; ok = np.isfinite(qt)
        oi = obs_idx.get(key)
        if oi is None or not ok.any():
            for w in WINDOWS:
                for d in ("past", "next"):
                    out[f"{d}_{w}_n"][qi[ok]] = 0.0
            continue
        oi = np.asarray(oi); order = np.argsort(obs_t[oi], kind="stable"); t = obs_t[oi][order]; v = obs_v[oi][order]
        cs = np.concatenate([[0.0], np.cumsum(v)])
        qt_ok = qt[ok]; qi_ok = qi[ok]
        at = np.searchsorted(t, qt_ok, side="left")      # obs strictly before t
        after = np.searchsorted(t, qt_ok, side="right")  # first obs strictly after t
        for w, width in WINDOWS.items():
            lo = np.searchsorted(t, qt_ok - width, side="left")
            n = at - lo; s = cs[at] - cs[lo]
            out[f"past_{w}_n"][qi_ok] = n; out[f"past_{w}_mean"][qi_ok] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            hi = np.searchsorted(t, qt_ok + width, side="right")
            n2 = hi - after; s2 = cs[hi] - cs[after]
            out[f"next_{w}_n"][qi_ok] = n2; out[f"next_{w}_mean"][qi_ok] = np.where(n2 > 0, s2 / np.maximum(n2, 1), np.nan)
        has_last = at > 0; last[qi_ok[has_last]] = v[at[has_last] - 1]
        has_next = after < len(t); nxt[qi_ok[has_next]] = v[after[has_next]]
    return out, last, nxt


def _active(starts: np.ndarray, ends: np.ndarray, q: np.ndarray) -> np.ndarray:
    s = np.sort(starts); e = np.sort(ends)
    return np.searchsorted(s, q, side="right") - np.searchsorted(e, q, side="right")


def add(df: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Attach all context features to a departure frame, from the full movement frame."""
    df = df.copy()
    n = len(df)
    dep = movements[movements["PHASE_mvt"] == "DEP"]
    arr = movements[movements["PHASE_mvt"] == "ARR"]

    # --- neighbour NM taxi proxy -------------------------------------------------------
    t_all = _sec(dep["MVT_TIME_UTC_mvt"]); a_all = _sec(dep["AOBT_3_flt"])
    proxy = t_all - a_all
    good = np.isfinite(proxy) & (proxy > 0) & (proxy <= PROXY_MAX)
    gk_dep = _group_keys(dep, "ADEP_mvt")
    gk_q = _group_keys(df, "ADEP_mvt")
    q_t = _sec(df["MVT_TIME_UTC_mvt"])
    own = q_t - _sec(df["AOBT_3_flt"])
    for g in ("apt", "rwy", "stand"):
        stats, last, nxt = _window_stats(t_all[good], proxy[good], gk_dep[g][good], q_t, gk_q[g], n)
        for k, v in stats.items():
            df[f"nbr_{g}_{k}"] = v
        if g == "apt":
            df["nbr_apt_last_proxy"] = last; df["nbr_apt_next_proxy"] = nxt
    df["nbr_apt_own_minus_past_15m"] = own - df["nbr_apt_past_15m_mean"].to_numpy()

    # --- queues -----------------------------------------------------------------------
    qgood = np.isfinite(proxy) & (proxy > 0) & (proxy <= QUEUE_MAX)
    q_a = _sec(df["AOBT_3_flt"])
    for g, name in (("apt", "apt"), ("rwy", "rwy")):
        at_aobt = np.full(n, np.nan); at_to = np.full(n, np.nan)
        dep_groups = pd.Series(np.arange(len(dep))).groupby(gk_dep[g]).indices
        for key, qi in pd.Series(np.arange(n)).groupby(gk_q[g]).indices.items():
            qi = np.asarray(qi); oi = dep_groups.get(key)
            if oi is None:
                at_aobt[qi] = 0; at_to[qi] = 0; continue
            oi = np.asarray(oi); oi = oi[qgood[oi]]
            s, e = a_all[oi], t_all[oi]
            fa = np.isfinite(q_a[qi])
            at_aobt[qi[fa]] = _active(s, e, q_a[qi[fa]]) - 1     # own interval starts at its AOBT
            at_to[qi] = _active(s, e, q_t[qi])                     # own interval already ended
        df[f"q_{name}_at_aobt"] = np.clip(at_aobt, 0, None); df[f"q_{name}_at_takeoff"] = at_to

    # --- arrival context --------------------------------------------------------------
    arr_taxi = pd.to_numeric(arr["TAXITIME_SEC_mvt"], errors="coerce").clip(0, ARR_TAXI_CAP)
    arr_month = _utc(arr["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m")
    dep_month = _utc(df["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m")
    for level, col in (("stand", "STAND_mvt"), ("rwy", "RUNWAY_mvt")):
        key_a = arr["ADES_mvt"].astype("string") + "|" + arr_month + "|" + arr[col].astype("string")
        agg = arr_taxi.groupby(key_a.to_numpy()).agg(["mean", "median", "size"])
        key_d = (df["ADEP_mvt"].astype("string") + "|" + dep_month + "|" + df[col].astype("string")).to_numpy()
        for stat, out in (("mean", "mean"), ("median", "median"), ("size", "n")):
            df[f"arrctx_{level}_{out}"] = agg[stat].reindex(key_d).to_numpy()
        df[f"arrctx_{level}_n"] = df[f"arrctx_{level}_n"].fillna(0)
    # previous arrival at the same stand by in-block time
    stand_a = (arr["ADES_mvt"].astype("string") + "|" + arr["STAND_mvt"].astype("string")).to_numpy()
    inblock = _sec(arr["BLOCK_TIME_UTC_mvt"]); a_type = arr["AIRCRAFT_TYPE_mvt"].astype("string").fillna("?").to_numpy()
    a_taxi = arr_taxi.to_numpy()
    stand_q = (df["ADEP_mvt"].astype("string") + "|" + df["STAND_mvt"].astype("string")).to_numpy()
    d_type = df["AIRCRAFT_TYPE_mvt"].astype("string").fillna("?").to_numpy()
    prev_age = np.full(n, np.nan); prev_taxi = np.full(n, np.nan); prev_match = np.full(n, np.nan)
    arr_groups = pd.Series(np.arange(len(arr))).groupby(stand_a).indices
    for key, qi in pd.Series(np.arange(n)).groupby(stand_q).indices.items():
        oi = arr_groups.get(key)
        if oi is None:
            continue
        qi = np.asarray(qi); oi = np.asarray(oi); oi = oi[np.isfinite(inblock[oi])]
        if len(oi) == 0:
            continue
        order = np.argsort(inblock[oi]); oi = oi[order]; tb = inblock[oi]
        pos = np.searchsorted(tb, q_t[qi], side="right") - 1
        ok = pos >= 0
        age = q_t[qi[ok]] - tb[pos[ok]]
        keep = age <= PREV_ARR_MAX_AGE
        sel = qi[ok][keep]; src = oi[pos[ok][keep]]
        prev_age[sel] = age[keep]; prev_taxi[sel] = a_taxi[src]
        prev_match[sel] = (a_type[src] == d_type[sel]).astype(float)
    df["arrctx_prev_age"] = prev_age; df["arrctx_prev_taxi"] = prev_taxi; df["arrctx_prev_type_match"] = prev_match

    # --- clocks -----------------------------------------------------------------------
    clocks = {c: _utc(df[c]) for c in CLOCK_COLS}
    for c, s in clocks.items():
        df[f"clk_{c}_minute"] = s.dt.minute.astype("float64"); df[f"clk_{c}_second"] = s.dt.second.astype("float64")
    def delta(a, b):
        return ((clocks[a] - clocks[b]).dt.total_seconds()).clip(-DELTA_CLIP, DELTA_CLIP).to_numpy()
    df["clk_mvt_minus_ARVT_1"] = delta("MVT_TIME_UTC_mvt", "ARVT_1_flt")
    df["clk_mvt_minus_ARVT_3"] = delta("MVT_TIME_UTC_mvt", "ARVT_3_flt")
    df["clk_lobt_minus_iobt"] = delta("LOBT_flt", "IOBT_flt")
    df["clk_arvt3_minus_arvt1"] = delta("ARVT_3_flt", "ARVT_1_flt")
    df["clk_sched_minus_aobt"] = delta("SCHED_TIME_UTC_mvt", "AOBT_3_flt")
    df["clk_lobt_minus_sched"] = delta("LOBT_flt", "SCHED_TIME_UTC_mvt")
    df["clk_eobt_minus_sched"] = delta("EOBT_1_flt", "SCHED_TIME_UTC_mvt")
    df["clk_eobt_minus_lobt"] = delta("EOBT_1_flt", "LOBT_flt")
    df["clk_iobt_minus_sched"] = delta("IOBT_flt", "SCHED_TIME_UTC_mvt")

    # --- take-off order against plan (kind-mango): departures that took off in the 30 min
    # before this one but filed a later EOBT_1, and departures airborne between own EOBT and MVT
    e_all = _sec(dep["EOBT_1_flt"]); e_q = _sec(df["EOBT_1_flt"])
    later = np.full(n, np.nan); total = np.full(n, np.nan); qe = np.full(n, np.nan)
    dep_groups = pd.Series(np.arange(len(dep))).groupby(gk_dep["apt"]).indices
    for key, qi in pd.Series(np.arange(n)).groupby(gk_q["apt"]).indices.items():
        qi = np.asarray(qi); oi = dep_groups.get(key)
        if oi is None:
            continue
        oi = np.asarray(oi); order = np.argsort(t_all[oi]); oi = oi[order]; tt = t_all[oi]; ee = e_all[oi]
        hi = np.searchsorted(tt, q_t[qi], side="left"); lo = np.searchsorted(tt, q_t[qi] - 1800.0, side="left")
        total[qi] = hi - lo
        # count of window members with EOBT later than own EOBT: sort-free via cumulative trick per query
        for j, (a, b) in enumerate(zip(lo, hi, strict=True)):
            if b > a and np.isfinite(e_q[qi[j]]):
                later[qi[j]] = np.count_nonzero(ee[a:b] > e_q[qi[j]])
        fe = np.isfinite(e_q[qi])
        qe[qi[fe]] = np.searchsorted(tt, q_t[qi[fe]], side="left") - np.searchsorted(tt, e_q[qi[fe]], side="right")
    df["ord_later_eobt_30m"] = later; df["ord_total_30m"] = total; df["ord_queue_eobt_mvt"] = qe
    return df
