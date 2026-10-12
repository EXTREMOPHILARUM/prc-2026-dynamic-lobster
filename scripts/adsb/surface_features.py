"""Stage B of the surface-state extractor: airport surface congestion at each departure's take-off,
built only from *other* aircraft's ADS-B ground runs (ground_points.py output).

The scored flight is never joined to its own trace (the rules question in section 37). It is
excluded by time instead: any run whose ADS-B take-off falls in [MVT - EX_BEFORE, MVT + EX_AFTER]
is dropped from every feature of that row. ADS-B take-off (last ground point) sits a median 32s
before MVT, 5-95% within [-214s, +1s] on 2025-07-15, so the window removes the flight itself on
essentially every covered row; it also removes its immediate neighbours, identically for every
row, which costs information but introduces no dependence on the row's own taxi time.

Counts are divided by the airport-day coverage (ADS-B take-offs / NM departures that day), so a
day with a thin feed does not read as a quiet day. Coverage below MIN_COV leaves the row NaN.

Features (all at airport a, reference time t = MVT, q = t - EX_BEFORE):
    sx_cov              ADS-B take-offs / NM departures, airport-day
    sx_dep_prev_n       take-offs in [t - 1800, q), per coverage
    sx_dep_prev_taxi    median dwell-end -> take-off of those, where the dwell was seen
    sx_dep_prev_hold    median seconds stationary during taxi-out of those
    sx_out_active       aircraft in taxi-out at q, per coverage
    sx_in_active        aircraft in taxi-in at t - 600, per coverage
    sx_out_gs           median ground speed of moving taxi-out points in [t - 1200, q)

Writes data/external/adsb/surface_features.parquet keyed by MVT_ID_mvt, for 2025 training
departures and 2026 ranking departures on every day with a ground.parquet.

    uv run python -u scripts/adsb/surface_features.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
EX_BEFORE, EX_AFTER = 420.0, 60.0
PREV_WIN = 1800.0
GS_WIN = 1200.0
IN_LAG = 600.0
MOVE_KT = 3.0
MAX_TAXI = 3600.0
MIN_COV = 0.10
PHASE_IN, PHASE_OUT = 1, 3
FEATURES = ["sx_cov", "sx_dep_prev_n", "sx_dep_prev_taxi", "sx_dep_prev_hold", "sx_out_active", "sx_in_active", "sx_out_gs"]


def departures() -> pd.DataFrame:
    parts = [pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
             for f in sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]]
    d = pd.concat(parts, ignore_index=True)
    d = d[d.PHASE_mvt == "DEP"].copy()
    d["t"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
    d["day"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    return d[["MVT_ID_mvt", "ADEP_mvt", "t", "day"]].drop_duplicates("MVT_ID_mvt")


def load_points(day: str) -> pd.DataFrame | None:
    """This day's ground points plus the previous day's (windows reach back past midnight)."""
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    for tag, dd in (("p", prev), ("c", day)):
        f = f"data/external/adsb/{dd}/ground.parquet"
        if os.path.exists(f):
            g = pd.read_parquet(f)
            g["airport"] = g["airport"].astype(str)
            g["rid"] = tag + g["icao"].astype(str) + "_" + g["run"].astype(str)
            parts.append(g)
    if not parts or not os.path.exists(f"data/external/adsb/{day}/ground.parquet"):
        return None
    return pd.concat(parts, ignore_index=True)


def run_table(G: pd.DataFrame) -> pd.DataFrame:
    """One row per run: taxi-out interval and hold time, taxi-in interval."""
    G = G.sort_values(["rid", "t"])
    out = G[G.phase == PHASE_OUT]
    dt = out.groupby("rid")["t"].diff().shift(-1).fillna(0.0).clip(upper=120.0)
    still = out["gs"].fillna(0.0) <= MOVE_KT
    hold = (dt * still).groupby(out["rid"]).sum()
    o = out.groupby("rid").agg(airport=("airport", "first"), out_start=("t", "min"), t_takeoff=("t_takeoff", "first"),
                               t_dwell_end=("t_dwell_end", "first"))
    o["hold"] = hold
    inn = G[G.phase == PHASE_IN].groupby("rid").agg(airport=("airport", "first"), in_start=("t", "min"), in_end=("t", "max"))
    return o[o.t_takeoff.notna()], inn


def features_for_day(day: str, D: pd.DataFrame) -> pd.DataFrame | None:
    G = load_points(day)
    if G is None:
        return None
    runs_out, runs_in = run_table(G)
    rows = []
    for apt, dep in D.groupby("ADEP_mvt"):
        n_nm = len(dep)
        o = runs_out[runs_out.airport == apt].sort_values("t_takeoff")
        today = o.t_takeoff.between(dep.t.min() - 1800, dep.t.max() + 1800)
        cov = float(today.sum()) / n_nm if n_nm else 0.0
        f = pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(), "sx_cov": cov})
        for c in FEATURES[1:]:
            f[c] = np.nan
        if cov < MIN_COV or o.empty:
            rows.append(f); continue
        to = o.t_takeoff.to_numpy(); st = o.out_start.to_numpy()
        taxi = np.where((to - o.t_dwell_end.to_numpy()) <= MAX_TAXI, to - o.t_dwell_end.to_numpy(), np.nan)
        hold = o.hold.to_numpy()
        i = runs_in[runs_in.airport == apt]; i_s = i.in_start.to_numpy(); i_e = i.in_end.to_numpy()
        pts = G[(G.airport == apt) & (G.phase == PHASE_OUT) & (G.gs > MOVE_KT)][["t", "gs", "t_takeoff"]].sort_values("t")
        p_t = pts.t.to_numpy(); p_gs = pts.gs.to_numpy(dtype="float64"); p_to = pts.t_takeoff.to_numpy()
        out = np.full((len(dep), 6), np.nan)
        for k, t in enumerate(dep.t.to_numpy()):
            q = t - EX_BEFORE
            lo, hi = np.searchsorted(to, [t - PREV_WIN, q])
            out[k, 0] = (hi - lo) / cov
            if hi > lo:
                out[k, 1] = np.nanmedian(taxi[lo:hi]) if np.isfinite(taxi[lo:hi]).any() else np.nan
                out[k, 2] = np.median(hold[lo:hi])
            # in taxi-out at q: started by q, took off after q, and not inside the exclusion window
            late = to > t + EX_AFTER
            out[k, 3] = float(((st <= q) & (to > q) & late).sum()) / cov
            r = t - IN_LAG
            out[k, 4] = float(((i_s <= r) & (i_e >= r)).sum()) / cov
            a, b = np.searchsorted(p_t, [t - GS_WIN, q])
            sel = (p_to[a:b] < t - EX_BEFORE) | (p_to[a:b] > t + EX_AFTER)
            out[k, 5] = np.median(p_gs[a:b][sel]) if sel.any() else np.nan
        f[FEATURES[1:]] = out
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def main():
    D = departures()
    days = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("data/external/adsb/*/ground.parquet"))
    out = []
    for day in days:
        f = features_for_day(day, D[D.day == day])
        if f is not None:
            out.append(f)
            print(f"{day}: {len(f):,} departures, covered {int(f.sx_dep_prev_n.notna().sum()):,}", flush=True)
    F = pd.concat(out, ignore_index=True)
    dst = "data/external/adsb/surface_features.parquet"
    F.to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst)
    print(f"wrote {len(F):,} rows over {len(out)} days to {dst}")


if __name__ == "__main__":
    main()
