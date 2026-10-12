"""Stand exit from a departing aircraft's own ADS-B ground run, using the NM-recorded stand's
coordinates from the X-Plane Scenery Gateway (scripts/fetch_xplane_stands.py, GPL v2+).

The aircraft is at its stand until it starts pushing back; the last point within RADIUS_M of the
stand position is therefore an estimate of the off-block, and a sharper one than the dwell rules
in ground_points.py / own_flight_features.py, which never know where the stand is. Only points
after the run's last silent gap of GAP_SPLIT_S or more count: before that gap the aircraft was
arriving, and its last point near the stand hours earlier is the arrival, not the departure.

`stand_key` normalises stand names so NM's `STAND_mvt` and apt.dat names meet: upper-case,
alphanumerics only, leading zeros dropped inside digit runs (H01 -> H1), and apt.dat names reduced
to their last token ("Gate A 12" -> "12", also tried as "A12").

    uv run python -u scripts/adsb/stand_exit.py 2025-04-15 2025-10-15      # radius screen
"""
from __future__ import annotations

import re
import sys
from math import cos, radians

import numpy as np
import pandas as pd

STANDS = "data/external/xplane/stands.parquet"
RADII_M = (15.0, 25.0, 40.0, 60.0, 90.0)
RADIUS_M = 40.0   # chosen on 2025-04-15 + 2025-10-15 (non-scored): same accuracy as 25m, more rows (section 50)
GAP_SPLIT_S = 1200.0
SPREAD_M = 60.0


def stand_key(name: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", s)


def stand_table(path: str = STANDS, spread_m: float = SPREAD_M, variants: bool = True) -> pd.DataFrame:
    """(airport, key) -> lat, lon. Each apt.dat stand is filed under its full name and its last one
    or two tokens; with `variants`, lettered sub-positions also under their base (EDDM `117A`,
    `117B` -> `117`, which is how NM records them). A key that lands on places more than
    `spread_m` apart is ambiguous and dropped; below that its positions are averaged."""
    S = pd.read_parquet(path)
    toks = S.name.astype(str).str.split()
    keys = {"full": S.name.map(stand_key),
            "tail": toks.map(lambda t: stand_key(t[-1]) if t else ""),
            "tail2": toks.map(lambda t: stand_key("".join(t[-2:])) if len(t) >= 2 else "")}
    if variants:
        keys["base"] = keys["full"].str.replace(r"(?<=[0-9])[A-Z]+$", "", regex=True)
    T = pd.concat([S[["airport", "lat", "lon"]].assign(key=k) for k in keys.values()], ignore_index=True)
    T = T[T.key != ""].drop_duplicates()
    g = T.groupby(["airport", "key"])
    dy = (g.lat.transform("max") - g.lat.transform("min")) * 111_000
    dx = (g.lon.transform("max") - g.lon.transform("min")) * 111_000 * np.cos(np.radians(T.lat))
    return T[np.hypot(dx, dy) < spread_m].groupby(["airport", "key"], as_index=False)[["lat", "lon"]].mean()


def stand_positions(path: str = STANDS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    S = pd.read_parquet(path)
    return {a: (g.lat.to_numpy(), g.lon.to_numpy()) for a, g in S.groupby("airport")}


def nearest_stand(lat0: float, lon0: float, stands: tuple[np.ndarray, np.ndarray],
                  radius: float = RADIUS_M) -> tuple[float, float]:
    """The apt.dat stand within `radius` of a position (NaN, NaN if none): where an aircraft first
    seen after a silent turnaround was parked, whatever NM calls the stand."""
    s_lat, s_lon = stands
    d = np.hypot((s_lon - lon0) * 111_000.0 * cos(radians(lat0)), (s_lat - lat0) * 111_000.0)
    j = int(np.argmin(d))
    return (float(s_lat[j]), float(s_lon[j])) if d[j] < radius else (np.nan, np.nan)


def departure_start(t: np.ndarray) -> int:
    gaps = np.flatnonzero(np.diff(t) >= GAP_SPLIT_S)
    return int(gaps[-1]) + 1 if len(gaps) else 0


def exit_time(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, s_lat: float, s_lon: float,
              radius: float = RADIUS_M) -> tuple[float, float]:
    """(last time within `radius` of the stand after the last long gap, min distance there)."""
    if len(t) == 0 or not np.isfinite(s_lat):
        return np.nan, np.nan
    k = departure_start(t)
    d = np.hypot((lon[k:] - s_lon) * 111_000.0 * cos(radians(s_lat)), (lat[k:] - s_lat) * 111_000.0)
    near = np.flatnonzero(d < radius)
    return (float(t[k + near[-1]]) if len(near) else np.nan), float(d.min())


def _screen(days: list[str]) -> None:
    """Stand-exit accuracy and coverage on labelled days, for three ways of locating the stand:
    NM name as in v33 (30m ambiguity, no sub-positions), NM name with sub-positions and 60m,
    and the name-free fallback (nearest apt.dat stand to the first point after the last gap)."""
    import glob
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    tables = {"name_v33": stand_table(spread_m=30.0, variants=False), "name_v2": stand_table()}
    pos = stand_positions()
    out = []
    for day in days:
        G = pd.read_parquet(f"data/external/adsb/{day}/ground.parquet",
                            columns=["icao", "callsign", "airport", "run", "t", "lat", "lon", "took_off", "t_takeoff"])
        G = G[G.took_off].sort_values(["icao", "run", "t"])
        R = G.drop_duplicates(["icao", "run"])[["icao", "run", "callsign", "airport", "t_takeoff"]].copy()
        R["callsign"] = R.callsign.astype(str).str.strip(); R["airport"] = R.airport.astype(str)
        src = glob.glob(f"data/raw/training_{day[:7]}-01_*.parquet")[0]
        d = pd.read_parquet(src, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                          "AOBT_3_flt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt"])
        d = d[(d.PHASE_mvt == "DEP") & (d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == day)].copy()
        d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
        for c, col in (("mvt", "MVT_TIME_UTC_mvt"), ("blk", "BLOCK_TIME_UTC_mvt"), ("aobt", "AOBT_3_flt")):
            d[c] = (pd.to_datetime(d[col], utc=True) - epoch).dt.total_seconds()
        d["key"] = d.STAND_mvt.map(stand_key)
        for name, T in tables.items():
            d = d.merge(T.rename(columns={"lat": f"lat_{name}", "lon": f"lon_{name}"}),
                        left_on=["ADEP_mvt", "key"], right_on=["airport", "key"], how="left").drop(columns="airport")
        m = d.merge(R, left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"])
        m["dto"] = (m.t_takeoff - m.mvt).abs(); m = m[m.dto < 600].sort_values("dto").drop_duplicates("MVT_ID_mvt")
        pts = {k: g for k, g in G.groupby(["icao", "run"], observed=True)}
        for r in m.itertuples():
            g = pts[(r.icao, r.run)]; t = g.t.to_numpy(); la = g.lat.to_numpy(); lo = g.lon.to_numpy()
            row = {"apt": r.ADEP_mvt, "blk": r.blk, "aobt": r.aobt}
            for name in tables:
                row[name] = exit_time(t, la, lo, getattr(r, f"lat_{name}"), getattr(r, f"lon_{name}"))[0] - r.blk
            k = departure_start(t)
            p_lat, p_lon = nearest_stand(la[k], lo[k], pos[r.ADEP_mvt]) if r.ADEP_mvt in pos else (np.nan, np.nan)
            row["position"] = exit_time(t, la, lo, p_lat, p_lon)[0] - r.blk
            row["combined"] = row["name_v2"] if np.isfinite(row["name_v2"]) else row["position"]
            out.append(row)
    D = pd.DataFrame(out)
    print(f"{len(D):,} joined departures on {days}")
    a = (D.aobt - D.blk).dropna()
    print(f"   NM AOBT - BLOCK:            |e|<60 {(a.abs()<60).mean()*100:5.1f}%  |e|<120 {(a.abs()<120).mean()*100:5.1f}%")
    for name in ("name_v33", "name_v2", "position", "combined"):
        e = D[name].dropna()
        print(f"   {name:10s} exit seen {len(e)/len(D)*100:5.1f}%  median {e.median():+5.0f}s  |e|<60 {(e.abs()<60).mean()*100:5.1f}%"
              f"  |e|<120 {(e.abs()<120).mean()*100:5.1f}%  RMSE {np.sqrt((e**2).mean()):6.0f}")
    both = D[["name_v2", "position"]].dropna()
    print(f"   where both exist ({len(both):,}): exits agree within 30s on {((both.name_v2 - both.position).abs() < 30).mean()*100:.1f}%")
    print("   coverage by airport (%):")
    print((D.assign(v33=D.name_v33.notna(), comb=D.combined.notna()).groupby("apt")[["v33", "comb"]].mean() * 100).round(1).T.to_string())


if __name__ == "__main__":
    _screen(sys.argv[1:])
