"""Per-departure features from the departing flight's OWN ADS-B ground run (ground_points.py).

Join: ADS-B callsign == CALLSIGN_flt (falling back to FLIGHT_mvt), same airport, ADS-B take-off
within JOIN_S of MVT, nearest take-off kept. Widening JOIN_S from 600 to 1200s, or keying on
FLIGHT_mvt as well, adds 0.3 points of join rate: 87% of unjoined departures have no ADS-B
take-off within [-120, +30]s at all, so the ceiling is coverage, not the join (section 47).

    own_joined          1 if a take-off run joined
    own_to_minus_mvt    ADS-B take-off - MVT (join quality)
    own_dwell_taxi      MVT - end of the stand dwell, where one was seen
    own_dwell_len       length of that dwell
    own_first_taxi      MVT - first taxi-out point of the run. The aircraft was off-block before
                        it was first seen moving, so y >= own_first_taxi whatever the transponder
                        did: a lower bound, available on runs joined mid-taxi with no dwell.
    own_gap_seg         1 where the features come from the silent-gap rule below, 0 otherwise
    own_stand_exit_taxi (--stands) MVT - last point within 40m of the NM-recorded stand (X-Plane
                        coordinates) after the last long gap: within 60s of BLOCK_TIME on 78% of
                        the rows where it exists, against 24% for NM's AOBT (section 50)
    own_stand_dmin      (--stands) closest approach to that stand after the last long gap
    own_first_dist      (--stands) distance from that stand to the first moving point after the gap
    own_hold            seconds stationary during taxi-out (queue model input, not an expert feature)
    own_adsb_taxi       dwell end -> ADS-B take-off, the trace's own taxi-out duration

Silent-gap rule (section 49). 40% of joined 2025 runs landed and took off in one ground run with
no stop that ground_points.py accepts as a dwell, so every point is phase "unknown". On those runs
the transponder went quiet at the stand: the largest silent gap is 1-4 hours, and the last point
before it (still taxiing in) sits a median 416m from the first after it, so the 60m stationary
rule rejects the gap. A ground gap of GAP_SPLIT_S or more at an airport is a turnaround, so the
points after the last such gap are the departure: its first moving point bounds the taxi (the
aircraft was off-block before it moved), and a stationary point seen before that is a dwell end.
Applied only where no taxi-out phase exists, so it never overrides ground_points.py.

--ground-v2 (section 55) reads ground_v2.parquet: the join key is the callsign first set after
the run (`air_cs`, falling back to the run's), and runs whose ground reception ended mid-taxi
count as take-offs (`to_late`, joined on their first airborne point `t_air`). Adds `own_to_late`.

Writes data/interim/adsb_own_<year><tag>.parquet keyed by MVT_ID_mvt (v31 serves the untagged
files, built before the silent-gap rule; v32 onward uses --tag _v2).

2026 is the ranking set. Joining a scored flight to its own post-event trace is the open
question in section 37; the policy is not to build it until the organisers answer, so 2026
requires --allow-ranking and must not be run before a ruling says it may.

    uv run python -u scripts/adsb/own_flight_features.py 2025
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import stand_exit  # scripts/adsb/stand_exit.py, next to this script
import taxi_network  # scripts/adsb/taxi_network.py

from prc import config

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
JOIN_S = 600.0
PHASE_OUT = 3
MOVE_KT = 3.0
HOLD_GAP_S = 120.0
GAP_SPLIT_S = 1200.0
FEATURES = ["own_joined", "own_to_minus_mvt", "own_dwell_taxi", "own_dwell_len", "own_first_taxi", "own_gap_seg"]


def gap_departure(G: pd.DataFrame) -> pd.DataFrame:
    """Silent-gap rule for take-off runs with no taxi-out phase: per run, the gap length, the last
    stationary point after the last long gap and before the first motion, and that first motion."""
    runs = G.groupby(["icao", "run"], observed=True).phase.transform(lambda p: (p == PHASE_OUT).any())
    P = G[~runs].sort_values(["icao", "run", "t"])
    out = []
    for (icao, run), s in P.groupby(["icao", "run"], observed=True):
        t = s.t.to_numpy(); dt = np.diff(t)
        big = np.flatnonzero(dt >= GAP_SPLIT_S)
        if not len(big):
            continue
        k = int(big[-1]) + 1                                    # first point after the last long gap
        moving = np.nan_to_num(s.gs.to_numpy()[k:], nan=0.0) > MOVE_KT
        if not moving.any():
            continue
        first_move = k + int(np.argmax(moving))
        still_end = t[first_move - 1] if first_move > k else np.nan
        out.append((icao, run, float(dt[big[-1]]), still_end, t[first_move]))
    return pd.DataFrame(out, columns=["icao", "run", "gap_len", "gap_still_end", "gap_first_move"])


def takeoff_runs(path: str, v2: bool = False) -> pd.DataFrame:
    G = pd.read_parquet(path, columns=["icao", "callsign", "airport", "run", "t", "gs", "phase", "took_off",
                                       "t_takeoff", "t_dwell_start", "t_dwell_end",
                                       *(("air_cs", "t_air", "to_late", "landed", "t_land") if v2 else ())])
    G = G[G.took_off]
    out = G[G.phase == PHASE_OUT].sort_values(["icao", "run", "t"])
    key = [out.icao, out.run]
    first_out = out.groupby(["icao", "run"], observed=True).t.min().rename("t_first_out")
    # seconds spent stationary during taxi-out (queueing / holding); a silent gap counts at most HOLD_GAP_S
    dt = out.groupby(key, observed=True).t.diff().shift(-1).fillna(0.0).clip(upper=HOLD_GAP_S)
    hold = (dt * (out.gs.fillna(0.0) <= MOVE_KT)).groupby(key, observed=True).sum().rename("hold")
    R = G.drop_duplicates(["icao", "run"]).set_index(["icao", "run"]).join(first_out).join(hold).reset_index()
    R = R.merge(gap_departure(G), on=["icao", "run"], how="left")
    R["gap_seg"] = (R.t_first_out.isna() & R.gap_first_move.notna()).astype("float64")
    seg = R.gap_seg == 1
    R.loc[seg, "t_first_out"] = R.loc[seg, "gap_first_move"]
    R.loc[seg & R.t_dwell_end.isna(), "t_dwell_end"] = R.loc[seg, "gap_still_end"]
    # the dwell is the silent turnaround: from the last point before the gap to the first after it
    R.loc[seg & R.t_dwell_start.isna(), "t_dwell_start"] = R.loc[seg, "t_dwell_end"] - R.loc[seg, "gap_len"]
    R["callsign"] = R.callsign.astype(str).str.strip(); R["airport"] = R.airport.astype(str)
    if v2:
        # the departing callsign is the one set once airborne; a late take-off joins on its first airborne point
        air = R.air_cs.astype("string").str.strip()
        R["callsign"] = air.where(air.notna() & (air != ""), R.callsign).astype(str)
        R["t_join"] = np.where(R.to_late, R.t_air, R.t_takeoff); R["to_late"] = R.to_late.astype("float64")
    else:
        R["t_join"] = R.t_takeoff; R["to_late"] = 0.0
    if v2:
        # v10: the run's own landing, where the inbound leg is in the same ground run (NaN otherwise)
        R["t_land_run"] = np.where(R.landed.astype(bool), R.t_land, np.nan)
    else:
        R["t_land_run"] = np.nan
    return R[["icao", "run", "callsign", "airport", "t_takeoff", "t_join", "to_late", "t_dwell_start", "t_dwell_end",
              "t_first_out", "hold", "gap_seg", "t_land_run"]]


def departures(day: str) -> pd.DataFrame:
    src = f"data/raw/{config.RANKING_FILE}" if day.startswith("2026") else glob.glob(f"data/raw/training_{day[:7]}-01_*.parquet")[0]
    t = pd.read_parquet(src, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt",
                                      "STAND_mvt"])
    d = t[(t.PHASE_mvt == "DEP") & (t.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == day)].copy()
    d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
    d["mvt"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
    return d[["MVT_ID_mvt", "ADEP_mvt", "cs", "mvt", "STAND_mvt"]]


def observed_taxi(t: np.ndarray, la: np.ndarray, lo: np.ndarray, gs: np.ndarray, k0: int,
                  t_end: float) -> tuple[float, float, float]:
    """v8 (log section 63): over the departure segment from its first moving point (after the run's last
    long gap) to `t_end` (the take-off), the path length flown on the ground in metres, the ground speed at
    that first moving point, and the number of stops (moving -> at most MOVE_KT). On rows first seen
    mid-taxi these give the speed of the observed part, from which the unobserved part can be scaled."""
    moving = np.flatnonzero(gs[k0:] > MOVE_KT)
    if not len(moving):
        return np.nan, np.nan, np.nan
    j0 = k0 + int(moving[0])
    sel = np.arange(j0, len(t))[t[j0:] <= t_end]
    if len(sel) < 2:
        return 0.0, float(gs[j0]), 0.0
    lat0 = float(np.radians(la[j0]))
    seg = np.hypot(np.diff(lo[sel]) * 111_000.0 * np.cos(lat0), np.diff(la[sel]) * 111_000.0)
    still = gs[sel] <= MOVE_KT
    return float(seg.sum()), float(gs[j0]), float(np.sum(~still[:-1] & still[1:]))


def reception(t: np.ndarray, gs: np.ndarray, k0: int, t_end: float) -> tuple[float, float, float]:
    """v9 (log section 63): reception quality of the departure segment from its first moving point to `t_end`:
    the number of points, the largest gap between consecutive points, and the gap after the first moving point.
    A sparsely received run is first seen later after the off-block than a densely received one."""
    moving = np.flatnonzero(gs[k0:] > MOVE_KT)
    if not len(moving):
        return np.nan, np.nan, np.nan
    j0 = k0 + int(moving[0])
    tt = t[j0:][t[j0:] <= t_end]
    if len(tt) < 2:
        return float(len(tt)), np.nan, np.nan
    d = np.diff(tt)
    return float(len(tt)), float(d.max()), float(d[0])


def first_position(la: np.ndarray, lo: np.ndarray, gs: np.ndarray, k0: int) -> tuple[float, float]:
    """v10: latitude and longitude of the departure segment's first moving point (reception zones)."""
    moving = np.flatnonzero(gs[k0:] > MOVE_KT)
    if not len(moving):
        return np.nan, np.nan
    j0 = k0 + int(moving[0])
    return float(la[j0]), float(lo[j0])


def stand_exits(path: str, m: pd.DataFrame, table: pd.DataFrame, positions: dict | None = None,
                v7: bool = False, v8: bool = False, v9: bool = False, v10: bool = False) -> tuple[np.ndarray, ...]:
    """Per joined row: MVT - last point within stand_exit.RADIUS_M of the NM stand, the minimum
    distance to that stand after the run's last long gap, and the distance from the stand to the
    run's first moving point after that gap (NaN where the stand is not in apt.dat).
    With `positions`, a row whose NM stand is not located falls back to the apt.dat stand nearest
    the first point after the last long gap (section 51). The first-move distance lets a model
    extrapolate the off-block back from a first sighting mid-taxi: on non-scored days the delay
    of that sighting grows with it (corr +0.57, ~4.6 m/s; section 54).
    With `v7`, also: the same distance along the X-Plane taxiway network (`taxi_network`), and MVT -
    the first moving point after the run's last long gap, whatever the stand (section 60: on runs
    without a landing the first taxi-out point is often an earlier stop or tow, ~6,000s early)."""
    k = m[["ADEP_mvt"]].assign(key=m.STAND_mvt.map(stand_exit.stand_key)).merge(
        table, left_on=["ADEP_mvt", "key"], right_on=["airport", "key"], how="left")
    G = pd.read_parquet(path, columns=["icao", "run", "t", "lat", "lon", "gs", "took_off"])  # took_off includes to_late in v2
    G = G[G.took_off].sort_values(["icao", "run", "t"])
    pts = {key: (g.t.to_numpy(), g.lat.to_numpy(dtype="float64"), g.lon.to_numpy(dtype="float64"),
                 np.nan_to_num(g.gs.to_numpy(dtype="float64")))
           for key, g in G.groupby(["icao", "run"], observed=True)}
    taxi = np.full(len(m), np.nan); dmin = np.full(len(m), np.nan); first_dist = np.full(len(m), np.nan)
    net_dist = np.full(len(m), np.nan); first_gap = np.full(len(m), np.nan)
    obs = np.full((len(m), 3), np.nan); rec = np.full((len(m), 3), np.nan); pos = np.full((len(m), 2), np.nan)
    for i, (icao, run, mvt, apt, s_lat, s_lon, t_to) in enumerate(zip(m.icao, m.run, m.mvt, m.ADEP_mvt, k.lat, k.lon,
                                                                     m.t_takeoff)):
        if (icao, run) not in pts:
            continue
        t, la, lo, gs = pts[(icao, run)]
        k0 = stand_exit.departure_start(t)
        if v8:
            obs[i] = observed_taxi(t, la, lo, gs, k0, float(t_to))
        if v9:
            rec[i] = reception(t, gs, k0, float(t_to))
        if v10:
            pos[i] = first_position(la, lo, gs, k0)
        moving0 = np.flatnonzero(gs[k0:] > MOVE_KT)
        if v7 and len(moving0):
            first_gap[i] = mvt - t[k0 + int(moving0[0])]
        if not np.isfinite(s_lat) and positions is not None and apt in positions:
            s_lat, s_lon = stand_exit.nearest_stand(la[k0], lo[k0], positions[apt])
        if np.isfinite(s_lat):
            te, dm = stand_exit.exit_time(t, la, lo, s_lat, s_lon)
            taxi[i] = mvt - te; dmin[i] = dm
            moving = np.flatnonzero(gs[k0:] > MOVE_KT)
            if len(moving):
                j = k0 + int(moving[0])
                first_dist[i] = float(np.hypot((lo[j] - s_lon) * 111_000.0 * np.cos(np.radians(s_lat)),
                                               (la[j] - s_lat) * 111_000.0))
                if v7:
                    net_dist[i] = taxi_network.network(apt).distance(s_lat, s_lon, float(la[j]), float(lo[j]))
    if v9 or v10:
        out = (taxi, dmin, first_dist, net_dist, first_gap, obs[:, 0], obs[:, 1], obs[:, 2], rec[:, 0], rec[:, 1], rec[:, 2])
        return (*out, pos[:, 0], pos[:, 1]) if v10 else out
    if v8:
        return taxi, dmin, first_dist, net_dist, first_gap, obs[:, 0], obs[:, 1], obs[:, 2]
    if v7:
        return taxi, dmin, first_dist, net_dist, first_gap
    return taxi, dmin, first_dist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("year", choices=["2025", "2026"])
    ap.add_argument("--allow-ranking", action="store_true")
    ap.add_argument("--tag", default="", help="output suffix, so a new version never overwrites a served file")
    ap.add_argument("--stands", action="store_true", help="add stand-exit features as in v33 (section 50)")
    ap.add_argument("--ground-v2", action="store_true", help="read ground_v2.parquet (section 55)")
    ap.add_argument("--days", nargs="*", default=None, help="only these days (diagnostics)")
    ap.add_argument("--v7", action="store_true", help="add own_first_net_dist and own_first_taxi_gap (section 60)")
    ap.add_argument("--v8", action="store_true", help="add own_obs_path, own_first_gs, own_stops (section 63)")
    ap.add_argument("--v9", action="store_true", help="add own_npts, own_max_dt, own_first_dt (section 63)")
    ap.add_argument("--v10", action="store_true", help="add own_first_lat, own_first_lon, own_since_land (section 63)")
    ap.add_argument("--stands-v2", action="store_true",
                    help="stand exits with sub-position names, 60m ambiguity and the position fallback (section 51)")
    args = ap.parse_args()
    if args.stands and args.stands_v2:
        raise SystemExit("--stands and --stands-v2 are alternatives")
    if args.v7 and not args.stands_v2:
        raise SystemExit("--v7 extends the --stands-v2 features")
    if args.v8 and not args.v7:
        raise SystemExit("--v8 extends the --v7 features")
    if args.v9 and not args.v8:
        raise SystemExit("--v9 extends the --v8 features")
    if args.v10 and not (args.v9 and args.ground_v2):
        raise SystemExit("--v10 extends the --v9 features and needs --ground-v2")
    # v33 was built with 30m ambiguity, no sub-position keys and no position fallback
    table = (stand_exit.stand_table(spread_m=30.0, variants=False) if args.stands
             else stand_exit.stand_table() if args.stands_v2 else None)
    positions = stand_exit.stand_positions() if args.stands_v2 else None
    if args.year == "2026" and not args.allow_ranking:
        raise SystemExit("2026 is the ranking set: own-flight features wait on the organisers' ruling (section 37)")
    out = []
    name = "ground_v2.parquet" if args.ground_v2 else "ground.parquet"
    for f in sorted(glob.glob(f"data/external/adsb/{args.year}-*/{name}")):
        day = f.split("/")[-2]
        if args.days and day not in args.days:
            continue
        d = departures(day)
        m = d.merge(takeoff_runs(f, args.ground_v2), left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"])
        m["dto"] = (m.t_join - m.mvt).abs()
        m = m[m.dto < JOIN_S].sort_values("dto").drop_duplicates("MVT_ID_mvt").reset_index(drop=True)
        extra = {}
        if table is not None:
            cols = stand_exits(f, m, table, positions, v7=args.v7, v8=args.v8, v9=args.v9, v10=args.v10)
            extra["own_stand_exit_taxi"], extra["own_stand_dmin"], extra["own_first_dist"] = cols[:3]
            if args.v7:
                extra["own_first_net_dist"], extra["own_first_taxi_gap"] = cols[3:5]
            if args.v8:
                extra["own_obs_path"], extra["own_first_gs"], extra["own_stops"] = cols[5:8]
            if args.v9:
                extra["own_npts"], extra["own_max_dt"], extra["own_first_dt"] = cols[8:11]
            if args.v10:
                extra["own_first_lat"], extra["own_first_lon"] = cols[11:13]
                extra["own_since_land"] = (m.mvt - m.t_land_run).to_numpy()
        out.append(pd.DataFrame(extra | {"MVT_ID_mvt": m.MVT_ID_mvt.to_numpy(), "own_joined": 1.0,
                                 "own_to_minus_mvt": (m.t_takeoff - m.mvt).to_numpy(),
                                 "own_dwell_taxi": (m.mvt - m.t_dwell_end).to_numpy(),
                                 "own_dwell_len": (m.t_dwell_end - m.t_dwell_start).to_numpy(),
                                 "own_first_taxi": (m.mvt - m.t_first_out).to_numpy(),
                                 "own_gap_seg": m.gap_seg.to_numpy(dtype="float64"),
                                 **({"own_to_late": m.to_late.to_numpy()} if args.ground_v2 else {}),
                                 # analysis columns (queue model), not expert inputs
                                 "own_hold": m.hold.to_numpy(dtype="float64"),
                                 "own_adsb_taxi": (m.t_takeoff - m.t_dwell_end).to_numpy()}))
    F = pd.concat(out, ignore_index=True)
    dst = f"data/interim/adsb_own_{args.year}{args.tag}.parquet"
    F.to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst)
    print(f"{len(F):,} joined departures over {len(out)} days; dwell seen {int(F.own_dwell_taxi.notna().sum()):,};"
          f" first-taxi point {int(F.own_first_taxi.notna().sum()):,} -> {dst}")


if __name__ == "__main__":
    main()
