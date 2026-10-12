"""Stage A of the surface-state extractor: every ADS-B ground point near the ten airports, with
its ground run segmented into taxi-in / stand dwell / taxi-out.

pilot_day.py keeps one summary row per ground run and takes "first motion" as the motion after
the *last* stationary point, which on a turnaround run is often the inbound taxi (section 39).
Here each run is segmented on its longest stop instead: the stand dwell is the longest stationary
stretch of at least DWELL_MIN_S (a silent gap between two points less than STILL_M apart counts
as stationary, since parked transponders often stop reporting). Points before the dwell on a run
that began with a landing are taxi-in; points after it on a run that ends in a take-off are
taxi-out. A run with no dwell is taxi-out if it ends airborne and did not begin airborne, taxi-in
in the mirror case, and unknown otherwise.

Output data/external/adsb/<day>/ground.parquet, one row per retained point (thinned to one per
THIN_S per aircraft run), plus run-level columns so stage B never re-reads the traces:

    icao, callsign, airport, run, t, lat, lon, gs, phase (0 unknown, 1 in, 2 dwell, 3 out),
    landed, took_off, t_land, t_takeoff, t_dwell_start, t_dwell_end

    uv run python -u scripts/adsb/ground_points.py 2025-07-15 [2025-07-16 ...]

--v2 (section 55) writes ground_v2.parquet with two fixes found by the raw-archive audit of
2026-07-16, where 823 departures (13%) had ground points for the right aircraft but no join:

  * the run's callsign is the latest one set *on the ground*, which on a turnaround is often the
    inbound leg's (EDDF: the new callsign appears only once airborne). `air_cs` is the first
    callsign set within AIR_CS_S after the run, and the join prefers it.
  * ground reception lost mid-taxi leaves a gap longer than AIR_GAP_S before the first airborne
    point, so the run never counted as a take-off (LEMD, LFPG). A run followed within LATE_GAP_S
    by an airborne point within LATE_KM of the same airport now counts, flagged `to_late`;
    `t_air` is that first airborne point, and t_takeoff stays the last ground point.

--traces-root reads the traces from another directory (<root>/<day>/traces).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from math import cos, radians

import numpy as np
import pandas as pd

AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4775, -0.4614), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
            "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
RADIUS_KM = 8.0
MOVE_KT = 3.0          # same motion threshold as pilot_day.py
STILL_M = 60.0         # a silent gap with less displacement than this is a stop, not a coverage hole
DWELL_MIN_S = 300.0    # shorter stops are holds or queueing, not a stand dwell
AIR_GAP_S = 300.0      # an airborne point within this of the run's edge marks a landing / take-off
AIR_CS_S = 900.0       # v2: the departing callsign is the first one set this soon after the run
LATE_GAP_S = 1800.0    # v2: reception lost mid-taxi; the first airborne point may come this late
LATE_KM = 15.0         #     but must be near the airport the run was at
THIN_S = 10.0
PHASE_UNKNOWN, PHASE_IN, PHASE_DWELL, PHASE_OUT = 0, 1, 2, 3
COLUMNS = ["icao", "callsign", "airport", "run", "t", "lat", "lon", "gs", "phase", "landed", "took_off",
           "t_land", "t_takeoff", "t_dwell_start", "t_dwell_end"]
COLUMNS_V2 = [*COLUMNS, "air_cs", "t_air", "to_late"]


def nearest_airport(lat: float, lon: float) -> str | None:
    best, best_d2 = None, RADIUS_KM * RADIUS_KM
    for ap, (alat, alon) in AIRPORTS.items():
        dx = (lon - alon) * 111.0 * cos(radians(alat)); dy = (lat - alat) * 111.0
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = ap, d2
    return best


def longest_stop(ts: np.ndarray, lat: np.ndarray, lon: np.ndarray, moving: np.ndarray) -> tuple[int, int] | None:
    """Index span [i, j] of the longest stationary stretch lasting at least DWELL_MIN_S."""
    n = len(ts)
    if n < 2:
        return None
    coslat = cos(radians(float(lat[0])))
    x = lon * 111_000.0 * coslat; y = lat * 111_000.0
    step_m = np.hypot(np.diff(x), np.diff(y))
    # an interval (k, k+1) is stationary if the aircraft did not move across it and either both
    # ends report no motion or it is a silent gap (parked transponders often stop reporting)
    still = (step_m < STILL_M) & ((~moving[:-1] & ~moving[1:]) | (np.diff(ts) > 30.0))
    best, best_len, k = None, DWELL_MIN_S, 0
    while k < n - 1:
        if not still[k]:
            k += 1; continue
        j = k
        while j < n - 1 and still[j]:
            j += 1
        dur = ts[j] - ts[k]
        # a slow queue creeps in steps below STILL_M; a stand dwell also stays in one place overall
        extent = float(np.hypot(x[k:j + 1] - x[k], y[k:j + 1] - y[k]).max())
        if dur >= best_len and extent < 2 * STILL_M:
            best, best_len = (k, j), dur
        k = j
    return best


def km_from(ap: str, lat: float, lon: float) -> float:
    alat, alon = AIRPORTS[ap]
    return (((lon - alon) * 111.0 * cos(radians(alat))) ** 2 + ((lat - alat) * 111.0) ** 2) ** 0.5


def runs_from_trace_v2(path: str) -> list[tuple]:
    return runs_from_trace(path, v2=True)


def runs_from_trace(path: str, v2: bool = False) -> list[tuple]:
    try:
        with gzip.open(path) as fh:
            d = json.loads(fh.read())
    except (OSError, ValueError, KeyError, zlib.error):
        # zlib.error is not an OSError; a corrupt upstream member must skip the file, not the day
        return []
    t0 = float(d["timestamp"]); tr = d["trace"]; icao = d["icao"]
    rows: list[tuple] = []
    callsign = None; run: list[list] = []; run_cs = None; prev_air_t = None; run_id = 0

    def flush(next_air_t, air_idx=None):
        nonlocal run_id
        if len(run) < 3:
            return
        lat = np.array([p[1] for p in run], dtype="float64"); lon = np.array([p[2] for p in run], dtype="float64")
        ap = nearest_airport(float(np.median(lat)), float(np.median(lon)))
        if ap is None:
            return
        ts = np.array([p[0] for p in run], dtype="float64")
        gs = np.array([p[4] if p[4] is not None else np.nan for p in run], dtype="float64")
        moving = np.nan_to_num(gs, nan=0.0) > MOVE_KT
        landed = prev_air_t is not None and ts[0] - prev_air_t < AIR_GAP_S
        took_off = next_air_t is not None and next_air_t - ts[-1] < AIR_GAP_S
        air_cs, t_air, to_late = None, np.nan, False
        if v2 and air_idx is not None:
            pa = tr[air_idx]; t_air = t0 + pa[0]
            for q in tr[air_idx:]:
                if q[0] - pa[0] > AIR_CS_S:
                    break
                if isinstance(q[8], dict) and q[8].get("flight"):
                    air_cs = q[8]["flight"].strip(); break
            if not took_off and pa[0] - ts[-1] < LATE_GAP_S and km_from(ap, pa[1], pa[2]) < LATE_KM:
                took_off = to_late = True
        stop = longest_stop(ts, lat, lon, moving)
        phase = np.full(len(run), PHASE_UNKNOWN, dtype="int8")
        if stop is not None:
            i, j = stop
            phase[i:j + 1] = PHASE_DWELL
            if landed:
                phase[:i] = PHASE_IN
            if took_off:
                phase[j + 1:] = PHASE_OUT
            t_ds, t_de = t0 + ts[i], t0 + ts[j]
        else:
            if took_off and not landed:
                phase[:] = PHASE_OUT
            elif landed and not took_off:
                phase[:] = PHASE_IN
            t_ds = t_de = np.nan
        keep = np.zeros(len(run), dtype=bool); last = -1e18
        for k in range(len(run)):
            if ts[k] - last >= THIN_S or (k > 0 and phase[k] != phase[k - 1]):
                keep[k] = True; last = ts[k]
        keep[-1] = True
        t_land = t0 + ts[0] if landed else np.nan
        t_to = t0 + ts[-1] if took_off else np.nan
        run_id += 1
        for k in np.flatnonzero(keep):
            row = (icao, run_cs, ap, run_id, t0 + ts[k], lat[k], lon[k], gs[k], phase[k], landed, took_off,
                   t_land, t_to, t_ds, t_de)
            rows.append((*row, air_cs, t_air, to_late) if v2 else row)

    for idx, p in enumerate(tr):
        if isinstance(p[8], dict) and p[8].get("flight"):
            callsign = p[8]["flight"].strip()
        if p[3] == "ground":
            if not run:
                run_cs = callsign
            run.append(p)
            if callsign:
                run_cs = callsign          # the departing flight's callsign is the latest one set
        else:
            if run:
                flush(p[0], idx); run = []
            prev_air_t = p[0]
    if run:
        flush(None)                        # still on the ground when the trace ends: no take-off
    return rows


def extract_day(day: str, workers: int, v2: bool = False, traces_root: str = "data/external/adsb") -> int:
    files = glob.glob(f"{traces_root}/{day}/traces/*/trace_full_*.json")
    if not files:
        print(f"{day}: no trace files", flush=True)
        return 0
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(runs_from_trace_v2 if v2 else runs_from_trace, files, chunksize=200) for r in chunk]
    G = pd.DataFrame(rows, columns=COLUMNS_V2 if v2 else COLUMNS)
    G["run"] = G["run"].astype("int32"); G["phase"] = G["phase"].astype("int8")
    for c in ("lat", "lon", "gs"):
        G[c] = G[c].astype("float32")
    for c in ("icao", "callsign", "airport", *(("air_cs",) if v2 else ())):
        G[c] = G[c].astype("category")
    out = f"data/external/adsb/{day}/ground{'_v2' if v2 else ''}.parquet"
    G.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
    runs = G.drop_duplicates(["icao", "run"])
    print(f"{day}: {len(files):,} files, {len(G):,} points, {len(runs):,} runs; take-offs {int(runs.took_off.sum()):,}, "
          f"take-offs with a stand dwell {int((runs.took_off & runs.t_dwell_end.notna()).sum()):,}", flush=True)
    return len(G)


def main():
    workers = int(os.environ.get("ADSB_WORKERS", "14"))
    args = sys.argv[1:]; v2 = "--v2" in args; root = "data/external/adsb"
    if "--traces-root" in args:
        i = args.index("--traces-root"); root = args[i + 1]; del args[i:i + 2]
    for day in (a for a in args if a != "--v2"):
        extract_day(day, workers, v2, root)


if __name__ == "__main__":
    main()
