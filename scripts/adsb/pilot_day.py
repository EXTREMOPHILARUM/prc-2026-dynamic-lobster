"""ADS-B pilot for one day: how many of our departures does ADSB.lol see at the stand, and
how well do trace-derived off-block estimates match the airport's BLOCK_TIME?

Reads data/external/adsb/<day>/traces/**/trace_full_*.json (gzip; readsb trace points
[dt, lat, lon, alt_baro|'ground', gs, track, flags, vrate, aircraft|None, source, ...]),
extracts departure ground segments within 8 km of the ten airports, joins to the training
departures of that day by callsign (+/- 20 min of take-off) and reports per airport.

    uv run python -u scripts/adsb/pilot_day.py 2025-07-15
"""
from __future__ import annotations

import glob
import gzip
import json
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
MOVE_KT = 3.0


def near(lat, lon):
    for ap, (alat, alon) in AIRPORTS.items():
        dx = (lon - alon) * 111.0 * cos(radians(alat)); dy = (lat - alat) * 111.0
        if dx * dx + dy * dy < RADIUS_KM * RADIUS_KM:
            return ap
    return None


def segments(path: str) -> list[dict]:
    try:
        with gzip.open(path) as fh:
            d = json.loads(fh.read())
    except (OSError, ValueError, KeyError, zlib.error):
        # zlib.error is NOT an OSError: a single corrupt gzip member inside an upstream archive
        # otherwise propagates out of the process pool and discards the whole day (2025-10-15).
        return []
    t0 = d["timestamp"]; tr = d["trace"]; out = []
    callsign = None; ground_run = []
    def flush(run, takeoff_t):
        if len(run) < 3:
            return
        ap = near(run[0][1], run[0][2])
        if ap is None:
            return
        ts = np.array([p[0] for p in run]); gs = np.array([p[4] if p[4] is not None else np.nan for p in run])
        moving = np.nan_to_num(gs, nan=0.0) > MOVE_KT
        # the departure taxi is the trailing block of moving points; the parked stretch is the
        # stationary block immediately before it (the same ground run also holds the arrival taxi-in)
        idx = np.arange(len(run)); last_stat_idx = idx[~moving]
        if len(last_stat_idx) == 0:
            first_motion_i = 0; last_stationary_i = None          # joined mid-taxi: never seen parked
        else:
            last_stationary_i = int(last_stat_idx.max())
            first_motion_i = last_stationary_i + 1 if last_stationary_i + 1 < len(run) else None
        park_start_i = None
        if last_stationary_i is not None:
            k = last_stationary_i
            while k > 0 and not moving[k - 1] and (ts[k] - ts[k - 1]) < 1800:
                k -= 1
            park_start_i = k
        out.append({"icao": d["icao"], "callsign": callsign, "airport": ap, "t_first_ground": t0 + ts[0],
                    "t_park_start": t0 + ts[park_start_i] if park_start_i is not None else np.nan,
                    "t_last_stationary": t0 + ts[last_stationary_i] if last_stationary_i is not None else np.nan,
                    "t_first_motion": t0 + ts[first_motion_i] if first_motion_i is not None else np.nan,
                    "t_takeoff": t0 + takeoff_t, "n_ground": len(run), "n_parked": int(last_stationary_i - park_start_i + 1) if park_start_i is not None else 0,
                    "seen_parked": bool(park_start_i is not None and last_stationary_i - park_start_i >= 2),
                    "lat_park": run[last_stationary_i][1] if last_stationary_i is not None else np.nan, "lon_park": run[last_stationary_i][2] if last_stationary_i is not None else np.nan})
    for p in tr:
        if isinstance(p[8], dict) and p[8].get("flight"):
            callsign = p[8]["flight"].strip()
        if p[3] == "ground":
            ground_run.append(p)
        else:
            if ground_run:
                flush(ground_run, p[0]); ground_run = []
    return out


def main():
    day = sys.argv[1]; files = glob.glob(f"data/external/adsb/{day}/traces/*/trace_full_*.json")
    print(f"{len(files):,} trace files", flush=True)
    import os
    with ProcessPoolExecutor(int(os.environ.get("ADSB_WORKERS", "14"))) as ex:
        segs = [s for chunk in ex.map(segments, files, chunksize=200) for s in chunk]
    S = pd.DataFrame(segs); print(f"{len(S):,} departure ground segments near the ten airports; by airport:", S.airport.value_counts().to_dict(), flush=True)
    S.to_parquet(f"data/external/adsb/{day}/segments.parquet", index=False)
    if len(sys.argv) > 2 and sys.argv[2] == "--segments-only":
        return
    month = day[:7]; tf = glob.glob(f"data/raw/training_{month}-01_*.parquet")[0]
    t = pd.read_parquet(tf); d = t[(t.PHASE_mvt == "DEP") & (t.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == day)].copy()
    d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
    d["mvt_s"] = (d.MVT_TIME_UTC_mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d["block_s"] = (d.BLOCK_TIME_UTC_mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d["aobt_s"] = (d.AOBT_3_flt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    m = d.merge(S, left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"], how="left")
    m = m[(m.t_takeoff - m.mvt_s).abs() < 1200].sort_values("n_ground", ascending=False).drop_duplicates("MVT_ID_mvt")
    print(f"\ndepartures that day: {len(d):,}; joined to a ground segment: {len(m):,} ({len(m)/len(d)*100:.1f}%)")
    rows = []
    for ap, g in d.groupby("ADEP_mvt"):
        j = m[m.ADEP_mvt == ap]; r = {"airport": ap, "departures": len(g), "joined_%": round(len(j) / len(g) * 100, 1), "seen_parked_%": round(j.seen_parked.mean() * 100, 1) if len(j) else 0}
        for est in ("t_last_stationary", "t_first_motion"):
            jj = j[j.seen_parked]; e = (jj[est] - jj.block_s); e = e[np.isfinite(e)]
            r[f"{est}_rmse"] = round(float(np.sqrt((e ** 2).mean())), 0) if len(e) else np.nan; r[f"{est}_med"] = round(float(e.median()), 0) if len(e) else np.nan
        jj = j[j.seen_parked]; e = (jj.aobt_s - jj.block_s); e = e[np.isfinite(e)]; r["aobt_rmse_same_rows"] = round(float(np.sqrt((e ** 2).mean())), 0) if len(e) else np.nan
        e2 = (jj.t_first_motion - jj.block_s); e2 = e2[np.isfinite(e2)]; r["motion_p90_abs"] = round(float(np.quantile(np.abs(e2), 0.9)), 0) if len(e2) else np.nan
        rows.append(r)
    R = pd.DataFrame(rows); print(R.to_string()); R.to_csv(f"data/external/adsb/{day}/pilot_report.csv", index=False)
    m.to_parquet(f"data/external/adsb/{day}/joined.parquet", index=False)


if __name__ == "__main__":
    main()
