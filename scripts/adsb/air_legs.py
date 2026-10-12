"""Airborne legs from raw adsb.lol traces: one row per contiguous airborne stretch per aircraft.

Used to measure how long a departing aircraft sat at the airport before take-off (log section 63,
"cause of the day slips"): ground time = this leg's start - the previous leg's end, when the
previous leg ended near the same airport. Airborne coverage is good where ground coverage is not
(LFPG), so this does not depend on ground points.

A leg breaks on any ground point or on a silence longer than GAP_S. Output
<out>/<day>.parquet with columns

    icao, reg, callsign (first set in the leg), t0, lat0, lon0, alt0, t1, lat1, lon1, alt1, n

alt is barometric feet (NaN if not reported). Restartable: existing day files are skipped.

    uv run python -u scripts/adsb/air_legs.py --roots ~/Data/adsb_raw2025 data/external/adsb \\
        --out data/interim/air_legs 2025-01-01 2025-12-31
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import pandas as pd

GAP_S = 1200.0
COLUMNS = ["icao", "reg", "callsign", "t0", "lat0", "lon0", "alt0", "t1", "lat1", "lon1", "alt1", "n"]


def _alt(v) -> float:
    return float(v) if isinstance(v, (int, float)) else float("nan")


def legs_from_trace(path: str) -> list[tuple]:
    try:
        with gzip.open(path) as fh:
            d = json.loads(fh.read())
    except (OSError, ValueError, zlib.error):
        return []
    t0 = float(d["timestamp"]); icao = d["icao"]; reg = d.get("r")
    rows: list[tuple] = []; leg: list = []; cs = None

    def flush():
        if leg:
            a, b = leg[0], leg[-1]
            rows.append((icao, reg, cs, t0 + a[0], a[1], a[2], _alt(a[3]), t0 + b[0], b[1], b[2], _alt(b[3]), len(leg)))

    for p in d["trace"]:
        if p[3] == "ground":
            flush(); leg = []; cs = None
            continue
        if leg and p[0] - leg[-1][0] > GAP_S:
            flush(); leg = []; cs = None
        leg.append(p)
        if cs is None and isinstance(p[8], dict) and p[8].get("flight"):
            cs = p[8]["flight"].strip() or None
    flush()
    return rows


def day_files(day: str, roots: list[str]) -> list[str]:
    for r in roots:
        files = glob.glob(f"{os.path.expanduser(r)}/{day}/traces/*/trace_full_*.json")
        if files:
            return files
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    with ProcessPoolExecutor(a.workers) as ex:
        while d <= end:
            day = d.isoformat(); d += timedelta(days=1)
            dst = f"{a.out}/{day}.parquet"
            if os.path.exists(dst):
                continue
            files = day_files(day, a.roots)
            if not files:
                print(f"{day}: no trace files", flush=True)
                continue
            rows = [r for chunk in ex.map(legs_from_trace, files, chunksize=500) for r in chunk]
            L = pd.DataFrame(rows, columns=COLUMNS)
            L.to_parquet(dst + ".tmp", index=False); os.replace(dst + ".tmp", dst)
            print(f"{day}: {len(files):,} traces, {len(L):,} legs", flush=True)


if __name__ == "__main__":
    main()
