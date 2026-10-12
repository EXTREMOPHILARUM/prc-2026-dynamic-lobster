"""Per-departure ADS-B trace features from ingested days' segments.parquet.

For every day under data/external/adsb/ with a segments table, join to that day's departures
(training files for 2025, ranking for 2026) by callsign + airport within +/- 20 min of
take-off, keeping the segment with most ground points. Features (seconds, relative to
take-off so they are usable as taxi estimates): trace taxi from first motion and from the
last parked point, the stationary-to-motion gap, parked point count, seen-parked flag, and
the parked position. Writes data/external/adsb/features.parquet keyed by MVT_ID_mvt.

    uv run python scripts/adsb/build_features.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
FEATURES = ["adsb_taxi_motion", "adsb_taxi_parked", "adsb_gap", "adsb_n_parked", "adsb_seen_parked", "adsb_joined", "adsb_lat_park", "adsb_lon_park"]


def departures_for(day: str) -> pd.DataFrame:
    if day.startswith("2026"):
        src = pd.read_parquet("data/raw/ranking.parquet")
    else:
        src = pd.read_parquet(glob.glob(f"data/raw/training_{day[:7]}-01_*.parquet")[0])
    d = src[(src.PHASE_mvt == "DEP") & (src.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == day)].copy()
    d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
    d["mvt_s"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
    return d[["MVT_ID_mvt", "ADEP_mvt", "cs", "mvt_s"]]


def main():
    out = []
    for seg_path in sorted(glob.glob("data/external/adsb/*/segments.parquet")):
        day = seg_path.split("/")[-2]
        S = pd.read_parquet(seg_path); d = departures_for(day)
        m = d.merge(S, left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"], how="inner")
        m = m[(m.t_takeoff - m.mvt_s).abs() < 1200].sort_values("n_ground", ascending=False).drop_duplicates("MVT_ID_mvt")
        f = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt.to_numpy(),
                          "adsb_taxi_motion": (m.mvt_s - m.t_first_motion).to_numpy(),
                          "adsb_taxi_parked": (m.mvt_s - m.t_last_stationary).to_numpy(),
                          "adsb_gap": (m.t_first_motion - m.t_last_stationary).to_numpy(),
                          "adsb_n_parked": m.n_parked.to_numpy(dtype="float64"),
                          "adsb_seen_parked": m.seen_parked.astype(float).to_numpy(),
                          "adsb_joined": np.ones(len(m)),
                          "adsb_lat_park": m.lat_park.to_numpy(), "adsb_lon_park": m.lon_park.to_numpy()})
        out.append(f); print(f"{day}: departures {len(d):,} joined {len(m):,} parked {int(m.seen_parked.sum()):,}", flush=True)
    F = pd.concat(out, ignore_index=True); F.to_parquet("data/external/adsb/features.parquet", index=False)
    print("wrote features for", len(F), "departures over", len(out), "days")


if __name__ == "__main__":
    main()
