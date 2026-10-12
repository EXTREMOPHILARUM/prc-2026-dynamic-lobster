"""Per-airport ADS-B pilot report from a day's segments.parquet.

With training labels (2025 days): join to that day's departures by callsign and compare the
trace off-block estimates with BLOCK_TIME and with NM's AOBT on the same rows. Without labels
(2026 days): coverage and parked-detection shares only.

    uv run python scripts/adsb/report_day.py 2025-07-15 [--no-labels]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("day"); ap.add_argument("--no-labels", action="store_true"); args = ap.parse_args()
    S = pd.read_parquet(f"data/external/adsb/{args.day}/segments.parquet")
    S["gap_stat_motion"] = S.t_first_motion - S.t_last_stationary
    month = args.day[:7]
    if args.no_labels:
        src = pd.read_parquet("data/raw/ranking.parquet")
    else:
        src = pd.read_parquet(glob.glob(f"data/raw/training_{month}-01_*.parquet")[0])
    d = src[(src.PHASE_mvt == "DEP") & (src.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == args.day)].copy()
    d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
    for c, col in (("mvt_s", "MVT_TIME_UTC_mvt"), ("block_s", "BLOCK_TIME_UTC_mvt"), ("aobt_s", "AOBT_3_flt")):
        d[c] = (pd.to_datetime(d[col], utc=True) - EPOCH).dt.total_seconds()
    m = d.merge(S, left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"], how="inner")
    m = m[(m.t_takeoff - m.mvt_s).abs() < 1200].sort_values("n_ground", ascending=False).drop_duplicates("MVT_ID_mvt")
    good = m.seen_parked & (m.gap_stat_motion < 900)
    print(f"{args.day}: departures {len(d):,}; joined {len(m):,} ({len(m)/max(len(d),1)*100:.1f}%); parked-and-tight {int(good.sum()):,} ({good.sum()/max(len(d),1)*100:.1f}%)")
    rows = []
    for apt, g in d.groupby("ADEP_mvt"):
        j = m[m.ADEP_mvt == apt]; jj = j[good.reindex(j.index).fillna(False).astype(bool)] if len(j) else j
        r = {"airport": apt, "dep": len(g), "joined_%": round(len(j) / len(g) * 100, 1), "parked_tight_%": round(len(jj) / len(g) * 100, 1)}
        if not args.no_labels and len(jj):
            for est in ("t_last_stationary", "t_first_motion"):
                e = (jj[est] - jj.block_s).to_numpy(); e = e[np.isfinite(e)]
                r[f"{est[2:]}_rmse"] = round(float(np.sqrt((e ** 2).mean()))) if len(e) else None; r[f"{est[2:]}_med"] = round(float(np.median(e))) if len(e) else None
            mid = ((jj.t_last_stationary + jj.t_first_motion) / 2 - jj.block_s).to_numpy(); mid = mid[np.isfinite(mid)]
            r["midpoint_rmse"] = round(float(np.sqrt((mid ** 2).mean()))) if len(mid) else None
            e = (jj.aobt_s - jj.block_s).to_numpy(); e = e[np.isfinite(e)]; r["aobt_rmse_same_rows"] = round(float(np.sqrt((e ** 2).mean()))) if len(e) else None
        rows.append(r)
    R = pd.DataFrame(rows); print(R.to_string()); R.to_csv(f"data/external/adsb/{args.day}/pilot_report.csv", index=False)
    if not args.no_labels:
        m.to_parquet(f"data/external/adsb/{args.day}/joined.parquet", index=False)


if __name__ == "__main__":
    main()
