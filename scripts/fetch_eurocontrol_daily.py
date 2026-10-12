"""Consolidate EUROCONTROL's open daily per-airport performance files into one parquet.

Source: https://www.eurocontrol.int/performance/data/download/xls/<name>.xlsx (Aviation
Intelligence Portal, https://ansperformance.eu/data/). Three files cover both ranking months
(Jan and Jul 2026); the pre-departure delay files stop at June 2026 and are not used.

    uv run python scripts/fetch_eurocontrol_daily.py [--download]
"""
from __future__ import annotations

import argparse
import hashlib
from urllib.request import urlretrieve

import pandas as pd

from prc import config

BASE = "https://www.eurocontrol.int/performance/data/download/xls/"
DIR = config.ROOT / "data" / "external" / "eurocontrol"
FILES = {
    "ATFM_Slot_Adherence": ["FLT_DEP_1", "FLT_DEP_REG_1", "FLT_DEP_OUT_EARLY_1", "FLT_DEP_IN_1", "FLT_DEP_OUT_LATE_1"],
    "Airport_Arrival_ATFM_Delay": ["FLT_ARR_1", "DLY_APT_ARR_1", "DLY_APT_ARR_A_1", "DLY_APT_ARR_C_1", "DLY_APT_ARR_D_1", "DLY_APT_ARR_W_1"],
    "Airport_Traffic": ["FLT_DEP_1", "FLT_ARR_1", "FLT_TOT_1"],
}
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
OUT = DIR / "daily_atfm.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--download", action="store_true"); args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    merged = None
    for name, cols in FILES.items():
        path = DIR / f"{name}.xlsx"
        if args.download or not path.exists():
            urlretrieve(BASE + f"{name}.xlsx", path)
        d = pd.read_excel(path, sheet_name="DATA")
        d = d[d["APT_ICAO"].isin(AIRPORTS) & (d["YEAR"] >= 2025)]
        d["date"] = pd.to_datetime(d["FLT_DATE"]).dt.date
        keep = [c for c in cols if c in d.columns]
        d = d[["date", "APT_ICAO"] + keep].rename(columns={c: f"{name.lower()}__{c.lower()}" for c in keep} | {"APT_ICAO": "airport"})
        print(f"{name}: {len(d):,} rows, {d['date'].min()} -> {d['date'].max()}, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
        merged = d if merged is None else merged.merge(d, on=["date", "airport"], how="outer")
    merged = merged.sort_values(["airport", "date"]).reset_index(drop=True)
    sa, ar, tr = "atfm_slot_adherence__", "airport_arrival_atfm_delay__", "airport_traffic__"
    merged["ec_reg_share"] = merged[f"{sa}flt_dep_reg_1"] / merged[f"{sa}flt_dep_1"].clip(lower=1)
    merged["ec_out_late_share"] = merged[f"{sa}flt_dep_out_late_1"] / merged[f"{sa}flt_dep_reg_1"].clip(lower=1)
    merged["ec_arr_atfm_min_per_flt"] = merged[f"{ar}dly_apt_arr_1"] / merged[f"{ar}flt_arr_1"].clip(lower=1)
    merged["ec_arr_atfm_weather_min"] = merged.get(f"{ar}dly_apt_arr_w_1", 0) / merged[f"{ar}flt_arr_1"].clip(lower=1)
    merged["ec_daily_movements"] = merged[f"{tr}flt_tot_1"]
    merged.to_parquet(OUT, index=False)
    print(f"wrote {OUT} {merged.shape}; Jan-2026 rows {(pd.to_datetime(merged['date']).dt.to_period('M')=='2026-01').sum()}, Jul-2026 rows {(pd.to_datetime(merged['date']).dt.to_period('M')=='2026-07').sum()}")


if __name__ == "__main__":
    main()
