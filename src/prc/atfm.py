"""EUROCONTROL daily per-airport ATFM context (open data per organiser ruling 2026-09-17).

Source: https://www.eurocontrol.int/performance/data/download/xls/{ATFM_Slot_Adherence,
Airport_Arrival_ATFM_Delay,Airport_Traffic}.xlsx, consolidated by
`scripts/fetch_eurocontrol_daily.py` into `data/external/eurocontrol/daily_atfm.parquet`.
Attribution: EUROCONTROL Aviation Intelligence Unit performance data. Joined on
(departure airport, UTC calendar day of take-off). Component screen: -2.1s on July's matched
base, ~0 elsewhere (section 16 of the experiment log).

The daily figures are finalised aggregates over the whole day, so a same-day join gives an
06:00 departure information from operations after it. `lag_days=1` joins the previous day
instead, which is available before any departure of the day; the organisers have been asked
which is acceptable (docs/discord-question-prediction-time.md). The default stays 0, which is
what v26-v30 were fitted with.
"""
from __future__ import annotations

import pandas as pd

from prc import config

PATH = config.ROOT / "data" / "external" / "eurocontrol" / "daily_atfm.parquet"
FEATURES = ["ec_reg_share", "ec_out_late_share", "ec_arr_atfm_min_per_flt", "ec_arr_atfm_weather_min", "ec_daily_movements"]


def attach(df: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    ec = pd.read_parquet(PATH, columns=["date", "airport"] + FEATURES)
    ec["date"] = pd.to_datetime(ec["date"])
    key = pd.DataFrame({
        "airport": df["ADEP_mvt"].astype("string").astype(str).to_numpy(),
        "date": (pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None).dt.normalize()
                 - pd.Timedelta(days=lag_days)).to_numpy(),
    })
    joined = key.merge(ec, on=["airport", "date"], how="left")
    out = df.copy()
    for c in FEATURES:
        out[c] = joined[c].to_numpy(dtype="float64")
    return out
