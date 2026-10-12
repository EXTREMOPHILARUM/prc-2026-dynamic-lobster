"""Fit the published-mean calibration offsets on 2025 and write the v39 wrapper (log section 62).

    uv run python scripts/fit_pubcal.py [--base mechanism_v38] [--out mechanism_v39]

Offsets: per airport, mean over 2025 months with valid share >= pubcal.VMIN of (published mean - true
mean of rows the v38-equivalent OOF predicts under pubcal.CAP). Also prints the 2025 leave-one-month-out
check: the squared-error change from shifting each eligible month with offsets fitted on the other months.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, data, features, pubcal

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="mechanism_v38")
ap.add_argument("--out", default="mechanism_v39")
ap.add_argument("--oof", default="data/interim/v38_equiv_oof.parquet")
args = ap.parse_args()

dep = features.departures(data.load_training())
oof = pd.read_parquet(config.ROOT / args.oof)
t = dep[[config.ID_COL, "ADEP_mvt", "MVT_TIME_UTC_mvt", config.TARGET]].merge(oof, on=config.ID_COL)
assert len(t) == len(oof), "OOF rows did not all join the training departures"
apt = t["ADEP_mvt"].astype(str).to_numpy()
y, pred = t[config.TARGET].to_numpy(dtype="float64"), t["prediction"].to_numpy(dtype="float64")
published = pubcal.load_published()
month = pd.to_datetime(t["MVT_TIME_UTC_mvt"], utc=True).dt.month.to_numpy()

before = after = 0.0
won = n = 0
for m in range(1, 13):
    tr, te = month != m, month == m
    off = pubcal.fit_offsets(apt[tr], t["MVT_TIME_UTC_mvt"][tr], y[tr], pred[tr], published)
    p, _ = pubcal.apply(pred[te], apt[te], t["MVT_TIME_UTC_mvt"][te], published, off)
    b, a = ((pred[te] - y[te]) ** 2).sum(), ((p - y[te]) ** 2).sum()
    before, after = before + b, after + a
    won += a < b; n += 1
    print(f"month {m:2d}: dSSE {(a - b) / 1e6:+8.1f}M")
N = len(y)
print(f"2025 LOMO: RMSE {np.sqrt(before / N):.3f} -> {np.sqrt(after / N):.3f}, months won {won}/{n}")

offsets = pubcal.fit_offsets(apt, t["MVT_TIME_UTC_mvt"], y, pred, published)
print("offsets:", {k: round(v, 1) for k, v in sorted(offsets.items())})
meta = json.loads((config.MODELS / f"{args.base}.json").read_text())
meta["pubcal_offsets"] = offsets
meta["pubcal_vmin"] = pubcal.VMIN
meta["external_data"]["eurocontrol_taxiout_monthly"] = {
    "source": "EUROCONTROL Aviation Intelligence Unit, Taxi-Out Additional Time (monthly per airport), ansperformance.eu",
    "url": "https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx",
    "file": "data/external/ansperformance/Taxi-Out_Additional_Time.xlsx",
    "licence": "open data per organiser ruling 2026-09-17 (EUROCONTROL performance files), attribution",
    "ruling": "organiser 2026-09-27: the published Jan/Jul 2026 monthly taxi-out may be used as model input "
              "(docs/rules-and-compliance.md)",
}
(config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
print(f"wrote models/{args.out}.json")
