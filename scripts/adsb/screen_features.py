"""Screen ADS-B trace features against the v21 OOF residual on ingested 2025 days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config, matrix


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    F = pd.read_parquet("data/external/adsb/features.parquet")
    d = raw[[config.ID_COL, "ADEP_mvt"]].merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left").merge(F, on=config.ID_COL, how="left")
    d["y"] = raw[config.TARGET].to_numpy(dtype="float64"); d["matched"] = X["aobt_missing"].to_numpy() == 0; d["aobt_taxi"] = X["taxi_proxy_AOBT_3_flt"].to_numpy(dtype="float64")
    j = d[d.adsb_joined == 1].copy(); print("2025 joined departures:", len(j), "seen parked:", int(j.adsb_seen_parked.sum()))
    print("\nMATCHED rows seen parked: v21 error vs (trace taxi - NM taxi)")
    for ap, g in j[(j.adsb_seen_parked == 1) & j.matched].groupby("ADEP_mvt"):
        if len(g) < 50:
            continue
        e = (g.y - g.prediction).to_numpy(); s = (g.adsb_taxi_motion - g.aobt_taxi).to_numpy(); ok = np.isfinite(s)
        r = np.corrcoef(e[ok], s[ok])[0, 1]; b = np.polyfit(s[ok], e[ok], 1); corr = g.prediction.to_numpy()[ok] + np.polyval(b, s[ok])
        print(f"  {ap} n={ok.sum():5d} v21 {rmse(g.y.to_numpy()[ok], g.prediction.to_numpy()[ok]):6.1f}  corr {r:+.3f}  in-sample linear fix {rmse(g.y.to_numpy()[ok], corr):6.1f}  slope {b[0]:+.2f}")
    print("\nUNMATCHED rows joined: v21 vs bias-corrected trace taxi")
    for ap, g in j[~j.matched].groupby("ADEP_mvt"):
        if len(g) < 15:
            continue
        gm = g[np.isfinite(g.adsb_taxi_motion)]
        est = gm.adsb_taxi_motion - np.median(gm.adsb_taxi_motion - gm.y)
        print(f"  {ap} n={len(gm):4d} v21 {rmse(gm.y, gm.prediction):7.1f}  trace-motion bias-corrected {rmse(gm.y, est):7.1f}  50/50 {rmse(gm.y, 0.5 * gm.prediction + 0.5 * est):7.1f}  parked share {gm.adsb_seen_parked.mean():.2f}")


if __name__ == "__main__":
    main()
