"""Seasonal blend weights on stored fold outputs: LOBT alpha and gate weight fitted
separately on winter (Nov, Dec, Feb, Mar) and summer (May, Jun, Aug, Sep) non-scored months,
applied to January and July respectively, against the single-alpha v21 OOF.

    uv run python scripts/experiment_seasonal_blend.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import arrivals, config, evaluation, lirfbands, lirfregime, lobt, matrix, mechanism

WINTER = (11, 12, 2, 3); SUMMER = (5, 6, 8, 9)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy(); airport = X["ADEP_mvt"].astype("string").to_numpy()
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    v21 = col("v21_oof_predictions.parquet", "prediction"); ordinary = col("v21_oof_predictions.parquet", "ordinary"); gate = np.nan_to_num(col("v21_oof_predictions.parquet", "gate_new"))
    expert = col("lobt_expert_oof.parquet", "expert"); short = col("v16_oof_predictions.parquet", "short_expert"); p_reg = np.nan_to_num(col("lirf_regime_oof_predictions.parquet", "p_identity"))
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    def post(stage):
        out = lirfregime.apply(stage, p_reg, X)
        for m in range(1, 13):
            t = lirfbands.fit(X.loc[month != m], y[month != m]); va = month == m; out[va] = lirfbands.apply(out[va], X.loc[va], t)
        out = lirfbands.rome_rules(out, raw, X); s = lirfbands.short_scope(X); out[s] = short[s]
        return np.clip(lirfbands.lobt_clip(out, X), 0, None)
    blended = mechanism.blend(gate, proxy, ordinary)
    out = blended.copy()
    rep = {}
    for season, fit_months, target_month in (("winter", WINTER, 1), ("summer", SUMMER, 7)):
        rows = lobt.scope(X) & np.isin(month, fit_months)
        alphas = lobt.fit_alphas(y, blended, expert, airport, rows); rep[season] = alphas
        tm = month == target_month
        out[tm] = lobt.apply(blended[tm], expert[tm], airport[tm], alphas)
    other = ~np.isin(month, (1, 7)); glob = json.loads((config.DATA_INTERIM / "lobt_alphas_oof_metrics.json").read_text())["alphas_oof"]
    out[other] = lobt.apply(blended[other], expert[other], airport[other], glob)
    new = post(out); ref = post(lobt.apply(blended, expert, airport, glob))
    sc = np.isin(month, (1, 7))
    rep["scored_jan_jul"] = {"single_alpha": rmse(y[sc], ref[sc]), "seasonal": rmse(y[sc], new[sc])}
    rep["jan"] = {"single_alpha": rmse(y[month == 1], ref[month == 1]), "seasonal": rmse(y[month == 1], new[month == 1])}
    rep["jul"] = {"single_alpha": rmse(y[month == 7], ref[month == 7]), "seasonal": rmse(y[month == 7], new[month == 7])}
    rep["v21_check"] = rmse(y[sc], v21[sc])
    (config.DATA_INTERIM / "seasonal_blend_metrics.json").write_text(json.dumps(rep, indent=2))
    print({k: (v if not isinstance(v, dict) or "single_alpha" in v else {a: round(b, 2) for a, b in v.items()}) for k, v in rep.items()})


if __name__ == "__main__":
    main()
