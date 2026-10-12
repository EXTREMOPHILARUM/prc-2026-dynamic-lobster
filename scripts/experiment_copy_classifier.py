"""Sharpen the LIRF copy classifier, which is what the no-record mixture's error is made of.

The mixture predicts `p * max(MVT - SCHED, 0) + (1 - p) * regressor`, so a row's squared error is
`(gap - reg)^2 * (copy - p)^2`: every misclassification costs the *square of the gap*, and the
rows that matter are the wide-gap ones. The shipped classifier is trained with unweighted binary
logloss over all LIRF rows, which optimises neither of those. Variants:

  base      the shipped recipe, reproduced (all LIRF rows, no-record weighted 8x)
  nr        trained on no-record rows only - matched LIRF rows are 20% "copy" because a flight
            that pushes back on time genuinely has y = MVT - SCHED, which is a different concept
  sse       weighted by (gap - reg)^2, i.e. the squared error actually at stake on each row
  nr_sse    both
  *_iso     isotonic calibration cross-fitted over all months instead of fixed months 4 and 10

The regressor is taken from the stored mixture OOF, so only the classifier is refitted.

    uv run python -u scripts/experiment_copy_classifier.py
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from prc import (
    arrivals,
    config,
    decomposition,
    evaluation,
    lirfbands,
    lirfmoe,
    matrix,
    mechanism,
    tencode,
)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    lirf = (X["ADEP_mvt"].astype("string") == lirfbands.AIRPORT).to_numpy()
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    copy = np.abs(y - gap) <= lirfmoe.COPY_TOL
    target = lirfmoe.target_rows(X, 8.0)
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched_any = np.isfinite(anchor_raw)
    ids = raw[[config.ID_COL]]
    stored = ids.merge(pd.read_parquet(config.DATA_INTERIM / "lirf_unmatched_moe_oof.parquet"), on=config.ID_COL, how="left")
    reg = stored["regressor"].to_numpy(dtype="float64"); p_ship = stored["p_copy"].to_numpy(dtype="float64")
    ref = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()

    cols = lirfmoe.columns(X); vcats = [c for c in cats if c in cols]
    sc = np.isin(month, (1, 7))
    # the regressor is only defined on target rows; fill elsewhere so the weight is computable
    reg_f = np.where(np.isfinite(reg), reg, np.clip(gap, 0, 3600))
    stake = (np.clip(gap, 0, None) - reg_f) ** 2
    out = {}

    def evaluate(name, p):
        mix = p * np.clip(gap, 0, None) + (1 - p) * reg
        new = ref.copy(); new[target] = mix[target]
        new = np.clip(lirfbands.lobt_clip(lirfbands.rome_rules(new, raw, X), X), 0, None)
        t_sc = target & sc
        rep = {"target_sse": float(((y[t_sc] - new[t_sc]) ** 2).sum()), "target_rmse": rmse(y[t_sc], new[t_sc]),
               "auc": float(roc_auc_score(copy[target], p[target])),
               "jan_jul_on_ref": rmse(y[sc], new[sc])}
        bym = [float(((y[target & (month == m)] - new[target & (month == m)]) ** 2).sum()) for m in range(1, 13)]
        rep["by_month_sse"] = [round(v / 1e6, 1) for v in bym]
        out[name] = rep
        print(f"{name:10s} target RMSE {rep['target_rmse']:8.1f}  scored SSE {rep['target_sse']/1e6:8.1f}M  AUC {rep['auc']:.4f}  Jan+Jul(on v21 ref) {rep['jan_jul_on_ref']:.2f}", flush=True)
        return new

    evaluate("shipped", p_ship)
    oracle = np.where(copy, 1.0, 0.0)
    evaluate("oracle", oracle)

    variants = {
        "base":    {"rows": "all", "weight": "flat", "iso": "fixed"},
        "nr":      {"rows": "nr",  "weight": "flat", "iso": "fixed"},
        "sse":     {"rows": "all", "weight": "sse",  "iso": "fixed"},
        "nr_sse":  {"rows": "nr",  "weight": "sse",  "iso": "fixed"},
        "sse_iso": {"rows": "all", "weight": "sse",  "iso": "cv"},
    }
    for name, cfg in variants.items():
        p_new = np.full(len(y), np.nan)
        for m in range(1, 13):
            pool = (lirf & missing) if cfg["rows"] == "nr" else lirf
            tr = pool & (month != m); va = target & (month == m)
            enc = tencode.fit(raw, y, np.where(matched_any, anchor_raw - y, np.nan), month != m)
            F = lirfmoe.frame(raw, X, cols, enc)
            w = np.where(missing, lirfmoe.WEIGHT, 1.0) if cfg["rows"] == "all" else np.ones(len(y))
            if cfg["weight"] == "sse":
                s = stake / max(np.median(stake[tr & (stake > 0)]), 1.0)
                w = w * np.clip(s, 0.05, 400.0)
            if cfg["iso"] == "fixed":
                fit_rows = tr & ~np.isin(month, lirfmoe.CAL_MONTHS); cal_rows = tr & np.isin(month, lirfmoe.CAL_MONTHS)
            else:   # hold out a rotating pair of months for calibration, never the fold's own month
                cal_m = [(m + 3 - 1) % 12 + 1, (m + 9 - 1) % 12 + 1]
                fit_rows = tr & ~np.isin(month, cal_m); cal_rows = tr & np.isin(month, cal_m)
            clf = lgb.train({**lirfmoe.CLF_PARAMS, "num_threads": 8},
                            lgb.Dataset(F.loc[fit_rows], copy[fit_rows].astype("int8"), weight=w[fit_rows], categorical_feature=vcats),
                            num_boost_round=lirfmoe.CLF_ROUNDS)
            iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict(F.loc[cal_rows]), copy[cal_rows].astype(float), sample_weight=w[cal_rows])
            p_new[va] = iso.predict(clf.predict(F.loc[va]))
        evaluate(name, np.where(np.isfinite(p_new), p_new, p_ship))
        print(f"           ({time.time()-start:.0f}s)", flush=True)
        pd.DataFrame({config.ID_COL: raw[config.ID_COL], "p_copy": p_new}).to_parquet(config.DATA_INTERIM / f"copyclf_{name}_oof.parquet", index=False)

    (config.DATA_INTERIM / "copy_classifier_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
