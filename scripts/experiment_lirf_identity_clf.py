"""Screen: flight identity in the Rome copy classifier (elegant-alligator's winter-only correction).

Refits only the copy classifier of the LIRF mixture, leave-one-month-out, with FLIGHT_mvt (and
any identifier the base excludes) added as a categorical. The regressor is taken from the stored
mixture OOF, so the run is minutes. Reports the all-season swap and a winter-only swap
(Nov-Feb: new classifier, other months: shipped classifier).

    uv run python -u scripts/experiment_lirf_identity_clf.py
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

WINTER = (11, 12, 1, 2)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    lirf = (X["ADEP_mvt"].astype("string") == lirfbands.AIRPORT).to_numpy(); missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); copy = np.abs(y - gap) <= lirfmoe.COPY_TOL
    target = lirfmoe.target_rows(X, 8.0)
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    ids = raw[[config.ID_COL]]
    stored = ids.merge(pd.read_parquet(config.DATA_INTERIM / "lirf_unmatched_moe_oof.parquet"), on=config.ID_COL, how="left")
    p_old = stored["p_copy"].to_numpy(dtype="float64"); reg = stored["regressor"].to_numpy(dtype="float64")
    ref = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    assert np.isfinite(reg[target]).all(), "stored regressor does not cover the 8h scope"

    cols = lirfmoe.columns(X); ident = [c for c in decomposition.BASE_EXCLUDED if c in X.columns and not c.endswith("_flt")]
    cols_id = cols + ident; vcats = [c for c in cats if c in cols_id]
    print(f"identity columns added: {ident}; categoricals {vcats}", flush=True)
    p_new = np.full(len(y), np.nan)
    for m in range(1, 13):
        tr = lirf & (month != m); va = target & (month == m)
        enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), month != m)
        F = lirfmoe.frame(raw, X, cols_id, enc); w = np.where(missing, lirfmoe.WEIGHT, 1.0)
        fit_rows = tr & ~np.isin(month, lirfmoe.CAL_MONTHS); cal_rows = tr & np.isin(month, lirfmoe.CAL_MONTHS)
        clf = lgb.train({**lirfmoe.CLF_PARAMS, "num_threads": 8}, lgb.Dataset(F.loc[fit_rows], copy[fit_rows].astype("int8"), weight=w[fit_rows], categorical_feature=vcats), num_boost_round=lirfmoe.CLF_ROUNDS)
        iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict(F.loc[cal_rows]), copy[cal_rows].astype(float))
        p_new[va] = iso.predict(clf.predict(F.loc[va]))
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)

    def chain(p):
        mix = p * np.clip(gap, 0, None) + (1 - p) * reg
        new = ref.copy(); new[target] = mix[target]
        return np.clip(lirfbands.lobt_clip(lirfbands.rome_rules(new, raw, X), X), 0, None)

    p_win = np.where(np.isin(month, WINTER), p_new, p_old)
    old, allseason, winter = chain(p_old), chain(p_new), chain(p_win)
    sc = np.isin(month, (1, 7)); rep = {}
    for name, new in (("all_season", allseason), ("winter_only", winter)):
        bym = [rmse(y[month == m], new[month == m]) - rmse(y[month == m], old[month == m]) for m in range(1, 13)]
        ys, os_, ns = y[sc], old[sc], new[sc]; order = np.argsort((ys - ns) ** 2 - (ys - os_) ** 2); rob = []
        for k in (1, 10, 50):
            keep = np.ones(len(ys), bool); keep[order[:k]] = False; rob.append(round(rmse(ys[keep], ns[keep]) - rmse(ys[keep], os_[keep]), 2))
        rep[name] = {"jan_jul": rmse(ys, ns), "ref_jan_jul": rmse(ys, os_), "jan": rmse(y[month == 1], new[month == 1]), "jul": rmse(y[month == 7], new[month == 7]),
                     "target_scored": rmse(y[target & sc], new[target & sc]), "ref_target_scored": rmse(y[target & sc], old[target & sc]),
                     "months_improved": int(sum(v < 0 for v in bym)), "by_month": [round(v, 2) for v in bym], "robustness": rob}
        r = rep[name]
        print(f"{name:12s} Jan+Jul {r['ref_jan_jul']:.3f} -> {r['jan_jul']:.3f} ({r['jan_jul'] - r['ref_jan_jul']:+.3f}) | Jan {r['jan']:.2f} Jul {r['jul']:.2f} | target {r['ref_target_scored']:.1f} -> {r['target_scored']:.1f} | months {r['months_improved']}/12 | rob {rob}", flush=True)
    ok = target & np.isfinite(p_new) & np.isfinite(p_old)
    rep["auc"] = {"old": float(roc_auc_score(copy[ok], p_old[ok])), "new": float(roc_auc_score(copy[ok], p_new[ok]))}
    okw = ok & np.isin(month, WINTER)
    rep["auc_winter"] = {"old": float(roc_auc_score(copy[okw], p_old[okw])), "new": float(roc_auc_score(copy[okw], p_new[okw]))}
    print("AUC", rep["auc"], "winter", rep["auc_winter"])
    (config.DATA_INTERIM / "lirf_identity_clf_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "p_copy_id": p_new}).to_parquet(config.DATA_INTERIM / "lirf_identity_clf_oof.parquet", index=False)


if __name__ == "__main__":
    main()
