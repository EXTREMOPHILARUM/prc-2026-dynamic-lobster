"""Gap-stratified calibration for the LIRF copy classifier.

The shipped mixture calibrates `P(copy)` with one isotonic fit on two held-out months, pooled over
every schedule gap. Wide-gap rows are rare but each misclassification costs `(gap - reg)^2`, and
pooling drags their probability toward the narrow-gap average: out of fold the 4-8h rows average
p = 0.68 against an actual copy rate of 0.81. This fits the isotonic separately per gap stratum,
on cross-fitted scores from *all* training months so the wide strata have enough rows to calibrate.

    uv run python -u scripts/experiment_copy_calibration.py

Ranking (`prc.lirfmoe`) is unchanged; only the calibration step differs, so the classifier and
regressor are the shipped ones.
"""
from __future__ import annotations

import json
import time
from itertools import pairwise

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

STRATA = {"two": [0.0, 4.0, 1e9], "three": [0.0, 2.0, 4.0, 1e9]}
INNER = 3


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_clf(F, labels, weight, rows, vcats):
    return lgb.train({**lirfmoe.CLF_PARAMS, "num_threads": 8},
                     lgb.Dataset(F.loc[rows], labels[rows].astype("int8"), weight=weight[rows], categorical_feature=vcats),
                     num_boost_round=lirfmoe.CLF_ROUNDS)


def main():
    start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    lirf = (X["ADEP_mvt"].astype("string") == lirfbands.AIRPORT).to_numpy()
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); gap_h = gap / 3600.0
    copy = np.abs(y - gap) <= lirfmoe.COPY_TOL
    target = lirfmoe.target_rows(X, 8.0)
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched_any = np.isfinite(anchor_raw)
    ids = raw[[config.ID_COL]]
    stored = ids.merge(pd.read_parquet(config.DATA_INTERIM / "lirf_unmatched_moe_oof.parquet"), on=config.ID_COL, how="left")
    reg = stored["regressor"].to_numpy(dtype="float64"); p_ship = stored["p_copy"].to_numpy(dtype="float64")
    ref = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v21_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    cols = lirfmoe.columns(X); vcats = [c for c in cats if c in cols]
    w_all = np.where(missing, lirfmoe.WEIGHT, 1.0)
    sc = np.isin(month, (1, 7))
    out = {}

    def evaluate(name, p):
        mix = p * np.clip(gap, 0, None) + (1 - p) * reg
        new = ref.copy(); new[target] = mix[target]
        new = np.clip(lirfbands.lobt_clip(lirfbands.rome_rules(new, raw, X), X), 0, None)
        t_sc = target & sc
        bym = [rmse(y[month == m], new[month == m]) for m in range(1, 13)]
        out[name] = {"target_rmse": rmse(y[t_sc], new[t_sc]), "target_sse_M": float(((y[t_sc] - new[t_sc]) ** 2).sum() / 1e6),
                     "jan_jul": rmse(y[sc], new[sc]), "jan": rmse(y[month == 1], new[month == 1]), "jul": rmse(y[month == 7], new[month == 7]),
                     "auc": float(roc_auc_score(copy[target], p[target])), "by_month": [round(v, 2) for v in bym]}
        r = out[name]
        print(f"{name:12s} target RMSE {r['target_rmse']:7.1f}  SSE {r['target_sse_M']:7.1f}M  AUC {r['auc']:.4f}  Jan+Jul {r['jan_jul']:.2f} (Jan {r['jan']:.2f} Jul {r['jul']:.2f})", flush=True)
        return new

    base = evaluate("shipped", p_ship)

    for sname, edges in STRATA.items():
        p_new = np.full(len(y), np.nan)
        for m in range(1, 13):
            tr = lirf & (month != m); va = target & (month == m)
            enc = tencode.fit(raw, y, np.where(matched_any, anchor_raw - y, np.nan), month != m)
            F = lirfmoe.frame(raw, X, cols, enc)
            tr_months = [q for q in range(1, 13) if q != m]
            # honest scores on every training month, so each stratum has enough rows to calibrate
            oof = np.full(len(y), np.nan)
            for i in range(INNER):
                hold = [q for j, q in enumerate(tr_months) if j % INNER == i]
                fit_rows = tr & ~np.isin(month, hold); score_rows = tr & np.isin(month, hold)
                clf_i = train_clf(F, copy, w_all, fit_rows, vcats)
                oof[score_rows] = clf_i.predict(F.loc[score_rows])
            clf = train_clf(F, copy, w_all, tr, vcats)
            s_va = clf.predict(F.loc[va])
            cal_rows = tr & np.isfinite(oof) & missing      # calibrate on the lane the mixture serves
            lo_hi = list(pairwise(edges))
            for lo, hi in lo_hi:
                cr = cal_rows & (gap_h >= lo) & (gap_h < hi)
                vr = (gap_h[va] >= lo) & (gap_h[va] < hi)
                if not vr.any():
                    continue
                if cr.sum() < 30:                            # too thin to calibrate: keep the pooled fit
                    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[cal_rows], copy[cal_rows].astype(float))
                else:
                    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[cr], copy[cr].astype(float))
                idx = np.where(va)[0][vr]
                p_new[idx] = iso.predict(s_va[vr])
            print(f"  {sname} fold {m:02d} t={time.time()-start:.0f}s", flush=True)
        p_new = np.where(np.isfinite(p_new), p_new, p_ship)
        new = evaluate(f"strat_{sname}", p_new)
        ys, bs, ns = y[sc], base[sc], new[sc]
        order = np.argsort((ys - ns) ** 2 - (ys - bs) ** 2); rob = []
        for k in (1, 10, 50):
            keep = np.ones(len(ys), bool); keep[order[:k]] = False
            rob.append(round(rmse(ys[keep], ns[keep]) - rmse(ys[keep], bs[keep]), 2))
        months = sum(out[f"strat_{sname}"]["by_month"][i] < out["shipped"]["by_month"][i] for i in range(12))
        out[f"strat_{sname}"]["robustness"] = rob; out[f"strat_{sname}"]["months_improved"] = months
        print(f"             months {months}/12  rob {rob}", flush=True)
        pd.DataFrame({config.ID_COL: raw[config.ID_COL], "p_copy": p_new}).to_parquet(config.DATA_INTERIM / f"copycal_{sname}_oof.parquet", index=False)

    (config.DATA_INTERIM / "copy_calibration_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
