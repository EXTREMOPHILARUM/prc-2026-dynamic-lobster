"""v18 candidate screens, paired against the v17 OOF.

  --what tailgate    faulty-AOBT rows (MVT-AOBT < 300 or > 2200, or MVT-EOBT > 3600, AOBT
                     present): a LightGBM correction on y - v17 fitted on those rows of the
                     other eleven months, blended at --alpha
  --what gatehold    matched LIRF rows: T = (MVT - SCHED) - G_hat with G learned on all LIRF
                     rows (NM-free features + D), blended at a per-fold alpha fitted on the
                     training months' in-sample residual (conservative)
  --what lineartree  base_matched refit with linear_tree=True and signed-log copies of the
                     clock deltas; component-level capped-residual RMSE on --months

    uv run python -u scripts/experiment_v18.py --what tailgate --threads 4
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, lirfbands, matrix, mechanism

NM_DEPENDENT_PREFIXES = ("aobt_minus_", "taxi_proxy_AOBT", "taxi_proxy_EOBT", "taxi_proxy_IOBT",
                         "taxi_proxy_LOBT", "arr_next_20m_aobt", "arr_during_taxi", "q_apt_at_aobt",
                         "q_rwy_at_aobt", "nbr_apt_own", "clk_AOBT", "clk_LOBT", "clk_IOBT", "clk_EOBT",
                         "clk_ARVT", "clk_mvt_minus_ARVT", "clk_lobt", "clk_arvt", "clk_sched_minus_aobt")
NM_DEPENDENT = {"offblock_delay", "nm_time_count", "aobt_missing", "nm_record_missing"}


def nm_free(X):
    return [c for c in X.columns if not c.endswith("_flt") and c not in NM_DEPENDENT
            and not c.startswith(NM_DEPENDENT_PREFIXES) and c not in decomposition.BASE_EXCLUDED]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def report(tag, y, base, new, month, extra=None):
    scored = np.isin(month, (1, 7)); rep = {"tag": tag}
    for name, mask in [("scored_jan_jul", scored), ("jan", month == 1), ("jul", month == 7), ("all_months", np.ones(len(y), bool))]:
        rep[name] = {"base": rmse(y[mask], base[mask]), "new": rmse(y[mask], new[mask])}; rep[name]["delta"] = rep[name]["new"] - rep[name]["base"]
    rep["by_month"] = {str(m): rmse(y[month == m], new[month == m]) - rmse(y[month == m], base[month == m]) for m in range(1, 13)}
    ys, bs, ns = y[scored], base[scored], new[scored]; order = np.argsort((ys - ns) ** 2 - (ys - bs) ** 2); rep["robustness_scored"] = {}
    for k in (1, 10, 50):
        keep = np.ones(len(ys), bool); keep[order[:k]] = False; rep["robustness_scored"][f"drop_{k}"] = rmse(ys[keep], ns[keep]) - rmse(ys[keep], bs[keep])
    if extra: rep.update(extra)
    (config.DATA_INTERIM / f"{tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    for k in ("scored_jan_jul", "jan", "jul", "all_months"):
        r = rep[k]; print(f"{k:15s} {r['base']:.3f} -> {r['new']:.3f} ({r['delta']:+.3f})")
    print("by month", {k: round(v, 2) for k, v in rep["by_month"].items()}); print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--what", required=True, choices=["tailgate", "gatehold", "lineartree"])
    ap.add_argument("--baseline", default="v17_oof_predictions.parquet"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7])
    args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.baseline), on=config.ID_COL, how="left")["prediction"].to_numpy().copy()
    aobt = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); eobt = X["taxi_proxy_EOBT_1_flt"].to_numpy(dtype="float64")
    matched = np.isfinite(aobt); lirf = (X["ADEP_mvt"].astype("string") == "LIRF").to_numpy()
    params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 5}

    if args.what == "tailgate":
        gate = matched & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
        cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
        Xg = X[cols].copy(); Xg["v17"] = base
        new = base.copy(); resid = y - base
        print(f"gate rows {gate.sum():,} ({gate.mean()*100:.2f}%)", flush=True)
        for m in range(1, 13):
            tr = gate & (month != m); va = gate & (month == m)
            b = lgb.train({**params, "num_leaves": 63, "min_data_in_leaf": 200}, lgb.Dataset(Xg.loc[tr], np.clip(resid[tr], -7200, 7200), categorical_feature=vcats), num_boost_round=300)
            new[va] = np.clip(base[va] + args.alpha * b.predict(Xg.loc[va]), 0, None)
            print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
        new = lirfbands.lobt_clip(new, X)
        report(f"v18_tailgate_a{args.alpha}", y, base, new, month, {"gate_rows": int(gate.sum()), "gate_rmse_base": rmse(y[gate], base[gate]), "gate_rmse_new": rmse(y[gate], new[gate])})

    elif args.what == "gatehold":
        cols = nm_free(X); vcats = [c for c in cats if c in cols]
        D = X[mechanism.PROXY_COL].to_numpy(dtype="float64"); G = np.clip(D - y, -172800, 172800)
        target = lirf & matched; new = base.copy(); alphas = {}
        gp = {"objective": "regression", "learning_rate": 0.04, "num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1, "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200, "max_cat_threshold": 16,
              "verbosity": -1, "seed": 42, "num_threads": args.threads}
        for m in range(1, 13):
            tr = lirf & (month != m); va = target & (month == m)
            b = lgb.train(gp, lgb.Dataset(X.loc[tr, cols], G[tr], categorical_feature=vcats), num_boost_round=500)
            t_tr = np.clip(D[tr] - b.predict(X.loc[tr, cols]), 0, None); fit_rows = matched[tr] & ~np.isin(month[tr], (1, 7))
            d = t_tr[fit_rows] - base[tr][fit_rows]; r = y[tr][fit_rows] - base[tr][fit_rows]
            a = float(np.clip((d @ r) / max(d @ d, 1e-9), 0, 1)); alphas[str(m)] = a
            t_va = np.clip(D[va] - b.predict(X.loc[va, cols]), 0, None)
            new[va] = base[va] + a * (t_va - base[va])
            print(f"  fold {m:02d} alpha {a:.3f} t={time.time()-start:.0f}s", flush=True)
        new = lirfbands.lobt_clip(new, X)
        report("v18_gatehold_matched", y, base, new, month, {"alphas": alphas, "target_rmse_base": rmse(y[target], base[target]), "target_rmse_new": rmse(y[target], new[target])})

    else:
        cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
        Xl = X[cols].copy()
        for c in ["taxi_proxy_AOBT_3_flt", "taxi_proxy_EOBT_1_flt", "taxi_proxy_IOBT_flt", "taxi_proxy_LOBT_flt", "taxi_proxy_SCHED_TIME_UTC_mvt", "offblock_delay"]:
            v = Xl[c].to_numpy(dtype="float64"); Xl[c + "_slog"] = np.sign(v) * np.log1p(np.abs(v))
        anchor = X[decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD); resid = (np.minimum(y, 3600) - anchor.to_numpy())
        rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]["base_matched"]
        out = {}
        for m in args.months:
            tr = matched & (month != m); va = matched & (month == m)
            b0 = lgb.train(params, lgb.Dataset(X.loc[tr, cols], resid[tr], categorical_feature=vcats), num_boost_round=rounds)
            r0 = rmse(resid[va], b0.predict(X.loc[va, cols]))
            b1 = lgb.train({**params, "linear_tree": True, "linear_lambda": 1.0}, lgb.Dataset(Xl.loc[tr], resid[tr], categorical_feature=vcats), num_boost_round=rounds)
            r1 = rmse(resid[va], b1.predict(Xl.loc[va]))
            out[str(m)] = {"plain": r0, "linear_tree+slog": r1}
            print(f"month {m:02d} plain {r0:.3f} linear_tree+slog {r1:.3f} ({r1-r0:+.3f}) t={time.time()-start:.0f}s", flush=True)
        (config.DATA_INTERIM / "v18_lineartree_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
