"""Second-stage stacker over stored OOF component predictions.

Reference: the shipped chain approximated from stored OOF files (v21 base, CatBoost member at
0.5 total, Rome mixture on its target rows, LOBT clip). Candidates, all leave-one-month-out
over the 12 months of 2025 so Jan and Jul are scored out of fold:

  linear   per-lane least squares over the components (matched / LIRF no-record / other no-record)
  lgbm     small LightGBM on the components + clocks + airport, target = residual to the reference

    uv run python -u scripts/experiment_stacker.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, lirfmoe, matrix

PROXIES = ["taxi_proxy_LOBT_flt", "taxi_proxy_EOBT_1_flt", "taxi_proxy_IOBT_flt", "taxi_proxy_SCHED_TIME_UTC_mvt", "aobt_missing", "nm_record_missing"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def report(name, y, ref, new, month, out):
    sc = np.isin(month, (1, 7))
    bym = [rmse(y[month == m], new[month == m]) - rmse(y[month == m], ref[month == m]) for m in range(1, 13)]
    ys, rs, ns = y[sc], ref[sc], new[sc]
    order = np.argsort((ys - ns) ** 2 - (ys - rs) ** 2)
    rob = []
    for k in (1, 10, 50):
        keep = np.ones(len(ys), bool); keep[order[:k]] = False
        rob.append(round(rmse(ys[keep], ns[keep]) - rmse(ys[keep], rs[keep]), 2))
    out[name] = {"jan_jul": rmse(ys, ns), "ref_jan_jul": rmse(ys, rs), "jan": rmse(y[month == 1], new[month == 1]), "jul": rmse(y[month == 7], new[month == 7]),
                 "all": rmse(y, new), "ref_all": rmse(y, ref), "months_improved": int(sum(v < 0 for v in bym)), "by_month": [round(v, 2) for v in bym], "robustness": rob}
    r = out[name]
    print(f"{name:14s} Jan+Jul {r['ref_jan_jul']:.3f} -> {r['jan_jul']:.3f} ({r['jan_jul'] - r['ref_jan_jul']:+.3f}) | Jan {r['jan']:.2f} Jul {r['jul']:.2f} | 12m {r['ref_all']:.3f} -> {r['all']:.3f} | months {r['months_improved']}/12 | rob {rob}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--members", nargs="+", default=["catgpu_d8"], help="member OOF stems whose mean forms the reference blend"); ap.add_argument("--weights", type=float, nargs="+", default=None, help="per-member blend weights (default: 0.5 spread evenly)"); ap.add_argument("--tag", default=""); ap.add_argument("--mid", action="store_true", help="reduced inputs plus ordinary/gate/LOBT expert, which the chain also computes"); ap.add_argument("--reduced", action="store_true", help="only inputs the live chain exposes: pre-member prediction, CatBoost member, blend, clocks, airport; matched rows")
    args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    ids = raw[[config.ID_COL]]

    def col(f, c):
        return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy(dtype="float64").copy()

    base = col("v21_oof_predictions.parquet", "prediction"); ordinary = col("v21_oof_predictions.parquet", "ordinary"); gate = col("v21_oof_predictions.parquet", "gate_new")
    mem = {m: col(f"{m}_oof_predictions.parquet", "member") for m in args.members}
    wts = args.weights or [0.5 / len(args.members)] * len(args.members)
    assert len(wts) == len(args.members)
    p_copy = col("lirf_unmatched_moe_oof.parquet", "p_copy"); mix = col("lirf_unmatched_moe_oof.parquet", "mixture")
    lobt_x = col("lobt_expert_oof.parquet", "expert"); p_id = col("lirf_regime_oof_predictions.parquet", "p_identity")
    tgt = lirfmoe.target_rows(X, 8.0)
    ok = np.all([np.isfinite(v) for v in mem.values()], axis=0)
    ref = base.copy()
    for m, wi in zip(args.members, wts, strict=True):
        ref[ok] += wi * (mem[m][ok] - base[ok])
    ref[tgt] = mix[tgt]
    ref = np.clip(lirfbands.lobt_clip(ref, X), 0, None)
    print(f"reference Jan+Jul {rmse(y[np.isin(month, (1, 7))], ref[np.isin(month, (1, 7))]):.3f}", flush=True)

    apt = raw["ADEP_mvt"].astype("category").cat.codes.to_numpy()
    lirf = (raw["ADEP_mvt"].to_numpy() == "LIRF"); missing = X["aobt_missing"].to_numpy() > 0
    lane = np.where(~missing, 0, np.where(lirf, 1, 2))

    comps = {"ref": ref, "base": base, "ordinary": ordinary, "gate": gate, "cat": np.where(ok, np.mean([mem[m] for m in args.members], axis=0), base), "mix": np.where(tgt, mix, ref), "p_copy": np.nan_to_num(p_copy), "lobt_x": np.nan_to_num(lobt_x, nan=0.0), "p_id": np.nan_to_num(p_id)}
    C = pd.DataFrame(comps)
    F = pd.concat([C, X[PROXIES].reset_index(drop=True)], axis=1); F["apt"] = apt; F["lane"] = lane
    out = {}
    if args.reduced or args.mid:
        keep = ["ref", "base", "cat", "taxi_proxy_LOBT_flt", "taxi_proxy_EOBT_1_flt", "taxi_proxy_IOBT_flt", "taxi_proxy_SCHED_TIME_UTC_mvt", "apt"] + (["ordinary", "gate", "lobt_x"] if args.mid else [])
        F = F[keep]; label = (args.tag or ("mid" if args.mid else "reduced"))
        params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 8, "seed": 7}
        for rounds in (300, 800):
            st = ref.copy()
            for m in range(1, 13):
                te = (month == m) & (lane == 0); tr = (month != m) & (lane == 0)
                bst = lgb.train(params, lgb.Dataset(F[tr], y[tr] - ref[tr], categorical_feature=["apt"]), num_boost_round=rounds)
                st[te] = ref[te] + bst.predict(F[te])
                print(f"  {label}_{rounds} fold {m} done", flush=True)
            st = np.clip(lirfbands.lobt_clip(st, X), 0, None)
            report(f"{label}_{rounds}", y, ref, st, month, out)
            pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "stack": st, "ref": ref}).to_parquet(config.DATA_INTERIM / f"stacker_{label}_{rounds}_oof.parquet", index=False)
        (config.DATA_INTERIM / f"stacker_{label}_metrics.json").write_text(json.dumps(out, indent=2))
        return

    # --- linear per-lane stack, LOMO
    lin = ref.copy()
    cols = ["ref", "base", "cat", "mix", "lobt_x", "ordinary"]
    for m in range(1, 13):
        te = month == m
        for ln in (0, 1, 2):
            tr = ~te & (lane == ln); ts = te & (lane == ln)
            if ts.sum() == 0:
                continue
            A = np.column_stack([C.loc[tr, cols].to_numpy(), np.ones(tr.sum())]); w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            lin[ts] = np.column_stack([C.loc[ts, cols].to_numpy(), np.ones(ts.sum())]) @ w
    lin = np.clip(lirfbands.lobt_clip(lin, X), 0, None)
    report("linear_lane", y, ref, lin, month, out)

    # --- LightGBM residual stacker, LOMO
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 8, "seed": 7}
    for rounds, tag in ((300, "lgbm_300"), (800, "lgbm_800")):
        st = ref.copy()
        for m in range(1, 13):
            te = month == m; tr = ~te
            ds = lgb.Dataset(F[tr], y[tr] - ref[tr], categorical_feature=["apt", "lane"], free_raw_data=False)
            bst = lgb.train(params, ds, num_boost_round=rounds)
            st[te] = ref[te] + bst.predict(F[te])
            print(f"  {tag} fold {m} done", flush=True)
        st = np.clip(lirfbands.lobt_clip(st, X), 0, None)
        report(tag, y, ref, st, month, out)
        # matched-only application (leave Rome layers untouched)
        stm = ref.copy(); stm[lane == 0] = st[lane == 0]
        report(tag + "_matched", y, ref, stm, month, out)
        pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "stack": st, "ref": ref}).to_parquet(config.DATA_INTERIM / f"stacker_{tag}_oof.parquet", index=False)

    (config.DATA_INTERIM / "stacker_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
