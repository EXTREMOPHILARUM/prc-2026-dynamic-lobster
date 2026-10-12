"""Matched own-flight expert, retune screen: richer inputs and robust targets (2026-10-04).

The shipped matched expert (`ownexpert.FEATURES_V7`, 255 leaves, 1,200 rounds, lr 0.025; section 60) reads
the trace, the anchor, the chain's prediction, airport, hour and weekday. It never sees the matrix the
chain was fitted on (clock proxies, congestion, arrivals, weather, stand/runway), so it cannot learn
where the trace's relation to the off-block depends on context. This screen adds those inputs, and
tries robust targets (the residual is unclipped, so a day-slip row can pull a leaf).

Same protocol as `experiment_own_expert_bundle.py`: fitted on the other months of the v29
reconstruction's OOF residual, applied to the held-out month, paired against the v7 expert's stored
OOF on the same months. Selection on non-scored months; January/July are diagnostics.

    uv run python -u scripts/experiment_own_expert_rich.py v7 rich_num --months 3 5 9 11
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)
BASELINE = ("own_expert_bundle_l255_r1200_lr025_v7_oof.parquet", "own_l255_r1200_lr025")
# PRC_THREADS sets the CPU threads per machine (this Mac 14, the GPU host 16) instead of editing the file
PARAMS = {**ownexpert.PARAMS, "num_leaves": 255, "learning_rate": 0.025, "num_threads": int(os.environ.get("PRC_THREADS", "14"))}
ROUNDS = 1200
CAT_KEEP = ownexpert.RICH_CATEGORICALS
DRIFTED = ["own_to_minus_mvt", "own_to_late", "own_npts", "own_obs_path", "own_first_dist", "own_first_net_dist",
           "own_max_dt", "own_stand_dmin", "own_pts_per_min", "own_obs_speed", "own_unobs_time", "own_first_gap_plus_unobs"]
CONFIGS = {
    "v7": {"x": None},
    "rich_num": {"x": "num"},
    "rich_cat": {"x": "cat"},
    "v7_clip1800": {"x": None, "clip": 1800.0},
    "v7_huber": {"x": None, "params": {"objective": "huber", "alpha": 600.0}},
    "rich_num_clip1800": {"x": "num", "clip": 1800.0},
    "rich_cat_clip1800": {"x": "cat", "clip": 1800.0},
    "rich_num_l127": {"x": "num", "params": {"num_leaves": 127}},
    "rich_num_r2400": {"x": "num", "rounds": 2400},
    "rich_num_matched": {"x": "num", "scope": "matched"},
    "rich_cat_matched": {"x": "cat", "scope": "matched"},
    "cb_rich_cat": {"x": "cat", "engine": "cb"},
    "rich_cat_s8": {"x": "cat", "params": {"seed": 8}},
    "rich_cat_r2400": {"x": "cat", "rounds": 2400},
    "rich_cat_ff5": {"x": "cat", "params": {"feature_fraction": 0.5}},
    "rich_cat_ff3": {"x": "cat", "params": {"feature_fraction": 0.3}},
    "rich_cat_ff5_md50": {"x": "cat", "params": {"feature_fraction": 0.5, "min_data_in_leaf": 50}},
    "rich_cat_ff5_l511": {"x": "cat", "params": {"feature_fraction": 0.5, "num_leaves": 511}},
    "rich_cat_ff5_cs10_l511": {"x": "cat", "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                       "cat_l2": 50, "min_data_per_group": 200}},
    "v8_cs10_l511": {"x": "cat", "own": 8, "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                       "cat_l2": 50, "min_data_per_group": 200}},
    "v9_cs10_l511": {"x": "cat", "own": 9, "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                       "cat_l2": 50, "min_data_per_group": 200}},
    "v8_cs10_l511_lag": {"x": "cat", "own": 8, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511,
                                                                      "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200}},
    "v9_cs10_l511_lag": {"x": "cat", "own": 9, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511,
                                                                      "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200}},
    "v10_lag": {"x": "cat", "own": 10, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                                "cat_l2": 50, "min_data_per_group": 200}},
    "v10_lag_s8": {"x": "cat", "own": 10, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511,
                                                                   "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200,
                                                                   "seed": 8}},
    "v10_lag_s9": {"x": "cat", "own": 10, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511,
                                                                   "cat_smooth": 50, "cat_l2": 50, "min_data_per_group": 200,
                                                                   "seed": 9}},
    "v9_lag_daily": {"x": "cat", "own": 9, "lag": True, "daily": True,
                     "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                                "min_data_per_group": 200}},
    "v10_lag_daily": {"x": "cat", "own": 10, "lag": True, "daily": True,
                      "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                                 "min_data_per_group": 200}},
    "v9_lag_fgbase": {"x": "cat", "own": 9, "lag": True, "base": "first_gap",
                      "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                                 "min_data_per_group": 200}},
    "lean_v9_lag": {"x": None, "own": 9, "lag": True, "params": {"num_leaves": 255}},
    "cb_lean_v9": {"x": None, "own": 9, "lag": True, "engine": "cb", "iterations": 6000, "depth": 8},
    "lean_v9_lag_s8": {"x": None, "own": 9, "lag": True, "params": {"num_leaves": 255, "seed": 8}},
    # GREKI (Discord 2026-09-27): reception-dependent inputs drift between 2025 and 2026 (adsb.lol's receiver network
    # changed); movement ones transfer. Drop the features whose 2025->2026 drift |2*AUC-1| reached >= 0.45 at some airport
    # (label-free check, log section 63), and the inputs derived from them
    "lean_v9_lag_mov": {"x": None, "own": 9, "lag": True, "params": {"num_leaves": 255}, "drop": DRIFTED},
    "v9_lag_daily_s8": {"x": "cat", "own": 9, "lag": True, "daily": True,
                        "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                                   "min_data_per_group": 200, "seed": 8}},
    "cb_v9_lag": {"x": "cat", "own": 9, "lag": True, "engine": "cb", "iterations": 6000, "depth": 8},
    "v9_lag_comp": {"x": "cat", "own": 9, "lag": True, "comp": True,
                    "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                               "min_data_per_group": 200}},
    "v9_lagext": {"x": "cat", "own": 9, "lag": "ext", "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                                  "cat_l2": 50, "min_data_per_group": 200}},
    "v9_lag_lr0125": {"x": "cat", "own": 9, "lag": True, "rounds": 2400,
                      "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50, "cat_l2": 50,
                                 "min_data_per_group": 200, "learning_rate": 0.0125}},
    "v9_lag_s8": {"x": "cat", "own": 9, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                                "cat_l2": 50, "min_data_per_group": 200, "seed": 8}},
    "v9_lag_s9": {"x": "cat", "own": 9, "lag": True, "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                                "cat_l2": 50, "min_data_per_group": 200, "seed": 9}},
    "rich_cat_ff5_cs10_l1023": {"x": "cat", "params": {"feature_fraction": 0.5, "num_leaves": 1023, "cat_smooth": 50,
                                                        "cat_l2": 50, "min_data_per_group": 200}},
    "rich_cat_ff5_cs10_l511_md50": {"x": "cat", "params": {"feature_fraction": 0.5, "num_leaves": 511, "cat_smooth": 50,
                                                            "cat_l2": 50, "min_data_per_group": 200, "min_data_in_leaf": 50}},
    "rich_cat_ff5_cs200": {"x": "cat", "params": {"feature_fraction": 0.5, "cat_smooth": 200, "cat_l2": 100,
                                                   "min_data_per_group": 500}},
    "rich_cat_ff5_hold": {"x": "cat", "params": {"feature_fraction": 0.5}, "own_extra": ["own_hold"]},
    "rich_cat_ff5_cs10": {"x": "cat", "params": {"feature_fraction": 0.5, "cat_smooth": 50, "cat_l2": 50,
                                                  "min_data_per_group": 200}},
    "rich_cat_matched_clip": {"x": "cat", "scope": "matched", "clip": 3600.0},
    "rich_num_matched_clip": {"x": "num", "scope": "matched", "clip": 3600.0},
}


def _load(name):
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fit_predict(cfg, params, rounds, F, feats, cats, target, tr, va):
    """Residual expert fitted on `tr`, predicted on `va`: LightGBM, or CatBoost on the GPU (cfg engine 'cb')."""
    if cfg.get("engine") != "cb":
        b = lgb.train(params, lgb.Dataset(F.loc[tr, feats], target[tr], categorical_feature=cats), rounds)
        return b.predict(F.loc[va, feats])
    from catboost import CatBoostRegressor, Pool

    def pool(m, label=None):
        D = F.loc[m, feats].copy()
        for c in cats:
            D[c] = D[c].astype(object).where(D[c].notna(), "NA").astype(str)
        return Pool(D, label, cat_features=cats)
    model = CatBoostRegressor(iterations=cfg.get("iterations", 4000), depth=cfg.get("depth", 8), learning_rate=0.06,
                              loss_function="RMSE", task_type="GPU", random_seed=7, verbose=0)
    model.fit(pool(tr, target[tr]))
    return model.predict(pool(va))


def lag_key_set(cfg) -> dict:
    return ownexpert.LAG_KEYS_EXT if cfg.get("lag") == "ext" else ownexpert.LAG_KEYS


def lag_keys(raw: pd.DataFrame, keys: dict) -> dict[str, np.ndarray]:
    """Integer codes of the lag encoding keys, for fast nested encodings in the folds."""
    return {name: pd.factorize(ownexpert.lag_key_strings(raw, cols), use_na_sentinel=False)[0]
            for name, cols in keys.items()}


def lag_encode(codes: dict, lag: np.ndarray, ok: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> dict[str, np.ndarray]:
    """`ownexpert.lag_tables` + `lag_apply` on integer codes: smoothed mean of the clipped lag over `src & ok`."""
    m = src & ok
    v = np.clip(lag[m], -ownexpert.LAG_CLIP, ownexpert.LAG_CLIP); g = float(v.mean())
    out = {}
    for name, c in codes.items():
        n = int(c.max()) + 1
        s_ = np.bincount(c[m], weights=v, minlength=n); cnt = np.bincount(c[m], minlength=n)
        out[name] = ((s_ + ownexpert.LAG_K * g) / (cnt + ownexpert.LAG_K))[c[tgt]]
    return out


COMP = ["comp_gate", "comp_ord_minus_pred", "comp_aobt_minus_pred", "comp_lobt_minus_pred", "comp_stack"]


def components(raw: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """The chain's internal components as offsets from its output `p` (v29 reconstruction's stored OOFs): the schedule
    gate, the ordinary base, the AOBT and LOBT CatBoost members, and the stacker's correction."""
    ids = raw[[config.ID_COL]]

    def col(f, c):
        return ids.merge(pd.read_parquet(config.DATA_INTERIM / f, columns=[config.ID_COL, c]), on=config.ID_COL,
                         how="left")[c].to_numpy("float64")
    st = col("stacker_mid_800_oof.parquet", "stack") - col("stacker_mid_800_oof.parquet", "ref")
    return pd.DataFrame({"comp_gate": col("v21_oof_predictions.parquet", "gate_new"),
                         "comp_ord_minus_pred": col("v21_oof_predictions.parquet", "ordinary") - p,
                         "comp_aobt_minus_pred": col("catgpu_d8_oof_predictions.parquet", "member") - p,
                         "comp_lobt_minus_pred": col("catgpu_lobt_oof_predictions.parquet", "member") - p,
                         "comp_stack": st}, index=raw.index)


def own_features(cfg) -> list[str]:
    feats = {10: ownexpert.FEATURES_V10, 9: ownexpert.FEATURES_V9, 8: ownexpert.FEATURES_V8}.get(cfg.get("own"),
                                                                                               ownexpert.FEATURES_V7)
    return [c for c in feats if c not in cfg.get("drop", [])]


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


extra_columns = ownexpert.rich_columns
with_extra = ownexpert.with_rich


def scope_rows(f: pd.DataFrame, kind: str) -> np.ndarray:
    """joined: the shipped expert's rows; matched: every row with an NM anchor (own_* NaN where unjoined)."""
    return ownexpert.scope(f) if kind == "joined" else np.isfinite(f["anchor"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    ap.add_argument("--months", nargs="*", type=int, default=list(range(1, 13)), help="held-out months; none with --save")
    ap.add_argument("--tag", default="")
    ap.add_argument("--src", default="adsb_own_2025_v7.parquet", help="own-flight feature file (v8 configs need _v8)")
    ap.add_argument("--save", default=None, metavar="STEM", help="fit the single config on all months -> models/STEM.txt")
    args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    p = _load("analyse_error_anatomy").v29_oof(raw, X)
    month = evaluation.movement_month(raw).to_numpy()
    apt = raw["ADEP_mvt"].astype(str).to_numpy()
    anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan, X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"))
    own = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.src), on=config.ID_COL, how="left")
    f = ownexpert.frame(apt, raw["MVT_TIME_UTC_mvt"], p, anchor, own)
    rows = ownexpert.scope(f)
    base = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / BASELINE[0], columns=[config.ID_COL, BASELINE[1]]),
                                      on=config.ID_COL, how="left")[BASELINE[1]].to_numpy("float64")
    resid = y - p
    sel = np.isin(month, args.months)
    print(f"{len(f):,} rows; scope {int(rows.sum()):,}; months {args.months}", flush=True)
    results = {}
    oofs = {}
    for name in args.configs if args.months else []:
        cfg = CONFIGS[name]
        xcols = extra_columns(X, cfg["x"])
        F = with_extra(f, X, xcols)
        rows = scope_rows(f, cfg.get("scope", "joined"))
        feats = list(own_features(cfg)) + cfg.get("own_extra", []) + ["x_" + c for c in xcols]
        if cfg.get("comp"):
            F = pd.concat([F, components(raw, p)], axis=1); feats = feats + COMP
        if cfg.get("daily"):
            F = pd.concat([F, ownexpert.daily_reception(f, raw["MVT_TIME_UTC_mvt"])], axis=1); feats = feats + ownexpert.DAILY
        cats = ["apt"] + ["x_" + c for c in xcols if c in CAT_KEEP]
        params = {**PARAMS, **cfg.get("params", {})}
        rounds = cfg.get("rounds", ROUNDS)
        # base the residual is taken against: the chain's prediction, or (cfg base "first_gap") the trace's first moving
        # point after the last gap where it exists; same inputs, a differently anchored error for blending
        b0 = p.copy()
        if cfg.get("base") == "first_gap":
            fg = F["own_first_taxi_gap"].to_numpy("float64"); okb = rows & np.isfinite(fg)
            b0[okb] = fg[okb]
        resid_b = y - b0
        target = np.clip(resid_b, -cfg["clip"], cfg["clip"]) if "clip" in cfg else resid_b
        q = p.copy()
        if cfg.get("lag"):
            codes = lag_keys(raw, lag_key_set(cfg))
            lag = y - F["own_first_taxi_gap"].to_numpy("float64")
            lag_ok = rows & np.isfinite(lag)
            feats = feats + list(codes)
        for m in args.months:
            tr = (month != m) & rows; va = (month == m) & rows
            if cfg.get("lag"):
                # nested: a training row's encoding never includes its own month, the held-out month never its own
                for c in codes:
                    F[c] = np.nan
                for t in range(1, 13):
                    if t == m:
                        continue
                    tgt = (month == t) & rows
                    for c, v in lag_encode(codes, lag, lag_ok, (month != m) & (month != t), tgt).items():
                        F.loc[tgt, c] = v
                for c, v in lag_encode(codes, lag, lag_ok, month != m, va).items():
                    F.loc[va, c] = v
            q[va] = np.clip(b0[va] + fit_predict(cfg, params, rounds, F, feats, cats, target, tr, va), 0, None)
            d = rmse((y - q)[month == m]) - rmse((y - base)[month == m])
            print(f"  {name} month {m}: {rmse((y - q)[month == m]):.3f} vs v7 {rmse((y - base)[month == m]):.3f} ({d:+.3f})", flush=True)
        e0 = (y - base)[sel]; e1 = (y - q)[sel]
        per = {m: rmse((y - q)[month == m]) - rmse((y - base)[month == m]) for m in args.months}
        ns = sel & ~np.isin(month, SCORED)
        dsse = float(((y - q) ** 2 - (y - base) ** 2)[ns].sum())
        results[name] = {"pooled": rmse(e1) - rmse(e0), "nonscored_dsse": dsse, "per_month": per,
                         "wins": sum(v < 0 for v in per.values())}
        oofs[name] = q
        print(f"== {name}: pooled {results[name]['pooled']:+.3f}  non-scored dSSE {dsse/1e6:+.1f}M  "
              f"wins {results[name]['wins']}/{len(args.months)}  " + " ".join(f"{m}:{v:+.2f}" for m, v in per.items()), flush=True)
    out = config.DATA_INTERIM / "own_expert_rich_results.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update({f"{k}{args.tag}@{'-'.join(map(str, args.months))}": v for k, v in results.items()})
    out.write_text(json.dumps(prev, indent=1, default=str))
    if args.months and (len(args.months) == 12 or args.tag):
        pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), **{f"own_{k}": v for k, v in oofs.items()}}).to_parquet(
            config.DATA_INTERIM / f"own_expert_rich_{'_'.join(args.configs)}{args.tag}_oof.parquet", index=False)
    if args.save:
        name = args.configs[0]; cfg = CONFIGS[name]; xcols = extra_columns(X, cfg["x"])
        F = with_extra(f, X, xcols)
        rows = scope_rows(f, cfg.get("scope", "joined"))
        feats = list(own_features(cfg)) + cfg.get("own_extra", []) + ["x_" + c for c in xcols]
        if cfg.get("daily"):
            F = pd.concat([F, ownexpert.daily_reception(f, raw["MVT_TIME_UTC_mvt"])], axis=1); feats = feats + ownexpert.DAILY
        cats = ["apt"] + ["x_" + c for c in xcols if c in CAT_KEEP]
        target = np.clip(resid, -cfg["clip"], cfg["clip"]) if "clip" in cfg else resid
        if cfg.get("lag"):
            lag = y - F["own_first_taxi_gap"].to_numpy("float64")
            tables = ownexpert.lag_tables(raw, lag, rows & np.isfinite(lag), lag_key_set(cfg))
            for c, v in ownexpert.lag_apply(tables, raw).items():
                F[c] = v
            feats = feats + list(lag_key_set(cfg))
            (config.MODELS / f"{args.save}.lag.json").write_text(json.dumps(tables))
        if cfg.get("engine") == "cb":
            from catboost import CatBoostRegressor, Pool
            D = F.loc[rows, feats].copy()
            for c in cats:
                D[c] = D[c].astype(object).where(D[c].notna(), "NA").astype(str)
            m_cb = CatBoostRegressor(iterations=cfg.get("iterations", 4000), depth=cfg.get("depth", 8), learning_rate=0.06,
                                     loss_function="RMSE", task_type="GPU", random_seed=7, verbose=0)
            m_cb.fit(Pool(D, target[rows], cat_features=cats))
            m_cb.save_model(str(config.MODELS / f"{args.save}.cbm")); print(f"saved models/{args.save}.cbm")
            return
        b = lgb.train({**PARAMS, **cfg.get("params", {})}, lgb.Dataset(F.loc[rows, feats], target[rows], categorical_feature=cats),
                      cfg.get("rounds", ROUNDS))
        b.save_model(str(config.MODELS / f"{args.save}.txt")); print(f"saved models/{args.save}.txt")


if __name__ == "__main__":
    main()
