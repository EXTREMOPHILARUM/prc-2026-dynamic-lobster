"""LIRF second stage on the rich own-flight expert (log section 63).

Under the v42-equivalent OOF, LIRF's joined matched rows carry ~37% of joined squared error (RMSE ~345) on 5.6% of
joined rows. The global expert is fitted on all ten airports; this screen fits a small LightGBM on the residual the
v42-equivalent leaves on LIRF joined rows (other months, twelve folds), with the expert's inputs, and replaces the
prediction there. Paired against the v42-equivalent; selection on the non-scored months.

    uv run python -u scripts/experiment_lirf_stage2.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)
CONFIGS = {
    "l15_r300": {"num_leaves": 15, "rounds": 300, "clip": None},
    "l15_r300_clip1800": {"num_leaves": 15, "rounds": 300, "clip": 1800.0},
    "l63_r600_lr025": {"num_leaves": 63, "rounds": 600, "learning_rate": 0.025, "clip": None},
}


def _load(name):
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    ids = raw[[config.ID_COL]]
    y = raw[config.TARGET].to_numpy("float64"); month = evaluation.movement_month(raw).to_numpy()
    base = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v42_equiv_oof.parquet"), on=config.ID_COL,
                     how="left")["prediction"].to_numpy("float64")
    anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan, X[decomposition.ANCHOR_COL].to_numpy("float64"))
    own = ids.merge(pd.read_parquet(config.DATA_INTERIM / "adsb_own_2025_v9.parquet"), on=config.ID_COL, how="left")
    f = ownexpert.frame(raw["ADEP_mvt"].astype(str).to_numpy(), raw["MVT_TIME_UTC_mvt"], base, anchor, own)
    xcols = ownexpert.rich_columns(X, "cat")
    f = ownexpert.with_rich(f, X, xcols)
    rows = ownexpert.scope(f) & (raw["ADEP_mvt"].astype(str).to_numpy() == "LIRF")
    feats = [c for c in ownexpert.FEATURES_V9 if c != "apt"] + ["x_" + c for c in xcols]
    cats = ["x_" + c for c in xcols if c in ownexpert.RICH_CATEGORICALS]
    resid = y - base
    sc = np.isin(month, SCORED); ns = ~sc
    print(f"LIRF joined rows {int(rows.sum()):,}; base RMSE there {rmse(resid[rows]):.1f}", flush=True)
    for name, cfg in CONFIGS.items():
        params = {**ownexpert.PARAMS, "num_leaves": cfg["num_leaves"], "learning_rate": cfg.get("learning_rate", 0.05),
                  "num_threads": 4}
        target = np.clip(resid, -cfg["clip"], cfg["clip"]) if cfg["clip"] else resid
        q = base.copy()
        for m in range(1, 13):
            tr = rows & (month != m); va = rows & (month == m)
            if not va.any():
                continue
            b = lgb.train(params, lgb.Dataset(f.loc[tr, feats], target[tr], categorical_feature=cats), cfg["rounds"])
            q[va] = np.clip(base[va] + b.predict(f.loc[va, feats]), 0, None)
        d = [rmse((y - q)[month == m]) - rmse((y - base)[month == m]) for m in range(1, 13)]
        dsse = float(((y - q) ** 2 - (y - base) ** 2)[ns].sum())
        print(f"{name:20s} twelve {rmse(y - q) - rmse(y - base):+.3f}  Jan+Jul {rmse((y - q)[sc]) - rmse((y - base)[sc]):+.3f}  "
              f"non-scored dSSE {dsse / 1e6:+.1f}M wins {sum(x < 0 for m, x in enumerate(d, 1) if m not in SCORED)}/10  "
              + " ".join(f"{m}:{x:+.2f}" for m, x in enumerate(d, 1)), flush=True)


if __name__ == "__main__":
    main()
