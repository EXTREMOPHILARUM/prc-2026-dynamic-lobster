"""Select the matched base's learning rate / rounds with the v20 feature set (context v2 +
in-fold target encodings), early-stopped on inner months disjoint from Jan/Jul.

    uv run python -u scripts/select_rounds_v20.py --lr 0.02 --max-rounds 20000 --inner 4 10
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

from prc import arrivals, config, decomposition, evaluation, matrix, tencode


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--inner", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--max-rounds", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=12); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    inner = np.isin(month, args.inner); outer = ~inner & ~np.isin(month, (1, 7))
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    delta = np.where(matched, anchor_raw - y, np.nan)   # AOBT-based delta = (MVT-AOBT) - y = BLOCK - AOBT
    enc = tencode.fit(raw, y, delta, outer)
    Xte = X.join(tencode.transform(raw, enc))
    resid = np.minimum(y, decomposition.TAIL_THRESHOLD) - np.clip(anchor_raw, 0, decomposition.TAIL_THRESHOLD)
    cols = [c for c in Xte.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
    print(f"matrix {Xte.shape}; fit {int((outer & matched).sum()):,} rows, select on {int((inner & matched).sum()):,}", flush=True)
    params = {**decomposition.BASE_PARAMS, "learning_rate": args.lr, "num_threads": args.threads, "seed": 42}
    rec = {}
    booster = lgb.train(params, lgb.Dataset(Xte.loc[outer & matched, cols], resid[outer & matched], categorical_feature=vcats),
                        num_boost_round=args.max_rounds,
                        valid_sets=[lgb.Dataset(Xte.loc[inner & matched, cols], resid[inner & matched], categorical_feature=vcats)],
                        valid_names=["inner"], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500), lgb.record_evaluation(rec)])
    curve = rec["inner"]["rmse"]
    out = {"lr": args.lr, "best_iter": int(booster.best_iteration), "best_rmse": float(min(curve)), "inner_months": args.inner,
           "features": len(cols), "rmse_at": {str(k): float(curve[k - 1]) for k in (500, 1000, 2000, 5000, 10000, 20000) if k <= len(curve)},
           "seconds": time.time() - start}
    (config.DATA_INTERIM / f"rounds_v20_lr{args.lr}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
