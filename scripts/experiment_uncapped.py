"""Does capping the matched base's target at 3600s cost us the learnable tail?

merry-quicksand fit the full clock delta and found winsorising 1% of training targets cost
30s. Our base caps y at TAIL_THRESHOLD and adds a shrunk gate x severity tail. This screens,
on matched rows of held-out months, three matched-lane predictors on the TRUE target:
  capped     anchor + f(x) trained on min(y,3600) - anchor            (our base alone)
  uncapped   anchor + f(x) trained on y - anchor, no cap
  v19        the shipped pipeline's OOF (base + gate x severity + all layers), for reference

    uv run python -u scripts/experiment_uncapped.py --months 2 10 --threads 4
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, nargs="+", default=[2, 10]); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=800); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    anchor = np.clip(anchor_raw, 0, decomposition.TAIL_THRESHOLD)
    v19 = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / "v19_oof_predictions.parquet"), on=config.ID_COL, how="left")["prediction"].to_numpy()
    cols = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; vcats = [c for c in cats if c in cols]
    params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 42}
    out = {}
    for m in args.months:
        tr = matched & (month != m); va = matched & (month == m); r = {}
        for name, target in (("capped", np.minimum(y, 3600) - anchor), ("uncapped", y - anchor)):
            b = lgb.train(params, lgb.Dataset(X.loc[tr, cols], target[tr], categorical_feature=vcats), num_boost_round=args.rounds)
            pred = np.clip(anchor[va] + b.predict(X.loc[va, cols]), 0, None)
            r[name] = {"all": rmse(y[va], pred), "tail_rows": rmse(y[va & (y > 3600)], pred[y[va] > 3600]), "bulk_rows": rmse(y[va & (y <= 3600)], pred[y[va] <= 3600])}
        r["v19_pipeline"] = {"all": rmse(y[va], v19[va]), "tail_rows": rmse(y[va & (y > 3600)], v19[va][y[va] > 3600]), "bulk_rows": rmse(y[va & (y <= 3600)], v19[va][y[va] <= 3600])}
        out[str(m)] = r
        print(f"month {m:02d} matched rows: " + " | ".join(f"{k}: all {v['all']:.2f} bulk {v['bulk_rows']:.2f} tail {v['tail_rows']:.0f}" for k, v in r.items()) + f"  t={time.time()-start:.0f}s", flush=True)
    (config.DATA_INTERIM / "uncapped_screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
