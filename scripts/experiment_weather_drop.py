"""Paired drop test of the METAR block on the matched base, v20 feature set, four months.

merry-quicksand logged weather as +0.49s (harmful) on their base; ours gained 3.6s on the
board in v9 on a much weaker base. Component-level screen: capped-residual RMSE with and
without the wx_* and rwy_length_ft columns.

    uv run python -u scripts/experiment_weather_drop.py --months 2 10 1 7 --threads 3
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

from prc import arrivals, config, decomposition, evaluation, matrix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7]); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=800); args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    resid = np.minimum(y, 3600) - np.clip(anchor_raw, 0, 3600)
    with_wx = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]
    no_wx = [c for c in with_wx if not c.startswith("wx_") and c != "rwy_length_ft"]
    vcats = [c for c in cats if c in no_wx]; params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 42}
    out = {}
    for m in args.months:
        tr = matched & (month != m); va = matched & (month == m); r = {}
        for name, cols in (("with_wx", with_wx), ("no_wx", no_wx)):
            b = lgb.train(params, lgb.Dataset(X.loc[tr, cols], resid[tr], categorical_feature=vcats), num_boost_round=args.rounds)
            r[name] = float(np.sqrt(np.mean((resid[va] - b.predict(X.loc[va, cols])) ** 2)))
        out[str(m)] = r; print(f"month {m:02d} with_wx {r['with_wx']:.3f} no_wx {r['no_wx']:.3f} (drop effect {r['no_wx']-r['with_wx']:+.3f}) t={time.time()-start:.0f}s", flush=True)
    (config.DATA_INTERIM / "weather_drop_screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
