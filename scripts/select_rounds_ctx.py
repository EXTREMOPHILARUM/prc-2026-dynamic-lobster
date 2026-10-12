"""Re-select component rounds for the context feature set on inner months disjoint from Jan/Jul.

    uv run python -u scripts/select_rounds_ctx.py --inner 4 10
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from prc import arrivals, config, decomposition, evaluation, matrix


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--inner", type=int, nargs="+", default=[4, 10]); ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    for p in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS, decomposition.SEVERITY_PARAMS):
        p["num_threads"] = args.threads
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].astype("float32"); month = evaluation.movement_month(raw).to_numpy()
    inner = np.isin(month, args.inner); outer = ~inner & ~np.isin(month, (1, 7))
    print(f"fit on {outer.sum():,} rows, select on {inner.sum():,} (months {args.inner}); {len(X.columns)} features", flush=True)
    models = decomposition.fit(X.loc[outer], y.loc[outer], cats, X.loc[inner], y.loc[inner], rounds=3000, stopping=100)
    rounds = {k: int(m.best_iteration or m.num_trees()) for k, m in models.items()}
    old = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    (config.DATA_INTERIM / "component_rounds_ctx.json").write_text(json.dumps({"inner_months": args.inner, "rounds": rounds, "previous": old, "features": len(X.columns)}, indent=2))
    print("rounds", rounds, "previous", old)


if __name__ == "__main__":
    main()
