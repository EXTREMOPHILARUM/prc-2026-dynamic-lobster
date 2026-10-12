"""Fit the CatBoost member on all 2025 (GPU) and write the v22 wrapper with its blend weight.

    uv run python -u scripts/fit_catmember.py --weight 0.5 --inner mechanism_v21 --out mechanism_v22
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prc import arrivals, catmember, config, decomposition, matrix, tencode


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--weight", type=float, required=True); ap.add_argument("--inner", default="mechanism_v21")
    ap.add_argument("--out", default="mechanism_v22"); ap.add_argument("--alphas", default=None); ap.add_argument("--task-type", default="GPU"); ap.add_argument("--iterations", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--stem", default="catmember"); ap.add_argument("--anchor", choices=["aobt", "lobt", "eobt", "iobt"], default="aobt"); args = ap.parse_args()
    if args.iterations: catmember.PARAMS["iterations"] = args.iterations
    if args.lr: catmember.PARAMS["learning_rate"] = args.lr
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64")
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), np.ones(len(y), bool))
    model = catmember.fit(raw, X, cats, y, enc, task_type=args.task_type, anchor=args.anchor)
    model.save_model(str(config.MODELS / f"{args.stem}.cbm"))
    (config.MODELS / f"{args.stem}.json").write_text(json.dumps({"params": catmember.PARAMS, "categoricals": cats, "target_encoding": enc, "features": list(X.columns), "anchor": args.anchor}))
    meta = json.loads((config.MODELS / f"{args.inner}.json").read_text())
    meta |= {"cat_member_stem": args.stem, "cat_weight": args.weight}
    if args.alphas:
        meta["lobt_alphas"] = json.loads(Path(args.alphas).read_text())["alphas_oof"]
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models/{args.stem}.cbm and models/{args.out}.json (weight {args.weight})")


if __name__ == "__main__":
    main()
