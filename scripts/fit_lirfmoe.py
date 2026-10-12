"""Fit the LIRF unmatched mixture on all 2025 and write the v24 wrapper.

    uv run python scripts/fit_lirfmoe.py --inner mechanism_v23 --out mechanism_v24
"""
from __future__ import annotations

import argparse
import json

from prc import arrivals, config, evaluation, lirfmoe, matrix


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--inner", default="mechanism_v23"); ap.add_argument("--out", default="mechanism_v24"); ap.add_argument("--max-gap-h", type=float, default=None); ap.add_argument("--stem", default="lirf_moe"); ap.add_argument("--regressor", default="lightgbm"); ap.add_argument("--task-type", default="GPU"); args = ap.parse_args()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    meta = json.loads((config.MODELS / f"{args.inner}.json").read_text())
    enc = json.loads((config.MODELS / f"{meta['inner_stem']}.json").read_text())["target_encoding"]
    model = lirfmoe.fit(raw, X, cats, raw[config.TARGET].to_numpy(dtype="float64"), evaluation.movement_month(raw).to_numpy(), enc, max_gap_h=args.max_gap_h, regressor=args.regressor, task_type=args.task_type)
    lirfmoe.save(model, config.MODELS / args.stem)
    meta["lirf_moe_stem"] = args.stem
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models/{args.stem}.* and models/{args.out}.json ({len(model['columns'])} columns)")


if __name__ == "__main__":
    main()
