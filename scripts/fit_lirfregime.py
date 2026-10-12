"""Fit the LIRF schedule-identity gate on all 2025 and write the v18 wrapper.

    uv run python scripts/fit_lirfregime.py --inner mechanism_v17 --out mechanism_v18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prc import arrivals, config, evaluation, lirfregime, matrix


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--inner", default="mechanism_v17"); ap.add_argument("--out", default="mechanism_v18")
    ap.add_argument("--alphas", default=None, help="json file with an 'alphas_oof' dict to replace lobt_alphas")
    args = ap.parse_args()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    model = lirfregime.fit(raw, X, raw[config.TARGET].to_numpy(dtype="float64"), cats, evaluation.movement_month(raw).to_numpy())
    lirfregime.save(model, config.MODELS / "lirf_regime")
    meta = json.loads((config.MODELS / f"{args.inner}.json").read_text())
    meta["regime_stem"] = "lirf_regime"
    if args.alphas:
        meta["lobt_alphas"] = json.loads(Path(args.alphas).read_text())["alphas_oof"]
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models/lirf_regime.* and models/{args.out}.json (base rate {model['base_rate']:.3f}, {len(model['columns'])} gate columns)")


if __name__ == "__main__":
    main()
