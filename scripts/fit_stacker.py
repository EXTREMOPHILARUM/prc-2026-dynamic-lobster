"""Fit the second-stage stacker on all 12 months of stored component OOFs (matched rows).

    uv run python scripts/fit_stacker.py --stem stacker_mid --inner mechanism_v27 --out mechanism_v28
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, lirfmoe, matrix, stacker


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem", default="stacker_mid"); ap.add_argument("--inner", default="mechanism_v27"); ap.add_argument("--out", default="mechanism_v28")
    ap.add_argument("--rounds", type=int, default=stacker.ROUNDS)
    ap.add_argument("--members", nargs="+", default=["catgpu_d8"]); ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--stacker-weight", type=float, default=0.5); args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy(); ids = raw[[config.ID_COL]]

    def col(f, c):
        return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy(dtype="float64").copy()

    base = col("v21_oof_predictions.parquet", "prediction"); ordinary = col("v21_oof_predictions.parquet", "ordinary"); gate = col("v21_oof_predictions.parquet", "gate_new")
    mem = {m: col(f"{m}_oof_predictions.parquet", "member") for m in args.members}
    wts = args.weights or [0.5 / len(args.members)] * len(args.members)
    mix = col("lirf_unmatched_moe_oof.parquet", "mixture"); lobt_x = col("lobt_expert_oof.parquet", "expert")
    tgt = lirfmoe.target_rows(X, 8.0); ok = np.all([np.isfinite(v) for v in mem.values()], axis=0)
    ref = base.copy()
    for m, wi in zip(args.members, wts, strict=True):
        ref[ok] += wi * (mem[m][ok] - base[ok])
    ref[tgt] = mix[tgt]; ref = np.clip(lirfbands.lobt_clip(ref, X), 0, None)
    cat = np.mean([mem[m] for m in args.members], axis=0)
    matched = X["aobt_missing"].to_numpy() == 0
    F = stacker.frame(ref, base, np.where(ok, cat, base), ordinary, gate, lobt_x, X)
    model = stacker.fit(F.loc[matched], (y - ref)[matched], rounds=args.rounds)
    fitted = ref.copy(); fitted[matched] += model.predict(F.loc[matched])
    sc = np.isin(month, (1, 7))
    print(f"in-sample Jan+Jul {np.sqrt(np.mean((y[sc] - ref[sc]) ** 2)):.3f} -> {np.sqrt(np.mean((y[sc] - fitted[sc]) ** 2)):.3f} (in-sample, not a fold number)")
    stacker.save(model, config.MODELS / args.stem, {"fit_rows": int(matched.sum()), "oof_sources": [f"{m}_oof_predictions" for m in args.members] + ["v21_oof_predictions", "lirf_unmatched_moe_oof", "lobt_expert_oof"]})
    meta = json.loads((config.MODELS / f"{args.inner}.json").read_text()); meta["stacker_stem"] = args.stem
    meta["stacker_weight"] = args.stacker_weight
    meta["stacker_member_mean"] = "all" if len(args.members) > 1 else "aobt"
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved models/{args.stem}.txt and models/{args.out}.json")


if __name__ == "__main__":
    main()
