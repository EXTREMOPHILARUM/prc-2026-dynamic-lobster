"""Assemble seed averages from the per-seed OOF component files and score them.

Paired against the frozen baseline data/interim/sequencing_oof_predictions.parquet.
Averaging is done on the COMPONENTS (base, tail probability, tail excess, fallback gate)
and the mixture is formed once, which is what the production path would do.

    uv run python scripts/analyze_seedavg.py --tag r1 --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, evaluation, matrix, mechanism

OUT_DIR = config.DATA_INTERIM / "seedavg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--baseline", default="sequencing_oof_predictions.parquet")
    args = ap.parse_args()

    raw, _X, _cats = matrix.build()
    y = raw[config.TARGET].astype("float64").to_numpy()
    month = evaluation.movement_month(raw).to_numpy()
    tail = y > 3600
    base_pred = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline)[[config.ID_COL, "prediction"]],
        on=config.ID_COL, how="left")["prediction"].to_numpy()

    frames = []
    for seed in args.seeds:
        path = OUT_DIR / f"oof_{args.tag}_seed{seed}.parquet"
        frame = pd.read_parquet(path)
        if not (frame[config.ID_COL].to_numpy() == raw[config.ID_COL].to_numpy()).all():
            raise SystemExit(f"{path.name}: row order differs from the matrix")
        if not np.isfinite(frame["base"].to_numpy()).all():
            raise SystemExit(f"{path.name}: incomplete (some folds missing)")
        frames.append(frame)

    proxy = frames[0]["proxy"].to_numpy(dtype="float64")
    tail_scale = frames[0]["tail_scale"].to_numpy(dtype="float64")

    def blend_of(subset: list[pd.DataFrame]) -> np.ndarray:
        b = np.mean([f["base"].to_numpy() for f in subset], axis=0)
        p = np.mean([f["tail_probability"].to_numpy() for f in subset], axis=0)
        e = np.mean([f["tail_excess"].to_numpy() for f in subset], axis=0)
        g = np.mean([f["gate_prob"].to_numpy() for f in subset], axis=0)
        ordinary = np.clip(b + tail_scale * p * e, 0, None)
        return mechanism.blend(g, proxy, ordinary)

    def rmse(pred: np.ndarray, mask: np.ndarray | None = None) -> float:
        m = np.ones(len(y), bool) if mask is None else mask
        return float(np.sqrt(np.mean((y[m] - pred[m]) ** 2)))

    sse_base = float(((y - base_pred) ** 2).sum())
    print(f"baseline (frozen)  RMSE {rmse(base_pred):.4f}  "
          f"bulk {rmse(base_pred, ~tail):.2f}  tail {rmse(base_pred, tail):.2f}")

    rows = []
    preds = {}
    for k in range(1, len(frames) + 1):
        pred = blend_of(frames[:k])
        preds[k] = pred
        sse = float(((y - pred) ** 2).sum())
        rows.append({
            "seeds": k,
            "rmse": rmse(pred),
            "d_rmse": rmse(pred) - rmse(base_pred),
            "marginal": np.nan if k == 1 else rmse(pred) - rows[-1]["rmse"],
            "bulk": rmse(pred, ~tail),
            "tail": rmse(pred, tail),
            "sse_pct": 100 * (sse - sse_base) / sse_base,
            "jan": rmse(pred, month == 1),
            "jul": rmse(pred, month == 7),
            "months_better": int(sum(
                ((y[month == m] - pred[month == m]) ** 2).sum()
                < ((y[month == m] - base_pred[month == m]) ** 2).sum()
                for m in range(1, 13))),
        })
    table = pd.DataFrame(rows)
    print("\n" + table.round(4).to_string(index=False))

    best = len(frames)
    pred = preds[best]
    pm = []
    for m in range(1, 13):
        v = month == m
        o = float(((y[v] - base_pred[v]) ** 2).sum())
        n = float(((y[v] - pred[v]) ** 2).sum())
        pm.append({"month": m, "rmse_old": np.sqrt(o / v.sum()),
                   "rmse_new": np.sqrt(n / v.sum()), "better": bool(n < o)})
    print(f"\nper month, {best} seeds:")
    print(pd.DataFrame(pm).round(3).to_string(index=False))

    (config.DATA_INTERIM / f"seedavg_{args.tag}_metrics.json").write_text(
        json.dumps({"baseline_rmse": rmse(base_pred), "cumulative": rows,
                    "per_month": pm, "seeds": args.seeds, "tag": args.tag},
                   indent=2, default=float))
    print(f"\nwrote seedavg_{args.tag}_metrics.json")


if __name__ == "__main__":
    main()
