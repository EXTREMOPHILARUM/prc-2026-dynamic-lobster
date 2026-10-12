"""Blend several stored OOF members into the v21 OOF on matched rows, weights fitted by
least squares on the ten non-scored months, LOBT bound re-applied, scored on Jan+Jul.

    uv run python scripts/blend_members.py --members catgpu_d8 catgpu_d10 catgpu_sched
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, matrix


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--members", nargs="+", required=True); ap.add_argument("--baseline", default="v21_oof_predictions.parquet"); args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy(); sc = np.isin(month, (1, 7))
    ids = raw[[config.ID_COL]]
    def col(f, c): return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy().copy()
    base = col(args.baseline, "prediction"); members = {m: col(f"{m}_oof_predictions.parquet", "member") for m in args.members}
    ok = np.all([np.isfinite(v) for v in members.values()], axis=0)
    fit = ok & ~sc
    D = np.column_stack([members[m][fit] - base[fit] for m in args.members]); r = y[fit] - base[fit]
    w, *_ = np.linalg.lstsq(D, r, rcond=None); w = np.clip(w, 0, 1)
    out = {"weights": dict(zip(args.members, map(float, w), strict=True))}
    def blend(weights):
        p = base.copy()
        for m, wi in zip(args.members, weights, strict=True):
            p[ok] += wi * (members[m][ok] - base[ok])
        return np.clip(lirfbands.lobt_clip(p, X), 0, None)
    for name, weights in (("lstsq", w), ("equal_0.5_total", np.full(len(args.members), 0.5 / len(args.members))), ("equal_0.6_total", np.full(len(args.members), 0.6 / len(args.members)))):
        p = blend(weights); bym = [rmse(y[month == m], p[month == m]) - rmse(y[month == m], base[month == m]) for m in range(1, 13)]
        ys, bs, ns = y[sc], base[sc], p[sc]; order = np.argsort((ys - ns) ** 2 - (ys - bs) ** 2); rob = []
        for k in (1, 10, 50):
            keep = np.ones(len(ys), bool); keep[order[:k]] = False; rob.append(round(rmse(ys[keep], ns[keep]) - rmse(ys[keep], bs[keep]), 2))
        out[name] = {"weights": [round(float(x), 3) for x in weights], "jan_jul": rmse(y[sc], p[sc]), "jan": rmse(y[month == 1], p[month == 1]), "jul": rmse(y[month == 7], p[month == 7]), "all": rmse(y, p), "months_improved": int(sum(v < 0 for v in bym)), "robustness": rob}
        print(f"{name:16s} w={out[name]['weights']} Jan+Jul {rmse(y[sc], base[sc]):.3f} -> {out[name]['jan_jul']:.3f} | Jan {out[name]['jan']:.2f} Jul {out[name]['jul']:.2f} | 12m {out[name]['all']:.3f} | months {out[name]['months_improved']}/12 | rob {rob}")
    (config.DATA_INTERIM / "blend_members_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
