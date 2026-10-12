"""Trace the stored v29 OOF chain layer by layer, without fitting a model.

The report is deliberately in SSE, which is additive across disjoint regimes.  It
shows where each deployed layer removes error and, more importantly, where it leaves
error for the next layer to address.  All predictions are the stored leave-one-month-out
components used by ``analyse_error_anatomy.py``.

    uv run python -u scripts/analyse_v29_layers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, lirfmoe, matrix, mechanism


def rmse(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def main() -> None:
    raw, X, _ = matrix.build(
        "train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True
    )
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    scored = np.isin(month, (1, 7))
    ids = raw[[config.ID_COL]]

    def col(filename: str, column: str) -> np.ndarray:
        frame = pd.read_parquet(config.DATA_INTERIM / filename)
        return ids.merge(frame, on=config.ID_COL, how="left")[column].to_numpy(dtype="float64")

    v21 = col("v21_oof_predictions.parquet", "prediction")
    d8 = col("catgpu_d8_oof_predictions.parquet", "member")
    lobt = col("catgpu_lobt_oof_predictions.parquet", "member")
    mixture = col("lirf_unmatched_moe_oof.parquet", "mixture")
    stack = col("stacker_mid_800_oof.parquet", "stack")
    stack_ref = col("stacker_mid_800_oof.parquet", "ref")

    members_ok = np.isfinite(d8) & np.isfinite(lobt)
    after_members_raw = v21.copy()
    after_members_raw[members_ok] += 0.3 * (d8[members_ok] - v21[members_ok])
    after_members_raw[members_ok] += 0.3 * (lobt[members_ok] - v21[members_ok])
    after_members = np.clip(lirfbands.lobt_clip(after_members_raw, X), 0, None)
    moe_rows = lirfmoe.target_rows(X, 8.0)
    after_moe_raw = after_members_raw.copy()
    after_moe_raw[moe_rows] = mixture[moe_rows]
    after_moe = np.clip(lirfbands.lobt_clip(after_moe_raw, X), 0, None)
    final = np.clip(
        lirfbands.lobt_clip(after_moe_raw + 0.5 * (stack - stack_ref), X), 0, None
    )

    anchor = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    airport = raw["ADEP_mvt"].to_numpy()
    schedule_proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    early = anchor - y < -600
    late = anchor - y > 600
    lirf = airport == lirfbands.AIRPORT
    blocks = [
        ("all scored", scored),
        ("matched", scored & ~missing),
        ("matched early", scored & ~missing & early),
        ("matched agree", scored & ~missing & ~early & ~late),
        ("matched late", scored & ~missing & late),
        ("non-LIRF no-record", scored & missing & ~lirf),
        ("LIRF no-record <=8h", scored & missing & lirf & (schedule_proxy <= 8 * 3600)),
        ("LIRF no-record >8h", scored & missing & lirf & (schedule_proxy > 8 * 3600)),
    ]
    layers = [("v21 reference", v21), ("+ clock members", after_members), ("+ Rome MoE", after_moe), ("+ stacker = v29", final)]

    print("v29 stored-OOF layer trace; all masks are Jan/Jul 2025 and include every target")
    print("\nAbsolute SSE / RMSE by layer")
    print(f"{'block':26s} {'n':>8s}" + "".join(f" {name:>25s}" for name, _ in layers))
    print("-" * (36 + 26 * len(layers)))
    for name, mask in blocks:
        fields = []
        for _, prediction in layers:
            sse = float(((y[mask] - prediction[mask]) ** 2).sum())
            fields.append(f"{sse / 1e6:8.0f}M/{rmse(y, prediction, mask):6.1f}")
        print(f"{name:26s} {int(mask.sum()):8,}" + "".join(f" {field:>25s}" for field in fields))

    print("\nIncremental SSE removed at each layer (positive is better; values do not add across rows a later layer replaces)")
    print(f"{'block':26s} {'members':>14s} {'Rome MoE':>14s} {'stacker':>14s}")
    print("-" * 72)
    for name, mask in blocks:
        values = []
        for before, after in ((v21, after_members), (after_members, after_moe), (after_moe, final)):
            saved = float(((y[mask] - before[mask]) ** 2).sum() - ((y[mask] - after[mask]) ** 2).sum())
            values.append(saved / 1e6)
        print(f"{name:26s}" + "".join(f" {value:13.0f}M" for value in values))

    final_sse = float(((y[scored] - final[scored]) ** 2).sum())
    early_sse = float(((y[scored & ~missing & early] - final[scored & ~missing & early]) ** 2).sum())
    print(
        f"\nRemaining v29 early-anchor block: {early_sse / 1e6:,.0f}M / {final_sse / 1e6:,.0f}M"
        f" = {100 * early_sse / final_sse:.1f}% of scored SSE."
    )


if __name__ == "__main__":
    main()
