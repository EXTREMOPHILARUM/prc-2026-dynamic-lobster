"""Leave-one-month-out test of the day-offset gate, layered on the v7 mechanism model.

Paired against the frozen mechanism OOF predictions, so the delta is attributable.

    uv run python scripts/cross_validate_dayoffset.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import config, data, dayoffset, evaluation, features


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, _ = features.build(raw)
    y = raw[config.TARGET].astype("float64").to_numpy()
    month = evaluation.movement_month(raw)
    label = dayoffset.day_offset_label(y)
    bands = dayoffset.band(raw, X)

    base = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "blended"]],
        on=config.ID_COL, how="left")["blended"].to_numpy()
    if not np.isfinite(base).all():
        raise SystemExit("mechanism OOF predictions incomplete")

    print(f"day-offset rows: {label.sum()} of {len(y):,}")
    print("\nband populations and rates (full data, for orientation only):")
    orient = pd.DataFrame({"band": bands, "y": label})
    print(orient[orient.band != "none"].groupby("band").agg(
        rows=("y", "size"), positives=("y", "sum"),
        rate_pct=("y", lambda s: 100 * s.mean())).round(3).to_string())

    prob = np.zeros(len(y))
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        rates = dayoffset.fit_rates(bands[~valid], label[~valid])
        prob[valid] = dayoffset.probability(bands[valid], rates)
    new = dayoffset.apply(base, prob)

    sse_old = float(((y - base) ** 2).sum())
    sse_new = float(((y - new) ** 2).sum())
    print(f"\nOOF RMSE  mechanism {np.sqrt(sse_old/len(y)):.2f}  ->  +day-offset "
          f"{np.sqrt(sse_new/len(y)):.2f}   SSE {100*(sse_new-sse_old)/sse_old:+.2f}%")
    print(f"expected positives from gate {prob.sum():.2f} vs actual {label.sum()}")

    rows = []
    for name, m in [("all", np.ones(len(y), bool)), ("day-offset rows", label),
                    ("gated rows (p>0)", prob > 0), ("everything else", (prob == 0) & ~label)]:
        if not m.sum():
            continue
        o = float(((y[m] - base[m]) ** 2).sum()); n = float(((y[m] - new[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()),
                     "rmse_old": np.sqrt(o/m.sum()), "rmse_new": np.sqrt(n/m.sum()),
                     "sse_delta": n - o})
    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False))

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y[v] - base[v]) ** 2).sum()); n = float(((y[v] - new[v]) ** 2).sum())
        pm.append({"month": m, "rmse_old": np.sqrt(o/v.sum()), "rmse_new": np.sqrt(n/v.sum()),
                   "positives": int(label[v].sum()), "worse": bool(n > o * 1.001)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(2).to_string(index=False))

    jan = pmd.loc[pmd.month == 1]; jul = pmd.loc[pmd.month == 7]
    print("\n--- acceptance gate ---")
    print(f"  total SSE lower              : {sse_new < sse_old}")
    print(f"  January not worse            : {not bool(jan.worse.iloc[0])}")
    print(f"  July not worse               : {not bool(jul.worse.iloc[0])}")
    print(f"  months materially worse      : {int(pmd.worse.sum())}")

    (config.DATA_INTERIM / "dayoffset_oof_metrics.json").write_text(json.dumps({
        "rmse_old": float(np.sqrt(sse_old/len(y))), "rmse_new": float(np.sqrt(sse_new/len(y))),
        "per_month": pmd.to_dict("records"),
    }, indent=2, default=float))
    print("wrote dayoffset_oof_metrics.json")


if __name__ == "__main__":
    main()
