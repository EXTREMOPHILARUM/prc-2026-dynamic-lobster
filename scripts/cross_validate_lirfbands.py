"""Leave-one-month-out evaluation of the LIRF band rule and the LOBT clip.

The rule has no trained components beyond a band table, so each fold fits the table on
the other eleven months and applies it to the stored out-of-fold predictions of the
current model (default: `noid_arr`, the v14 configuration).

    uv run python scripts/cross_validate_lirfbands.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, matrix


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="noid_arr_oof_predictions.parquet")
    ap.add_argument("--tag", default="lirfbands")
    args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    base = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline), on=config.ID_COL, how="left"
    )["prediction"].to_numpy()
    if not np.isfinite(base).all():
        raise SystemExit("baseline did not join onto every row")
    banded = base.copy()
    for m in range(1, 13):
        table = lirfbands.fit(X.loc[month != m], y[month != m])
        v = month == m
        banded[v] = lirfbands.apply(base[v], X.loc[v], table)
    clipped = lirfbands.lobt_clip(banded, X)
    scored = np.isin(month, (1, 7))
    rep = {}
    for name, mask in [("scored_jan_jul", scored), ("all_months", np.ones(len(y), bool)),
                       ("other_months", ~scored), ("jan", month == 1), ("jul", month == 7)]:
        rep[name] = {"base": rmse(y[mask], base[mask]), "bands": rmse(y[mask], banded[mask]),
                     "bands+lobt": rmse(y[mask], clipped[mask])}
    rep["by_month"] = {str(m): rmse(y[month == m], clipped[month == m]) - rmse(y[month == m], base[month == m])
                       for m in range(1, 13)}
    helped = ((y - clipped) ** 2 - (y - base) ** 2)[scored]
    order = np.argsort(helped)
    ys, bs, cs = y[scored], base[scored], clipped[scored]
    rep["robustness_scored"] = {}
    for k in (1, 5, 10, 50):
        keep = np.ones(len(ys), bool); keep[order[:k]] = False
        rep["robustness_scored"][f"drop_{k}"] = rmse(ys[keep], cs[keep]) - rmse(ys[keep], bs[keep])
    rep["rows_in_scope"] = int(lirfbands.scope(X).sum())
    rep["lobt_violations_before"] = int((np.abs(base - X[lirfbands.LOBT_COL].to_numpy()) > lirfbands.LOBT_BOUND)[
        np.isfinite(X[lirfbands.LOBT_COL].to_numpy())].sum())
    (config.DATA_INTERIM / f"{args.tag}_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": clipped}).to_parquet(
        config.DATA_INTERIM / f"{args.tag}_oof_predictions.parquet", index=False)
    for k in ("scored_jan_jul", "jan", "jul", "all_months", "other_months"):
        r = rep[k]; print(f"{k:15s} {r['base']:.3f} -> bands {r['bands']:.3f} -> +lobt {r['bands+lobt']:.3f}")
    print("by month delta", {k: round(v, 2) for k, v in rep["by_month"].items()})
    print("robustness", {k: round(v, 3) for k, v in rep["robustness_scored"].items()})
    print("rows in scope", rep["rows_in_scope"], "lobt violations before", rep["lobt_violations_before"])


if __name__ == "__main__":
    main()
