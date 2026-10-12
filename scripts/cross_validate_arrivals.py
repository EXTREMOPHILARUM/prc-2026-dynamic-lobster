"""Paired 12-fold A/B of the arrival-pressure family, built from production code.

Runs the real `matrix.build(use_arrivals=True)` path rather than a throwaway script, so
what is measured here is what a submission would use.

    OMP_NUM_THREADS=2 uv run python -u scripts/cross_validate_arrivals.py --variant six

`--variant fixed` drops the two `arr_during_taxi_*` features. Their window *length* is
`MVT - AOBT`, the taxi proxy the model already carries, so they may be a nonlinear
re-encoding of a signal already present; the fixed-window variant is the control that
says whether the gain survives without them.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix, mechanism

VARIANTS = {
    "six": arrivals.FEATURES,
    "fixed": arrivals.FIXED_WINDOW_FEATURES,
    "none": [],
}


def _oof(name: str, ids: pd.Series, column: str = "prediction") -> np.ndarray:
    frame = pd.read_parquet(config.DATA_INTERIM / f"{name}.parquet")
    return ids.to_frame().merge(
        frame[[config.ID_COL, column]], on=config.ID_COL, how="left", validate="one_to_one"
    )[column].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="six")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    keep = VARIANTS[args.variant]
    tag = args.tag or args.variant
    print(f"variant {args.variant}: {len(keep)} arrival features", flush=True)

    raw, X, cats = matrix.build(use_arrivals=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if c not in keep])
    print(f"matrix {X.shape}", flush=True)

    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    baseline = _oof("sequencing_oof_predictions", raw[config.ID_COL])
    gate_p = _oof("mechanism_oof_predictions", raw[config.ID_COL], "probability")
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    pred = np.full(len(raw), np.nan)
    for valid_month in range(1, 13):
        valid = (month == valid_month).to_numpy()
        print(f"month {valid_month:02d}: train {(~valid).sum():,} valid {valid.sum():,}", flush=True)
        models = decomposition.fit_fixed(X.loc[~valid], y.loc[~valid], cats, rounds=rounds)
        ordinary = decomposition.predict(models, X.loc[valid])
        pred[valid] = mechanism.blend(gate_p[valid], proxy[valid], ordinary)
    if not np.isfinite(pred).all():
        raise RuntimeError("OOF predictions are incomplete")

    tail = y_np > 3600
    missing = raw["AOBT_3_flt"].isna().to_numpy()
    lirf = raw["ADEP_mvt"].eq("LIRF").to_numpy()
    segments = {
        "all": np.ones(len(raw), dtype=bool),
        "bulk": ~tail,
        "tail": tail,
        "AOBT present": ~missing,
        "AOBT missing": missing,
        "LIRF": lirf,
    }
    rows = []
    for name, mask in segments.items():
        old = float(np.square(y_np[mask] - baseline[mask]).sum())
        new = float(np.square(y_np[mask] - pred[mask]).sum())
        rows.append({
            "segment": name,
            "n": int(mask.sum()),
            "rmse_old": float(np.sqrt(old / mask.sum())),
            "rmse_new": float(np.sqrt(new / mask.sum())),
            "sse_delta_pct": 100 * (new - old) / old,
        })
    segment_frame = pd.DataFrame(rows)
    print("\nSEGMENTS", flush=True)
    print(segment_frame.round(4).to_string(index=False), flush=True)

    per_month = []
    for valid_month in range(1, 13):
        mask = (month == valid_month).to_numpy()
        old = float(np.square(y_np[mask] - baseline[mask]).sum())
        new = float(np.square(y_np[mask] - pred[mask]).sum())
        per_month.append({
            "month": valid_month,
            "n": int(mask.sum()),
            "rmse_old": float(np.sqrt(old / mask.sum())),
            "rmse_new": float(np.sqrt(new / mask.sum())),
            "better": bool(new < old),
        })
    month_frame = pd.DataFrame(per_month)
    print("\nMONTHS", flush=True)
    print(month_frame.round(4).to_string(index=False), flush=True)
    print(f"months improved: {int(month_frame.better.sum())}/12", flush=True)

    out = config.DATA_INTERIM / "arrival_review"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        out / f"arrivals_{tag}_oof_predictions.parquet", index=False)
    (out / f"arrivals_{tag}_oof_metrics.json").write_text(json.dumps({
        "variant": args.variant,
        "features": keep,
        "segments": segment_frame.to_dict("records"),
        "per_month": month_frame.to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
