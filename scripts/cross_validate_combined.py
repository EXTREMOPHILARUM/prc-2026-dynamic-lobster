"""Joint 12-fold of every surviving feature family, against the frozen v11 baseline.

Four families were each measured in isolation against the same baseline. They cannot be
summed: `arrivals` and the arrival half of `neighbour` both read landing traffic, and
`cfgstate` reads runway occupancy that the sequencing features already partly carry. The
only number that can support a submission is a single joint run, and the useful
diagnostic is the *overlap* - the combined gain against the sum of the isolated gains.

Families:

* ``arrivals``  - `arrivals.FIXED_WINDOW_FEATURES`, the four AOBT-anchored windows. The
  two taxi-length windows are dropped: the owning agent measured them as adding nothing
  over the fixed windows while re-encoding the anchor.
* ``neighbour`` - `neighbour.FEATURES`, neighbour departure taxi plus arrival taxi-in.
* ``cfgstate``  - the runway configuration state features.
* ``storm``     - the thunderstorm and shower codes, already merged into `weather.py`
  and therefore present in the matrix; excluding the family drops them again.

Only the ordinary component is refitted; the schedule-fallback gate is reused frozen from
the mechanism OOF, so every delta is attributable to the features.

    uv run python -u scripts/cross_validate_combined.py
    uv run python -u scripts/cross_validate_combined.py --drop neighbour   # ablation
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix, mechanism

#: measured in isolation by the owning agents, each against this same baseline
ISOLATED_GAIN = {
    "arrivals": -0.563,
    "neighbour": -0.530,
    "cfgstate": -0.326,
    "storm": -0.085,
}

STORM_FEATURES = ["wx_thunder", "wx_shower", "wx_hail", "wx_thunder_hours_3h"]

#: arrivals columns that the `arrivals` family deliberately excludes
TAXI_WINDOW_FEATURES = [c for c in arrivals.FEATURES if "during_taxi" in c]


def load_family_columns(families: set[str], X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return X restricted to the requested families, plus the columns each contributed."""
    used: dict[str, list[str]] = {}
    frame = X

    # arrivals: matrix.build supplies all six; keep the four fixed windows, or none
    if "arrivals" in families:
        used["arrivals"] = [c for c in arrivals.FIXED_WINDOW_FEATURES if c in frame.columns]
        frame = frame.drop(columns=[c for c in TAXI_WINDOW_FEATURES if c in frame.columns])
    else:
        frame = frame.drop(columns=[c for c in arrivals.FEATURES if c in frame.columns])

    # storm: already merged into weather.py, so excluding means dropping the columns
    if "storm" in families:
        used["storm"] = [c for c in STORM_FEATURES if c in frame.columns]
    else:
        frame = frame.drop(columns=[c for c in STORM_FEATURES if c in frame.columns])

    for family, path in (("neighbour", "neighbour_features.parquet"),
                         ("cfgstate", "cfgstate_features.parquet")):
        if family not in families:
            continue
        side = pd.read_parquet(config.DATA_INTERIM / path)
        columns = [c for c in side.columns if c != config.ID_COL]
        merged = frame.join(
            side.set_index(config.ID_COL).reindex(X["__id__"].to_numpy()).reset_index(drop=True)
        )
        if merged[columns].isna().all().any():
            raise SystemExit(f"{family}: some columns joined entirely empty")
        frame = merged
        used[family] = columns
    return frame, used


def segments(y: np.ndarray, base: np.ndarray, pred: np.ndarray, month: np.ndarray,
             raw: pd.DataFrame) -> dict:
    tail = y > 3600
    scored = np.isin(month, (1, 7))

    def one(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        old = float(((y[mask] - base[mask]) ** 2).sum())
        new = float(((y[mask] - pred[mask]) ** 2).sum())
        return {"n": n, "rmse_base": float(np.sqrt(old / n)),
                "rmse_new": float(np.sqrt(new / n)),
                "delta": float(np.sqrt(new / n) - np.sqrt(old / n)),
                "sse_delta_pct": 100 * (new - old) / old}

    out = {
        "scored_jan_jul": one(scored),
        "all_months": one(np.ones(len(y), dtype=bool)),
        "excluding_inner_months": one(~np.isin(month, (4, 10))),
        "bulk": one(~tail),
        "tail": one(tail),
        "scored_bulk": one(scored & ~tail),
        "scored_tail": one(scored & tail),
        "by_month": {str(m): one(month == m) for m in range(1, 13)},
    }
    out["months_improved"] = sum(
        1 for m in range(1, 13) if out["by_month"][str(m)]["sse_delta_pct"] < 0)
    return out


def leave_out(y: np.ndarray, base: np.ndarray, pred: np.ndarray, mask: np.ndarray,
              drops: tuple[int, ...]) -> dict:
    """Recompute the scored delta after removing the rows the change helped most."""
    y_s, b_s, p_s = y[mask], base[mask], pred[mask]
    helped = ((y_s - p_s) ** 2) - ((y_s - b_s) ** 2)
    order = np.argsort(helped)
    out = {}
    for k in drops:
        keep = np.ones(len(y_s), dtype=bool)
        keep[order[:k]] = False
        old = float(((y_s[keep] - b_s[keep]) ** 2).mean())
        new = float(((y_s[keep] - p_s[keep]) ** 2).mean())
        out[f"drop_{k}"] = float(np.sqrt(new) - np.sqrt(old))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", nargs="*", default=[], choices=sorted(ISOLATED_GAIN),
                    help="families to leave out (for ablations)")
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(ISOLATED_GAIN))
    ap.add_argument("--baseline", default="sequencing_oof_predictions.parquet")
    ap.add_argument("--baseline-col", default="prediction")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    families = set(args.only) if args.only is not None else set(ISOLATED_GAIN) - set(args.drop)
    tag = args.tag or ("combined" if families == set(ISOLATED_GAIN)
                       else "drop_" + "_".join(sorted(args.drop)) if args.drop
                       else "only_" + "_".join(sorted(families)))
    print(f"families: {sorted(families)}   tag: {tag}", flush=True)

    for params in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS,
                   decomposition.SEVERITY_PARAMS):
        params["num_threads"] = args.threads

    start = time.time()
    raw, X, cats = matrix.build(use_arrivals=True)
    X = X.copy()
    X["__id__"] = raw[config.ID_COL].to_numpy()
    X, used = load_family_columns(families, X)
    X = X.drop(columns=["__id__"])
    print(f"matrix {X.shape} in {time.time()-start:.0f}s; "
          f"columns per family { {k: len(v) for k, v in used.items()} }", flush=True)

    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    base = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline)[[config.ID_COL, args.baseline_col]],
        on=config.ID_COL, how="left")[args.baseline_col].to_numpy()
    if not (np.isfinite(base).all() and np.isfinite(gate_p).all()):
        raise SystemExit("baseline or gate probabilities did not join onto every row")
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    pred = np.full(len(raw), np.nan)
    for fold in range(1, 13):
        valid = month == fold
        models = decomposition.fit_fixed(X.loc[~valid], y.loc[~valid], cats, rounds=rounds)
        pred[valid] = mechanism.blend(
            gate_p[valid], proxy[valid], decomposition.predict(models, X.loc[valid]))
        print(f"  fold {fold:02d} done t={time.time()-start:.0f}s", flush=True)
    if not np.isfinite(pred).all():
        raise SystemExit("some rows were never predicted")

    report = segments(y_np, base, pred, month, raw)
    scored = np.isin(month, (1, 7))
    report["robustness_scored"] = leave_out(y_np, base, pred, scored, (1, 10))
    report["families"] = sorted(families)
    report["family_columns"] = used
    report["isolated_sum"] = float(sum(ISOLATED_GAIN[f] for f in families))
    report["combined_delta"] = report["scored_jan_jul"]["delta"]
    report["overlap"] = float(report["combined_delta"] - report["isolated_sum"])

    config.DATA_INTERIM.joinpath(f"{tag}_oof_metrics.json").write_text(
        json.dumps(report, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        config.DATA_INTERIM / f"{tag}_oof_predictions.parquet", index=False)

    s, a = report["scored_jan_jul"], report["all_months"]
    print(f"\nJan+Jul  {s['rmse_base']:.4f} -> {s['rmse_new']:.4f}  ({s['delta']:+.3f}s)")
    print(f"12-month {a['rmse_base']:.4f} -> {a['rmse_new']:.4f}  ({a['delta']:+.3f}s)")
    print(f"months improved {report['months_improved']}/12")
    print(f"sum of isolated {report['isolated_sum']:+.3f}s   combined "
          f"{report['combined_delta']:+.3f}s   overlap {report['overlap']:+.3f}s")
    print(f"robustness (scored): {report['robustness_scored']}")


if __name__ == "__main__":
    main()
