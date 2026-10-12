"""Per-airport specialist models against the global model, paired 12-fold.

The error is extremely concentrated by airport: on the scored months LIRF alone carries
40.4% of squared error. This asks whether an airport with enough rows to support its own
fit does better with one (160,704 rows for LIRF), or whether the global model with
`ADEP_mvt` as a categorical already extracts everything there is.

A specialist refits the four decomposition components on that airport's rows alone, with
each component's rounds early-stopped on the airport's own data. The frozen mechanism gate
is reused from the mechanism OOF, so the delta is attributable to the specialist alone.

Rounds are selected on months 4+10, disjoint from the ranking months January and July.

    uv run python -u scripts/experiment_perairport.py
    uv run python -u scripts/experiment_perairport.py --airports LIRF
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

#: the five airports carrying the most squared error, worst first
DEFAULT_AIRPORTS = ("LIRF", "LFPG", "LTFM", "EGLL", "LSZH")

#: months used to early-stop each specialist's components, disjoint from January and July
INNER_MONTHS = (4, 10)

#: a fold whose training rows hold fewer tail examples than this cannot support a severity
#: model; the fold falls back to the global prediction rather than extrapolating from a
#: handful of rows, which is the failure mode recorded in docs/experiment-log.md 2.4
MIN_TAIL_ROWS = 20


def specialist_oof(
    X: pd.DataFrame,
    y: pd.Series,
    cats: list[str],
    month: np.ndarray,
    airport_col: np.ndarray,
    airports: tuple[str, ...],
    gate_p: np.ndarray,
    proxy: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Return out-of-fold specialist predictions and the rounds chosen per airport."""
    start = time.time()
    prediction = np.full(len(X), np.nan)
    chosen: dict[str, dict[str, int]] = {}

    for name in airports:
        rows = np.flatnonzero(airport_col == name)
        X_a, y_a, month_a = X.iloc[rows], y.iloc[rows], month[rows]
        inner = np.isin(month_a, INNER_MONTHS)
        print(f"[{name}] n={len(rows)} tail={(y_a.to_numpy() > 3600).sum()}", flush=True)

        models = decomposition.fit(
            X_a[~inner], y_a[~inner], cats, X_a[inner], y_a[inner], rounds=1500, stopping=100
        )
        rounds = {k: max(int(m.best_iteration), 20) for k, m in models.items()}
        chosen[name] = rounds
        print(f"[{name}] rounds {rounds} t={time.time()-start:.0f}s", flush=True)

        for fold in range(1, 13):
            held_out = month_a == fold
            if not held_out.any():
                continue
            train = ~held_out
            if (y_a[train].to_numpy() > 3600).sum() < MIN_TAIL_ROWS:
                print(f"[{name}] month {fold}: too few tail rows, using the global model",
                      flush=True)
                prediction[rows[held_out]] = baseline[rows[held_out]]
                continue
            fitted = decomposition.fit_fixed(X_a[train], y_a[train], cats, rounds=rounds)
            ordinary = decomposition.predict(fitted, X_a[held_out])
            index = rows[held_out]
            prediction[index] = mechanism.blend(gate_p[index], proxy[index], ordinary)
        print(f"[{name}] folds done t={time.time()-start:.0f}s", flush=True)

    return prediction, chosen


def _segment(y: np.ndarray, pred: np.ndarray, base: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    old = float(((y[mask] - base[mask]) ** 2).sum())
    new = float(((y[mask] - pred[mask]) ** 2).sum())
    return {
        "n": n,
        "rmse_global": float(np.sqrt(old / n)) if n else float("nan"),
        "rmse_specialist": float(np.sqrt(new / n)) if n else float("nan"),
        "sse_delta_pct": 100 * (new - old) / old if old else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", nargs="*", default=list(DEFAULT_AIRPORTS))
    ap.add_argument("--apply", nargs="*", default=["LIRF"],
                    help="airports whose specialist is used in the saved artifact")
    ap.add_argument("--baseline", default="sequencing_oof_predictions.parquet")
    ap.add_argument("--baseline-col", default="prediction")
    ap.add_argument("--out", default="perairport_lirf_oof")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    for params in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS,
                   decomposition.SEVERITY_PARAMS):
        params["num_threads"] = args.threads

    raw, X, cats = matrix.build()
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    airport_col = raw["ADEP_mvt"].astype("string").to_numpy()
    missing = X[decomposition.ANCHOR_COL].isna().to_numpy()

    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline)[[config.ID_COL, args.baseline_col]],
        on=config.ID_COL, how="left")[args.baseline_col].to_numpy()
    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    if not np.isfinite(baseline).all():
        raise SystemExit("baseline predictions did not join onto every row")

    specialist, rounds = specialist_oof(
        X, y, cats, month, airport_col, tuple(args.airports), gate_p, proxy, baseline
    )
    applied = np.isin(airport_col, args.apply) & np.isfinite(specialist)
    blended = np.where(applied, specialist, baseline)

    tail = y_np > 3600
    scored = np.isin(month, (1, 7))
    lirf = airport_col == "LIRF"
    metrics = {
        "applied_airports": args.apply,
        "screened_airports": args.airports,
        "inner_months": list(INNER_MONTHS),
        "rounds": rounds,
        "overall": {
            "all_months": _segment(y_np, blended, baseline, np.ones(len(y_np), dtype=bool)),
            "scored_jan_jul": _segment(y_np, blended, baseline, scored),
            "excluding_inner_months": _segment(
                y_np, blended, baseline, ~np.isin(month, INNER_MONTHS)),
            "bulk": _segment(y_np, blended, baseline, ~tail),
            "tail": _segment(y_np, blended, baseline, tail),
        },
        "scored_segments": {
            "lirf": _segment(y_np, blended, baseline, scored & lirf),
            "lirf_missing_aobt": _segment(y_np, blended, baseline, scored & lirf & missing),
            "lirf_matched_clock": _segment(y_np, blended, baseline, scored & lirf & ~missing),
        },
        "by_month": {
            str(m): _segment(y_np, blended, baseline, month == m) for m in range(1, 13)
        },
        "by_airport": {
            str(a): _segment(y_np, blended, baseline, airport_col == a)
            for a in sorted(set(airport_col[~pd.isna(airport_col)]))
        },
        "lirf_by_month": {
            str(m): _segment(y_np, blended, baseline, lirf & (month == m)) for m in range(1, 13)
        },
    }
    improved = [m for m in range(1, 13) if metrics["by_month"][str(m)]["sse_delta_pct"] < 0]
    metrics["months_improved"] = {"months": improved, "count": len(improved)}

    config.DATA_INTERIM.joinpath(f"{args.out}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": blended}).to_parquet(
        config.DATA_INTERIM / f"{args.out}_predictions.parquet", index=False)

    overall = metrics["overall"]
    print(f"\nall months   {overall['all_months']['rmse_global']:.3f} -> "
          f"{overall['all_months']['rmse_specialist']:.3f}")
    print(f"Jan+Jul      {overall['scored_jan_jul']['rmse_global']:.3f} -> "
          f"{overall['scored_jan_jul']['rmse_specialist']:.3f}")
    print(f"months improved: {len(improved)}/12")
    print(f"wrote {args.out}_predictions.parquet and {args.out}_metrics.json")


if __name__ == "__main__":
    main()
