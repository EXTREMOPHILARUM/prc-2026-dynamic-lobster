"""Paired 12-fold A/B of the neighbour-observation features on the v11 model.

Rebuilds the two features section 3 of the experiment log rejected on a Pearson residual
screen — "neighbour departure taxi proxy (30m median)" (+0.020) and "arrival taxi-in
aggregates" (+0.018) — and measures them the only way that settles the question: a paired
twelve-fold, whole months held out, every holdout row retained, the mechanism gate frozen
from the existing OOF so the delta is attributable to the features alone.

    uv run python -u scripts/experiment_neighbour.py --features both
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, data, decomposition, evaluation, matrix, mechanism, neighbour

SETS = {
    "neighbour": neighbour.NEIGHBOUR_FEATURES,
    "arrival": neighbour.ARRIVAL_FEATURES,
    "both": neighbour.FEATURES,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=sorted(SETS), default="both")
    ap.add_argument("--baseline", default="sequencing_oof_predictions.parquet")
    ap.add_argument("--baseline-col", default="prediction")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_stem = args.out or f"neighbour_{args.features}_oof"
    extra = SETS[args.features]

    for params in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS,
                   decomposition.SEVERITY_PARAMS):
        params["num_threads"] = args.threads

    raw, X, cats = matrix.build("train")
    movements = data.load_training(
        columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                 "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"])
    feats = neighbour.add(raw[[config.ID_COL, "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                               "AOBT_3_flt", "EOBT_1_flt"]], movements)
    del movements
    X_new = X.join(feats[extra])
    missing = [c for c in extra if c not in X_new.columns]
    if missing:
        raise SystemExit(f"features missing from X: {missing}")

    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline)[[config.ID_COL, args.baseline_col]],
        on=config.ID_COL, how="left")[args.baseline_col].to_numpy()
    if not np.isfinite(baseline).all() or not np.isfinite(gate_p).all():
        raise SystemExit("baseline or gate probabilities did not join onto every row")
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    pred = np.full(len(raw), np.nan)
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        models = decomposition.fit_fixed(X_new.loc[~valid], y.loc[~valid], cats, rounds=rounds)
        pred[valid] = mechanism.blend(
            gate_p[valid], proxy[valid], decomposition.predict(models, X_new.loc[valid]))
        print(f"  month {m:02d} done", flush=True)

    sse_o = float(((y_np - baseline) ** 2).sum())
    sse_n = float(((y_np - pred) ** 2).sum())
    print(f"\n[{args.features}] OOF RMSE  baseline {np.sqrt(sse_o/len(y_np)):.4f}  "
          f"new {np.sqrt(sse_n/len(y_np)):.4f}   SSE {100*(sse_n-sse_o)/sse_o:+.3f}%")

    tail = y_np > 3600
    aobt = X["taxi_proxy_AOBT_3_flt"].notna().to_numpy()
    rows = []
    for name, m in [("all", np.ones(len(y_np), bool)), ("bulk", ~tail), ("tail", tail),
                    ("aobt present", aobt), ("aobt missing", ~aobt)]:
        o = float(((y_np[m] - baseline[m]) ** 2).sum()); n = float(((y_np[m] - pred[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()), "rmse_old": np.sqrt(o/m.sum()),
                     "rmse_new": np.sqrt(n/m.sum()), "sse_delta_pct": 100*(n-o)/o})
    segments = pd.DataFrame(rows)
    print("\n" + segments.round(3).to_string(index=False))

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y_np[v] - baseline[v]) ** 2).sum()); n = float(((y_np[v] - pred[v]) ** 2).sum())
        t = v & tail
        pm.append({"month": m, "rmse_old": np.sqrt(o/v.sum()), "rmse_new": np.sqrt(n/v.sum()),
                   "tail_rmse_old": np.sqrt(((y_np[t]-baseline[t])**2).sum()/max(t.sum(), 1)),
                   "tail_rmse_new": np.sqrt(((y_np[t]-pred[t])**2).sum()/max(t.sum(), 1)),
                   "better": bool(n < o)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(3).to_string(index=False))
    print(f"\nmonths improved: {int(pmd.better.sum())}/12")

    config.DATA_INTERIM.joinpath(f"{out_stem}_metrics.json").write_text(json.dumps({
        "features": extra,
        "baseline": args.baseline,
        "rmse_baseline": float(np.sqrt(sse_o/len(y_np))),
        "rmse_new": float(np.sqrt(sse_n/len(y_np))),
        "sse_delta_pct": 100*(sse_n-sse_o)/sse_o,
        "months_improved": int(pmd.better.sum()),
        "segments": segments.to_dict("records"),
        "per_month": pmd.to_dict("records"),
    }, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        config.DATA_INTERIM / f"{out_stem}_predictions.parquet", index=False)
    print(f"wrote {out_stem} artifacts")


if __name__ == "__main__":
    main()
