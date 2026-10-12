"""Validate a fixed three-seed average of capped bases, freezing all gates/tails.

Uses the exact v13 feature schema. See the architecture experiment protocol.
Completed folds can be resumed only with the same source and input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_architecture import delta_report, fit_model

from prc import config, decomposition, evaluation, matrix, mechanism

SEEDS = (None, 3, 5)


def fit_base(Xt, Xv, yt, cats, rounds, threads, seed):
    params = dict(decomposition.BASE_PARAMS)
    if seed is not None:
        params.update({name: seed for name in (
            "seed", "bagging_seed", "feature_fraction_seed", "data_random_seed", "extra_seed"
        )})
    matched = Xt[decomposition.ANCHOR_COL].notna()
    matched_v = Xv[decomposition.ANCHOR_COL].notna().to_numpy()
    cap = decomposition.TAIL_THRESHOLD
    base = np.empty(len(Xv))
    for name, tm, vm in (("base_matched", matched, matched_v),
                          ("base_missing", ~matched, ~matched_v)):
        target = yt.loc[tm].clip(upper=cap)
        if name == "base_matched":
            target = target - Xt.loc[tm, decomposition.ANCHOR_COL].clip(0, cap)
        model = fit_model(params, Xt.loc[tm], target, cats, rounds[name], threads)
        base[vm] = model.predict(Xv.loc[vm], num_threads=threads)
        if name == "base_matched":
            base[vm] += Xv.loc[vm, decomposition.ANCHOR_COL].clip(0, cap).to_numpy()
    return np.clip(base, 0, cap)


def run_fold(raw, X, cats, rounds, month, threads):
    valid = evaluation.movement_month(raw).eq(month).to_numpy()
    Xt, Xv = X.loc[~valid], X.loc[valid]
    yt = raw.loc[~valid, config.TARGET].astype("float32")
    bases = []
    for seed in SEEDS:
        bases.append(fit_base(Xt, Xv, yt, cats, rounds, threads, seed))
        print(f"month {month:02d}: base seed {seed} fitted", flush=True)

    def fit(params, columns, target, count, mask=None):
        xx = Xt[columns] if mask is None else Xt.loc[mask, columns]
        return fit_model(params, xx, target, [c for c in cats if c in columns], count, threads)

    cols = [c for c in decomposition.TAIL_FEATURES if c in X]
    tail = yt > decomposition.TAIL_THRESHOLD
    gate = fit(decomposition.GATE_PARAMS, cols, tail.astype("int8"), rounds["gate"])
    severity = fit(decomposition.SEVERITY_PARAMS, cols,
                   yt.loc[tail] - decomposition.TAIL_THRESHOLD, rounds["severity"], tail)
    probability = gate.predict(Xv[cols], num_threads=threads)
    excess = np.maximum(severity.predict(Xv[cols], num_threads=threads), 0)
    scale = np.where(Xv.ADEP_mvt.astype("string").eq("LIRF"),
                     decomposition.LIRF_TAIL_SCALE, decomposition.OTHER_TAIL_SCALE)
    cols = mechanism.gate_features(X)
    gate = fit(mechanism.GATE_PARAMS, cols,
               mechanism.fallback_label(raw.loc[~valid]).astype("int8"), 200)
    schedule_p = gate.predict(Xv[cols], num_threads=threads)
    proxy = Xv[mechanism.PROXY_COL].to_numpy(dtype=float)
    correction = scale * probability * excess
    control = mechanism.blend(schedule_p, proxy, bases[0] + correction)
    candidate = mechanism.blend(schedule_p, proxy, np.mean(bases, axis=0) + correction)
    out = pd.DataFrame({
        config.ID_COL: raw.loc[valid, config.ID_COL].to_numpy(),
        config.TARGET: raw.loc[valid, config.TARGET].to_numpy(),
        "month": month, "airport": raw.loc[valid, "ADEP_mvt"].to_numpy(),
        "aobt_missing": Xv[decomposition.ANCHOR_COL].isna().to_numpy(),
        "airport_day": (raw.loc[valid, "ADEP_mvt"].astype(str) + ":" +
                        raw.loc[valid, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")).to_numpy(),
        "control": control, "base_average": candidate,
        "base_default": bases[0], "base_seed3": bases[1], "base_seed5": bases[2],
        "tail_correction": correction, "schedule_probability": schedule_p, "proxy": proxy,
    })
    if not np.isfinite(out[["control", "base_average"]].to_numpy()).all():
        raise ValueError("Non-finite fold predictions")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, nargs="+", default=[2, 5, 9])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(set(args.months)) != len(args.months) or not set(args.months) <= set(range(1, 13)):
        ap.error("months must be unique values from 1 through 12")
    if args.threads < 1:
        ap.error("threads must be positive")
    meta_path = config.MODELS / "decomposition_arr.json"
    meta = json.loads(meta_path.read_text())
    rounds_path = config.DATA_INTERIM / "component_rounds.json"
    rounds = json.loads(rounds_path.read_text())["rounds"]
    paths = [Path(__file__), Path(__file__).with_name("experiment_architecture.py"),
             *sorted((config.ROOT / "src/prc").glob("*.py")), meta_path, rounds_path,
             *sorted(config.DATA_RAW.glob("*.parquet")),
             *sorted((config.ROOT / "data/external").rglob("*.parquet")),
             *sorted((config.ROOT / "data/external").rglob("*.csv"))]
    manifest = {"hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "rounds": rounds, "features": meta["features"], "seeds": list(SEEDS),
                "threads": args.threads, "lightgbm_deterministic": True,
                "force_col_wise": True, "schedule_rounds": 200}
    args.output.mkdir(parents=True, exist_ok=True)
    mp = args.output / "manifest.json"
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise ValueError("Inputs/source/settings changed: use a new output directory")
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    raw, X, cats = matrix.build(use_arrivals=True)
    X = X[meta["features"]]
    cats = [c for c in cats if c in X]
    print(f"matrix {X.shape}, seeds {SEEDS}", flush=True)
    frames = []
    for month in args.months:
        path = args.output / f"month_{month:02d}.parquet"
        start = time.monotonic()
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = run_fold(raw, X, cats, rounds, month, args.threads)
            frame.to_parquet(path, index=False)
        frames.append(frame)
        report = delta_report(frame, "base_average")
        print(f"month {month:02d}: {report['all']['delta']:+.4f}; "
              f"drop50 {report['drop_50']['delta']:+.4f}; "
              f"{time.monotonic() - start:.1f}s", flush=True)
    pooled = pd.concat(frames, ignore_index=True)
    reports = {"all_requested": delta_report(pooled, "base_average")}
    for name, months in (("selection", [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]),
                         ("jan_jul", [1, 7])):
        subset = pooled[pooled.month.isin(months)].reset_index(drop=True)
        if len(subset):
            reports[name] = delta_report(subset, "base_average")
    suffix = "_".join(map(str, args.months))
    (args.output / f"metrics_{suffix}.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    for name, report in reports.items():
        print(f"POOLED {name}: {report['all']['delta']:+.4f}; "
              f"drop50 {report['drop_50']['delta']:+.4f}; "
              f"drop best day {report['drop_best_airport_day']['delta']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
