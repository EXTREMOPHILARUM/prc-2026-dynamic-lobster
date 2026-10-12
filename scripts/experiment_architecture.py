"""Paired temporal screens of richer tail inputs and uncapped clock regression.

See docs/architecture-experiment-2026-09-16.md for the frozen protocol.
Each output directory is write-once; individual completed folds are reusable only
when their input/source fingerprint agrees exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

VARIANTS = (
    "control", "rich_probability", "rich_severity", "rich_both",
    "direct", "direct_mechanism", "equal_direct",
)


def fit_model(params, X, y, cats, rounds, threads):
    """Copy parameters rather than modifying shared production defaults."""
    settings = dict(params, num_threads=threads, deterministic=True, force_col_wise=True)
    return lgb.train(
        settings, lgb.Dataset(X, y, categorical_feature=cats), num_boost_round=rounds
    )


def delta_report(frame: pd.DataFrame, candidate: str) -> dict:
    y = frame[config.TARGET].to_numpy(dtype=float)
    a = frame["control"].to_numpy(dtype=float)
    b = frame[candidate].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([y, a, b])).all():
        raise ValueError("Incomplete or non-finite evaluation rows")
    ea, eb = (y - a) ** 2, (y - b) ** 2

    def score(mask):
        n = int(mask.sum())
        if not n:
            return {"n": 0, "baseline": None, "candidate": None, "delta": None}
        old, new = float(np.sqrt(ea[mask].mean())), float(np.sqrt(eb[mask].mean()))
        return {"n": n, "baseline": old, "candidate": new, "delta": new - old,
                "sse_delta": float((eb[mask] - ea[mask]).sum())}

    all_rows = np.ones(len(frame), dtype=bool)
    missing = frame["aobt_missing"].to_numpy(dtype=bool)
    lirf = frame["airport"].eq("LIRF").to_numpy()
    masks = {"all": all_rows, "bulk": y <= 3600, "tail": y > 3600,
             "clock_present": ~missing, "clock_missing": missing,
             "LIRF": lirf, "non_LIRF": ~lirf}
    result = {name: score(mask) for name, mask in masks.items()}
    helped = np.argsort(eb - ea)
    for count in (1, 10, 50):
        kept = all_rows.copy()
        kept[helped[:count]] = False
        result[f"drop_{count}"] = score(kept)
    day_gain = pd.Series(ea - eb).groupby(frame["airport_day"].to_numpy()).sum()
    kept = frame["airport_day"].ne(day_gain.idxmax()).to_numpy()
    result["drop_best_airport_day"] = score(kept)
    result["months"] = {
        str(m): score(frame["month"].eq(m).to_numpy()) for m in sorted(frame.month.unique())
    }
    return result


def run_fold(raw, X, cats, rounds, month, threads, requested):
    valid = evaluation.movement_month(raw).eq(month).to_numpy()
    train = ~valid
    y = raw[config.TARGET].astype("float32")
    Xt, Xv = X.loc[train], X.loc[valid]
    yt = y.loc[train]
    matched = Xt[decomposition.ANCHOR_COL].notna()
    matched_v = Xv[decomposition.ANCHOR_COL].notna().to_numpy()
    anchor_t = Xt[decomposition.ANCHOR_COL]
    anchor_v = Xv[decomposition.ANCHOR_COL].to_numpy(dtype=float)
    cap = decomposition.TAIL_THRESHOLD
    capped = yt.clip(upper=cap)
    tail_cols = [c for c in decomposition.TAIL_FEATURES if c in X.columns]
    rich_cols = tail_cols + [c for c in X if c.startswith(("wx_", "arr_", "seq_", "rwy_"))
                             and c not in tail_cols]

    def fit(params, xx, yy, count):
        return fit_model(params, xx, yy, [c for c in cats if c in xx], count, threads)

    base = np.empty(len(Xv))
    for name, tr_mask, va_mask in (
        ("base_matched", matched, matched_v),
        ("base_missing", ~matched, ~matched_v),
    ):
        target = capped.loc[tr_mask]
        if name == "base_matched":
            target = target - anchor_t.loc[tr_mask].clip(0, cap)
        model = fit(decomposition.BASE_PARAMS, Xt.loc[tr_mask], target, rounds[name])
        base[va_mask] = model.predict(Xv.loc[va_mask], num_threads=threads)
        if name == "base_matched":
            base[va_mask] += np.clip(anchor_v[va_mask], 0, cap)
    base = np.clip(base, 0, cap)
    print(f"month {month:02d}: base fitted", flush=True)

    tail = yt > cap
    gate_cols = mechanism.gate_features(X)
    schedule_gate = fit(mechanism.GATE_PARAMS, Xt[gate_cols],
                        mechanism.fallback_label(raw.loc[train]).astype("int8"), 200)
    schedule_p = schedule_gate.predict(Xv[gate_cols], num_threads=threads)
    proxy = Xv[mechanism.PROXY_COL].to_numpy(dtype=float)
    scale = np.where(Xv["ADEP_mvt"].astype("string").eq("LIRF"),
                     decomposition.LIRF_TAIL_SCALE, decomposition.OTHER_TAIL_SCALE)
    probability, severity = {}, {}
    for label, columns in (("old", tail_cols), ("rich", rich_cols)):
        if label == "old" or {"rich_probability", "rich_both"} & set(requested):
            model = fit(decomposition.GATE_PARAMS, Xt[columns], tail.astype("int8"),
                        rounds["gate"])
            probability[label] = model.predict(Xv[columns], num_threads=threads)
        if label == "old" or {"rich_severity", "rich_both"} & set(requested):
            model = fit(decomposition.SEVERITY_PARAMS, Xt.loc[tail, columns],
                        yt.loc[tail] - cap, rounds["severity"])
            severity[label] = np.maximum(model.predict(Xv[columns], num_threads=threads), 0)

    predictions = {}
    for name, p, s in (("control", "old", "old"), ("rich_probability", "rich", "old"),
                       ("rich_severity", "old", "rich"), ("rich_both", "rich", "rich")):
        if name in requested:
            ordinary = np.maximum(base + scale * probability[p] * severity[s], 0)
            predictions[name] = mechanism.blend(schedule_p, proxy, ordinary)
    print(f"month {month:02d}: tail arms fitted", flush=True)

    if {"direct", "direct_mechanism", "equal_direct"} & set(requested):
        direct = np.empty(len(Xv))
        for name, tr_mask, va_mask in (
            ("base_matched", matched, matched_v),
            ("base_missing", ~matched, ~matched_v),
        ):
            target = yt.loc[tr_mask]
            if name == "base_matched":
                target = target - anchor_t.loc[tr_mask]
            model = fit(decomposition.BASE_PARAMS, Xt.loc[tr_mask], target, rounds[name])
            direct[va_mask] = model.predict(Xv.loc[va_mask], num_threads=threads)
            if name == "base_matched":
                direct[va_mask] += anchor_v[va_mask]
        direct = np.maximum(direct, 0)
        derived = {"direct": direct, "direct_mechanism": mechanism.blend(schedule_p, proxy, direct),
                   "equal_direct": (direct + predictions["control"]) / 2}
        predictions.update({k: v for k, v in derived.items() if k in requested})

    result = pd.DataFrame({config.ID_COL: raw.loc[valid, config.ID_COL].to_numpy(),
                           config.TARGET: raw.loc[valid, config.TARGET].to_numpy(),
                           "month": month, "airport": raw.loc[valid, "ADEP_mvt"].to_numpy(),
                           "aobt_missing": ~matched_v,
                           "airport_day": (raw.loc[valid, "ADEP_mvt"].astype(str) + ":" +
                               raw.loc[valid, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")).to_numpy(),
                           **predictions})
    if not np.isfinite(result[list(requested)].to_numpy()).all():
        raise ValueError(f"Non-finite fold {month}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, nargs="+", default=[2, 5, 9])
    ap.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if len(set(args.months)) != len(args.months) or not set(args.months) <= set(range(1, 13)):
        ap.error("months must be unique values from 1 through 12")
    if args.threads < 1:
        ap.error("threads must be positive")
    requested = list(dict.fromkeys(["control", *args.variants]))
    meta_path = config.MODELS / "decomposition_arr.json"
    meta = json.loads(meta_path.read_text())
    rounds_path = config.DATA_INTERIM / "component_rounds.json"
    rounds = json.loads(rounds_path.read_text())["rounds"]
    sources = [Path(__file__), *sorted((Path(__file__).resolve().parents[1] / "src/prc").glob("*.py")),
               meta_path, rounds_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    inputs = [*sorted(config.DATA_RAW.glob("*.parquet")),
              *sorted((config.ROOT / "data/external").rglob("*.parquet")),
              *sorted((config.ROOT / "data/external").rglob("*.csv"))]
    hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    manifest = {"hashes": hashes, "variants": requested, "rounds": rounds,
                "threads": args.threads, "features": meta["features"],
                "deterministic": True, "force_col_wise": True, "schedule_rounds": 200}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing output belongs to different inputs/settings; use a new directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    raw, X, cats = matrix.build(use_arrivals=True)
    X = X[meta["features"]]
    cats = [c for c in cats if c in X]
    print(f"matrix {X.shape}; rounds {rounds}; variants {requested}", flush=True)
    frames = []
    for month in args.months:
        path = args.output / f"month_{month:02d}.parquet"
        started = time.monotonic()
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = run_fold(raw, X, cats, rounds, month, args.threads, requested)
            frame.to_parquet(path, index=False)
        frames.append(frame)
        print(f"month {month:02d} complete in {time.monotonic() - started:.1f}s", flush=True)
        for candidate in requested[1:]:
            report = delta_report(frame, candidate)
            print(f"  {candidate}: {report['all']['delta']:+.4f}; "
                  f"drop50 {report['drop_50']['delta']:+.4f}", flush=True)
    pooled = pd.concat(frames, ignore_index=True)
    reports = {candidate: delta_report(pooled, candidate) for candidate in requested[1:]}
    suffix = "_".join(map(str, args.months))
    (args.output / f"metrics_{suffix}.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    for candidate, report in reports.items():
        print(f"POOLED {candidate}: {report['all']['delta']:+.4f}; "
              f"drop50 {report['drop_50']['delta']:+.4f}; "
              f"drop best day {report['drop_best_airport_day']['delta']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
