"""Twelve-fold leave-one-month-out measurement of `prc.ownexpert` on top of the v29 chain.

For each 2025 month m the expert is fitted on the chain's OOF residual of the other eleven months
(matched, own-joined rows) and applied to month m. Reported: per-month RMSE over all rows, the
twelve-month and January+July fold, drop-1/10/50, and the Jan+Jul effect re-weighted to 2026
coverage. The v29 prediction is `analyse_error_anatomy.v29_oof`, the stored-component
reconstruction (an approximation of the shipped chain; see the 2026-09-20 audit).

`--save` also fits on all twelve months and writes models/ownexpert.txt. Nothing serves it
until the organisers rule on own-flight external data (section 37).

    uv run python -u scripts/experiment_own_expert.py [--save]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)
COV_2026 = {1: {"EDDM": .970, "LEMD": .961, "LSZH": .940, "LEBL": .921, "EHAM": .909, "EGLL": .761, "LIRF": .660,
                "EDDF": .567, "LFPG": .069, "LTFM": 0.0},
            7: {"EGLL": .968, "EDDM": .965, "EHAM": .958, "LEBL": .947, "LSZH": .936, "LEMD": .598, "EDDF": .420,
                "LFPG": .064, "LIRF": .001, "LTFM": 0.0}}


def _anatomy():
    path = pathlib.Path(__file__).resolve().parent / "analyse_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("analyse_error_anatomy", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true")
    ap.add_argument("--version", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="feature version: 2 silent-gap rule (s.49), 3 + stand exit (s.50), 4 wider stand location (s.51)")
    ap.add_argument("--member", default=None,
                    help="OOF parquet of an extra CatBoost member (experiment_catboost_gpu.py) blended into the chain before the expert")
    args = ap.parse_args()
    feats = {1: ownexpert.FEATURES, 2: ownexpert.FEATURES_V2, 3: ownexpert.FEATURES_V3, 4: ownexpert.FEATURES_V3,
             5: ownexpert.FEATURES_V5}[args.version]
    tag = "" if args.version == 1 else f"_v{args.version}"
    src = f"adsb_own_2025{tag}.parquet"
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    p = _anatomy().v29_oof(raw, X)
    month = evaluation.movement_month(raw).to_numpy()
    p29 = p.copy()
    if args.member:
        mem = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / args.member), on=config.ID_COL,
                                          how="left")["member"].to_numpy(dtype="float64")
        # the member is fitted on the residual to clip(MVT - AOBT, 0, 3600), so it cannot represent a
        # row whose NM clock itself reads over an hour (day slips); those stay with the uncapped base
        # (section 53: one such LIRF row at 87,002s was predicted 12,172s)
        anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
        ok = np.isfinite(mem) & (anchor_raw <= decomposition.TAIL_THRESHOLD); sel = ok & ~np.isin(month, SCORED)
        d = mem[sel] - p[sel]; r = (y - p)[sel]
        w = float(np.clip((d @ r) / max(d @ d, 1e-9), 0, 1))       # fitted on the ten non-scored months only
        p = p.copy(); p[ok] = np.clip(lirfbands.lobt_clip(p + w * (np.nan_to_num(mem) - p), X)[ok], 0, None)
        sc = np.isin(month, SCORED)
        print(f"member {args.member}: weight {w:.3f} (non-scored months); chain before the expert Jan+Jul "
              f"{rmse(y[sc]-p29[sc]):.2f} -> {rmse(y[sc]-p[sc]):.2f}", flush=True)
        tag += "_member"
    apt = raw["ADEP_mvt"].astype(str).to_numpy()
    anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan, X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"))
    own = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / src), on=config.ID_COL, how="left")
    f = ownexpert.frame(apt, raw["MVT_TIME_UTC_mvt"], p, anchor, own)
    rows = ownexpert.scope(f)
    print(f"{len(f):,} rows; expert scope (matched, own-joined) {int(rows.sum()):,} ({rows.mean()*100:.1f}%)\n", flush=True)

    q = p.copy()
    for m in range(1, 13):
        tr = month != m
        booster = ownexpert.fit(f[tr].reset_index(drop=True), (y - p)[tr], feats)
        va = month == m
        q[va] = ownexpert.apply(booster, f[va].reset_index(drop=True))
        print(f"   month {m:>2}: {int(va.sum()):7,} rows  scope {int((va & rows).sum()):7,}  RMSE {rmse(y[va]-p[va]):7.2f} -> {rmse(y[va]-q[va]):7.2f}"
              f"  ({rmse(y[va]-q[va]) - rmse(y[va]-p[va]):+.2f})", flush=True)

    sc = np.isin(month, SCORED)
    wins = sum(rmse(y[month == m] - q[month == m]) < rmse(y[month == m] - p[month == m]) for m in range(1, 13))
    print(f"\ntwelve months: {rmse(y-p):.2f} -> {rmse(y-q):.2f} ({rmse(y-q)-rmse(y-p):+.2f}s), {wins}/12 months improve")
    print(f"Jan+Jul fold: {rmse(y[sc]-p[sc]):.2f} -> {rmse(y[sc]-q[sc]):.2f} ({rmse(y[sc]-q[sc])-rmse(y[sc]-p[sc]):+.2f}s)")
    if args.member:
        print(f"against v29 (member + expert together): twelve months {rmse(y-p29):.2f} -> {rmse(y-q):.2f}"
              f" ({rmse(y-q)-rmse(y-p29):+.2f}s); Jan+Jul {rmse(y[sc]-p29[sc]):.2f} -> {rmse(y[sc]-q[sc]):.2f}"
              f" ({rmse(y[sc]-q[sc])-rmse(y[sc]-p29[sc]):+.2f}s)")
    gain_row = (y - p) ** 2 - (y - q) ** 2
    order = np.argsort(-gain_row[sc]); n = int(sc.sum()); base_sse = float(((y[sc] - p[sc]) ** 2).sum())
    for k in (1, 10, 50):
        g = float(gain_row[sc].sum() - gain_row[sc][order[:k]].sum())
        print(f"   drop the {k:>2} rows gaining most: {np.sqrt((base_sse - g)/n) - np.sqrt(base_sse/n):+.2f}s")
    adj = 0.0
    for mo in SCORED:
        for a in COV_2026[mo]:
            k = sc & (month == mo) & (apt == a)
            j = float(f.own_joined[k].mean()) if k.any() else 0.0
            adj += float(gain_row[k].sum()) * (min(COV_2026[mo][a] / j, 1.0) if j > 0 else 0.0)
    print(f"Jan+Jul re-weighted to 2026 coverage (scaled down only): {np.sqrt((base_sse - adj)/n) - np.sqrt(base_sse/n):+.2f}s")

    pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "v29": p, "own": q, "in_scope": rows}).to_parquet(
        config.DATA_INTERIM / f"own_expert{tag}_oof.parquet", index=False)

    if args.save:
        booster = ownexpert.fit(f, y - p, feats)
        out = config.MODELS / f"ownexpert{tag}.txt"
        booster.save_model(str(out)); print(f"saved {out} (serve it through a wrapper's own_expert_stem)")


if __name__ == "__main__":
    main()
