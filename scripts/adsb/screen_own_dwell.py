"""Screen the scored flight's OWN ADS-B stand dwell against the v29 residual, on 2025 training
data only. Measurement, not a feature build: using this on 2026 ranking rows joins a scored
flight to its own post-event trace, which waits on the organisers (section 37). 2025 rows are
labelled training data and fall outside that policy.

Features per departure (data/interim/adsb_own_2025.parquet, `own_flight_features.py 2025` on
the GPU host):
    own_dwell_taxi      MVT - end of the stand dwell        (the trace's taxi estimate)
    own_dwell_len       length of that dwell
    own_to_minus_mvt    ADS-B take-off - MVT                 (join quality)
    own_minus_anchor    own_dwell_taxi - (MVT - AOBT_3)      (matched rows: is the NM clock late?)
    own_first_taxi      MVT - first taxi-out point          (lower bound on y, runs seen mid-taxi too)
    own_first_minus_anchor  own_first_taxi - (MVT - AOBT_3)  (> 0 proves the NM clock late)

Two lanes, each with a LOMO residual corrector fitted with and without the own_* features on the
ten non-scored months (January and July reported only). Criteria fixed before the first run:

    PASS if the with-own correctors cut total SSE of the non-scored months (all rows, both lanes
    summed) by >= 0.5% more than the without-own ones (~0.75s of fold RMSE), win >= 7/10 months,
    and stay positive after dropping the 10 and 50 largest-|residual| rows.

    uv run python -u scripts/adsb/screen_own_dwell.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, lirfmoe, matrix, mechanism

OWN = ["own_joined", "own_dwell_taxi", "own_dwell_len", "own_to_minus_mvt", "own_minus_anchor",
       "own_first_taxi", "own_first_minus_anchor"]
SCORED = (1, 7)
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
          "seed": 7, "num_threads": 8}
ROUNDS = 300


def _anatomy():
    path = pathlib.Path(__file__).resolve().parents[1] / "analyse_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("analyse_error_anatomy", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def corrector_oof(frame: pd.DataFrame, cols: list[str], months: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frame))
    for m in np.unique(months):
        tr, va = months != m, months == m
        ds = lgb.Dataset(frame.loc[tr, cols], frame.loc[tr, "resid"], categorical_feature=["apt"])
        out[va] = lgb.train(PARAMS, ds, ROUNDS).predict(frame.loc[va, cols])
    return out


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    p = _anatomy().v29_oof(raw, X)
    month = evaluation.movement_month(raw).to_numpy()
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    anchor = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    t = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    d = pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "apt": raw["ADEP_mvt"].astype(str).to_numpy(),
                      "hour": t.dt.hour.to_numpy(), "dow": t.dt.dayofweek.to_numpy(), "gap": gap, "anchor": anchor,
                      "pred": p, "resid": y - p, "y": y, "month": month, "missing": missing})
    d = d.merge(pd.read_parquet(config.DATA_INTERIM / "adsb_own_2025.parquet"), on=config.ID_COL, how="left")
    d["own_joined"] = d.own_joined.fillna(0.0)
    d["own_minus_anchor"] = d.own_dwell_taxi - d.anchor
    d["own_first_minus_anchor"] = d.own_first_taxi - d.anchor
    d["apt"] = d["apt"].astype("category")
    lane_nr = d.missing & ~(np.abs(d.y - d.gap) <= lirfmoe.COPY_TOL) & (d.y <= 43200)
    lane_m = ~d.missing & d.anchor.notna()

    print("coverage of the own dwell, by lane (all 2025)")
    for name, lane in (("matched", lane_m), ("no-record ordinary", lane_nr)):
        g = d[lane]
        print(f"   {name:20s} rows {len(g):9,}  joined {g.own_joined.mean()*100:5.1f}%  dwell seen {g.own_dwell_taxi.notna().mean()*100:5.1f}%")

    m = d[lane_m]; early = (m.anchor - m.y) < -600; seen = m.own_dwell_taxi.notna()
    print(f"\nmatched early-anchor rows (AOBT proxy more than 600s short of y): {int(early.sum()):,}; own dwell seen on {seen[early].mean()*100:.1f}%")
    for lab, sel in (("early", early & seen), ("agree/late", ~early & seen)):
        g = m[sel]
        print(f"   {lab:10s} n={len(g):8,}  median(y - own_dwell_taxi) {np.median(g.y - g.own_dwell_taxi):+6.0f}s"
              f"  median(y - anchor) {np.median(g.y - g.anchor):+6.0f}s  corr(own_minus_anchor, resid) {g.own_minus_anchor.corr(g.resid):+.3f}")

    ns_all = ~d.month.isin(SCORED)
    total_ns = float((d.resid[ns_all] ** 2).sum())
    lanes = {"matched": (lane_m, ["apt", "hour", "dow", "pred", "anchor"]),
             "no-record ordinary": (lane_nr, ["apt", "hour", "dow", "pred", "gap"])}
    per_month = {}; sse_parts = {"without": [], "with": [], "resid": []}
    for name, (lane, base) in lanes.items():
        e = d[lane & ns_all].reset_index(drop=True); mo = e.month.to_numpy()
        print(f"\nlane {name}: {len(e):,} non-scored rows, fitting correctors", flush=True)
        c0 = corrector_oof(e, base, mo); c1 = corrector_oof(e, base + OWN, mo)
        r = e.resid.to_numpy(); s0 = (r - c0) ** 2; s1 = (r - c1) ** 2
        print(f"   RMSE v29 {np.sqrt((r**2).mean()):.1f}  without-own {np.sqrt(s0.mean()):.1f}  with-own {np.sqrt(s1.mean()):.1f}")
        for mm in np.unique(mo):
            k = mo == mm; a, b = per_month.get(mm, (0.0, 0.0)); per_month[mm] = (a + s0[k].sum(), b + s1[k].sum())
        sse_parts["without"].append(s0); sse_parts["with"].append(s1); sse_parts["resid"].append(r)
    s0 = np.concatenate(sse_parts["without"]); s1 = np.concatenate(sse_parts["with"]); r = np.concatenate(sse_parts["resid"])
    gain = (s0.sum() - s1.sum()) / total_ns
    wins = sum(b < a for a, b in per_month.values())
    print("\nper month, both lanes: with-own vs without-own, % of that month's lane SSE")
    for mm, (a, b) in sorted(per_month.items()):
        print(f"   {mm:>3}  {100*(b-a)/a:+6.2f}%")
    order = np.argsort(-np.abs(r)); drops = {}
    for n_drop in (10, 50):
        keep = np.ones(len(r), bool); keep[order[:n_drop]] = False
        drops[n_drop] = (s0[keep].sum() - s1[keep].sum()) / total_ns
    fold_s = 307.27 * (1 - np.sqrt(1 - gain))
    print(f"\nwith-own cuts non-scored total SSE by {100*gain:+.3f}% (~{fold_s:.2f}s of fold RMSE); wins {wins}/{len(per_month)};"
          f" drop-10 {100*drops[10]:+.3f}%, drop-50 {100*drops[50]:+.3f}%")
    passed = gain >= 0.005 and wins >= 7 and all(v > 0 for v in drops.values())
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} (criteria: >= 0.5% of total SSE, >= 7/10 months, positive after drop-10/50)")
    confirm(d, lane_m, lanes["matched"][1])


def confirm(d: pd.DataFrame, lane: pd.Series, base: list[str]) -> None:
    """Matched lane only (where the gain is): correctors fitted on the ten non-scored months,
    applied to January and July, which played no part in the decision. Per airport, then the
    fold effect re-weighted to each airport's 2026 joined share in that month (sections 41, 44)."""
    cov26 = {1: {"EDDM": .970, "LEMD": .961, "LSZH": .940, "LEBL": .921, "EHAM": .909, "EGLL": .761, "LIRF": .660,
                 "EDDF": .567, "LFPG": .069, "LTFM": 0.0},
             7: {"EGLL": .968, "EDDM": .965, "EHAM": .958, "LEBL": .947, "LSZH": .936, "LEMD": .598, "EDDF": .420,
                 "LFPG": .064, "LIRF": .001, "LTFM": 0.0}}
    tr = d[lane & ~d.month.isin(SCORED)]; te = d[lane & d.month.isin(SCORED)].copy()
    c0 = lgb.train(PARAMS, lgb.Dataset(tr[base], tr.resid, categorical_feature=["apt"]), ROUNDS).predict(te[base])
    c1 = lgb.train(PARAMS, lgb.Dataset(tr[base + OWN], tr.resid, categorical_feature=["apt"]), ROUNDS).predict(te[base + OWN])
    te["s0"] = (te.resid - c0) ** 2; te["s1"] = (te.resid - c1) ** 2; te["sv"] = te.resid ** 2
    total = float((d.resid[d.month.isin(SCORED)] ** 2).sum()); n = int(d.month.isin(SCORED).sum())
    print("\nCONFIRMATION on January + July 2025 (matched lane; correctors fitted on the ten other months)")
    print(f"   {'apt':5} {'mon':>3} {'rows':>7} {'joined':>7} {'v29':>7} {'no-own':>7} {'own':>7} {'dSSE M':>8} {'2026 cov':>8}")
    raw_gain = 0.0; adj_gain = 0.0
    for (a, mo), g in te.groupby(["apt", "month"], observed=True):
        dsse = float(g.s0.sum() - g.s1.sum()); j = float(g.own_joined.mean())
        # a gain that exists only on joined rows scales with the share of rows that can be joined in 2026
        scale = min(cov26[mo][a] / j, 1.0) if j > 0 else 0.0
        raw_gain += dsse; adj_gain += dsse * scale
        print(f"   {a:5} {mo:>3} {len(g):>7,} {j*100:6.1f}% {np.sqrt(g.sv.mean()):7.1f} {np.sqrt(g.s0.mean()):7.1f}"
              f" {np.sqrt(g.s1.mean()):7.1f} {dsse/1e6:8.1f} {cov26[mo][a]*100:7.1f}%")
    base_rmse = np.sqrt(total / n)
    for lab, gsse in (("as measured on 2025 coverage", raw_gain), ("re-weighted to 2026 coverage", adj_gain)):
        print(f"   {lab:30s}: SSE -{gsse/1e6:,.0f}M of {total/1e6:,.0f}M -> fold {base_rmse:.2f} -> {np.sqrt((total-gsse)/n):.2f}"
              f" ({np.sqrt((total-gsse)/n)-base_rmse:+.2f}s)")


if __name__ == "__main__":
    main()
