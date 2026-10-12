"""Paired comparison of two own-flight expert OOFs, plain and reweighted toward the 2026 trace distribution.

adsb.lol's receiver network changed between 2025 and 2026 (log section 63: a classifier tells 2025 from 2026 trace
features at AUC 0.90), so a 2025 fold cannot show how an expert that leans on reception-dependent inputs transfers.
Covariate-shift check, no 2026 labels used: fit a classifier for P(2026 | trace features, airport, month-of-year) on
2025 vs 2026 Jan/Jul joined rows, then weight each 2025 joined row by its odds p / (1 - p) (clipped), and compare the two
experts' squared error under those weights as well as unweighted. Only rows in the expert's scope (joined) differ.

    uv run python -u scripts/compare_drift_weighted.py \
        --a own_expert_rich_lean_v9_lag_lean12_oof.parquet:own_lean_v9_lag \
        --b own_expert_rich_lean_v9_lag_mov_mov12_oof.parquet:own_lean_v9_lag_mov
"""
from __future__ import annotations

import argparse
import glob

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config

ID = config.ID_COL
CLIP = (0.05, 20.0)


def oof(spec: str) -> pd.DataFrame:
    f, c = spec.split(":")
    return pd.read_parquet(config.DATA_INTERIM / f, columns=[ID, c]).rename(columns={c: "p"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="file:column of the reference expert OOF")
    ap.add_argument("--b", required=True, help="file:column of the candidate expert OOF")
    ap.add_argument("--own", default="v10", help="feature-file version used for the drift classifier")
    a = ap.parse_args()

    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    tr = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(config.DATA_RAW / "training_2025-*.parquet")))])
    rk = pd.read_parquet(config.DATA_RAW / "ranking.parquet", columns=cols[:-1])
    own = pd.concat([pd.read_parquet(config.DATA_INTERIM / f"adsb_own_{y}_{a.own}.parquet") for y in (2025, 2026)])
    feats = [c for c in own.columns if c.startswith("own_") and c != "own_joined"]

    def frame(df):
        df = df[df.PHASE_mvt == "DEP"].merge(own, on=ID)
        df = df[df.own_joined == 1].copy()
        t = pd.to_datetime(df.MVT_TIME_UTC_mvt, utc=True)
        df["year"] = t.dt.year; df["month"] = t.dt.month
        return df
    d25, d26 = frame(tr), frame(rk)
    # classifier trained on 2025 Jan/Jul vs 2026 Jan/Jul (same months of year), applied to every 2025 month
    ref = pd.concat([d25[d25.month.isin([1, 7])], d26])
    cats = sorted(ref.ADEP_mvt.astype(str).unique())
    def X(df):
        out = df[feats + ["month"]].copy()
        out["month"] = np.where(df.month.isin([1, 2, 3, 10, 11, 12]), 1, 7)  # season of year, not the month itself
        out["apt"] = pd.Categorical(df.ADEP_mvt.astype(str), categories=cats).codes
        return out
    clf = lgb.train({"objective": "binary", "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 200, "verbose": -1, "num_threads": 12},
                    lgb.Dataset(X(ref), (ref.year == 2026).astype(int)), 300)
    p = clf.predict(X(d25))
    prior = (ref.year == 2026).mean()
    w = np.clip((p / (1 - p)) * ((1 - prior) / prior), *CLIP)
    d25 = d25.assign(w=w)
    print(f"2025 joined rows {len(d25):,}; weight quantiles {np.quantile(w, [0.05, 0.25, 0.5, 0.75, 0.95]).round(2)}; "
          f"effective n {w.sum() ** 2 / (w ** 2).sum():,.0f}")

    m = d25.merge(oof(a.a), on=ID).merge(oof(a.b).rename(columns={"p": "q"}), on=ID)
    y = m.TAXITIME_SEC_mvt.astype(float)
    ea, eb = (y - m.p) ** 2, (y - m.q) ** 2
    diff = (eb - ea).groupby(m.month).sum()
    print(f"\nscope rows compared {len(m):,}")
    print(f"unweighted: RMSE A {np.sqrt(ea.mean()):.3f}  B {np.sqrt(eb.mean()):.3f}  dSSE {diff.sum() / 1e6:+,.1f}M  "
          f"months won by B {int((diff < 0).sum())}/12")
    print("  by month (M):", (diff / 1e6).round(1).to_dict())
    wa, wb = (m.w * ea).sum() / m.w.sum(), (m.w * eb).sum() / m.w.sum()
    wdiff = (m.w * (eb - ea)).groupby(m.month).sum() / m.w.sum()
    print(f"2026-weighted: RMSE A {np.sqrt(wa):.3f}  B {np.sqrt(wb):.3f}  ({np.sqrt(wb) - np.sqrt(wa):+.3f}); "
          f"months won by B {int((wdiff < 0).sum())}/12")
    for k in (1, 10, 50):
        order = np.argsort((m.w * (eb - ea)).to_numpy())
        keep = np.ones(len(m), bool); keep[order[:k]] = False
        print(f"  weighted drop-{k:<3} {np.sqrt((m.w * eb)[keep].sum() / m.w[keep].sum()) - np.sqrt((m.w * ea)[keep].sum() / m.w[keep].sum()):+.3f}")
    print("\nper airport, weighted RMSE A -> B:")
    for apt, q in m.groupby("ADEP_mvt"):
        print(f"  {apt} n {len(q):>7,}  {np.sqrt((q.w * (q.TAXITIME_SEC_mvt - q.p) ** 2).sum() / q.w.sum()):7.1f} -> "
              f"{np.sqrt((q.w * (q.TAXITIME_SEC_mvt - q.q) ** 2).sum() / q.w.sum()):7.1f}")


if __name__ == "__main__":
    main()
