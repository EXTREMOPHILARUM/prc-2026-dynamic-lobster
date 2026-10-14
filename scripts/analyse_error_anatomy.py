"""Reproduce the error anatomy that `docs/state.md` quotes.

Rebuilds the v29 chain's leave-one-month-out prediction from the stored component OOFs and
reports, for scored months (January and July 2025):

  1. squared error by lane and airport, and what each block would be worth if solved
  2. every day-offset row (y > 12h) with the model's prediction, so the solved/hedged/missed
     split is visible
  3. the sensitivity arithmetic: what one mispredicted day-offset row costs, against the gap
     to each rival
  4. the copy/day/normal composition of LIRF no-record rows by schedule-gap band
  5. calibration of the mixture's copy probability by gap band
  6. the raw AOBT anchor's RMSE against ours, per airport, on matched rows

Everything is read from `data/interim`; nothing is refitted, so it runs in a couple of minutes.

    uv run python -u scripts/analyse_error_anatomy.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, lirfmoe, matrix, mechanism

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
RIVALS: list[tuple[float, str]] = []  # reference scores to compare the anatomy against; none by default
BOARD = 267.87      # v29
N_SCORED = 344_841  # rows in the ranking set


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def v29_oof(raw, X):
    """The shipped chain's OOF prediction, assembled from stored component files."""
    ids = raw[[config.ID_COL]]

    def col(f, c):
        return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=config.ID_COL, how="left")[c].to_numpy(dtype="float64").copy()

    base = col("v21_oof_predictions.parquet", "prediction")
    mix = col("lirf_unmatched_moe_oof.parquet", "mixture")
    mem = {m: col(f"catgpu_{m}_oof_predictions.parquet", "member") for m in ("d8", "lobt")}
    stack = col("stacker_mid_800_oof.parquet", "stack") - col("stacker_mid_800_oof.parquet", "ref")
    tgt = lirfmoe.target_rows(X, 8.0)
    ok = np.all([np.isfinite(v) for v in mem.values()], axis=0)
    p = base.copy()
    for k, w in (("d8", 0.3), ("lobt", 0.3)):
        p[ok] += w * (mem[k][ok] - base[ok])
    p[tgt] = mix[tgt]
    return np.clip(lirfbands.lobt_clip(p + 0.5 * stack, X), 0, None)


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    sc = np.isin(month, (1, 7))
    apt = raw["ADEP_mvt"].to_numpy()
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    anchor = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    lirf = apt == lirfbands.AIRPORT
    p = v29_oof(raw, X)
    n = int(sc.sum()); tot = float(((y[sc] - p[sc]) ** 2).sum())
    print(f"v29 chain, leave-one-month-out, scored months: {n:,} rows  RMSE {np.sqrt(tot/n):.2f}  SSE {tot/1e6:,.0f}M\n")

    print("1. where the squared error sits")
    blocks = [("non-LIRF matched", ~lirf & ~missing), ("non-LIRF no-record", ~lirf & missing),
              ("LIRF matched", lirf & ~missing), ("LIRF no-record, gap <= 8h", lirf & missing & (gap <= 8 * 3600)),
              ("LIRF no-record, gap > 8h", lirf & missing & (gap > 8 * 3600))]
    for name, m in blocks:
        s = float(((y[m & sc] - p[m & sc]) ** 2).sum())
        print(f"   {name:28s} rows {int((m & sc).sum()):7,}  SSE {s/1e6:9,.0f}M  {100*s/tot:5.1f}%  RMSE if solved {np.sqrt((tot-s)/n):7.2f}")

    # The signed error of the NM anchor, not the model residual.  This separates
    # ordinary matched rows from flights whose recorded AOBT is substantially late.
    # Keep the denominator within the matched lane: no-record rows have no anchor
    # from which delta can be defined.
    matched_sc = sc & ~missing & np.isfinite(anchor)
    delta = anchor - y
    regime = np.select(
        [delta < -600, delta > 600],
        ["early (< -600s)", "late (> +600s)"],
        default="agree (|delta| <= 600s)",
    )
    print("\n1a. matched-lane error by signed NM-anchor disagreement (delta = AOBT_3 proxy - y)")
    matched_sse = float(((y[matched_sc] - p[matched_sc]) ** 2).sum())
    for name in ("early (< -600s)", "agree (|delta| <= 600s)", "late (> +600s)"):
        m = matched_sc & (regime == name)
        s = float(((y[m] - p[m]) ** 2).sum())
        print(f"   {name:26s} rows {int(m.sum()):7,}  SSE {s/1e6:9,.0f}M  {100*s/matched_sse:5.1f}% of matched SSE  {100*s/tot:5.1f}% of all SSE")
    print("\n   by airport (share is of that airport's matched-lane SSE)")
    for a in AIRPORTS:
        am = matched_sc & (apt == a)
        a_sse = float(((y[am] - p[am]) ** 2).sum())
        cells = []
        for name in ("early (< -600s)", "agree (|delta| <= 600s)", "late (> +600s)"):
            m = am & (regime == name)
            s = float(((y[m] - p[m]) ** 2).sum())
            cells.append(f"{name.split()[0]} {int(m.sum()):,}/{s / 1e6:.0f}M/{100*s/a_sse:.1f}%")
        print(f"   {a}  " + "  ".join(cells))

    print("\n2. every day-offset row (y > 12h) and what we predicted")
    big = y > 43200
    d = pd.DataFrame({"apt": apt[big], "month": month[big], "y": y[big].round(0), "pred": p[big].round(0),
                      "gap_h": (gap[big] / 3600).round(2), "matched": ~missing[big]})
    d["err"] = (d.pred - d.y).round(0)
    d["group"] = np.where(d.err.abs() < 100, "solved", np.where(d.pred < 5000, "MISSED", "hedged"))
    print(d.sort_values(["group", "apt", "month"]).to_string(index=False))
    print(f"\n   scored day-offset rows: {int((big & sc).sum())}, holding {100*float(((y[big&sc]-p[big&sc])**2).sum())/tot:.1f}% of scored SSE;"
          f" perfect on them -> RMSE {np.sqrt((tot-float(((y[big&sc]-p[big&sc])**2).sum()))/n):.2f}")

    print("\n3. what one day-offset row is worth on the board")
    mse = BOARD ** 2
    one = (86400 - 1050) ** 2
    print(f"   one row truth 86,400s predicted 1,050s: SSE {one:,} = {one/N_SCORED:,.0f} of MSE"
          f" -> fixing it alone gives RMSE {np.sqrt(mse - one/N_SCORED):.2f}")
    for target, name in RIVALS:
        need = (mse - target ** 2) * N_SCORED
        print(f"   reaching {target:.2f} ({name}): remove {need/1e6:,.0f}M SSE = {need/one:.2f} such rows")

    print("\n4. LIRF no-record rows: mechanism by schedule-gap band (all 12 months)")
    copy = np.abs(y - gap) <= lirfmoe.COPY_TOL
    t = pd.DataFrame({"gap_h": gap / 3600, "cls": np.where(copy, "copy", np.where(big, "day", "normal"))})[lirf & missing]
    t["band"] = pd.cut(t.gap_h, [-99, 1, 2, 4, 8, 12, 16, 20, 24, 99])
    print(pd.crosstab(t.band, t.cls).to_string())

    print("\n5. calibration of the mixture's copy probability (out of fold, its own scope)")
    ids = raw[[config.ID_COL]]
    st = ids.merge(pd.read_parquet(config.DATA_INTERIM / "lirf_unmatched_moe_oof.parquet"), on=config.ID_COL, how="left")
    pc = st["p_copy"].to_numpy(dtype="float64"); reg = st["regressor"].to_numpy(dtype="float64")
    tgt = lirfmoe.target_rows(X, 8.0)
    c = pd.DataFrame({"gap_h": gap / 3600, "p": pc, "copy": copy, "stake": (np.clip(gap, 0, None) - reg) ** 2 * (copy - pc) ** 2})[tgt]
    c["band"] = pd.cut(c.gap_h, [-99, 1, 2, 4, 6, 8])
    g = c.groupby("band", observed=True).agg(n=("p", "size"), mean_p=("p", "mean"), actual=("copy", "mean"), stake_M=("stake", lambda v: v.sum() / 1e6))
    print(g.round({"mean_p": 3, "actual": 3, "stake_M": 0}).to_string())
    print("   (the wide-gap shortfall is sampling noise: every attempt to correct it lost, log section 29)")

    print("\n6. the raw AOBT anchor vs our prediction, matched scored rows")
    for a in AIRPORTS:
        m = (apt == a) & ~missing & sc & np.isfinite(anchor)
        ratio = rmse(p[m], y[m]) / rmse(anchor[m], y[m])
        print(f"   {a}  n {int(m.sum()):6,}  anchor {rmse(anchor[m], y[m]):7.1f}  ours {rmse(p[m], y[m]):7.1f}  ratio {ratio:.2f}")


if __name__ == "__main__":
    main()
