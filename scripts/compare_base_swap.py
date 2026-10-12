"""Paired twelve-fold verdict for swapping the base chain under the shipped experts (log section 63).

The vN-equivalent OOF is the v29 reconstruction (`analyse_error_anatomy.v29_oof`: CatBoost members, Rome
mixture, stored stacker correction, LOBT clip) built on the base chain, on rows outside the experts' scopes,
and the experts' values elsewhere. A candidate base is judged by rebuilding the reconstruction on the
candidate chain and replacing exactly those rows: the rows where the current equivalent equals the
reconstruction of the current chain. Reports twelve-month and Jan+Jul RMSE deltas, months
won, the drop-1/10/50 concentration diagnostic, and a day-block bootstrap CI on the Jan+Jul delta.

    uv run python -u scripts/compare_base_swap.py --equiv v48_equiv_oof.parquet \
        --old base_rws_39 base_rws_rest --new l1023rws l1023rws_rest --save v49_equiv_oof.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from prc import config, lirfbands, lirfmoe, matrix

ID = config.ID_COL


def chain(tags: list[str], month: pd.Series) -> pd.DataFrame:
    """Concatenate `<tag>_oof_predictions.parquet` files, each restricted to its own fold months.

    A fold file also carries a few values outside its months (the Rome short-gap rows are copied from a
    stored expert for every month), so a file owns a month only where it predicts most of the rows.
    `month` is indexed by ID."""
    parts = []
    for t in tags:
        p = pd.read_parquet(config.DATA_INTERIM / f"{t}_oof_predictions.parquet")[[ID, "prediction"]]
        m = p[ID].map(month)
        own = p.prediction.notna().groupby(m).mean()
        parts.append(p[m.isin(own[own > 0.5].index)])
    out = pd.concat(parts, ignore_index=True)
    assert out[ID].is_unique, "fold files overlap"
    return out


def reconstruct(base: np.ndarray, ids: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """`analyse_error_anatomy.v29_oof` with the base chain passed in instead of the stored v21 OOF."""
    def col(f, c):
        return ids.merge(pd.read_parquet(config.DATA_INTERIM / f), on=ID, how="left")[c].to_numpy(dtype="float64").copy()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equiv", required=True, help="current equivalent OOF in data/interim")
    ap.add_argument("--old", nargs="+", required=True, help="tags of the current base chain's fold files")
    ap.add_argument("--new", nargs="+", required=True, help="tags of the candidate base chain's fold files")
    ap.add_argument("--save", help="write the candidate equivalent OOF here (data/interim)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    d = raw[[ID, "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]].reset_index(drop=True).merge(pd.read_parquet(config.DATA_INTERIM / a.equiv)[[ID, "prediction"]].rename(columns={"prediction": "equiv"}), on=ID)
    t = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    d["month"] = t.dt.month; d["day"] = t.dt.strftime("%m-%d"); y = d.TAXITIME_SEC_mvt.astype("float64")
    month = d.set_index(ID).month
    d = d.merge(chain(a.old, month).rename(columns={"prediction": "old"}), on=ID, how="left")
    d = d.merge(chain(a.new, month).rename(columns={"prediction": "new"}), on=ID, how="left")

    assert len(d) == len(X), "row order must follow the matrix"
    old = reconstruct(d.old.to_numpy(dtype="float64"), d[[ID]], X)
    new = reconstruct(d.new.to_numpy(dtype="float64"), d[[ID]], X)
    swap = np.abs(d.equiv.to_numpy() - old) < 1e-6
    assert np.isfinite(new[swap]).all(), "candidate chain missing rows in the swap scope"
    cand = np.where(swap, new, d.equiv)
    print(f"rows {len(d):,}; swapped (outside expert scopes) {int(swap.sum()):,} ({swap.mean():.1%})")

    e0 = (y - d.equiv) ** 2; e1 = (y - cand) ** 2; dd = e1 - e0
    def rmse(e, m): return float(np.sqrt(e[m].mean()))
    alln = np.ones(len(d), bool); jj = d.month.isin([1, 7]).to_numpy()
    print(f"twelve   {rmse(e0, alln):.3f} -> {rmse(e1, alln):.3f} ({rmse(e1, alln) - rmse(e0, alln):+.3f})")
    print(f"Jan+Jul  {rmse(e0, jj):.3f} -> {rmse(e1, jj):.3f} ({rmse(e1, jj) - rmse(e0, jj):+.3f})")
    per = {int(m): round(rmse(e1, (d.month == m).to_numpy()) - rmse(e0, (d.month == m).to_numpy()), 3) for m in range(1, 13)}
    print(f"by month {per}; won {sum(v < 0 for v in per.values())}/12")
    order = np.argsort(dd.to_numpy())  # most-helped rows first
    for k in (1, 10, 50):
        keep = np.ones(len(d), bool); keep[order[:k]] = False
        print(f"drop-{k:<3} twelve delta {np.sqrt(e1[keep].mean()) - np.sqrt(e0[keep].mean()):+.3f}")
    # day-block bootstrap on Jan+Jul
    q = pd.DataFrame({"day": d.day[jj], "e0": e0[jj], "e1": e1[jj]}).groupby("day").agg(["sum", "size"])
    s0, s1, n = q[("e0", "sum")].to_numpy(), q[("e1", "sum")].to_numpy(), q[("e0", "size")].to_numpy()
    rng = np.random.default_rng(0); deltas = []
    for _ in range(a.boot):
        i = rng.integers(0, len(n), len(n))
        deltas.append(np.sqrt(s1[i].sum() / n[i].sum()) - np.sqrt(s0[i].sum() / n[i].sum()))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"Jan+Jul day-block bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    if a.save:
        pd.DataFrame({ID: d[ID], "prediction": cand}).to_parquet(config.DATA_INTERIM / a.save, index=False)
        print(f"saved data/interim/{a.save}")


if __name__ == "__main__":
    main()
