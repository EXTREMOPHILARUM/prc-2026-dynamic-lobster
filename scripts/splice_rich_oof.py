"""Splice per-month rich-expert OOF parts into the v38-equivalent OOF and compare (log section 63).

Each part is FILE:COLUMN:MONTHS (months comma-separated) from `experiment_own_expert_rich.py --tag`. The
candidate is `v38_equiv_oof.parquet` with the matched-expert scope rows of those months replaced; every other
row (no-record expert, garbled rule, unjoined rows) is the baseline's, so a part computed on a machine whose
out-of-scope chain differs cannot leak into the comparison.

    uv run python scripts/splice_rich_oof.py OUT.parquet a.parquet:own_x:1,2,3 b.parquet:own_x:4,5,6
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    out, *parts = sys.argv[1:]
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    ids = raw[[config.ID_COL]]
    y = raw[config.TARGET].to_numpy("float64")
    month = evaluation.movement_month(raw).to_numpy()
    base = ids.merge(pd.read_parquet(config.DATA_INTERIM / "v38_equiv_oof.parquet"), on=config.ID_COL,
                     how="left")["prediction"].to_numpy("float64")
    anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan, X[decomposition.ANCHOR_COL].to_numpy("float64"))
    own = ids.merge(pd.read_parquet(config.DATA_INTERIM / "adsb_own_2025_v7.parquet"), on=config.ID_COL, how="left")
    rows = ownexpert.scope(ownexpert.frame(raw["ADEP_mvt"].astype(str).to_numpy(), raw["MVT_TIME_UTC_mvt"], base, anchor, own))
    cand = base.copy(); covered = np.zeros(len(y), bool)
    for spec in parts:
        f, c, ms = spec.rsplit(":", 2)
        ms = [int(m) for m in ms.split(",")]
        q = ids.merge(pd.read_parquet(config.DATA_INTERIM / f, columns=[config.ID_COL, c]), on=config.ID_COL,
                      how="left")[c].to_numpy("float64")
        sel = rows & np.isin(month, ms)
        cand[sel] = q[sel]; covered |= np.isin(month, ms)
    if not covered.all():
        raise SystemExit(f"months not covered: {sorted(set(month[~covered]))}")
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": cand}).to_parquet(config.DATA_INTERIM / out, index=False)
    sc = np.isin(month, SCORED); ns = ~sc
    d = [rmse((y - cand)[month == m]) - rmse((y - base)[month == m]) for m in range(1, 13)]
    gain = (y - base) ** 2 - (y - cand) ** 2
    o = np.argsort(-gain[sc]); n = int(sc.sum()); bs = float(((y[sc] - base[sc]) ** 2).sum())
    drops = [np.sqrt((bs - (gain[sc].sum() - gain[sc][o[:k]].sum())) / n) - np.sqrt(bs / n) for k in (1, 10, 50)]
    print(f"baseline v38-equivalent: twelve {rmse(y - base):.3f}, Jan+Jul {rmse((y - base)[sc]):.3f}")
    print(f"candidate {out}: twelve {rmse(y - cand):.3f} ({rmse(y - cand) - rmse(y - base):+.3f}), "
          f"Jan+Jul {rmse((y - cand)[sc]):.3f} ({rmse((y - cand)[sc]) - rmse((y - base)[sc]):+.3f}), "
          f"non-scored {rmse((y - cand)[ns]) - rmse((y - base)[ns]):+.3f}")
    print("per month " + " ".join(f"{m}:{x:+.2f}" for m, x in enumerate(d, 1)))
    print(f"months won {sum(x < 0 for x in d)}/12 (non-scored {sum(x < 0 for m, x in enumerate(d, 1) if m not in SCORED)}/10); "
          f"Jan+Jul drop-1/10/50 {drops[0]:+.3f}/{drops[1]:+.3f}/{drops[2]:+.3f}")


if __name__ == "__main__":
    main()
