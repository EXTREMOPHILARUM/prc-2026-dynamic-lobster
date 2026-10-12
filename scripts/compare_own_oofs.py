"""Paired twelve-fold comparison of stored own-expert OOFs (section 55: v5 against v6a / v6b).

Each argument is FILE:COLUMN, relative to data/interim (as written by experiment_own_expert_bundle.py);
the first is the baseline. Reported per candidate, paired against the baseline: RMSE change per
month, twelve months, January+July, wins on the ten non-scored months (the selection months) and
on all twelve, non-scored SSE change, and drop-1/10/50 of the rows gaining most on Jan+Jul
(a concentration diagnostic, not a veto). Use scripts/paired_bootstrap.py for intervals.

    uv run python scripts/compare_own_oofs.py base.parquet:own_x cand.parquet:own_y [...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from prc import config, evaluation, matrix

SCORED = (1, 7)


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    specs = [a.rsplit(":", 1) for a in sys.argv[1:]]
    if len(specs) < 2:
        raise SystemExit(__doc__)
    raw, _, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    ids = raw[[config.ID_COL]]
    preds = []
    for f, c in specs:
        p = ids.merge(pd.read_parquet(config.DATA_INTERIM / f, columns=[config.ID_COL, c]), on=config.ID_COL,
                      how="left")[c].to_numpy(dtype="float64")
        if not np.isfinite(p).all():
            raise ValueError(f"{f}:{c} does not cover every training row")
        preds.append((f"{f}:{c}", p))
    sc = np.isin(month, SCORED); ns = ~sc
    (bname, pb) = preds[0]
    print(f"baseline {bname}: twelve {rmse(y - pb):.3f}, Jan+Jul {rmse(y[sc] - pb[sc]):.3f}\n")
    for name, p in preds[1:]:
        d = [rmse(y[month == m] - p[month == m]) - rmse(y[month == m] - pb[month == m]) for m in range(1, 13)]
        gain = (y - pb) ** 2 - (y - p) ** 2
        o = np.argsort(-gain[sc]); n = int(sc.sum()); bs = float(((y[sc] - pb[sc]) ** 2).sum())
        drops = [np.sqrt((bs - (gain[sc].sum() - gain[sc][o[:k]].sum())) / n) - np.sqrt(bs / n) for k in (1, 10, 50)]
        print(f"{name}")
        print("  per month: " + " ".join(f"{m}:{d[m - 1]:+.2f}" for m in range(1, 13)))
        print(f"  twelve {rmse(y - p) - rmse(y - pb):+.3f}s  Jan+Jul {rmse(y[sc] - p[sc]) - rmse(y[sc] - pb[sc]):+.3f}s  "
              f"non-scored dSSE {-(gain[ns].sum()) / 1e6:+.1f}M  wins {sum(x < 0 for m, x in enumerate(d, 1) if m not in SCORED)}/10 "
              f"non-scored, {sum(x < 0 for x in d)}/12 all  drop-1/10/50 {drops[0]:+.3f}/{drops[1]:+.3f}/{drops[2]:+.3f}\n")


if __name__ == "__main__":
    main()
