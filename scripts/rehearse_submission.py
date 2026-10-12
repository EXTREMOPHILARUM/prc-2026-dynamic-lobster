"""End-to-end rehearsal: score a stand-in ranking set built from 2025 months (default Jan+Jul).

Runs the genuine submission path (prc.submit.predict_frame) over a frame blanked exactly
the way the organisers blank theirs, then scores it against truth we hold back.

Two checks:

1. **Leakage** - predict the same rows twice, once with BLOCK_TIME_UTC_mvt blanked and
   once intact. Identical predictions prove no feature reads the withheld column. A
   difference means the OOF is flattering us and a real submission will be worse.
2. **Pipeline** - ids, dtype, row count and finiteness, through the organisers' own
   rejection rules.

    uv run python scripts/rehearse_submission.py --model models/mechanism_wx
    uv run python scripts/rehearse_submission.py --model models/mechanism_v48 --months 1 2 6 7 --jitter-seed 0

`--months 1 2 6 7` stages the final phase's four months; `--jitter-seed` shifts movement and block times by up to
12s as the final set does (`rehearsal.JITTER_S`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prc import config, features, rehearsal, submit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.MODELS / "mechanism_wx"))
    ap.add_argument("--months", type=int, nargs="+", default=list(rehearsal.REHEARSAL_MONTHS))
    ap.add_argument("--jitter-seed", type=int, default=None, help="emulate the final phase's +/-12s time jitter")
    args = ap.parse_args()

    mock, template, truth = rehearsal.build_mock_ranking(tuple(args.months), args.jitter_seed)
    print(f"stand-in ranking set: {len(mock):,} rows, {len(truth):,} departures to predict")

    meta, handle = submit.load_model(Path(args.model))
    blanked = features.departures(mock).copy()
    intact = blanked.copy()
    # restore the withheld column (as staged, jittered or not) only for the leakage probe
    intact["BLOCK_TIME_UTC_mvt"] = (
        intact[config.ID_COL].map(truth.set_index(config.ID_COL)["BLOCK_TIME_UTC_mvt"]).to_numpy()
    )

    print("\n1. leakage probe")
    p_blank = submit.predict_frame(meta, handle, blanked, movements=mock)
    p_intact = submit.predict_frame(meta, handle, intact, movements=mock)
    delta = np.abs(p_blank - p_intact)
    if delta.max() == 0:
        print("   PASS - predictions identical with and without BLOCK_TIME_UTC_mvt")
    else:
        print(f"   FAIL - {int((delta > 0).sum()):,} rows differ, max {delta.max():,.1f}s")
        print("   a feature is reading the withheld column; OOF results are optimistic")

    print("\n2. submission validity and score")
    out = template.copy()
    out[config.TARGET] = np.round(np.clip(p_blank, 0, None)).astype("int32")
    result = rehearsal.score(out, truth)
    for key, value in result.items():
        print(f"   {key}: {value}")
    if result.get("status") == "Succeeded":
        m = truth.merge(out.rename(columns={config.TARGET: "_pred"}), on=config.ID_COL)
        e2 = (m["_pred"].astype("float64") - m[config.TARGET].astype("float64")) ** 2
        month = pd.to_datetime(m[features.TIME_COL], utc=True).dt.month
        print("   by month:", {int(k): round(float(np.sqrt(v)), 2) for k, v in e2.groupby(month).mean().items()})
    if result.get("status") == "Succeeded":
        print("\n   NOTE: the deployed model was trained on all 12 months, so this score "
              "includes\n   the staged months and is optimistic. Use it to verify the path, not to "
              "rank models.")


if __name__ == "__main__":
    main()
