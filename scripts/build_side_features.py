"""Materialise another agent's feature family as a side-join parquet keyed on MVT_ID_mvt.

The combined run needs several families that live in modules and scripts owned by other
agents. Rather than editing their code or `matrix.py`, each family is computed once,
read-only, and written to `data/interim/<name>_features.parquet` for joining into X.

    uv run python -u scripts/build_side_features.py --family neighbour
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from prc import config, data, matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_neighbour() -> tuple[pd.DataFrame, list[str]]:
    from prc import neighbour

    raw, _, _ = matrix.build("train")
    movements = data.load_training(
        columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
                 "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"])
    feats = neighbour.add(
        raw[[config.ID_COL, "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
             "AOBT_3_flt", "EOBT_1_flt"]], movements)
    del movements
    return feats[[config.ID_COL, *neighbour.FEATURES]], list(neighbour.FEATURES)


def build_cfgstate() -> tuple[pd.DataFrame, list[str]]:
    import _cfgstate as cfg

    movements = data.load_training()
    feats = cfg.add(movements)
    return feats[[config.ID_COL, *cfg.FEATURES]], list(cfg.FEATURES)


BUILDERS = {"neighbour": build_neighbour, "cfgstate": build_cfgstate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(BUILDERS), required=True)
    args = ap.parse_args()

    start = time.time()
    frame, columns = BUILDERS[args.family]()
    out = config.DATA_INTERIM / f"{args.family}_features.parquet"
    if len(frame) != 2_085_047:
        raise SystemExit(f"expected 2,085,047 rows, built {len(frame)}")
    frame.to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(frame)} rows, {len(columns)} features, "
          f"{time.time()-start:.0f}s")
    print(columns)


if __name__ == "__main__":
    main()
