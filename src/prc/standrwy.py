"""Stand-to-runway taxi distance (log section 63): joins data/interim/stand_rwy_dist.parquet, built by
scripts/build_stand_rwy.py from the X-Plane Scenery Gateway geometry (GPL v2+), on the raw (airport, NM stand,
runway) strings. A combination the table does not hold is NaN."""
from __future__ import annotations

import pandas as pd

from prc import config

FEATURES = ["srw_net_m", "srw_line_m"]
TABLE = config.DATA_INTERIM / "stand_rwy_dist.parquet"
KEYS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} missing: run scripts/build_stand_rwy.py")
    t = pd.read_parquet(TABLE)
    k = df[KEYS].astype("string")
    out = k.merge(t.astype({c: "string" for c in KEYS}), on=KEYS, how="left")
    return pd.DataFrame({c: out[c].to_numpy(dtype="float64") for c in FEATURES}, index=df.index)
