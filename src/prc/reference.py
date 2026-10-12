"""Stand x runway reference (unimpeded) taxi-out encoding.

EUROCONTROL PRU defines the unimpeded taxi-out time per departure airport x runway x
stand, estimated as a low percentile of that cell's distribution; additional taxi-out is
actual minus unimpeded. On this data the stand x runway cell explains ~40% of taxi-out
variance - the largest single effect measured in the project (docs/research.md).

Encoding a cell statistic computed over the same rows being predicted leaks the target.
Everything here is therefore **leave-one-month-out**: a row's encoding is built only from
months other than its own. At scoring time the ranking set is encoded from all 12
training months, which is the analogous "all data the row didn't come from".

Thin cells are shrunk toward progressively coarser levels:
    airport x stand x runway  ->  airport x runway  ->  airport x stand  ->  airport
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config

CELL = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
FALLBACKS = [["ADEP_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "STAND_mvt"], ["ADEP_mvt"]]

#: pseudo-count for shrinkage toward the coarser level
SMOOTHING = 20

STATS = {"ref_q10": 0.10, "ref_q50": 0.50}
FEATURES = ["ref_q10", "ref_q50", "ref_mean", "ref_log_count"]


def _agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, observed=True)[config.TARGET]
    out = pd.DataFrame({
        "ref_mean": g.mean(),
        "ref_count": g.size(),
        **{name: g.quantile(q) for name, q in STATS.items()},
    })
    return out.reset_index()


def build_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cell statistics at every level, from the rows given."""
    table = {"cell": _agg(df, CELL)}
    for keys in FALLBACKS:
        table["+".join(keys)] = _agg(df, keys)
    return table


def apply_table(df: pd.DataFrame, table: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach shrunk reference statistics to `df` (does not mutate the input)."""
    cols = ["ref_mean", "ref_count", *STATS]
    merged = df[CELL].merge(table["cell"], on=CELL, how="left")[cols]

    for keys in FALLBACKS:
        coarse = df[keys].merge(table["+".join(keys)], on=keys, how="left")[cols]
        n = merged["ref_count"].fillna(0).to_numpy()
        w = n / (n + SMOOTHING)
        for c in ["ref_mean", *STATS]:
            fine = merged[c].to_numpy()
            back = coarse[c].to_numpy()
            blended = np.where(np.isnan(fine), back, w * fine + (1 - w) * back)
            merged[c] = np.where(np.isnan(blended), back, blended)
        # once blended, the effective sample size is the coarser level's
        merged["ref_count"] = np.where(
            np.isnan(merged["ref_count"]), coarse["ref_count"], merged["ref_count"]
        )

    out = df.copy()
    for c in ["ref_mean", *STATS]:
        out[c] = merged[c].to_numpy()
    out["ref_log_count"] = np.log1p(merged["ref_count"].fillna(0).to_numpy())
    return out


def add_out_of_fold(df: pd.DataFrame, month: pd.Series) -> pd.DataFrame:
    """Encode every row from the other months only - no row sees its own month."""
    out = df.copy()
    for c in FEATURES:
        out[c] = np.nan
    for m in month.unique():
        rows = month == m
        table = build_table(df[~rows])
        encoded = apply_table(df[rows], table)
        for c in FEATURES:
            out.loc[rows, c] = encoded[c].to_numpy()
    return out
