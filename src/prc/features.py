"""Feature engineering for taxi-out prediction.

Only columns actually present in the frame are used, so this survives schema
surprises in the ranking set (where BLOCK_TIME_UTC_mvt and the target are blanked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config

CATEGORICAL = [
    "ADEP_mvt",
    "ADES_mvt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "FLIGHT_mvt",
    "CALLSIGN_flt",
    "AIRCRAFT_OPERATOR_flt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "MARKET_SEGMENT_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
]

TIME_COL = "MVT_TIME_UTC_mvt"

#: NM off-block/scheduling times left intact in the ranking set. BLOCK_TIME_UTC_mvt is
#: blanked there, but AOBT_3_flt is NM's own actual off-block time for the same flight,
#: so MVT_TIME - AOBT_3_flt is a direct (noisy) reading of the target.
OFFBLOCK_COLS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def add_time_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["hour"] = t.dt.hour
    df["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["doy"] = t.dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # cyclical encodings so the trees do not have to split around midnight
    df["tod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    return df


def add_congestion_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Movements at the same airport in the surrounding hour — a proxy for queueing."""
    if "ADEP_mvt" not in df.columns:
        return df
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    hour_bucket = t.dt.floor("h")
    grp = df.assign(_h=hour_bucket).groupby(["ADEP_mvt", "_h"], observed=True)
    df["adep_movements_hour"] = grp["_h"].transform("size").astype("int32")
    return df


def add_offblock_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Seconds from each known off-block estimate to the movement (takeoff) time."""
    mvt = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    for c in _present(df, OFFBLOCK_COLS):
        df[f"taxi_proxy_{c}"] = (mvt - pd.to_datetime(df[c], utc=True, errors="coerce")).dt.total_seconds()
    present = _present(df, OFFBLOCK_COLS[:-1])
    if present:
        available = pd.concat(
            [pd.to_datetime(df[c], utc=True, errors="coerce").notna() for c in present], axis=1
        )
        df["nm_time_count"] = available.sum(axis=1).astype("int8")
        df["nm_record_missing"] = (~available.any(axis=1)).astype("int8")
    if "AOBT_3_flt" in df.columns:
        aobt = pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
        df["aobt_missing"] = aobt.isna().astype("int8")
        for c in _present(df, ["EOBT_1_flt", "IOBT_flt", "LOBT_flt"]):
            df[f"aobt_minus_{c}"] = (
                aobt - pd.to_datetime(df[c], utc=True, errors="coerce")
            ).dt.total_seconds()
    # how late the flight actually pushed back versus plan — congestion shows up here
    if {"AOBT_3_flt", "EOBT_1_flt"} <= set(df.columns):
        df["offblock_delay"] = (
            pd.to_datetime(df["AOBT_3_flt"], utc=True, errors="coerce")
            - pd.to_datetime(df["EOBT_1_flt"], utc=True, errors="coerce")
        ).dt.total_seconds()
    return df


def build(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature frame, categorical column names).

    LightGBM encodes categoricals by pandas category *code*, so scoring must reuse
    the exact level ordering learned at training time — pass the mapping that
    `category_levels` returned for the training frame. Deriving it per-frame
    silently relabels every category and destroys the predictions.
    """
    df = df.copy()
    df = add_time_features(df, time_col)
    df = add_congestion_features(df, time_col)
    df = add_offblock_features(df, time_col)

    cats = _present(df, CATEGORICAL)
    if categories is not None:
        # Persisted models may predate newly added categoricals.
        cats = [c for c in cats if c in categories]
    for c in cats:
        if categories is not None:
            df[c] = pd.Categorical(df[c], categories=categories[c])
        else:
            df[c] = df[c].astype("category")

    numeric = [
        "hour",
        "minute_of_day",
        "dow",
        "month",
        "doy",
        "is_weekend",
        "tod_sin",
        "tod_cos",
        "adep_movements_hour",
        "nm_time_count",
        "nm_record_missing",
        "aobt_missing",
        "offblock_delay",
        *[f"aobt_minus_{c}" for c in ["EOBT_1_flt", "IOBT_flt", "LOBT_flt"]],
        *[f"taxi_proxy_{c}" for c in OFFBLOCK_COLS],
    ]
    feature_cols = cats + _present(df, numeric)
    return df[feature_cols], cats


def departures(df: pd.DataFrame) -> pd.DataFrame:
    """Departure movements only — arrivals carry taxi-IN time in the same column."""
    return df[df["PHASE_mvt"] == "DEP"]


def departures_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Training rows: every departure carrying a taxi-out time.

    Deliberately unfiltered. 0.07% of rows exceed two hours (block-time artifacts up
    to 24h) and they carry ~67% of the squared error, so dropping them optimises a
    metric the competition does not score and biases every prediction low.
    """
    return departures(df)[departures(df)[config.TARGET].notna()].copy()


def category_levels(X: pd.DataFrame, cats: list[str]) -> dict[str, list]:
    """The level ordering LightGBM baked into the model, for reuse at scoring time."""
    return {c: list(X[c].cat.categories) for c in cats}
