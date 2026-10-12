"""Time-aware splits and diagnostics for model comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import features


def movement_month(df: pd.DataFrame) -> pd.Series:
    """Calendar month of each movement, preserving the frame index."""
    return pd.to_datetime(df[features.TIME_COL], utc=True, errors="coerce").dt.month


def month_disjoint_split(
    df: pd.DataFrame, valid_months: tuple[int, ...] = (1, 7)
) -> tuple[pd.Index, pd.Index]:
    """Return train and validation indices with complete months held out."""
    is_valid = movement_month(df).isin(valid_months)
    return df.index[~is_valid], df.index[is_valid]


def regression_metrics(
    raw: pd.DataFrame,
    y: np.ndarray | pd.Series,
    pred: np.ndarray | pd.Series,
    tail_threshold: float = 3600,
) -> dict[str, object]:
    """Return additive SSE diagnostics and presentation RMSEs."""
    actual = np.asarray(y, dtype="float64")
    predicted = np.asarray(pred, dtype="float64")
    if len(raw) != len(actual) or actual.shape != predicted.shape:
        raise ValueError("raw, y, and pred must describe the same rows")

    squared_error = np.square(actual - predicted)
    tail = actual > tail_threshold
    airport = raw["ADEP_mvt"].astype("string").to_numpy()
    month = movement_month(raw).to_numpy()

    def segment(mask: np.ndarray) -> dict[str, float | int]:
        n = int(mask.sum())
        sse = float(squared_error[mask].sum())
        return {"n": n, "sse": sse, "rmse": float(np.sqrt(sse / n)) if n else float("nan")}

    return {
        "all": segment(np.ones(len(actual), dtype=bool)),
        "bulk": segment(~tail),
        "tail": segment(tail),
        "lirf": segment(airport == "LIRF"),
        "non_lirf": segment(airport != "LIRF"),
        "by_month": {
            str(int(value)): segment(month == value)
            for value in np.unique(month[~pd.isna(month)])
        },
        "by_airport": {
            str(value): segment(airport == value)
            for value in np.unique(airport[~pd.isna(airport)])
        },
    }
