"""Calibrate each airport-month's mean to EUROCONTROL's published monthly taxi-out time (log sections 61-62).

Source: https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx (EUROCONTROL AIU),
sheet DATA, monthly per airport. Published mean = 60 * (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) /
TOTAL_REF_NB_FL seconds. The organisers ruled on 2026-09-27 that the January/July 2026 figures may be used
as model inputs (docs/rules-and-compliance.md).

The published mean covers the flights the file could validate, which are close to our rows predicted
under an hour. Per airport it differs from the true mean of those rows by a stable offset, fitted on 2025.
Where the file's valid share (VALID_FL / TF) is at least VMIN, the rows predicted under CAP are shifted
together so their mean equals the published mean minus that offset. Below VMIN the published mean
drops flights and reads low (2025: RMS error 17-115s below 0.985 against 3-7s above), so those
airport-months are left alone. VMIN 0.985 was chosen on 2025 leave-one-month-out alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc import config

PATH = config.ROOT / "data" / "external" / "ansperformance" / "Taxi-Out_Additional_Time.xlsx"
VMIN = 0.985
CAP = 3600.0


def load_published(path=PATH) -> pd.DataFrame:
    """One row per (airport, year, month): published mean seconds and valid share."""
    d = pd.read_excel(path, sheet_name="DATA")
    d = d[d["TOTAL_REF_NB_FL"] > 0]
    return pd.DataFrame({
        "apt": d["APT_ICAO"].astype(str).to_numpy(),
        "year": d["YEAR"].astype(int).to_numpy(),
        "month": d["MONTH_NUM"].astype(int).to_numpy(),
        "pub": (60.0 * (d["TOTAL_REF_TIME_MIN"] + d["TOTAL_ADD_TIME_MIN"]) / d["TOTAL_REF_NB_FL"]).to_numpy(),
        "valid": (d["VALID_FL"] / d["TF"]).to_numpy(),
    })


def _keys(airport, mvt_time) -> pd.DataFrame:
    t = pd.to_datetime(pd.Series(np.asarray(mvt_time)), utc=True)
    return pd.DataFrame({"apt": pd.Series(np.asarray(airport)).astype(str).to_numpy(),
                         "year": t.dt.year.to_numpy(), "month": t.dt.month.to_numpy()})


def fit_offsets(airport, mvt_time, y, pred, published: pd.DataFrame, vmin: float = VMIN,
                cap: float = CAP) -> dict[str, float]:
    """Per airport: mean over its valid months of (published mean - true mean of rows predicted under cap)."""
    k = _keys(airport, mvt_time)
    k["y"] = np.asarray(y, dtype="float64")
    sel = k[np.asarray(pred, dtype="float64") < cap]
    true = sel.groupby(["apt", "year", "month"])["y"].mean().rename("true").reset_index()
    m = true.merge(published, on=["apt", "year", "month"])
    m = m[m["valid"] >= vmin]
    return {a: float(v) for a, v in (m["pub"] - m["true"]).groupby(m["apt"]).mean().items()}


def shifts(airport, mvt_time, pred, published: pd.DataFrame, offsets: dict[str, float],
           vmin: float = VMIN, cap: float = CAP) -> pd.DataFrame:
    """The shift each eligible airport-month would receive, with the inputs that set it."""
    k = _keys(airport, mvt_time)
    k["pred"] = np.asarray(pred, dtype="float64")
    sel = k[k["pred"] < cap]
    ours = sel.groupby(["apt", "year", "month"])["pred"].agg(["mean", "size"]).reset_index()
    m = ours.merge(published, on=["apt", "year", "month"])
    m["offset"] = m["apt"].map(offsets)
    m = m[(m["valid"] >= vmin) & m["offset"].notna()].copy()
    m["shift"] = m["pub"] - m["offset"] - m["mean"]
    return m.reset_index(drop=True)


def apply(pred, airport, mvt_time, published: pd.DataFrame, offsets: dict[str, float],
          vmin: float = VMIN, cap: float = CAP) -> tuple[np.ndarray, np.ndarray]:
    """Shifted predictions, and a mask of the rows that were shifted."""
    out = np.asarray(pred, dtype="float64").copy()
    k = _keys(airport, mvt_time)
    s = shifts(airport, mvt_time, out, published, offsets, vmin, cap)
    shift = k.merge(s[["apt", "year", "month", "shift"]], on=["apt", "year", "month"], how="left")["shift"]
    rows = (out < cap) & shift.notna().to_numpy()
    out[rows] = np.clip(out[rows] + shift.to_numpy()[rows], 0.0, None)
    return out, rows
