"""Smoothed in-fold target encodings of the taxi time and of the clock delta.

merry-quicksand's matched model (fold 222.6 against our ~240) carries 24 such columns: for
each identity column, the training-fold mean of `y` and of `delta = BLOCK - AOBT`, shrunk
toward the global mean with K pseudo-rows. Their permutation importance ranks the
airport x runway x stand encodings second and fifth behind the NM proxy itself. Encodings
must be fitted on the training months only; `fit` returns plain dicts so they serialise
into a model JSON and score identically at submission time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 50.0
KEYS = {
    "adep": lambda r: r["ADEP_mvt"].astype("string"),
    "stand": lambda r: r["ADEP_mvt"].astype("string") + "|" + r["STAND_mvt"].astype("string"),
    "stand_prefix": lambda r: r["ADEP_mvt"].astype("string") + "|" + r["STAND_mvt"].astype("string").str[:2],
    "airline": lambda r: r["FLIGHT_mvt"].astype("string").str.extract(r"^([A-Z]+)")[0].str[:3],
    "ades": lambda r: r["ADES_mvt"].astype("string"),
    "actype": lambda r: r["AIRCRAFT_TYPE_mvt"].astype("string"),
    "runway": lambda r: r["ADEP_mvt"].astype("string") + "|" + r["RUNWAY_mvt"].astype("string"),
    "operator": lambda r: r["AIRCRAFT_OPERATOR_flt"].astype("string"),
    "segment": lambda r: r["MARKET_SEGMENT_flt"].astype("string"),
    "wake": lambda r: r["WK_TBL_CAT_flt"].astype("string"),
    "ftype": lambda r: r["FLIGHT_TYPE_flt"].astype("string"),
    "ars": lambda r: r["ADEP_mvt"].astype("string") + "|" + r["RUNWAY_mvt"].astype("string") + "|" + r["STAND_mvt"].astype("string"),
}
FEATURES = [f"te_{k}_{t}" for k in KEYS for t in ("y", "delta")]


def keys(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    return {k: f(raw).fillna("?").to_numpy() for k, f in KEYS.items()}


SPREAD_KEYS = ("ars", "stand", "runway")
SPREAD_STATS = ("p90", "std")


def fit(raw: pd.DataFrame, y: np.ndarray, delta: np.ndarray, rows: np.ndarray, spread: bool = False) -> dict:
    """Tables from `rows` (training months). delta may be NaN where AOBT is missing.

    With `spread` (log section 63), also the 90th percentile and the standard deviation of y (clipped at 2h so a
    day slip cannot set a cell) per airport x runway x stand, airport x stand and airport x runway, shrunk toward
    the global value with K pseudo-rows: how noisy the cell is, where the means above say only where it sits."""
    kk = keys(raw)
    out = {"k": K, "tables": {}, "global": {}}
    if spread:
        ok = rows & np.isfinite(y)
        yc = np.clip(y[ok], 0, 7200)
        gstat = {"p90": float(np.percentile(yc, 90)), "std": float(np.std(yc))}
        for name in SPREAD_KEYS:
            grp = pd.Series(yc).groupby(kk[name][ok])
            n = grp.size()
            for stat, val in (("p90", grp.quantile(0.9)), ("std", grp.std(ddof=0))):
                g = gstat[stat]; out["global"][f"spread_{stat}"] = g
                out["tables"][f"{name}_y_{stat}"] = ((val * n + K * g) / (n + K)).to_dict()
    for target_name, target in (("y", y), ("delta", delta)):
        ok = rows & np.isfinite(target)
        g = float(np.mean(target[ok])); out["global"][target_name] = g
        for name, key in kk.items():
            s = pd.Series(target[ok]).groupby(key[ok]).agg(["sum", "size"])
            out["tables"][f"{name}_{target_name}"] = ((s["sum"] + K * g) / (s["size"] + K)).to_dict()
    return out


def transform(raw: pd.DataFrame, enc: dict) -> pd.DataFrame:
    kk = keys(raw); cols = {}
    for name in KEYS:
        for t in ("y", "delta"):
            table = enc["tables"][f"{name}_{t}"]; g = enc["global"][t]
            cols[f"te_{name}_{t}"] = pd.Series(kk[name]).map(table).fillna(g).to_numpy(dtype="float64")
    for name in SPREAD_KEYS:      # present only in encodings fitted with spread=True
        for stat in SPREAD_STATS:
            if f"{name}_y_{stat}" in enc["tables"]:
                cols[f"te_{name}_y_{stat}"] = pd.Series(kk[name]).map(enc["tables"][f"{name}_y_{stat}"]).fillna(
                    enc["global"][f"spread_{stat}"]).to_numpy(dtype="float64")
    return pd.DataFrame(cols, index=raw.index)
