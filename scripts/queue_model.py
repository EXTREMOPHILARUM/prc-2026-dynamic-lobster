"""Departure queue model on 2025: unimpeded taxi-out plus additional time in traffic.

Analysis, not a scoring feature. Each ordinary 2025 departure (0 < y <= 3600s; day offsets and
schedule copies are clock failures, not taxis) is described by the traffic it met between its
true off-block (BLOCK_TIME) and take-off, from NM movements of *all* flights (complete, unlike
ADS-B):

    q_ahead      other departures, same airport and runway, already taxiing when it left the stand
    to_next      take-offs from its runway in the fixed WINDOW_S after it left the stand
    arr_next     landings at the airport in the same fixed window (crossings, mixed-mode runways)

All three are fixed at stand exit or over a fixed window. A count over the flight's *own* taxi
(take-offs between off-block and take-off) is circular: a longer taxi contains more take-offs
by construction, so its coefficient is the runway's inter-departure spacing and it leaks y into
any model that uses it. That version was run first and discarded (section 48).

Unimpeded time is the median taxi of flights that left the stand with no departure ahead
(q_ahead == 0), per (airport, stand, runway), falling back to (airport, runway) below MIN_ROWS. Additional time
= taxi - unimpeded, which is the quantity PRC's excess-fuel/CO2 analysis needs. The own ADS-B
trace, where joined, splits the taxi into moving and holding, as a check that the additional
time is queueing.

Because to_during and q_ahead use the flight's own off-block, this cannot be served as is; a
scoring version must solve for the off-block (see the section 48 plan).

    uv run python -u scripts/queue_model.py
"""
from __future__ import annotations

import glob

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MIN_ROWS = 30
WINDOW_S = 600.0
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt",
        "RUNWAY_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_TYPE_mvt"]


def sec(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, utc=True) - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(str(config.DATA_RAW / "training_2025-*.parquet")))],
                  ignore_index=True)
    dep = t[t.PHASE_mvt == "DEP"].copy(); arr = t[t.PHASE_mvt == "ARR"].copy()
    dep["to"] = sec(dep.MVT_TIME_UTC_mvt); dep["ob"] = sec(dep.BLOCK_TIME_UTC_mvt); dep["y"] = dep.to - dep.ob
    arr["t"] = sec(arr.MVT_TIME_UTC_mvt)
    dep["month"] = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True).dt.month
    return dep, arr


def queue_state(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Traffic each departure met between its off-block and take-off (all flights, incl. broken clocks)."""
    dep = dep.copy(); dep["q_ahead"] = 0; dep["to_next"] = 0; dep["arr_next"] = 0
    ok_all = np.isfinite(dep.ob.to_numpy()) & (dep.y.to_numpy() > 0) & (dep.y.to_numpy() <= 3600)
    for (apt, rwy), g in dep[ok_all].groupby(["ADEP_mvt", "RUNWAY_mvt"], observed=True):
        ob = g.ob.to_numpy(); to = g.to.to_numpy()
        so = np.sort(ob); st = np.sort(to)
        # taxiing when i leaves: ob_j < ob_i and to_j > ob_i  ==  #(ob_j < ob_i) - #(to_j <= ob_i)
        q = np.searchsorted(so, ob, "left") - np.searchsorted(st, ob, "right")
        # take-offs in (ob_i, ob_i + WINDOW_S], excluding i itself when its taxi is shorter
        d = np.searchsorted(st, ob + WINDOW_S, "right") - np.searchsorted(st, ob, "right") - (to <= ob + WINDOW_S)
        dep.loc[g.index, "q_ahead"] = np.clip(q, 0, None); dep.loc[g.index, "to_next"] = np.clip(d, 0, None)
    for apt, g in dep[ok_all].groupby("ADEP_mvt"):
        at = np.sort(arr.loc[arr.ADES_mvt == apt, "t"].to_numpy())
        ob = g.ob.to_numpy()
        dep.loc[g.index, "arr_next"] = np.searchsorted(at, ob + WINDOW_S) - np.searchsorted(at, ob)
    return dep[ok_all].copy()


def unimpeded(train: pd.DataFrame, d: pd.DataFrame) -> pd.Series:
    """No-traffic reference for the rows of `d`, from the queue-free flights of `train` only."""
    free = train[train.q_ahead == 0]
    fine = free.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], observed=True).y.agg(["median", "size"])
    coarse = free.groupby(["ADEP_mvt", "RUNWAY_mvt"], observed=True).y.median().rename("coarse")
    k = d[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].merge(fine.reset_index(), how="left", on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]) \
        .merge(coarse.reset_index(), how="left", on=["ADEP_mvt", "RUNWAY_mvt"])
    u = np.where(k["size"].fillna(0).to_numpy() >= MIN_ROWS, k["median"].to_numpy(), k["coarse"].to_numpy())
    return pd.Series(u, index=d.index)


def main():
    dep, arr = load()
    d = queue_state(dep, arr)
    print(f"{len(d):,} ordinary 2025 departures (0 < y <= 3600s)\n")

    # unimpeded time fitted leave-one-month-out, so the additional time of month m never uses month m
    d["unimp"] = np.nan
    for m in range(1, 13):
        tr = d.month != m
        d.loc[~tr, "unimp"] = unimpeded(d[tr], d[~tr]).to_numpy()
    d = d[d.unimp.notna()].copy(); d["add"] = d.y - d.unimp

    print("1. per airport: median taxi, unimpeded, additional (s), and additional share of total taxi time")
    t = d.groupby("ADEP_mvt").agg(n=("y", "size"), taxi=("y", "median"), unimpeded=("unimp", "median"), additional=("add", "median"))
    t["add_share_%"] = (d.groupby("ADEP_mvt")["add"].sum().clip(lower=0) / d.groupby("ADEP_mvt").y.sum() * 100).round(1)
    print(t.round(0).to_string())

    print("\n2. additional taxi by departures already taxiing at stand exit, same runway (all airports): median / mean s")
    d["b"] = pd.cut(d.q_ahead, [-1, 0, 1, 2, 3, 5, 8, 12, 999])
    print(d.groupby("b", observed=True)["add"].agg(["size", "median", "mean"]).round(0).to_string())

    print("\n3. per airport: seconds of additional taxi per aircraft ahead at stand exit, per take-off and per")
    print(f"   landing in the {WINDOW_S/60:.0f} min after stand exit (joint OLS on additional time)")
    for apt, g in d.groupby("ADEP_mvt"):
        A = np.column_stack([np.ones(len(g)), g.q_ahead, g.to_next, g.arr_next]).astype("float64")
        b = np.linalg.lstsq(A, g["add"].to_numpy(dtype="float64"), rcond=None)[0]
        print(f"   {apt}  n={len(g):7,}  {b[1]:+6.1f}s/aircraft ahead  {b[2]:+6.1f}s/take-off  {b[3]:+6.1f}s/landing  (base {b[0]:+5.0f}s)")

    own = pd.read_parquet(config.DATA_INTERIM / "adsb_own_2025.parquet", columns=["MVT_ID_mvt", "own_hold", "own_adsb_taxi"])
    j = d.merge(own, on="MVT_ID_mvt").dropna(subset=["own_hold", "own_adsb_taxi"])
    j = j[(j.own_adsb_taxi > 60) & (j.own_adsb_taxi < 3600)]
    print(f"\n4. ADS-B check on {len(j):,} flights with their own dwell seen: is the additional time holding?")
    print(f"   corr(holding, aircraft ahead) {j.own_hold.corr(j.q_ahead):+.3f};"
          f" corr(holding, additional) {j.own_hold.corr(j['add']):+.3f};"
          f" corr(moving time, additional) {(j.own_adsb_taxi - j.own_hold).corr(j['add']):+.3f}")
    print(j.groupby("b", observed=True).agg(n=("own_hold", "size"), hold=("own_hold", "median"),
          moving=("own_adsb_taxi", lambda s: float(np.median(s - j.loc[s.index, "own_hold"])))).round(0).to_string())

    print("\n5. how much of the taxi the queue explains, leave-one-month-out (RMSE, s)")
    feats = ["unimp", "q_ahead", "to_next", "arr_next"]
    X = d[feats].astype("float64"); X["apt"] = d.ADEP_mvt.astype("category")
    pred = np.full(len(d), np.nan)
    for m in range(1, 13):
        tr = (d.month != m).to_numpy()
        bst = lgb.train({"objective": "regression", "num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 200,
                         "verbose": -1, "num_threads": 2, "seed": 7},
                        lgb.Dataset(X[tr], d.y[tr], categorical_feature=["apt"]), 200)
        pred[~tr] = bst.predict(X[~tr])
    apt_med = d.groupby("ADEP_mvt").y.transform("median")
    for lab, p in (("airport median", apt_med), ("unimpeded (stand x runway, no traffic)", d.unimp), ("unimpeded + queue", pred)):
        print(f"   {lab:40s} {np.sqrt(np.mean((d.y - p) ** 2)):7.1f}")

    out = config.DATA_INTERIM / "queue_scenarios_2025.parquet"
    d.assign(queue_pred=pred)[["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "y", "unimp", "add", "queue_pred",
                               "q_ahead", "to_next", "arr_next"]].to_parquet(out, index=False)
    print(f"\nper-flight scenarios -> {out}")


if __name__ == "__main__":
    main()
