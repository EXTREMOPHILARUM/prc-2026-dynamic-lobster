"""Does the departing aircraft's ground time before take-off flag day slips? (log section 63)

Both Paris day slips of Jan 2025 were aircraft grounded 32-41h at LFPG across the recorded block.
Question: among departures, how many have the aircraft on the ground at ADEP for more than H hours
before take-off, and how many of those are day slips (y > 12h)?

Join: the departure's leg is an airborne leg (scripts/adsb/air_legs.py) starting within KM of ADEP
below ALT_FT, with callsign == CALLSIGN_flt (falling back to FLIGHT_mvt), first airborne point in
[MVT - 120s, MVT + 900s]. Ground time = MVT - the end of the aircraft's last leg ending PREV_GAP_S before MVT, counted
only if that previous leg ended within KM of ADEP below ALT_FT (it landed here); otherwise unknown.

The hedge column is the in-sample squared-error saving of shifting the flagged rows by their mean
v48-equivalent residual (n x mean^2): an upper bound, fitted on the rows it is scored on.

    uv run python -u scripts/experiment_ground_idle.py --legs data/interim/air_legs
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prc.features import OFFBLOCK_COLS

AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4775, -0.4614), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
            "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
KM = 20.0
ALT_FT = 5000.0
DAY_SLIP_S = 12 * 3600.0
HOURS = [8, 12, 16, 20, 24, 30]
PREV_GAP_S = 600.0


def km_to(lat: np.ndarray, lon: np.ndarray, ap: np.ndarray) -> np.ndarray:
    alat = np.array([AIRPORTS[a][0] for a in ap]); alon = np.array([AIRPORTS[a][1] for a in ap])
    return np.hypot((lon - alon) * 111.0 * np.cos(np.radians(alat)), (lat - alat) * 111.0)


def low(alt: pd.Series) -> np.ndarray:
    return (alt.isna() | (alt < ALT_FT)).to_numpy()


def departures() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "CALLSIGN_flt", "FLIGHT_mvt",
            *OFFBLOCK_COLS]
    t = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
    d = t[(t.PHASE_mvt == "DEP") & t.ADEP_mvt.isin(AIRPORTS)].copy()
    d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
    d["mvt"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    d["norec"] = ~pd.concat([pd.to_datetime(d[c], utc=True, errors="coerce").notna() for c in OFFBLOCK_COLS[:-1]],
                            axis=1).any(axis=1)
    d["sched"] = (pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    d["y"] = d.TAXITIME_SEC_mvt.astype("float64")
    return d[["MVT_ID_mvt", "ADEP_mvt", "cs", "mvt", "sched", "norec", "y"]].reset_index(drop=True)


def legs(root: str) -> pd.DataFrame:
    """All 2025 legs, strings dictionary-encoded (~70M rows; object strings would not fit in memory)."""
    cols = ["icao", "reg", "callsign", "t0", "lat0", "lon0", "alt0", "t1", "lat1", "lon1", "alt1", "n"]
    T = pa.concat_tables([pq.read_table(f, columns=cols) for f in sorted(glob.glob(f"{root}/2025-*.parquet"))])
    for c in ["icao", "reg", "callsign"]:
        T = T.set_column(T.schema.get_field_index(c), c, T[c].dictionary_encode())
    L = T.unify_dictionaries().to_pandas()
    del T
    # a leg crossing midnight appears in both day files; keep the longer copy
    L["icode"] = L.icao.cat.codes.astype("int32")
    L = L.sort_values(["icode", "t0", "n"]).drop_duplicates(["icode", "t0"], keep="last").reset_index(drop=True)
    return L


def nearest_airport(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.full(len(lat), None, dtype=object); best = np.full(len(lat), KM)
    for a, (alat, alon) in AIRPORTS.items():
        dist = np.hypot((lon - alon) * 111.0 * np.cos(np.radians(alat)), (lat - alat) * 111.0)
        hit = dist < best
        out[hit] = a; best[hit] = dist[hit]
    return out


def join(d: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    # candidate take-off legs: first airborne point low and within KM of one of the ten airports
    C = L.loc[low(L.alt0), ["icode", "reg", "callsign", "t0", "lat0", "lon0"]]
    C = C.assign(apt0=nearest_airport(C.lat0.to_numpy(), C.lon0.to_numpy()))
    C = C[C.apt0.notna() & C.callsign.notna()]
    C = C.assign(callsign=C.callsign.astype(str), apt0=C.apt0.astype(str), reg=C.reg.astype(object))
    # the departure's leg: the first candidate with the same callsign at ADEP starting in [MVT - 120s, MVT + 900s]
    q = d[d.cs.notna()].assign(cs=lambda x: x.cs.astype(str), ADEP_mvt=lambda x: x.ADEP_mvt.astype(str), key=lambda x: x.mvt - 120).sort_values("key")
    m = pd.merge_asof(q, C.sort_values("t0"), left_on="key", right_on="t0", left_by=["cs", "ADEP_mvt"],
                      right_by=["callsign", "apt0"], direction="forward", tolerance=1020.0)
    m = m[m.t0.notna()].copy()
    m["icode"] = m.icode.astype("int32")
    # the previous leg is the aircraft's last one ending PREV_GAP_S before take-off: a departure whose
    # climb-out is split by a stray ground point or a reception gap must not count as its own inbound
    m["key"] = m.mvt - PREV_GAP_S
    P = L[["icode", "t1", "lat1", "lon1", "alt1", "callsign"]].rename(columns=lambda c: f"prev_{c}" if c != "icode" else c)
    m = pd.merge_asof(m.sort_values("key"), P.sort_values("prev_t1"), left_on="key", right_on="prev_t1",
                      by="icode", direction="backward")
    here = m.prev_t1.notna() & low(m.prev_alt1)
    here &= km_to(m.prev_lat1.fillna(0).to_numpy(), m.prev_lon1.fillna(0).to_numpy(), m.ADEP_mvt.to_numpy()) < KM
    m["ground_h"] = np.where(here, (m.mvt - m.prev_t1) / 3600.0, np.nan)
    m["t_land_prev"] = m.prev_t1.where(here)
    m["cs_prev"] = m.prev_callsign.astype(object).where(here)
    m = m.rename(columns={"icode": "icao"})
    return d.merge(m[["MVT_ID_mvt", "icao", "reg", "ground_h", "t_land_prev", "cs_prev"]], on="MVT_ID_mvt", how="left")


def table(g: pd.DataFrame, name: str) -> None:
    slip = g.y > DAY_SLIP_S
    print(f"\n{name}: {len(g):,} departures, {int(slip.sum())} day slips; joined {g.icao.notna().mean():.1%}, "
          f"ground time known {g.ground_h.notna().mean():.1%}; day slips joined {int(g.icao[slip].notna().sum())}, "
          f"ground time known {int(g.ground_h[slip].notna().sum())}")
    print(f"{'ground >':>9} {'flagged':>8} {'slips':>6} {'hit rate':>9} {'hedge SSE saved':>16} {'slip SSE':>10}")
    slip_sse = float((g.res[slip] ** 2).sum())
    for h in HOURS:
        f = g.ground_h > h
        n = int(f.sum()); k = int((f & slip).sum())
        saved = n * float(g.res[f].mean()) ** 2 if n else 0.0
        print(f"{h:>8}h {n:>8,} {k:>6} {k / max(n, 1):>9.2%} {saved / 1e6:>14,.1f}M {slip_sse / 1e6:>9,.0f}M")


def screen(g: pd.DataFrame) -> None:
    """General use: does ground time or turnaround slack track the v48 residual (not the target)?
    Rows with y <= 12h only, so the day slips do not dominate; residual means by bin, per airport
    correlation of the residual with log ground time and with slack (SCHED - inbound landing)."""
    r = g[(g.y <= DAY_SLIP_S) & g.ground_h.notna()].copy()
    r["slack_min"] = (r.sched - r.t_land_prev) / 60.0
    print(f"\nresidual screen: {len(r):,} rows with ground time (y <= 12h), residual RMSE {np.sqrt((r.res ** 2).mean()):.1f}")
    for col, edges in [("ground_h", [0, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 1e9]),
                       ("slack_min", [-1e9, 0, 20, 30, 40, 50, 60, 90, 180, 1e9])]:
        b = pd.cut(r[col], edges)
        t = r.groupby(b, observed=True).res.agg(["size", "mean", "std"])
        t["mean_sq_share"] = t["size"] * t["mean"] ** 2 / float((r.res ** 2).sum())
        print(f"\n{col}:\n{t.round(3).to_string()}")
    landing_bound(g)
    print("\nper airport, corr(residual, log ground_h) / corr(residual, slack_min) / rows / norec rows:")
    for a, q in r.groupby("ADEP_mvt"):
        print(f"  {a} {q.res.corr(np.log1p(q.ground_h)):+.3f} {q.res.corr(q.slack_min.clip(-120, 600)):+.3f} "
              f"{len(q):>8,} {int(q.norec.sum()):>6,}")


def landing_bound(g: pd.DataFrame) -> None:
    """Physical bound: off-block cannot precede the inbound landing, so y <= MVT - t_land_prev.
    How often does the *recorded* target break it, and what does clipping the prediction to
    since_land - c do to squared error (all rows with a known inbound landing, day slips included)?"""
    r = g[g.t_land_prev.notna()].copy()
    r["since_land"] = r.mvt - r.t_land_prev
    print(f"\nlanding bound: {len(r):,} rows; recorded y > since_land on {int((r.y > r.since_land).sum()):,} "
          f"({(r.y > r.since_land).mean():.3%}); prediction > since_land on {int((r.prediction > r.since_land).sum()):,}")
    v = r[r.y > r.since_land]
    if len(v):
        print("  violators by airport / norec:", v.groupby(["ADEP_mvt", "norec"]).size().to_dict())
        print(f"  violators: median y - since_land {float((v.y - v.since_land).median()):.0f}s, "
              f"median since_land {float(v.since_land.median()) / 60:.1f} min")
    base = float(((r.y - r.prediction) ** 2).sum())
    for c in [0, 180, 300, 420]:
        p = np.minimum(r.prediction, np.maximum(r.since_land - c, 0))
        moved = int((p < r.prediction).sum())
        d = float(((r.y - p) ** 2).sum()) - base
        print(f"  clip at since_land - {c:>3}s: {moved:>6,} rows moved, SSE delta {d / 1e6:+,.2f}M "
              f"(norec {float(((r.y - p) ** 2 - (r.y - r.prediction) ** 2)[r.norec].sum()) / 1e6:+,.2f}M)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="data/interim/air_legs")
    ap.add_argument("--oof", default="data/interim/v48_equiv_oof.parquet")
    ap.add_argument("--save", help="write the per-departure join (ground time, inbound landing and callsign) here")
    a = ap.parse_args()
    g = join(departures(), legs(a.legs))
    if a.save:
        g[["MVT_ID_mvt", "icao", "reg", "ground_h", "t_land_prev", "cs_prev"]].assign(reg=lambda x: x.reg.astype(str)) \
            .to_parquet(a.save, index=False)
    oof = pd.read_parquet(a.oof)
    g = g.merge(oof, on="MVT_ID_mvt", how="left")
    g["res"] = g.y - g.prediction
    print(f"OOF coverage {g.prediction.notna().mean():.1%}")
    g = g[g.prediction.notna()]
    table(g, "all departures")
    table(g[g.norec], "no-record departures")
    table(g[g.norec & (g.ADEP_mvt != "LIRF")], "no-record departures outside LIRF")
    table(g[~g.norec & (g.ADEP_mvt != "LIRF")], "matched departures outside LIRF")
    screen(g)
    s = g[g.y > DAY_SLIP_S]
    print("\nday slips:")
    print(s[["ADEP_mvt", "cs", "reg", "norec", "y", "prediction", "ground_h"]]
          .assign(mvt=pd.to_datetime(s.mvt, unit="s")).sort_values("mvt").to_string(index=False))


if __name__ == "__main__":
    main()
