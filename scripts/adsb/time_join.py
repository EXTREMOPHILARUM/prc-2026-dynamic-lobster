"""Time-only join of a departure to an ADS-B take-off run, for departures the callsign join misses.

A departure with no callsign match is joined to an unclaimed ADS-B take-off run at the same
airport whose take-off falls in [MVT + LO_S, MVT + HI_S] (ADS-B take-off, the last ground point,
sits a median 32s before MVT, 5-95% within [-214s, +1s]), provided the pairing is unique both
ways: exactly one run in the window, and that run in the window of exactly one departure. The
stand check (optional) additionally requires the run to pass within STAND_CHECK_M of the NM
stand, where apt.dat locates it.

`_screen` measures precision on departures whose callsign join is known: hide their callsigns,
re-join them by time among the runs no *other* departure claimed, and count how often the run
picked is the right one.

    uv run python -u scripts/adsb/time_join.py 2025-04-15 2025-10-15
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LO_S, HI_S = -120.0, 30.0
STAND_CHECK_M = 100.0


def time_join(dep: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Unique-both-ways pairs. `dep`: MVT_ID_mvt, ADEP_mvt, mvt. `runs`: icao, run, airport,
    t_takeoff. Returns MVT_ID_mvt, icao, run."""
    out = []
    for apt, g in dep.groupby("ADEP_mvt"):
        r = runs[runs.airport == apt]
        if r.empty:
            continue
        to = r.t_takeoff.to_numpy()
        cand = [np.flatnonzero((to >= m + LO_S) & (to <= m + HI_S)) for m in g.mvt.to_numpy()]
        hits = np.bincount(np.concatenate([c for c in cand if len(c)] or [np.array([], int)]), minlength=len(r))
        for mid, c in zip(g.MVT_ID_mvt.to_numpy(), cand):
            if len(c) == 1 and hits[c[0]] == 1:
                out.append((mid, r.icao.iloc[c[0]], r.run.iloc[c[0]]))
    return pd.DataFrame(out, columns=["MVT_ID_mvt", "icao", "run"])


def _screen(days: list[str]) -> None:
    import glob

    import stand_exit
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    table = stand_exit.stand_table()
    res = []
    for day in days:
        G = pd.read_parquet(f"data/external/adsb/{day}/ground.parquet",
                            columns=["icao", "callsign", "airport", "run", "t", "lat", "lon", "took_off", "t_takeoff"])
        G = G[G.took_off].sort_values(["icao", "run", "t"])
        R = G.drop_duplicates(["icao", "run"])[["icao", "run", "callsign", "airport", "t_takeoff"]].copy()
        R["callsign"] = R.callsign.astype(str).str.strip(); R["airport"] = R.airport.astype(str)
        src = glob.glob(f"data/raw/training_{day[:7]}-01_*.parquet")[0]
        d = pd.read_parquet(src, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt",
                                          "FLIGHT_mvt", "STAND_mvt"])
        d = d[(d.PHASE_mvt == "DEP") & (d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d") == day)].copy()
        d["cs"] = d.CALLSIGN_flt.astype("string").str.strip().fillna(d.FLIGHT_mvt.astype("string").str.strip())
        d["mvt"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - epoch).dt.total_seconds()
        m = d.merge(R, left_on=["cs", "ADEP_mvt"], right_on=["callsign", "airport"])
        m["dto"] = (m.t_takeoff - m.mvt).abs(); m = m[m.dto < 600].sort_values("dto").drop_duplicates("MVT_ID_mvt")
        truth = m.set_index("MVT_ID_mvt")[["icao", "run"]]
        unjoined = d[~d.MVT_ID_mvt.isin(m.MVT_ID_mvt)]
        claimed = set(zip(m.icao, m.run))
        # precision: hide a random tenth of the callsign-joined departures at a time, so the run pool
        # looks like the real one (their runs free, everyone else's claimed)
        rng = np.random.default_rng(0)
        fold = rng.integers(0, 10, len(m))
        k = d[["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt"]].assign(key=d.STAND_mvt.map(stand_exit.stand_key)).merge(
            table, left_on=["ADEP_mvt", "key"], right_on=["airport", "key"], how="left").set_index("MVT_ID_mvt")
        pts = {kk: (g.t.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy()) for kk, g in G.groupby(["icao", "run"], observed=True)}

        def stand_ok(mid, icao, run, k=k, pts=pts):
            s_lat, s_lon = k.loc[mid, "lat"], k.loc[mid, "lon"]
            if not np.isfinite(s_lat):
                return np.nan
            return float(stand_exit.exit_time(*pts[(icao, run)], s_lat, s_lon)[1] < STAND_CHECK_M)

        for f in range(10):
            hidden = m[fold == f]
            freed = set(zip(hidden.icao, hidden.run))
            free = R[[(i, r) not in claimed or (i, r) in freed for i, r in zip(R.icao, R.run)]]
            pool_dep = pd.concat([hidden[["MVT_ID_mvt", "ADEP_mvt", "mvt"]], unjoined[["MVT_ID_mvt", "ADEP_mvt", "mvt"]]])
            J = time_join(pool_dep, free)
            J = J[J.MVT_ID_mvt.isin(hidden.MVT_ID_mvt)]
            for mid, icao, run in J.itertuples(index=False):
                ok = (truth.loc[mid, "icao"], truth.loc[mid, "run"]) == (icao, run)
                res.append({"kind": "hidden", "correct": ok, "stand_ok": stand_ok(mid, icao, run), "n_hidden": len(hidden)})
        J = time_join(unjoined[["MVT_ID_mvt", "ADEP_mvt", "mvt"]], R[[(i, r) not in claimed for i, r in zip(R.icao, R.run)]])
        for mid, icao, run in J.itertuples(index=False):
            res.append({"kind": "new", "correct": np.nan, "stand_ok": stand_ok(mid, icao, run), "n_hidden": 0})
        res.append({"kind": "count", "correct": np.nan, "stand_ok": np.nan, "n_hidden": len(d), "joined": len(m)})
    D = pd.DataFrame(res)
    c = D[D.kind == "count"]; hid = D[D.kind == "hidden"]; new = D[D.kind == "new"]
    n_dep = int(c.n_hidden.sum()); n_cs = int(c.joined.sum())
    print(f"{n_dep:,} departures; callsign-joined {n_cs:,} ({n_cs/n_dep*100:.1f}%)")
    print(f"precision test (callsign hidden): time-joined {len(hid):,} of {n_cs:,} ({len(hid)/n_cs*100:.1f}%), correct {hid.correct.mean()*100:.1f}%")
    s = hid.dropna(subset=["stand_ok"])
    print(f"   with the stand located ({len(s):,}): stand check passes {s.stand_ok.mean()*100:.1f}%;"
          f" correct when it passes {s[s.stand_ok == 1].correct.mean()*100:.1f}%, when it fails {s[s.stand_ok == 0].correct.mean()*100:.1f}%")
    ns = new.dropna(subset=["stand_ok"])
    print(f"new rows from truly unjoined departures: {len(new):,} (+{len(new)/n_dep*100:.1f}% of departures, "
          f"+{len(new)/n_cs*100:.1f}% on the joined set); stand located {len(ns):,}, stand check passes {ns.stand_ok.mean()*100:.1f}%")


if __name__ == "__main__":
    _screen(sys.argv[1:])
