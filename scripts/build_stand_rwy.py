"""Stand-to-runway taxi distance per (airport, NM stand, departure runway) (log section 63; idea from team
outgoing-boat's public repo).

For every combination seen in the 2025 training departures and the 2026 ranking departures: the X-Plane (GPL v2+)
stand position (`stand_exit.stand_table`, the same name matching as the own-flight features), the threshold of the
departure runway end (apt.dat row 100, the end whose designator equals RUNWAY_mvt), and two distances between them:
along the taxiway network (`taxi_network`, NaN when either end is more than 300m from a node) and in a straight line.
Written to data/interim/stand_rwy_dist.parquet, which `prc.standrwy` joins on the raw strings.

    uv run python -u scripts/build_stand_rwy.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "adsb"))
import stand_exit
import taxi_network

from prc import config, data, features

DAT = "data/external/xplane/{apt}.dat"


def thresholds(apt: str) -> dict[str, tuple[float, float]]:
    """Runway end designator -> (lat, lon) of its threshold, from apt.dat row 100."""
    out = {}
    with open(DAT.format(apt=apt), encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("100 "):
                f = line.split()
                out[f[8]] = (float(f[9]), float(f[10]))
                out[f[17]] = (float(f[18]), float(f[19]))
    return out


def line_m(la1, lo1, la2, lo2) -> float:
    return float(np.hypot((lo2 - lo1) * 111_000.0 * np.cos(np.radians(la1)), (la2 - la1) * 111_000.0))


def main() -> None:
    train = features.departures(data.load_training())
    rank = features.departures(pd.read_parquet(config.DATA_RAW / "ranking.parquet"))
    cols = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    combos = pd.concat([train[cols], rank[cols]]).astype("string").dropna().drop_duplicates()
    table = stand_exit.stand_table()
    pos = {(a, k): (la, lo) for a, k, la, lo in zip(table.airport, table.key, table.lat, table.lon)}
    rows = []
    for apt, g in combos.groupby("ADEP_mvt"):
        thr = thresholds(apt); net = taxi_network.network(apt)
        for stand, rwy in zip(g.STAND_mvt, g.RUNWAY_mvt):
            p = pos.get((apt, stand_exit.stand_key(stand))); t = thr.get(rwy)
            if p is None or t is None or not np.isfinite(p[0]):
                continue
            rows.append((apt, stand, rwy, net.distance(p[0], p[1], t[0], t[1]), line_m(p[0], p[1], t[0], t[1])))
    out = pd.DataFrame(rows, columns=[*cols, "srw_net_m", "srw_line_m"])
    dst = config.DATA_INTERIM / "stand_rwy_dist.parquet"
    out.to_parquet(dst, index=False)
    print(f"{len(combos):,} combinations, {len(out):,} located ({out.srw_net_m.notna().sum():,} on the network) -> {dst}")
    print(out.groupby("ADEP_mvt").agg(n=("srw_line_m", "size"), net_med=("srw_net_m", "median"), line_med=("srw_line_m", "median")).round(0))


if __name__ == "__main__":
    main()
