"""Taxiway-network distances from the X-Plane Scenery Gateway apt.dat files (GPL v2+, declared).

apt.dat row 1201 is a taxi-routing node (lat, lon, usage, id, name) and row 1202 an edge between two
node ids. Every airport's routing network is one connected graph. The distance between two points
is the straight line from each to its nearest node plus the shortest path between those nodes: an
aircraft first seen mid-taxi has come along taxiways, not across the grass (log section 60;
section 54's straight-line `own_first_dist` gave -0.11s).
"""
from __future__ import annotations

from functools import cache
from math import cos, radians

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

DAT = "data/external/xplane/{apt}.dat"
MAX_SNAP_M = 300.0   # a point farther than this from every node is off the network: NaN


class Network:
    def __init__(self, apt: str, path: str = DAT):
        nodes: dict[int, tuple[float, float]] = {}; edges: list[tuple[int, int]] = []
        with open(path.format(apt=apt), encoding="latin-1") as fh:
            for line in fh:
                p = line.split()
                if not p:
                    continue
                if p[0] == "1201":
                    nodes[int(p[4])] = (float(p[1]), float(p[2]))
                elif p[0] == "1202":
                    edges.append((int(p[1]), int(p[2])))
        used = sorted({a for e in edges for a in e if a in nodes})
        ix = {k: i for i, k in enumerate(used)}
        lat = np.array([nodes[k][0] for k in used]); lon = np.array([nodes[k][1] for k in used])
        self.lat0 = float(lat.mean()); self.k = 111_000.0 * cos(radians(self.lat0))
        self.x = lon * self.k; self.y = lat * 111_000.0
        e = np.array([(ix[a], ix[b]) for a, b in edges if a in ix and b in ix])
        w = np.hypot(self.x[e[:, 0]] - self.x[e[:, 1]], self.y[e[:, 0]] - self.y[e[:, 1]])
        n = len(used)
        g = coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                       shape=(n, n)).tocsr()
        self.D = dijkstra(g, directed=False)

    def _snap(self, lat: float, lon: float) -> tuple[int, float]:
        d = np.hypot(self.x - lon * self.k, self.y - lat * 111_000.0)
        i = int(np.argmin(d))
        return i, float(d[i])

    def distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        if not (np.isfinite(lat1) and np.isfinite(lat2)):
            return np.nan
        i, di = self._snap(lat1, lon1); j, dj = self._snap(lat2, lon2)
        if di > MAX_SNAP_M or dj > MAX_SNAP_M or not np.isfinite(self.D[i, j]):
            return np.nan
        return di + float(self.D[i, j]) + dj


@cache
def network(apt: str) -> Network:
    return Network(apt)
