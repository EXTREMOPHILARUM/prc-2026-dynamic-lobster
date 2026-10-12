"""Fetch the X-Plane Scenery Gateway airport layout for the ten challenge airports and extract
their stands (apt.dat row code 1300, "startup locations": lat, lon, heading, type, name).

Licence: X-Plane Scenery Gateway airport data is GPL v2 or later; the copyright notice shipped in
each scenery pack is kept next to the data (data/external/xplane/<ICAO>.txt). Declared in the
model wrapper's `external_data`. Lead from the rank-3 team's post on 2026-09-24 (log section 50).

For each airport the Gateway's *recommended* scenery is used. Writes
data/external/xplane/<ICAO>.dat, <ICAO>.txt and data/external/xplane/stands.parquet.

    uv run python -u scripts/fetch_xplane_stands.py
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request
import zipfile

import pandas as pd

from prc import config

API = "https://gateway.x-plane.com/apiv1"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
OUT = config.ROOT / "data" / "external" / "xplane"


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def stands(dat: str, icao: str) -> list[dict]:
    rows = []
    for line in dat.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0] == "1300":
            rows.append({"airport": icao, "lat": float(p[1]), "lon": float(p[2]), "heading": float(p[3]),
                         "kind": p[4], "classes": p[5], "name": " ".join(p[6:]).strip()})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for icao in AIRPORTS:
        sid = get(f"{API}/airport/{icao}")["airport"]["recommendedSceneryId"]
        blob = get(f"{API}/scenery/{sid}")["scenery"]["masterZipBlob"]
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob)))
        dat = z.read(f"{icao}.dat").decode("utf-8", errors="replace")
        (OUT / f"{icao}.dat").write_text(dat)
        if f"{icao}.txt" in z.namelist():
            (OUT / f"{icao}.txt").write_bytes(z.read(f"{icao}.txt"))
        s = stands(dat, icao); allrows += s
        print(f"{icao}: scenery {sid}, {len(s)} stands", flush=True)
    S = pd.DataFrame(allrows); S.to_parquet(OUT / "stands.parquet", index=False)
    print(f"wrote {len(S):,} stands to {OUT / 'stands.parquet'}")


if __name__ == "__main__":
    main()
