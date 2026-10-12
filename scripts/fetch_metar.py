"""Download METAR observations for the ten challenge airports.

Source: Iowa Environmental Mesonet ASOS/METAR archive, Iowa State University.
  https://mesonet.agron.iastate.edu/ASOS/
Licence: public domain - "The materials found on this website are in the public domain
and may be used freely by anyone for any lawful purpose." Attribution appreciated.
  https://mesonet.agron.iastate.edu/disclaimer.php

Permitted for this challenge: open external data is allowed provided it is declared in
the solution documentation (organiser ruling, OSN Discord #prc-data-competition,
2026-09-08). Note that OpenSky state vectors are NOT permitted (ruling of 2026-09-03).

    uv run python scripts/fetch_metar.py
"""
from __future__ import annotations

import argparse
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

from prc import config

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

#: temperature, dewpoint, wind, precipitation, visibility, gusts, cloud, present weather
#: skyc/skyl pairs give cloud cover and its height in feet AGL, needed for the ceiling
#: (lowest BKN/OVC layer) that drives Low Visibility Procedures.
FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "vsby", "gust",
          "skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3", "wxcodes"]

OUT = config.ROOT / "data" / "external" / "metar"


def url_for(station: str) -> str:
    params = [("station", station), *[("data", f) for f in FIELDS]]
    params += [
        ("year1", "2025"), ("month1", "1"), ("day1", "1"),
        ("year2", "2026"), ("month2", "8"), ("day2", "1"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"),
    ]
    return f"{BASE}?{urllib.parse.urlencode(params)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="destination directory")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for station in AIRPORTS:
        target = out / f"{station}.csv"
        if not args.force and target.exists() and target.stat().st_size > 10_000:
            print(f"skip  {station} ({target.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"get   {station} ...", end="", flush=True)
        body = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url_for(station), timeout=300) as r:
                    body = r.read()
                break
            except urllib.error.HTTPError as exc:  # the free archive throttles under load
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                wait = 15 * (attempt + 1)
                print(f" [{exc.code}, retry in {wait}s]", end="", flush=True)
                time.sleep(wait)
        if body is None:
            raise RuntimeError(f"no data for {station}")
        target.write_bytes(body)
        print(f" {len(body)/1e6:.1f} MB, {body.count(10):,} lines")
        time.sleep(2)  # be polite to a free public archive
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
