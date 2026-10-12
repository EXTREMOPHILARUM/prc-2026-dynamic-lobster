"""Download OurAirports runway geometry for the challenge airports.

Source: https://ourairports.com/data/ (mirrored at github.com/davidmegginson/ourairports-data)
Licence: "All data is released to the Public Domain, and comes with no guarantee of
accuracy or fitness for use." Permitted as open external data under the organiser ruling
of 2026-09-08, and declared here and in docs/.

    uv run python scripts/fetch_ourairports.py
"""
from __future__ import annotations

import urllib.request

from prc import config

BASE = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main"
FILES = ("runways.csv", "airports.csv")
OUT = config.ROOT / "data" / "external" / "ourairports"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT / name
        if target.exists() and target.stat().st_size > 10_000:
            print(f"skip  {name}")
            continue
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=120) as r:
            target.write_bytes(r.read())
        print(f"get   {name}  {target.stat().st_size/1e6:.1f} MB")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
