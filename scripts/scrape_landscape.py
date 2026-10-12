"""Snapshot the public competition landscape: leaderboard + team registrations.

    uv run python scripts/scrape_landscape.py

Writes data/interim/leaderboard.json and data/interim/teams.json. Both pages are
public; nothing here touches the ranking process itself.
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from prc import config

SITE = "https://prc-data-challenge-2026.netlify.app"
LEADERBOARD = "https://prc-challenge-2026.vercel.app/"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def to_text(segment: str) -> str:
    segment = re.sub(r"(?is)<(script|style).*?</\1>", " ", segment)
    segment = re.sub(r"(?s)<[^>]+>", "\n", segment)
    return re.sub(r"\n\s*\n+", "\n", html.unescape(segment)).strip()


def team_slugs() -> list[str]:
    index = fetch(f"{SITE}/teams/")
    return sorted(set(re.findall(r"teams/([a-z0-9-]+)\.html", index)))


def parse_team(slug: str) -> dict:
    raw = fetch(f"{SITE}/teams/{slug}.html")
    main = re.search(r"(?is)<main.*?</main>", raw)
    body = to_text(main.group(0) if main else raw)
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

    def field(label: str) -> str | None:
        for i, ln in enumerate(lines):
            if ln.lower().startswith(label.lower()):
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ""
                return tail or (lines[i + 1] if i + 1 < len(lines) else None)
        return None

    members = []
    if "Affiliation" in lines:
        j = lines.index("Affiliation") + 1
        while j + 3 < len(lines) and lines[j].isdigit():
            members.append({"name": f"{lines[j + 1]} {lines[j + 2]}", "affiliation": lines[j + 3]})
            j += 4

    n = field("Number of team members")
    return {
        "slug": slug,
        "country": field("Country"),
        "type": field("Type"),
        "n_members": int(n) if n and n.isdigit() else len(members) or None,
        "members": members,
        "narrative": body,
    }


def main() -> None:
    slugs = team_slugs()
    with ThreadPoolExecutor(max_workers=12) as pool:
        teams = list(pool.map(parse_team, slugs))
    (config.DATA_INTERIM / "teams.json").write_text(json.dumps(teams, indent=1))
    print(f"teams: {len(teams)} -> data/interim/teams.json")

    # The leaderboard table is client-rendered; scrape it with Chrome DevTools MCP
    # (evaluate_script over `table tr`) and drop the rows into leaderboard.json as
    # {"scraped": iso8601, "rows": [{rank, team, best, subs, last_activity}, ...]}.
    print(f"leaderboard: render {LEADERBOARD} in a browser; see docstring above")
    print(f"snapshot attempted {datetime.now(UTC).isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
