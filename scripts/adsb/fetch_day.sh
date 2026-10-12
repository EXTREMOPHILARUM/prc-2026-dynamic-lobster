#!/usr/bin/env bash
# Download and unpack one ADSB.lol globe_history day (readsb-prod-0) into data/external/adsb/<date>/
# Unauthenticated: public release assets via the GitHub API + curl. Exit 0 iff traces/ exists afterwards.
# usage: scripts/adsb/fetch_day.sh 2025-07-15
set -u
d="$1"; tag="v${d//-/.}-planes-readsb-prod-0"; out="data/external/adsb/$d"; repo="globe_history_${d:0:4}"
mkdir -p "$out"; cd "$out" || exit 1
if [ -d traces ]; then echo "$d: traces already present"; exit 0; fi
urls=$(curl -sL "https://api.github.com/repos/adsblol/$repo/releases/tags/$tag" | python3 -c 'import json,sys; [print(a["browser_download_url"]) for a in json.load(sys.stdin).get("assets",[]) if ".tar." in a["name"]]')
[ -z "$urls" ] && { echo "$d: no assets found for $tag"; exit 1; }
for u in $urls; do curl -sL -o "$(basename "$u")" "$u" || { echo "$d: download failed $u"; exit 1; }; done
cat "$tag".tar.* > "$tag.tar" && rm -f "$tag".tar.a?
tar -xf "$tag.tar"; rm -f "$tag.tar"
n=$(find traces -type f 2>/dev/null | wc -l)
echo "$d: $n trace files, $(du -sh . | cut -f1)"
[ "$n" -gt 0 ]
