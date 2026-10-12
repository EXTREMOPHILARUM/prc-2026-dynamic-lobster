#!/usr/bin/env bash
# Fetch one ADSB.lol day WITHOUT the GitHub API: asset URLs are constructed from the tag, so a
# parallel run is not capped by the 60/hour unauthenticated API limit (that cap is what made the
# six-worker run report every day as "no assets"). Days over 2 GB are split into .tar.aa, .tar.ab,
# ... (read until one 404s); smaller days are a single .tar, which the parts-only version missed
# and reported as absent (2025-01-01/11/18/19/21, 2025-12-25). Days whose prod-0 is absent are
# published under another replica (upstream PREFERRED_RELEASES.txt), e.g. staging-0tmp for
# 2025-05-28..06-11: pass it as the second argument.
# usage: fetch_direct.sh 2025-07-21 [prod-0|staging-0|staging-0tmp|...]
set -u
d="$1"; v="${2:-prod-0}"; tag="v${d//-/.}-planes-readsb-$v"; out="data/external/adsb/$d"
base="https://github.com/adsblol/globe_history_${d:0:4}/releases/download/$tag"
mkdir -p "$out"; cd "$out" || exit 1
[ -d traces ] && { echo "$d: traces already present"; exit 0; }
got=0
for sfx in aa ab ac ad; do
  f="$tag.tar.$sfx"
  code=$(curl -sL -w "%{http_code}" -o "$f" "$base/$f")
  if [ "$code" = "200" ] && [ -s "$f" ]; then got=$((got+1)); else rm -f "$f"; break; fi
done
if [ "$got" -gt 0 ]; then
  cat "$tag".tar.a? > "$tag.tar" && rm -f "$tag".tar.a?
else
  code=$(curl -sL -w "%{http_code}" -o "$tag.tar" "$base/$tag.tar")
  [ "$code" = "200" ] && [ -s "$tag.tar" ] || { rm -f "$tag.tar"; echo "$d: no assets at $base"; exit 1; }
  got=1
fi
tar -xf "$tag.tar"; rm -f "$tag.tar"
n=$(find traces -type f 2>/dev/null | wc -l)
echo "$d: $n trace files, $got parts ($v), $(du -sh . | cut -f1)"
[ "$n" -gt 0 ]
