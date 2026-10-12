#!/usr/bin/env bash
# Section 55: rebuild ground_v2.parquet for every day, keeping all raw traces on the HDD.
#   downloader: 4 parallel authenticated, resumable fetches (per-asset curl, HTTP/1.1) -> ~/Data/adsb_raw<year>/<day>/traces, `.complete` written once unpacked;
#               stops fetching when the HDD has < MIN_FREE_GB left
#   extractor:  2 loops, each taking the first complete, unextracted day in priority order (7 threads each;
#               more concurrent readers made the HDD seek-bound)
# The recovered 2025 days (section 55) stay where they are, in data/external/adsb/<day>/traces.
# Restartable: complete days are not re-fetched, extracted days (ground_v2.parquet) are skipped.
# Launch: setsid nohup scripts/adsb/reingest_v2.sh >> data/interim/reingest_v2.log 2>&1 &
# Stop:   kill -- -$(cat data/interim/reingest_v2.pgid)
cd ~/prc_data_challenge
ps -o pgid= $$ | tr -d " " > data/interim/reingest_v2.pgid
export MIN_FREE_GB=60
export GH_TOKEN=$(gh auth token)
rm -f ~/Data/adsb_raw2025/*.fetch_failed ~/Data/adsb_raw2026/*.fetch_failed
days() {
  # DAYS_FILE (lines "<day> <variant>", e.g. the final phase's Feb/Jun 2026) replaces the default list
  if [ -n "$DAYS_FILE" ]; then cat "$DAYS_FILE"; return; fi
  for m in 01 07; do for i in $(seq -w 1 31); do echo "2026-$m-$i prod-0"; done; done
  for m in 01 07; do for i in $(seq -w 1 31); do echo "2025-$m-$i prod-0"; done; done
  for m in 02 03 04 05 06 08 09 10 11 12; do for i in $(seq -w 1 31); do date -d "2025-$m-$i" >/dev/null 2>&1 && echo "2025-$m-$i prod-0"; done; done
} 
list() { days | sed -E "s/^(2025-01-11) prod-0/\1 staging-0/; s/^(2025-05-(28|29|30|31)|2025-06-(0[1-9]|1[01])) prod-0/\1 staging-0tmp/" | grep -v "^2025-12-31"; }
rootof() {  # where a day's traces live (or will)
  d=$1
  if [ -d data/external/adsb/$d/traces ] && [ -f data/external/adsb/$d/.complete ]; then echo data/external/adsb; else echo ~/Data/adsb_raw${d:0:4}; fi
}
get_asset() {  # get_asset <repo> <asset_id> <file>: authenticated, HTTP/1.1, resumable; a fresh signed URL each try
  for try in 1 2 3 4 5 6 7 8 9 10; do
    curl -sfL --http1.1 -C - --retry 3 --retry-delay 5 -o "$3" \
      -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/octet-stream" \
      "https://api.github.com/repos/$1/releases/assets/$2" && return 0
    sleep 10
  done
  return 1
}
fetch() {
  d=$1; v=$2; base=~/Data/adsb_raw${d:0:4}; out=$base/$d
  [ -f data/external/adsb/$d/ground_v2.parquet ] && return 0
  [ -f $out/.complete ] || [ -f data/external/adsb/$d/.complete ] && return 0
  free=$(df -BG --output=avail ~/Data | tail -1 | tr -dc 0-9)
  [ "$free" -lt "$MIN_FREE_GB" ] && { echo "$d: FETCH SKIPPED, HDD free ${free}G"; touch $base/$d.fetch_failed; return 0; }
  rm -rf $out; mkdir -p $out
  ( cd $out || exit 1
    tag="v${d//-/.}-planes-readsb-$v"; repo=adsblol/globe_history_${d:0:4}
    # anonymous downloads share a per-IP cap; gh release download (HTTP/2, no resume) died with
    # PROTOCOL_ERROR on hour-long streams, so: asset ids from the API, then resumable curl per asset
    gh api repos/$repo/releases/tags/$tag --jq '.assets[] | select(.name | startswith("'$tag'.tar")) | "\(.id) \(.name) \(.size)"' > assets.txt || exit 1
    [ -s assets.txt ] || exit 1
    while read -r id name size; do
      get_asset $repo $id $name || exit 1
      [ "$(stat -c %s $name)" = "$size" ] || exit 1
    done < assets.txt
    if ls $tag.tar.a? >/dev/null 2>&1; then cat $tag.tar.a? > $tag.tar && rm -f $tag.tar.a?; fi
    [ -s $tag.tar ] || exit 1
    tar -xf $tag.tar && rm -f $tag.tar assets.txt && touch .complete
  ) || { rm -rf $out; echo "$d: FETCH FAILED"; touch $base/$d.fetch_failed; return 0; }
  echo "$d: fetched $(du -sh $out | cut -f1)"
}
extract() {  # extract one day whose traces are complete; atomic claim so two extractors never share a day
  d=$1; root=$(rootof $d)
  mkdir data/external/adsb/$d/.extracting 2>/dev/null || return 0
  ADSB_WORKERS=7 .venv/bin/python -u scripts/adsb/ground_points.py $d --v2 --traces-root $root | tail -1
  [ -f data/external/adsb/$d/ground_v2.parquet ] || echo "$d: EXTRACT FAILED"
  rmdir data/external/adsb/$d/.extracting
}
extractor() {  # whichever days are ready, in priority order, until the downloader is done and nothing is left
  while true; do
    ready=0
    for d in $(list | cut -d" " -f1); do
      [ -f data/external/adsb/$d/ground_v2.parquet ] && continue
      root=$(rootof $d); [ -f $root/$d/.complete ] || continue
      [ -d data/external/adsb/$d/.extracting ] && continue
      ready=1; mkdir -p data/external/adsb/$d; extract $d; break
    done
    if [ $ready = 0 ]; then [ -f data/interim/reingest_v2.dl_done ] && return 0; sleep 30; fi
  done
}
export DAYS_FILE
export -f days list rootof get_asset fetch extract extractor
rm -f data/interim/reingest_v2.dl_done; rmdir data/external/adsb/*/.extracting 2>/dev/null
( list | xargs -P 4 -L 1 bash -c 'fetch $0 $1'; touch data/interim/reingest_v2.dl_done; echo DOWNLOAD_DONE ) &
wait; echo REINGEST_DONE   # extraction: scripts/adsb/extract_v2.sh
