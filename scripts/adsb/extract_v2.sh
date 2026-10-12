#!/usr/bin/env bash
# Section 55 extractor for reingest_v2.sh's downloads: 2 loops, each taking the first day (priority order)
# whose traces are complete and which has no ground_v2.parquet yet (7 threads each; more concurrent readers
# made the HDD seek-bound). A day is claimed atomically (mkdir .extracting); a failed day is marked
# .extract_failed and not retried. Exits when every day is extracted, failed, or its fetch failed (a
# <day>.fetch_failed marker, cleared when reingest_v2.sh restarts).
# Launch: setsid nohup scripts/adsb/extract_v2.sh >> data/interim/extract_v2.log 2>&1 &
cd ~/prc_data_challenge
ps -o pgid= $$ | tr -d " " > data/interim/extract_v2.pgid
list() {
  if [ -n "$DAYS_FILE" ]; then cut -d" " -f1 "$DAYS_FILE"; return; fi
  { for m in 01 07; do for i in $(seq -w 1 31); do echo 2026-$m-$i; done; done
    for m in 01 07; do for i in $(seq -w 1 31); do echo 2025-$m-$i; done; done
    for m in 02 03 04 05 06 08 09 10 11 12; do for i in $(seq -w 1 31); do date -d "2025-$m-$i" >/dev/null 2>&1 && echo 2025-$m-$i; done; done
  } | grep -v "^2025-12-31"
}
rootof() { if [ -f data/external/adsb/$1/.complete ]; then echo data/external/adsb; else echo ~/Data/adsb_raw${1:0:4}; fi; }
extractor() {
  while true; do
    pending=0
    for d in $(list); do
      o=data/external/adsb/$d
      [ -f $o/ground_v2.parquet ] || [ -f $o/.extract_failed ] && continue
      [ -f ~/Data/adsb_raw${d:0:4}/$d.fetch_failed ] && continue
      pending=1; root=$(rootof $d)
      [ -f $root/$d/.complete ] || continue
      mkdir -p $o; mkdir $o/.extracting 2>/dev/null || continue
      ADSB_WORKERS=7 .venv/bin/python -u scripts/adsb/ground_points.py $d --v2 --traces-root $root | tail -1
      [ -f $o/ground_v2.parquet ] || { echo "$d: EXTRACT FAILED"; touch $o/.extract_failed; }
      rmdir $o/.extracting; continue 2
    done
    [ $pending = 0 ] && return 0
    sleep 30
  done
}
export DAYS_FILE
export -f list rootof extractor
shopt -s nullglob; for x in data/external/adsb/*/.extracting; do rmdir $x; done
extractor & extractor &
wait; echo EXTRACT_DONE
