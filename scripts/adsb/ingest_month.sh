#!/usr/bin/env bash
# usage: scripts/adsb/ingest_month.sh 2025-07   (all days of that month, sequential, traces deleted after extraction)
set -uo pipefail
ym="$1"; y=${ym:0:4}; m=${ym:5:2}
n=$(python3 -c "import calendar; print(calendar.monthrange($y,$((10#$m)))[1])")
for i in $(seq 1 "$n"); do
  d=$(printf "%s-%02d" "$ym" "$i")
  scripts/adsb/ingest_days.sh "$d"
done
