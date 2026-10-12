#!/usr/bin/env bash
# Run ground_points.py over every ingested day that has traces and no ground.parquet yet, 2025
# first (the labelled screen needs it), then 2026. Restartable: finished days are skipped, and a
# day only counts as finished once ground.parquet exists (written atomically). Launch under setsid
# and stop with `kill -- -$(cat data/interim/adsb_ground.pgid)`, never with pkill -f over ssh.
#   setsid nohup scripts/adsb/extract_ground_all.sh > data/interim/adsb_ground.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
ps -o pgid= $$ | tr -d ' ' > data/interim/adsb_ground.pgid
for dir in $(ls -d data/external/adsb/2025-* data/external/adsb/2026-* 2>/dev/null); do
  day=$(basename "$dir")
  [ -f "$dir/ground.parquet" ] && continue
  [ -d "$dir/traces" ] || { echo "$day: no traces, skipped"; continue; }
  .venv/bin/python -u scripts/adsb/ground_points.py "$day" || echo "$day: FAILED"
  [ -f "$dir/ground.parquet" ] || echo "$day: FAILED no ground.parquet"
done
echo GROUND_DONE
