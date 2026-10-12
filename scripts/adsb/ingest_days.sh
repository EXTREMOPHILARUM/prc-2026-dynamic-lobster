#!/usr/bin/env bash
# Fetch, extract segments, and delete raw traces for a list of days. Keeps only segments.parquet.
# usage: scripts/adsb/ingest_days.sh 2025-01-01 2025-01-02 ...
set -uo pipefail
for d in "$@"; do
  if [ -s "data/external/adsb/$d/segments.parquet" ]; then echo "$d: done"; continue; fi
  scripts/adsb/fetch_day.sh "$d" || { echo "$d: fetch failed"; continue; }
  uv run python -u scripts/adsb/pilot_day.py "$d" --segments-only 2>&1 | grep -v Warning | tail -1
  # verify the parquet actually exists: the extractor can fail while the log still looks healthy
  [ -s "data/external/adsb/$d/segments.parquet" ] || { echo "$d: EXTRACTION FAILED, no segments.parquet"; continue; }
  rm -rf "data/external/adsb/$d/traces" "data/external/adsb/$d/acas"
  echo "$d: $(date -u +%H:%M) segments kept, traces removed"
done
