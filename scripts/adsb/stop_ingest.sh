#!/usr/bin/env bash
pkill -f "ingest_month.sh" ; pkill -f "ingest_days.sh"; pkill -f "fetch_day.sh"; pkill -f "pilot_day.py"; pkill -f "curl -sL -o"; echo stopped
