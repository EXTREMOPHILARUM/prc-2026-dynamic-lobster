"""Central paths and S3/competition settings, loaded from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

for _p in (DATA_RAW, DATA_INTERIM, MODELS, SUBMISSIONS):
    _p.mkdir(parents=True, exist_ok=True)

S3_ENDPOINT = os.getenv("PRC_S3_ENDPOINT", "https://s3.opensky-network.org")
S3_ACCESS_KEY = os.getenv("PRC_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("PRC_S3_SECRET_KEY", "")

TEAM = os.getenv("PRC_TEAM", "dynamic-lobster")
SUBMIT_BUCKET = os.getenv("PRC_SUBMIT_BUCKET", f"prc-2026-{TEAM}")
DATA_BUCKET = os.getenv("PRC_DATA_BUCKET", "prc-2026-datasets")

# Which scored set to predict: "ranking" (Jan + Jul 2026, the public board) or "final" (the final phase announced
# 2026-10-08: Jan, Feb, Jun, Jul 2026, with movement and block times jittered by up to 12s)
RANKING_SET = os.getenv("PRC_RANKING_SET", "ranking")
if RANKING_SET not in ("ranking", "final"):
    raise ValueError(f"PRC_RANKING_SET must be 'ranking' or 'final', not {RANKING_SET!r}")
RANKING_FILE = "ranking.parquet" if RANKING_SET == "ranking" else "final_ranking.parquet"
TEMPLATE_FILE = "submitting.parquet" if RANKING_SET == "ranking" else "final_submitting.parquet"

TARGET = "TAXITIME_SEC_mvt"
ID_COL = "MVT_ID_mvt"
