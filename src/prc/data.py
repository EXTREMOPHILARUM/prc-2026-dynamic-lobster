"""Loading the raw competition parquet files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prc import config


def training_files() -> list[Path]:
    return sorted(config.DATA_RAW.glob("training_*.parquet"))


def load_training(months: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate the monthly training files (optionally only the last `months`)."""
    files = training_files()
    if not files:
        raise FileNotFoundError(
            f"No training_*.parquet in {config.DATA_RAW}. Run: uv run python -m prc.s3 pull"
        )
    if months is not None:
        files = files[-months:]
    return pd.concat(
        [pd.read_parquet(f, columns=columns) for f in files], ignore_index=True
    )


def load_ranking(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_RAW / config.RANKING_FILE, columns=columns)


def load_submission_template() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_RAW / config.TEMPLATE_FILE)
