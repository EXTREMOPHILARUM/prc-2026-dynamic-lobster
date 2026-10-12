"""Recompute metrics from saved component predictions without retraining models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import config, data, decomposition, evaluation, features


def score(name: str, raw: pd.DataFrame) -> None:
    path = config.DATA_INTERIM / f"decomposition_{name}_predictions.parquet"
    predictions = pd.read_parquet(path)
    frame = raw.merge(predictions, on=config.ID_COL, how="inner", validate="one_to_one")
    scale = np.where(
        frame["ADEP_mvt"].eq("LIRF"),
        decomposition.LIRF_TAIL_SCALE,
        decomposition.OTHER_TAIL_SCALE,
    )
    prediction = frame["base"] + scale * frame["tail_probability"] * frame["tail_excess"]
    metrics = evaluation.regression_metrics(frame, frame[config.TARGET], prediction)
    output = config.DATA_INTERIM / f"decomposition_{name}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2))
    print(f"{name}: RMSE {metrics['all']['rmse']:.2f} bulk {metrics['bulk']['rmse']:.2f}")


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    _, valid_idx = evaluation.month_disjoint_split(raw)
    score("holdout", raw.loc[valid_idx])
    score("oof", raw)


if __name__ == "__main__":
    main()
