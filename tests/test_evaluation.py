import numpy as np
import pandas as pd
import pytest

from prc import evaluation


def test_month_disjoint_split_preserves_indices() -> None:
    raw = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-07-01"], utc=True
            )
        },
        index=[4, 8, 12],
    )

    train, valid = evaluation.month_disjoint_split(raw)

    assert train.tolist() == [8]
    assert valid.tolist() == [4, 12]


def test_regression_metrics_segments_add_to_total() -> None:
    raw = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-02-01"], utc=True
            ),
            "ADEP_mvt": ["LIRF", "EDDF", "LIRF"],
        }
    )

    metrics = evaluation.regression_metrics(raw, [100, 4000, 200], [110, 3900, 180])

    assert metrics["all"]["sse"] == pytest.approx(10500)
    assert metrics["bulk"]["sse"] + metrics["tail"]["sse"] == metrics["all"]["sse"]
    assert metrics["lirf"]["sse"] + metrics["non_lirf"]["sse"] == metrics["all"]["sse"]
    assert metrics["all"]["rmse"] == pytest.approx(np.sqrt(3500))
