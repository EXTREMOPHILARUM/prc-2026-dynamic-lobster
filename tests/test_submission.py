"""Guards for the submission path.

Every check here corresponds to a failure that actually happened in this project:
a filtered holdout, a wrong dtype, a silently dropped feature column, an overwritten
version, a leaking predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc import config, rehearsal


@pytest.fixture
def truth() -> pd.DataFrame:
    return pd.DataFrame({config.ID_COL: [1.0, 2.0, 3.0], config.TARGET: [600, 900, 1200]})


def _submission(ids, values) -> pd.DataFrame:
    return pd.DataFrame({config.ID_COL: ids, config.TARGET: np.int32(values)})


def test_scorer_accepts_a_wellformed_submission(truth):
    result = rehearsal.score(_submission([1.0, 2.0, 3.0], [600, 900, 1200]), truth)
    assert result["status"] == "Succeeded"
    assert result["used_pairs"] == 3
    assert result["score"] == pytest.approx(0.0)


def test_scorer_computes_rmse_not_mae(truth):
    # errors of 0, 0, 300 -> RMSE 173.2, MAE would be 100
    result = rehearsal.score(_submission([1.0, 2.0, 3.0], [600, 900, 1500]), truth)
    assert result["score"] == pytest.approx(300 / np.sqrt(3), rel=1e-6)


def test_scorer_rejects_missing_rows(truth):
    result = rehearsal.score(_submission([1.0, 2.0], [600, 900]), truth)
    assert result["status"] == "Rejected"
    assert "missing" in result["error"]


def test_scorer_rejects_extra_rows(truth):
    result = rehearsal.score(_submission([1.0, 2.0, 3.0, 4.0], [600, 900, 1200, 100]), truth)
    assert result["status"] == "Rejected"
    assert "extra" in result["error"]


def test_scorer_rejects_duplicate_ids(truth):
    result = rehearsal.score(_submission([1.0, 2.0, 2.0], [600, 900, 900]), truth)
    assert result["status"] == "Rejected"
    assert "duplicate" in result["error"]


def test_scorer_rejects_non_finite_predictions(truth):
    bad = pd.DataFrame({config.ID_COL: [1.0, 2.0, 3.0],
                        config.TARGET: [600.0, np.nan, 1200.0]})
    assert rehearsal.score(bad, truth)["status"] == "Rejected"


def test_mock_ranking_blanks_departures_only():
    """The organisers blank BLOCK_TIME and the target for DEP rows, and only those."""
    pytest.importorskip("pyarrow")
    if not list((config.DATA_RAW).glob("training_*.parquet")):
        pytest.skip("training data not downloaded")
    mock, template, truth = rehearsal.build_mock_ranking(months=(1,))
    dep = mock[mock.PHASE_mvt == "DEP"]
    arr = mock[mock.PHASE_mvt == "ARR"]
    assert dep["BLOCK_TIME_UTC_mvt"].isna().all(), "departure block times must be blanked"
    assert dep[config.TARGET].isna().all(), "departure targets must be blanked"
    assert arr["BLOCK_TIME_UTC_mvt"].notna().all(), "arrival rows must be left intact"
    assert len(template) == len(truth) == len(dep)
    assert template[config.ID_COL].equals(truth[config.ID_COL])


def test_departures_filter_excludes_arrivals():
    """Arrivals carry taxi-IN in the same column; training on them was an early bug."""
    from prc import features
    frame = pd.DataFrame({
        "PHASE_mvt": ["DEP", "ARR", "DEP"],
        config.TARGET: [600.0, 300.0, 900.0],
        config.ID_COL: [1.0, 2.0, 3.0],
    })
    assert list(features.departures(frame)["PHASE_mvt"].unique()) == ["DEP"]
    assert len(features.departures_with_target(frame)) == 2


def test_departures_with_target_keeps_extreme_values():
    """Filtering the tail made the holdout read 346 when the real score was 602."""
    from prc import features
    frame = pd.DataFrame({
        "PHASE_mvt": ["DEP", "DEP", "DEP"],
        config.TARGET: [600.0, 88_000.0, np.nan],
        config.ID_COL: [1.0, 2.0, 3.0],
    })
    kept = features.departures_with_target(frame)
    assert len(kept) == 2, "rows with a target must be kept regardless of magnitude"
    assert 88_000.0 in set(kept[config.TARGET])
