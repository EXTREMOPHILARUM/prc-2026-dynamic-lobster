import numpy as np
import pandas as pd
import pytest

from prc import arrivals


def movement(
    mvt_id: str,
    phase: str,
    when: str,
    runway: str | None,
    *,
    adep: str | None = None,
    ades: str | None = None,
    aobt: str | None = None,
) -> dict[str, object]:
    return {
        "MVT_ID_mvt": mvt_id,
        "PHASE_mvt": phase,
        "MVT_TIME_UTC_mvt": pd.Timestamp(when, tz="UTC"),
        "RUNWAY_mvt": runway,
        "ADEP_mvt": adep,
        "ADES_mvt": ades,
        "AOBT_3_flt": pd.Timestamp(aobt, tz="UTC") if aobt else pd.NaT,
    }


def frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for column in ("MVT_TIME_UTC_mvt", "AOBT_3_flt"):
        out[column] = pd.to_datetime(out[column], utc=True).astype("datetime64[us, UTC]")
    return out


def run(rows: list[dict[str, object]]) -> pd.DataFrame:
    movements = frame(rows)
    departures = movements[movements["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    return arrivals.add(departures, movements)


@pytest.mark.parametrize(
    ("label", "canonical"),
    [
        ("25L", "07R/25L"),
        ("07R", "07R/25L"),
        ("09", "09/27"),
        ("27", "09/27"),
        ("18C", "18C/36C"),
        ("36C", "18C/36C"),
        ("01L", "01L/19R"),
        ("19R", "01L/19R"),
        ("36", "18/36"),
    ],
)
def test_reciprocal_labels_collapse_to_one_strip(label: str, canonical: str) -> None:
    assert arrivals.physical_runway(label) == canonical


@pytest.mark.parametrize("label", [None, "", "X", np.nan, "ABC", "99", "00L", 25])
def test_unparseable_runway_labels_do_not_match_a_real_runway(label: object) -> None:
    assert arrivals.physical_runway(label) == "?"
    assert arrivals.physical_runway(label) != arrivals.physical_runway("25L")


def test_reciprocal_arrival_counts_as_same_physical_runway() -> None:
    # the arrival lands on 07R; the departure uses 25L - the same asphalt
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:45"),
        movement("a1", "ARR", "2025-01-01 09:55", "07R", ades="EDDM"),
        movement("a2", "ARR", "2025-01-01 09:56", "26L", ades="EDDM"),
    ])

    assert result["arr_prev_15m_same_physical"].iloc[0] == 1
    assert result["arr_prev_15m_other_physical"].iloc[0] == 1
    assert result["arr_during_taxi_same_physical"].iloc[0] == 1
    assert result["arr_during_taxi_other_physical"].iloc[0] == 1


def test_arrivals_key_on_ades_and_departures_on_adep() -> None:
    # the arrival lands at EDDM (ADES); its ADEP is the far-away airport it came from
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("a1", "ARR", "2025-01-01 09:55", "25L", adep="LFPG", ades="EDDM"),
        movement("a2", "ARR", "2025-01-01 09:55", "25L", adep="EDDM", ades="LFPG"),
    ])

    # only a1 lands at the departure's airport; a2 merely originated there
    assert result["arr_prev_15m_same_physical"].iloc[0] == 1
    assert result["arr_prev_15m_other_physical"].iloc[0] == 0


def test_departure_rows_in_the_source_frame_are_not_counted() -> None:
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("d2", "DEP", "2025-01-01 09:55", "25L", adep="EDDM", aobt="2025-01-01 09:40"),
    ])

    assert result["arr_prev_15m_same_physical"].iloc[0] == 0


def test_missing_aobt_leaves_aobt_anchored_windows_undefined() -> None:
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM"),
        movement("a1", "ARR", "2025-01-01 09:55", "25L", ades="EDDM"),
    ])

    # take-off anchored windows are still computable
    assert result["arr_prev_15m_same_physical"].iloc[0] == 1
    # AOBT anchored ones are NaN, not a zero count the trees would believe
    assert np.isnan(result["arr_during_taxi_same_physical"].iloc[0])
    assert np.isnan(result["arr_next_20m_aobt_same_physical"].iloc[0])
    assert np.isnan(result["arr_next_20m_aobt_other_physical"].iloc[0])


def test_windows_cross_month_and_year_boundaries() -> None:
    # push back on 31 January, take off on 1 February
    result = run([
        movement("d1", "DEP", "2025-02-01 00:05", "25L", adep="EDDM", aobt="2025-01-31 23:55"),
        movement("a1", "ARR", "2025-01-31 23:58", "25L", ades="EDDM"),
        movement("a2", "ARR", "2025-02-01 00:02", "25L", ades="EDDM"),
        movement("a3", "ARR", "2025-01-31 23:50", "25L", ades="EDDM"),
    ])

    # a1 and a2 fall inside the taxi; a3 pushed before it but inside the 15m lookback
    assert result["arr_during_taxi_same_physical"].iloc[0] == 2
    assert result["arr_prev_15m_same_physical"].iloc[0] == 3


def test_window_edges_are_inclusive_on_both_sides() -> None:
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("a1", "ARR", "2025-01-01 09:50", "25L", ades="EDDM"),
        movement("a2", "ARR", "2025-01-01 10:00", "25L", ades="EDDM"),
        movement("a3", "ARR", "2025-01-01 09:45", "25L", ades="EDDM"),
        movement("a4", "ARR", "2025-01-01 10:10", "25L", ades="EDDM"),
    ])

    assert result["arr_during_taxi_same_physical"].iloc[0] == 2
    # 20 minutes from AOBT 09:50 reaches 10:10, which a4 sits exactly on
    assert result["arr_next_20m_aobt_same_physical"].iloc[0] == 3


def test_no_arrivals_at_the_airport_gives_zero_not_nan() -> None:
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("a1", "ARR", "2025-01-01 09:55", "25L", ades="LFPG"),
    ])

    for name in arrivals.FEATURES:
        assert result[name].iloc[0] == 0


def test_omitting_the_movement_frame_yields_no_arrivals() -> None:
    movements = frame([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("a1", "ARR", "2025-01-01 09:55", "25L", ades="EDDM"),
    ])
    departures = movements[movements["PHASE_mvt"] == "DEP"].reset_index(drop=True)

    result = arrivals.add(departures)

    assert result["arr_prev_15m_same_physical"].iloc[0] == 0


def test_every_declared_feature_is_produced_and_the_control_is_a_subset() -> None:
    result = run([
        movement("d1", "DEP", "2025-01-01 10:00", "25L", adep="EDDM", aobt="2025-01-01 09:50"),
        movement("a1", "ARR", "2025-01-01 09:55", "25L", ades="EDDM"),
    ])

    assert set(arrivals.FEATURES) <= set(result.columns)
    assert set(arrivals.FIXED_WINDOW_FEATURES) < set(arrivals.FEATURES)
    assert not any("during_taxi" in name for name in arrivals.FIXED_WINDOW_FEATURES)
