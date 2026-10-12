import pandas as pd

from prc import features


def test_offblock_availability_and_disagreement_features() -> None:
    raw = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00"], utc=True),
            "AOBT_3_flt": pd.to_datetime(["2025-01-01 00:45", None], utc=True),
            "EOBT_1_flt": pd.to_datetime(["2025-01-01 00:40", None], utc=True),
            "IOBT_flt": pd.to_datetime(["2025-01-01 00:30", None], utc=True),
            "LOBT_flt": pd.to_datetime(["2025-01-01 00:35", None], utc=True),
        }
    )

    result = features.add_offblock_features(raw)

    assert result["nm_time_count"].tolist() == [4, 0]
    assert result["nm_record_missing"].tolist() == [0, 1]
    assert result["aobt_missing"].tolist() == [0, 1]
    assert result["aobt_minus_EOBT_1_flt"].iloc[0] == 300
    assert result["taxi_proxy_AOBT_3_flt"].iloc[0] == 900


def test_build_accepts_categories_from_an_older_model() -> None:
    raw = pd.DataFrame(
        {
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-01"], utc=True),
            "ADEP_mvt": ["LIRF"],
            "CALLSIGN_flt": ["ITY123"],
        }
    )

    result, cats = features.build(raw, categories={"ADEP_mvt": ["LIRF"]})

    assert cats == ["ADEP_mvt"]
    assert str(result["ADEP_mvt"].dtype) == "category"
    assert "CALLSIGN_flt" not in result.columns
