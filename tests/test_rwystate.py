import pandas as pd

from prc import rwystate


def test_held_and_runway_state_on_a_small_frame():
    base = pd.Timestamp("2025-03-01 10:00", tz="UTC")
    m = lambda x: base + pd.Timedelta(minutes=x)
    df = pd.DataFrame({
        "ADEP_mvt": ["EHAM"] * 4,
        "RUNWAY_mvt": ["36L", "36L", "24", "36L"],
        "MVT_TIME_UTC_mvt": [m(10), m(20), m(30), m(40)],
        "AOBT_3_flt": [m(0), m(25), m(15), m(30)],
        "EOBT_1_flt": [m(-5), m(5), m(10), m(20)],
        "SCHED_TIME_UTC_mvt": [m(-5), m(5), m(10), m(20)],
    })
    f = rwystate.compute(df)
    # at row 2's AOBT (10:15): row 1 (EOBT 10:05, AOBT 10:25) has not pushed; row 3's EOBT 10:20 is later
    assert f.loc[2, "ret_held"] == 1
    # rows that pushed in the 15 min before 10:30 (row 3's AOBT): rows 1 (10:25) and 2 (10:15)
    assert f.loc[3, "ret_pushed15"] == 2 and f.loc[3, "ret_delay15"] == (20 * 60 + 5 * 60) / 2
    assert f.loc[3, "rws_dom_share"] == 3 / 4 and f.loc[3, "rws_n_rwys_60"] == 2


def test_tencode_spread_is_optional_and_shrunk():
    import numpy as np

    from prc import tencode
    raw = pd.DataFrame({c: ["x"] * 4 for c in ["FLIGHT_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
                                               "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]})
    raw["ADEP_mvt"] = "EHAM"; raw["STAND_mvt"] = ["A", "A", "B", "B"]; raw["RUNWAY_mvt"] = "36L"
    y = np.array([100.0, 300.0, 600.0, 9000.0]); rows = np.ones(4, bool)
    plain = tencode.transform(raw, tencode.fit(raw, y, y, rows))
    assert not any(c.endswith(("_p90", "_std")) for c in plain)          # v47-style encodings: no new columns
    enc = tencode.fit(raw, y, y, rows, spread=True)
    out = tencode.transform(raw, enc)
    assert "te_ars_y_std" in out and out["te_ars_y_std"].iloc[0] < out["te_ars_y_std"].iloc[2]   # cell B is noisier
