import numpy as np
import pandas as pd

from prc import ownexpert


def rows(n: int, joined: np.ndarray, anchor: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    own = pd.DataFrame({
        "own_joined": np.where(joined, 1.0, np.nan),
        "own_to_minus_mvt": np.where(joined, -30.0, np.nan),
        "own_dwell_taxi": np.where(joined, anchor + rng.normal(0, 60, n), np.nan),
        "own_dwell_len": np.where(joined, 2400.0, np.nan),
        "own_first_taxi": np.where(joined, anchor - 60, np.nan),
    })
    times = pd.date_range("2025-07-01", periods=n, freq="7min", tz="UTC")
    return ownexpert.frame(np.array(["EHAM"] * n), times, np.full(n, 600.0), anchor, own)


def test_scope_is_matched_and_joined_only():
    joined = np.array([True, True, False, False])
    anchor = np.array([600.0, np.nan, 600.0, np.nan])
    f = rows(4, joined, anchor)
    assert ownexpert.scope(f).tolist() == [True, False, False, False]
    assert f["own_joined"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_apply_leaves_out_of_scope_rows_unchanged_and_learns_the_shift():
    n = 4000
    rng = np.random.default_rng(1)
    joined = rng.random(n) < 0.5
    anchor = rng.uniform(300, 900, n)
    f = rows(n, joined, anchor)
    # the NM clock is late by exactly what the trace shows, so the residual is recoverable
    resid = np.where(joined, f["own_dwell_taxi"].to_numpy() - 600.0, 0.0)
    out = ownexpert.apply(ownexpert.fit(f, resid), f)
    assert np.array_equal(out[~joined], f["pred"].to_numpy()[~joined])
    err_before = np.sqrt(np.mean(resid[joined] ** 2))
    err_after = np.sqrt(np.mean((600.0 + resid[joined] - out[joined]) ** 2))
    assert err_after < 0.3 * err_before


def test_norec_scope_is_joined_no_record_outside_rome():
    f = pd.DataFrame({"apt": pd.Categorical(["EDDM", "EDDM", "EDDM", "LIRF"]), "own_joined": [1.0, 0.0, 1.0, 1.0]})
    missing = np.array([1, 1, 0, 1])
    assert ownexpert.norec_scope(f, missing).tolist() == [True, False, False, False]


def test_apply_norec_leaves_out_of_scope_rows_unchanged():
    class Const:
        def feature_name(self):
            return ["pred"]

        def predict(self, x):
            return np.full(len(x), 100.0)
    f = pd.DataFrame({"apt": pd.Categorical(["EDDM", "LIRF", "EDDM"]), "own_joined": [1.0, 1.0, 1.0],
                      "pred": [500.0, 500.0, 500.0]})
    out = ownexpert.apply_norec(Const(), f, np.array([1, 1, 0]))
    assert out.tolist() == [600.0, 500.0, 500.0]


def test_rich_columns_drop_identity_strings_and_keep_categoricals():
    X = pd.DataFrame({"FLIGHT_mvt": pd.Categorical(["A1"]), "CALLSIGN_flt": pd.Categorical(["B"]),
                      "STAND_mvt": pd.Categorical(["S1"]), "hour": [3], "doy": [10]})
    assert ownexpert.rich_columns(X, None) == []
    assert ownexpert.rich_columns(X, "num") == ["hour"]
    assert ownexpert.rich_columns(X, "cat")[:1] == ["hour"] and "STAND_mvt" in ownexpert.rich_columns(X, "cat")
    assert "FLIGHT_mvt" not in ownexpert.rich_columns(X, "cat")


def test_rich_booster_remaps_categories_after_a_save(tmp_path):
    # ranking frames carry their own category sets; a served booster must read stands by value, not code
    import lightgbm as lgb
    rng = np.random.default_rng(3)
    n = 3000
    stand = rng.choice(["S1", "S2", "S3"], n)
    f = pd.DataFrame({"x_STAND_mvt": pd.Categorical(stand, categories=["S1", "S2", "S3"]), "v": rng.normal(size=n)})
    y = np.select([stand == "S1", stand == "S2"], [100.0, -100.0], 0.0)
    b = lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 20}, lgb.Dataset(f, y), 50)
    b.save_model(str(tmp_path / "m.txt"))
    served = lgb.Booster(model_file=str(tmp_path / "m.txt"))
    g = pd.DataFrame({"x_STAND_mvt": pd.Categorical(["S2", "S1", "S9"], categories=["S9", "S2", "S1"]), "v": [0.0] * 3})
    p = served.predict(g)
    assert p[0] < -90 and p[1] > 90


def test_bagged_averages_and_requires_one_feature_list():
    class B:
        def __init__(self, names, v):
            self.names, self.v = names, v

        def feature_name(self):
            return self.names

        def predict(self, x):
            return np.full(len(x), self.v)
    bag = ownexpert.Bagged([B(["a"], 10.0), B(["a"], 30.0)])
    assert bag.feature_name() == ["a"] and bag.predict(pd.DataFrame({"a": [1, 2]})).tolist() == [20.0, 20.0]
    import pytest
    with pytest.raises(ValueError):
        ownexpert.Bagged([B(["a"], 1.0), B(["b"], 1.0)])


def test_lag_tables_smooth_toward_the_global_mean_and_handle_unseen_keys():
    frame = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 4 + ["EDDF"], "STAND_mvt": ["D4", "D4", "D6", None, "A1"],
                          "RUNWAY_mvt": ["36L"] * 5})
    lag = np.array([100.0, 300.0, 5000.0, 50.0, np.nan])
    ok = np.isfinite(lag)
    t = ownexpert.lag_tables(frame, lag, ok)
    clipped = np.array([100.0, 300.0, 1800.0, 50.0])
    g = clipped.mean()
    assert np.isclose(t["global"], g)
    assert np.isclose(t["enc_lag_stand"]["EHAM|D4"], (400.0 + 50 * g) / 52)
    assert np.isclose(t["enc_lag_stand"]["EHAM|"], (50.0 + 50 * g) / 51)     # a missing stand is its own key
    new = pd.DataFrame({"ADEP_mvt": pd.Series(["EHAM", "LTFM"], dtype="string"),
                        "STAND_mvt": pd.Series(["D4", "X9"], dtype="string"), "RUNWAY_mvt": ["36L", "35"]})
    enc = ownexpert.lag_apply(t, new)
    assert np.isclose(enc["enc_lag_stand"][0], t["enc_lag_stand"]["EHAM|D4"]) and np.isclose(enc["enc_lag_stand"][1], g)
    assert np.isclose(enc["enc_lag_stand_rwy"][0], t["enc_lag_stand_rwy"]["EHAM|D4|36L"])


def test_daily_reception_uses_only_joined_rows_of_the_same_airport_day():
    f = pd.DataFrame({"apt": pd.Categorical(["EHAM", "EHAM", "EHAM", "EDDF"]), "own_joined": [1.0, 1.0, 0.0, 1.0],
                      "own_first_net_dist": [100.0, 300.0, 9999.0, 50.0],
                      "own_stand_exit_taxi": [600.0, np.nan, 600.0, np.nan]})
    t = pd.to_datetime(["2025-03-01 08:00", "2025-03-01 23:00", "2025-03-01 09:00", "2025-03-01 08:00"], utc=True)
    d = ownexpert.daily_reception(f, t)
    assert d["day_first_net_dist"].tolist() == [200.0, 200.0, 200.0, 50.0]
    assert d["day_exit_share"].tolist() == [0.5, 0.5, 0.5, 0.0]
    assert d["day_joined_n"].tolist() == [2.0, 2.0, 2.0, 1.0]
