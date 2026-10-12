"""Own-flight ADS-B residual expert: corrects the chain's prediction on matched rows whose own
ADS-B ground run was joined (sections 46-47).

NM's AOBT_3 is a noisy record of the off-block. The departing aircraft's own trace observes the
physical pushback independently: the end of its stand dwell, and the first point of its taxi-out,
which bounds the taxi from below whatever the transponder did. A LightGBM fitted on the chain's
leave-one-month-out residual, with those features and the anchor, reads how far the NM clock is
off. It is applied only to the rows it serves (matched and joined); every other row is unchanged.

Using it on ranking rows joins a scored flight to its own post-event trace; the organiser ruled
on 2026-09-19 that this is permitted (docs/rules-and-compliance.md). In `prc.submit` it runs
after `lirfbands.lobt_clip`, on the clipped chain output it was fitted on, and the clip is then
applied again.
"""
from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config

OWN = ["own_joined", "own_to_minus_mvt", "own_dwell_taxi", "own_dwell_len", "own_first_taxi"]
FEATURES = ["apt", "hour", "dow", "pred", "anchor", *OWN, "own_minus_anchor", "own_first_minus_anchor"]
# v2 feature files (silent-gap rule, section 49) add a flag for the rows it produced
OWN_V2 = [*OWN, "own_gap_seg"]
FEATURES_V2 = [*FEATURES, "own_gap_seg"]
# v3 adds the stand exit against the NM stand's X-Plane coordinates (section 50)
OWN_V3 = [*OWN_V2, "own_stand_exit_taxi", "own_stand_dmin"]
FEATURES_V3 = [*FEATURES_V2, "own_stand_exit_taxi", "own_stand_dmin", "own_stand_minus_anchor"]
# v4: same columns as v3, stand exits located by more names and by position (section 51)
# v5 adds the first-sighting distance from the stand (section 54)
OWN_V5 = [*OWN_V3, "own_first_dist"]
FEATURES_V5 = [*FEATURES_V3, "own_first_dist"]
# v6 (section 55): ground_v2 extraction (departing callsign taken once airborne; take-offs after lost
# ground reception, flagged own_to_late)
OWN_V6 = [*OWN_V5, "own_to_late"]
FEATURES_V6 = [*FEATURES_V5, "own_to_late"]
# v7 (section 60): the stand-to-first-sighting distance along the X-Plane taxiway network, and the first
# moving point after the run's last long gap (an earlier stop or tow no longer counts as the taxi-out)
OWN_V7 = [*OWN_V6, "own_first_net_dist", "own_first_taxi_gap"]
FEATURES_V7 = [*FEATURES_V6, "own_first_net_dist", "own_first_taxi_gap", "own_first_gap_minus_anchor"]
# v8 (section 63): over the observed departure segment, the ground path length, the speed at first sighting and
# the number of stops; with the derived speed of that segment and the time the unobserved stand-to-first-sighting
# stretch would take at that speed (aimed at rows first seen mid-taxi)
OWN_V8 = [*OWN_V7, "own_obs_path", "own_first_gs", "own_stops", "own_hold"]
FEATURES_V8 = [*FEATURES_V7, "own_obs_path", "own_first_gs", "own_stops", "own_hold", "own_obs_speed", "own_unobs_time",
               "own_first_gap_plus_unobs"]
# v9 (section 63): reception quality of the observed segment, and its moving speed net of holds
OWN_V9 = [*OWN_V8, "own_npts", "own_max_dt", "own_first_dt"]
FEATURES_V9 = [*FEATURES_V8, "own_npts", "own_max_dt", "own_first_dt", "own_pts_per_min", "own_move_speed"]
# v10 (section 63): where the first moving point is (reception zones), and MVT minus the run's own landing
OWN_V10 = [*OWN_V9, "own_first_lat", "own_first_lon", "own_since_land"]
FEATURES_V10 = [*FEATURES_V9, "own_first_lat", "own_first_lon", "own_since_land"]
OWN_BY_VERSION = {1: OWN, 2: OWN_V2, 3: OWN_V3, 4: OWN_V3, 5: OWN_V5, 6: OWN_V6, 7: OWN_V7, 8: OWN_V8, 9: OWN_V9,
                  10: OWN_V10}
# Fixed with the section 46 screen, before any January/July result was seen; not tuned since.
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
          "seed": 7, "num_threads": 8}
ROUNDS = 300


def load_features(paths: Sequence[str], ids: pd.Series, own: Sequence[str] = OWN) -> pd.DataFrame:
    """`own` columns aligned to `ids`, from the builder's parquet files (paths relative to the repo).

    A missing file or column is an error: silently scoring without the trace would drop the expert."""
    parts = [pd.read_parquet(config.ROOT / p, columns=[config.ID_COL, *own]) for p in paths]
    F = pd.concat(parts, ignore_index=True)
    if F[config.ID_COL].duplicated().any():
        raise ValueError("own-flight feature files overlap on MVT_ID_mvt")
    return pd.DataFrame({config.ID_COL: ids.to_numpy()}).merge(F, on=config.ID_COL, how="left")


def frame(airport, mvt_time, pred, anchor, own: pd.DataFrame) -> pd.DataFrame:
    """Expert inputs. `own` is aligned to the rows and carries OWN (NaN where not joined)."""
    t = pd.to_datetime(pd.Series(mvt_time), utc=True)
    f = pd.DataFrame({"apt": pd.Categorical(np.asarray(airport, dtype=str)), "hour": t.dt.hour.to_numpy(),
                      "dow": t.dt.dayofweek.to_numpy(), "pred": np.asarray(pred, dtype="float64"),
                      "anchor": np.asarray(anchor, dtype="float64")})
    for c in own.columns:
        if c.startswith("own_"):
            f[c] = own[c].to_numpy(dtype="float64")
    f["own_joined"] = f["own_joined"].fillna(0.0)
    f["own_minus_anchor"] = f.own_dwell_taxi - f.anchor
    f["own_first_minus_anchor"] = f.own_first_taxi - f.anchor
    if "own_stand_exit_taxi" in f:
        f["own_stand_minus_anchor"] = f.own_stand_exit_taxi - f.anchor
    if "own_first_taxi_gap" in f:
        f["own_first_gap_minus_anchor"] = f.own_first_taxi_gap - f.anchor
    if "own_obs_path" in f:
        observed_s = (f.own_first_taxi_gap + f.own_to_minus_mvt).where(lambda v: v > 30)
        f["own_obs_speed"] = f.own_obs_path / observed_s
        f["own_unobs_time"] = f.own_first_net_dist / f.own_obs_speed.where(lambda v: v > 1)
        f["own_first_gap_plus_unobs"] = f.own_first_taxi_gap + f.own_unobs_time
    if "own_npts" in f:
        observed_s = (f.own_first_taxi_gap + f.own_to_minus_mvt).where(lambda v: v > 30)
        f["own_pts_per_min"] = 60.0 * f.own_npts / observed_s
        f["own_move_speed"] = f.own_obs_path / (observed_s - f.own_hold).where(lambda v: v > 30)
    return f


def scope(f: pd.DataFrame) -> np.ndarray:
    return (f["own_joined"].to_numpy() == 1) & np.isfinite(f["anchor"].to_numpy())


def fit(f: pd.DataFrame, resid: np.ndarray, features: Sequence[str] = FEATURES) -> lgb.Booster:
    rows = scope(f)
    return lgb.train(PARAMS, lgb.Dataset(f.loc[rows, list(features)], resid[rows], categorical_feature=["apt"]), ROUNDS)


class CatExpert:
    """A CatBoost expert served like a LightGBM booster: its own feature list, categoricals passed as strings
    (a missing value as "NA", as in fitting)."""

    def __init__(self, path: str):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(); self.model.load_model(path)
        self.cats = set(self.model.get_cat_feature_indices())

    def feature_name(self) -> list[str]:
        return list(self.model.feature_names_)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        D = x.copy()
        names = self.feature_name()
        cat_names = [names[i] for i in sorted(self.cats)]
        for c in cat_names:
            D[c] = D[c].astype(object).where(D[c].notna(), "NA").astype(str)
        return self.model.predict(Pool(D[names], cat_features=cat_names))


class Bagged:
    """Several boosters fitted on the same inputs (different seeds), served as the mean of their predictions."""

    def __init__(self, boosters: Sequence[lgb.Booster]):
        names = {tuple(b.feature_name()) for b in boosters}
        if len(names) != 1:
            raise ValueError("bagged boosters must share one feature list")
        self.boosters = list(boosters)

    def feature_name(self) -> list[str]:
        return self.boosters[0].feature_name()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(x) for b in self.boosters], axis=0)


def apply(booster: lgb.Booster, f: pd.DataFrame) -> np.ndarray:
    """Corrected prediction: `pred` plus the expert's residual on scope rows, unchanged elsewhere."""
    out = f["pred"].to_numpy(dtype="float64").copy()
    rows = scope(f)
    if rows.any():
        # the booster carries the feature list it was fitted on, so v1 and v2 models both serve
        out[rows] = np.clip(out[rows] + booster.predict(f.loc[rows, booster.feature_name()]), 0, None)
    return out


# Rich matched expert (log section 63): the v7 inputs plus the chain's own feature matrix, so the expert can
# learn where the trace's relation to the off-block depends on context (stand, runway, type, congestion).
# identity strings and airport codes (apt carries the airport); `doy` would let it memorise dates; the two
# taxi-length arrival windows are not served (the inner model ships `arrivals.FIXED_WINDOW_FEATURES` only)
RICH_EXCLUDE = {"FLIGHT_mvt", "CALLSIGN_flt", "ADEP_flt", "ADES_FILED_flt", "ADES_flt", "ADES_mvt", "ADEP_mvt", "doy",
                "arr_during_taxi_same_physical", "arr_during_taxi_other_physical"}
RICH_CATEGORICALS = ["RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt",
                     "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt"]


def rich_columns(X: pd.DataFrame, kind: str | None) -> list[str]:
    """Matrix columns the rich expert reads: every numeric one, plus the categoricals for kind 'cat'."""
    if kind is None:
        return []
    num = [c for c in X.columns if c not in RICH_EXCLUDE and not isinstance(X[c].dtype, pd.CategoricalDtype)]
    return num + (RICH_CATEGORICALS if kind == "cat" else [])


def with_rich(f: pd.DataFrame, X: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """`f` plus `X[cols]` as `x_<col>` (categoricals kept as pandas categoricals, which the booster remaps)."""
    return pd.concat([f, pd.DataFrame({"x_" + c: X[c].to_numpy() if not isinstance(X[c].dtype, pd.CategoricalDtype)
                                       else X[c].values for c in cols}, index=f.index)], axis=1)


# Daily reception (section 63): per airport and UTC day, over the joined departures, the median stand-to-first-sighting
# network distance, the share with a stand exit seen, and the count joined. Target-free, so computed on any frame;
# feeders come and go (EDDF's first sightings sit much further out in 2026), which static encodings cannot see.
DAILY = ["day_first_net_dist", "day_exit_share", "day_joined_n"]


def daily_reception(f: pd.DataFrame, mvt_time) -> pd.DataFrame:
    day = pd.to_datetime(pd.Series(mvt_time), utc=True).dt.floor("D").to_numpy()
    j = f["own_joined"].to_numpy() == 1
    g = pd.DataFrame({"apt": f["apt"].astype(str).to_numpy(), "day": day, "j": j,
                      "net": np.where(j, f["own_first_net_dist"].to_numpy("float64"), np.nan),
                      "exit": np.where(j, np.isfinite(f["own_stand_exit_taxi"].to_numpy("float64")), np.nan)})
    agg = g[g.j].groupby(["apt", "day"]).agg(day_first_net_dist=("net", "median"), day_exit_share=("exit", "mean"),
                                             day_joined_n=("j", "size"))
    out = g[["apt", "day"]].merge(agg, left_on=["apt", "day"], right_index=True, how="left")
    return pd.DataFrame({c: out[c].to_numpy("float64") for c in DAILY}, index=f.index)


# First-sighting lag encodings (section 63): per airport-stand and airport-stand-runway, the smoothed mean of the
# clipped lag between off-block and the first moving sighting, y - own_first_taxi_gap, from the training months.
LAG_CLIP = 1800.0
LAG_K = 50.0
LAG_KEYS = {"enc_lag_stand": ("ADEP_mvt", "STAND_mvt"), "enc_lag_stand_rwy": ("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt")}
# extended set: transponder habits differ by airline, and reception by aircraft type
LAG_KEYS_EXT = {**LAG_KEYS, "enc_lag_operator": ("ADEP_mvt", "AIRCRAFT_OPERATOR_flt"),
                "enc_lag_type": ("ADEP_mvt", "AIRCRAFT_TYPE_flt")}


def lag_key_strings(frame: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """'apt|stand[|runway]' with a missing value as the empty string, whatever the column's dtype."""
    parts = [frame[c].astype("string").fillna("").astype(object) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return pd.Series(out.to_numpy(dtype=object), index=frame.index)


def lag_tables(frame: pd.DataFrame, lag: np.ndarray, ok: np.ndarray, keys: dict | None = None) -> dict:
    """Encoding tables from the rows `ok` (fitted scope with a finite lag): {name: {key: value}, "global": g,
    "keys": {name: columns}}."""
    keys = keys or LAG_KEYS
    v = np.clip(np.asarray(lag, dtype="float64")[ok], -LAG_CLIP, LAG_CLIP)
    g = float(v.mean())
    tables: dict = {"global": g, "keys": {k: list(c) for k, c in keys.items()}}
    for name, cols in keys.items():
        keys = lag_key_strings(frame.loc[ok], cols).to_numpy()
        agg = pd.DataFrame({"k": keys, "v": v}).groupby("k").v.agg(["sum", "count"])
        tables[name] = ((agg["sum"] + LAG_K * g) / (agg["count"] + LAG_K)).to_dict()
    return tables


def lag_apply(tables: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The encodings for `frame`'s rows; a key never seen in fitting gets the global mean (zero pseudo-count rows)."""
    g = float(tables["global"])
    return {name: lag_key_strings(frame, cols).map(tables[name]).fillna(g).to_numpy(dtype="float64")
            for name, cols in tables.get("keys", LAG_KEYS).items()}


# No-record expert (section 59): the same trace, on departures with no NM record outside Rome. There is
# no anchor, so the inputs are the trace times, the schedule gap and the chain's prediction; the
# target was the chain's residual clipped to +/-NOREC_CAP (fixed before the first run).
FEATURES_NOREC = ["apt", "hour", "dow", "pred", "gap", *[c for c in OWN_V6 if c != "own_joined"]]
FEATURES_NOREC_V7 = [*FEATURES_NOREC, "own_first_net_dist", "own_first_taxi_gap"]
FEATURES_NOREC_V8 = [*FEATURES_NOREC_V7, "own_obs_path", "own_first_gs", "own_stops", "own_hold", "own_obs_speed",
                     "own_unobs_time", "own_first_gap_plus_unobs"]
NOREC_CAP = 1800.0
NOREC_PARAMS = {**PARAMS, "num_leaves": 31, "min_data_in_leaf": 50}
NOREC_ROUNDS = 300
NOREC_EXCLUDED_AIRPORT = "LIRF"   # Rome's no-record rows are the copy mixture's and the band table's


def norec_scope(f: pd.DataFrame, record_missing: np.ndarray) -> np.ndarray:
    return ((np.asarray(record_missing) == 1) & (f["apt"].astype(str).to_numpy() != NOREC_EXCLUDED_AIRPORT)
            & (f["own_joined"].to_numpy() == 1))


def fit_norec(f: pd.DataFrame, resid: np.ndarray, record_missing: np.ndarray, params: dict | None = None,
              rounds: int = NOREC_ROUNDS, features: Sequence[str] = FEATURES_NOREC) -> lgb.Booster:
    rows = norec_scope(f, record_missing)
    target = np.clip(resid[rows], -NOREC_CAP, NOREC_CAP)
    return lgb.train(params or NOREC_PARAMS, lgb.Dataset(f.loc[rows, list(features)], target, categorical_feature=["apt"]), rounds)


def apply_norec(booster: lgb.Booster, f: pd.DataFrame, record_missing: np.ndarray) -> np.ndarray:
    """`pred` plus the no-record expert's residual on its scope rows, unchanged elsewhere."""
    out = f["pred"].to_numpy(dtype="float64").copy()
    rows = norec_scope(f, record_missing)
    if rows.any():
        out[rows] = np.clip(out[rows] + booster.predict(f.loc[rows, booster.feature_name()]), 0, None)
    return out
