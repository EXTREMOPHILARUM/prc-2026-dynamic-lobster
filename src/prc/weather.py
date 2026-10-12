"""METAR ground-weather features.

Source: Iowa Environmental Mesonet ASOS/METAR archive (Iowa State University),
public domain - https://mesonet.agron.iastate.edu/disclaimer.php
Retrieved by scripts/fetch_metar.py. Declared here and in docs/ as required by the
challenge's external-data rule (organiser ruling, 2026-09-08).

Ground weather is the one external source with measurable signal in the model's
residual: on snow-affected departures the model under-predicts by ~239s on average
(February 2025, where snow is common, by ~440s). The scored January 2026 contains
5,480 snow departures against 1,170 in January 2025, so the exposure is real even
though a Jan+Jul 2025 holdout barely sees it.

Observations are matched backwards to each movement time, within two hours.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from prc import config

METAR_DIR = config.ROOT / "data" / "external" / "metar"
RUNWAY_FILE = config.ROOT / "data" / "external" / "ourairports" / "runways.csv"

#: point-in-time observation at the departure airport
POINT_FEATURES = [
    "wx_temp_c",
    "wx_wind_kt",
    "wx_gust_kt",
    "wx_visibility_mi",
    "wx_humidity",
    "wx_snow",
    "wx_freezing",
    "wx_fog",
    "wx_rain",
    "wx_low_visibility",
    "wx_thunder",
    "wx_shower",
    "wx_hail",
]

#: Temperature/dew-point spread, the standard operational frost indicator. Computed and
#: kept, but EXCLUDED from FEATURES: a paired 12-fold gives -0.032s on the scored months,
#: makes cold rows marginally *worse* (451.42 -> 451.56 on 48,895 rows), improves only
#: 7/12 months and reverses on dropping ten rows. The residual bias is real (+18.6s below
#: -5C) but wx_freezing already fires below 3C and captures it. Reported as worth 3.4s by
#: another team on the challenge Discord - their gain was recovering a weather block that
#: had been silently null for nineteen submissions, not the dew point itself.
DEWPOINT_FEATURES = [
    "wx_dewpoint_c",
    "wx_spread_c",
    "wx_frost_risk",
]

#: how conditions are changing, and what they have been over the preceding hours.
#: A single observation cannot distinguish clearing fog from thickening fog, and
#: de-icing demand depends on accumulated cold rather than the temperature right now.
TREND_FEATURES = [
    "wx_temp_delta_1h",
    "wx_temp_delta_3h",
    "wx_visibility_delta_1h",
    "wx_visibility_min_3h",
    "wx_wind_delta_1h",
    "wx_snow_hours_6h",
    "wx_freezing_hours_6h",
    "wx_fog_hours_3h",
    "wx_thunder_hours_3h",
]

#: Ceiling and Low Visibility Procedure features. Computed by load_metar but NOT in
#: FEATURES: measured negative. A paired 12-fold against the v11 baseline (285.6515)
#: gave -0.126s with the storm codes alone and only -0.117s with these six added, and
#: they made the low-ceiling rows themselves worse (ceiling<1000 SSE +0.27%). The
#: residual screen agrees - the mean residual on rows with a ceiling under 200 ft is
#: +3.7s, so the model already prices low ceilings through visibility and fog.
CEILING_FEATURES = [
    "wx_ceiling_ft",
    "wx_lvp",
    "wx_flight_category",
    "wx_ceiling_below_200",
    "wx_ceiling_min_3h",
    "wx_lvp_hours_3h",
]

#: geometry-dependent: wind resolved onto the departure runway
RUNWAY_FEATURES = [
    "wx_crosswind_kt",
    "wx_headwind_kt",
    "rwy_length_ft",
]

FEATURES = POINT_FEATURES + TREND_FEATURES + RUNWAY_FEATURES


def load_metar() -> pd.DataFrame:
    files = sorted(glob.glob(str(METAR_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"no METAR csv in {METAR_DIR}. Run: uv run python scripts/fetch_metar.py"
        )
    met = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    met["valid"] = pd.to_datetime(met["valid"], utc=True)
    for c in ("tmpf", "dwpf", "relh", "sknt", "gust", "vsby"):
        met[c] = pd.to_numeric(met[c], errors="coerce")
    ceiling = _ceiling_ft(met)

    codes = met["wxcodes"].fillna("").astype(str)
    out = pd.DataFrame({
        "ADEP_mvt": met["station"],
        "valid": met["valid"],
        "wx_temp_c": (met["tmpf"] - 32.0) * 5.0 / 9.0,
        "wx_dewpoint_c": (met["dwpf"] - 32.0) * 5.0 / 9.0,
        "wx_wind_kt": met["sknt"],
        "wx_gust_kt": met["gust"].fillna(0.0),
        "wx_visibility_mi": met["vsby"],
        "wx_humidity": met["relh"],
        "wx_snow": codes.str.contains("SN").astype("int8"),
        "wx_fog": codes.str.contains("FG|BR", regex=True).astype("int8"),
        "wx_rain": codes.str.contains("RA").astype("int8"),
        "wx_ceiling_ft": ceiling,
        # convective weather stops pushbacks and re-routes ground movement; TS/VCTS is
        # the ATC-relevant signal, SH is the weaker shower form of the same system
        "wx_thunder": codes.str.contains("TS").astype("int8"),
        "wx_shower": codes.str.contains("SH").astype("int8"),
        "wx_hail": codes.str.contains("GR|GS", regex=True).astype("int8"),
    })
    # Frost forms when the surface is at or below the dew point and both are near
    # freezing - a clear, calm morning that carries no precipitation code at all, so
    # wx_snow and wx_freezing both read zero while every aircraft still needs de-icing.
    # The temperature/dew-point spread is the standard operational indicator.
    out["wx_spread_c"] = out["wx_temp_c"] - out["wx_dewpoint_c"]
    out["wx_frost_risk"] = (
        (out["wx_temp_c"] < 3.0).to_numpy(na_value=False)
        & (out["wx_spread_c"] < 3.0).to_numpy(na_value=False)
    ).astype("int8")
    # de-icing is driven by temperature as much as by reported precipitation
    out["wx_freezing"] = (
        codes.str.contains("FZ").to_numpy() | (out["wx_temp_c"] <= 3.0).to_numpy()
    ).astype("int8")
    out["wx_low_visibility"] = (out["wx_visibility_mi"] < 1.0).astype("int8")
    # Low Visibility Procedures: the operational trigger is visibility below 3 statute
    # miles OR a ceiling below 1000 ft, not the 1-mile flag above. LVP forces increased
    # departure spacing and follow-me escorts, which is exactly a taxi-out regime.
    out["wx_lvp"] = (
        (out["wx_visibility_mi"] < 3.0).to_numpy(na_value=False)
        | (out["wx_ceiling_ft"] < 1000.0).to_numpy(na_value=False)
    ).astype("int8")
    out["wx_ceiling_below_200"] = (
        out["wx_ceiling_ft"].to_numpy(na_value=np.inf) < 200.0).astype("int8")
    out["wx_flight_category"] = _flight_category(
        out["wx_visibility_mi"], out["wx_ceiling_ft"])
    out = out.sort_values(["ADEP_mvt", "valid"]).reset_index(drop=True)

    # observations are hourly, so a shift of n rows is roughly n hours
    g = out.groupby("ADEP_mvt", observed=True)
    out["wx_temp_delta_1h"] = out["wx_temp_c"] - g["wx_temp_c"].shift(1)
    out["wx_temp_delta_3h"] = out["wx_temp_c"] - g["wx_temp_c"].shift(3)
    out["wx_visibility_delta_1h"] = out["wx_visibility_mi"] - g["wx_visibility_mi"].shift(1)
    out["wx_wind_delta_1h"] = out["wx_wind_kt"] - g["wx_wind_kt"].shift(1)
    out["wx_visibility_min_3h"] = g["wx_visibility_mi"].transform(
        lambda s: s.rolling(3, min_periods=1).min())
    out["wx_snow_hours_6h"] = g["wx_snow"].transform(
        lambda s: s.rolling(6, min_periods=1).sum())
    out["wx_freezing_hours_6h"] = g["wx_freezing"].transform(
        lambda s: s.rolling(6, min_periods=1).sum())
    out["wx_fog_hours_3h"] = g["wx_fog"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    out["wx_ceiling_min_3h"] = g["wx_ceiling_ft"].transform(
        lambda s: s.rolling(3, min_periods=1).min())
    out["wx_lvp_hours_3h"] = g["wx_lvp"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    out["wx_thunder_hours_3h"] = g["wx_thunder"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())

    # keep the wind direction for the runway projection, then drop it
    out["_wind_dir"] = pd.to_numeric(met["drct"], errors="coerce").to_numpy()[out.index]
    return out.sort_values("valid").reset_index(drop=True)


#: a ceiling is the lowest broken-or-worse layer; scattered and few do not count
CEILING_COVERS = ("BKN", "OVC", "VV")
#: layers that are reported but are not a ceiling: the sky was observed and found
#: clear, or only few/scattered cloud, which is an unlimited ceiling, not a missing one
CLEAR_COVERS = ("CLR", "SKC", "NSC", "NCD", "CAVOK", "FEW", "SCT")
#: stands in for "no ceiling"; a finite value keeps the feature monotone for the trees
UNLIMITED_FT = 25000.0


def _ceiling_ft(met: pd.DataFrame) -> pd.Series:
    """Height in feet AGL of the lowest broken/overcast layer, else UNLIMITED_FT.

    IEM reports up to four sky layers as (cover, height) pairs. A layer is only a
    ceiling if it is BKN, OVC or VV; SCT and FEW are not. Rows with no usable layer
    report at all stay NaN so the model can tell "clear" from "not reported".
    """
    ceiling = pd.Series(np.nan, index=met.index, dtype="float64")
    seen = np.zeros(len(met), dtype=bool)
    for i in (1, 2, 3, 4):
        cover_col, level_col = f"skyc{i}", f"skyl{i}"
        if cover_col not in met.columns or level_col not in met.columns:
            continue
        cover = met[cover_col].fillna("").astype(str).str.strip().str.upper()
        level = pd.to_numeric(met[level_col], errors="coerce")
        reported = cover.isin(CEILING_COVERS + CLEAR_COVERS).to_numpy()
        seen |= reported
        hit = (cover.isin(CEILING_COVERS).to_numpy() & level.notna().to_numpy()
               & ceiling.isna().to_numpy())
        ceiling = ceiling.where(~hit, level)
    # a report that listed only clear/scattered layers has no ceiling, not a missing one
    ceiling = ceiling.where(~(ceiling.isna().to_numpy() & seen), UNLIMITED_FT)
    return ceiling.clip(upper=UNLIMITED_FT)


def _flight_category(vis_mi: pd.Series, ceiling_ft: pd.Series) -> pd.Series:
    """0 VFR, 1 MVFR, 2 IFR, 3 LIFR - the standard ceiling/visibility bands.

    Ordinal rather than categorical: the bands are ordered by severity and a tree
    splits an ordinal in one cut where a one-hot needs three.
    """
    vis = vis_mi.to_numpy(dtype="float64", na_value=np.inf)
    cig = ceiling_ft.to_numpy(dtype="float64", na_value=np.inf)
    cat = np.zeros(len(vis), dtype="int8")
    cat[(vis < 5.0) | (cig < 3000.0)] = 1
    cat[(vis < 3.0) | (cig < 1000.0)] = 2
    cat[(vis < 1.0) | (cig < 500.0)] = 3
    unknown = np.isinf(vis) & np.isinf(cig)
    return pd.Series(np.where(unknown, -1, cat).astype("int8"), index=vis_mi.index)


def load_runways() -> pd.DataFrame:
    """Runway true headings and lengths, keyed by (airport, runway identifier)."""
    r = pd.read_csv(RUNWAY_FILE, low_memory=False)
    ends = []
    for ident, heading in (("le_ident", "le_heading_degT"), ("he_ident", "he_heading_degT")):
        part = r[["airport_ident", ident, heading, "length_ft"]].copy()
        part.columns = ["ADEP_mvt", "RUNWAY_mvt", "rwy_heading", "rwy_length_ft"]
        ends.append(part)
    out = pd.concat(ends, ignore_index=True).dropna(subset=["RUNWAY_mvt"])
    out["RUNWAY_mvt"] = out["RUNWAY_mvt"].astype(str).str.strip().str.upper()
    return out.drop_duplicates(["ADEP_mvt", "RUNWAY_mvt"])


def attach(
    df: pd.DataFrame, metar: pd.DataFrame | None = None, time_col: str = "MVT_TIME_UTC_mvt"
) -> pd.DataFrame:
    """Backward-match the most recent observation at the departure airport."""
    metar = load_metar() if metar is None else metar
    left = df.copy()
    left["_t"] = pd.to_datetime(left[time_col], utc=True)
    order = left["_t"].argsort(kind="stable")
    left = left.iloc[order]
    joined = pd.merge_asof(
        left,
        metar,
        left_on="_t",
        right_on="valid",
        by="ADEP_mvt",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    )
    joined = joined.drop(columns=["_t", "valid"]).iloc[np.argsort(order.to_numpy())]
    joined = joined.reset_index(drop=True)

    # resolve the wind onto the departure runway: a strong crosswind slows ground
    # movement and can force a configuration change, which a raw wind speed cannot show
    runways = load_runways()
    key = joined["RUNWAY_mvt"].astype(str).str.strip().str.upper()
    geo = pd.DataFrame({"ADEP_mvt": joined["ADEP_mvt"].astype(str), "RUNWAY_mvt": key}).merge(
        runways, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
    angle = np.radians(joined["_wind_dir"].to_numpy() - geo["rwy_heading"].to_numpy())
    speed = joined["wx_wind_kt"].to_numpy()
    joined["wx_crosswind_kt"] = np.abs(speed * np.sin(angle))
    joined["wx_headwind_kt"] = speed * np.cos(angle)
    joined["rwy_length_ft"] = geo["rwy_length_ft"].to_numpy()
    return joined.drop(columns=["_wind_dir"])
