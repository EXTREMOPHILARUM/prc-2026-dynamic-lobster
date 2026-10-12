"""Paired 12-fold A/B of the temperature / dew-point spread features.

Reported on the challenge Discord by another team: most de-icing happens on clear
frosty mornings that carry no precipitation code, so a binary snow flag misses them
and the temperature/dew-point spread does not. `dwpf` was already being downloaded
by scripts/fetch_metar.py and had simply never been turned into a feature.

    uv run python scripts/experiment_dewpoint.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism, weather

NEW = ["wx_dewpoint_c", "wx_spread_c", "wx_frost_risk"]


def main() -> None:
    raw, X, cats = matrix.build(use_weather=True, use_sequencing=True, use_arrivals=True)
    missing = [c for c in NEW if c not in X.columns]
    if missing:
        raise SystemExit(f"dew-point features absent from the matrix: {missing}")
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    X_off = X.drop(columns=NEW)

    preds = {}
    for name, frame in (("without", X_off), ("with", X)):
        pred = np.full(len(raw), np.nan)
        for m in range(1, 13):
            valid = month == m
            models = decomposition.fit_fixed(frame.loc[~valid], y.loc[~valid], cats, rounds=rounds)
            pred[valid] = mechanism.blend(
                gate_p[valid], proxy[valid], decomposition.predict(models, frame.loc[valid])
            )
            print(f"  {name}: month {m:02d} done", flush=True)
        preds[name] = pred

    jj = np.isin(month, [1, 7])

    def rmse(mask: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((y_np[mask] - pred[mask]) ** 2)))

    everything = np.ones(len(y_np), dtype=bool)
    a, b = preds["without"], preds["with"]
    print(f"\n12-month  {rmse(everything, a):.4f} -> {rmse(everything, b):.4f}  "
          f"({rmse(everything, b) - rmse(everything, a):+.4f})")
    print(f"Jan+Jul   {rmse(jj, a):.4f} -> {rmse(jj, b):.4f}  ({rmse(jj, b) - rmse(jj, a):+.4f})")
    cold = (raw["wx_temp_c"] < 3.0).to_numpy(na_value=False) & jj
    if cold.sum():
        print(f"cold rows (n={int(cold.sum()):,})  {rmse(cold, a):.2f} -> {rmse(cold, b):.2f}")

    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": b}).to_parquet(
        config.DATA_INTERIM / "dewpoint_oof_predictions.parquet", index=False)
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": a}).to_parquet(
        config.DATA_INTERIM / "dewpoint_base_oof_predictions.parquet", index=False)
    print(f"\nwrote OOF artifacts ({len(weather.FEATURES)} weather features with the new three)")


if __name__ == "__main__":
    main()
