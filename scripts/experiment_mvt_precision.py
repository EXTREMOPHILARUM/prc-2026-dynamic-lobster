"""Does the precision of MVT_TIME carry information?

The data page defines MVT_TIME_UTC_mvt as the "(best available) movement time" - the only
field documented with a hedge. Five of the ten airports report ~8.3% of movement times on
an exact minute boundary against ~1.7% (chance) at the other five, which looks like a
coarser upstream source. Those rows carry 2.6x our RMSE on the scored months and a higher
rate of very long taxis, and the model cannot currently see it: features.py derives hour
and minute_of_day but never the seconds field.

The marker survives in ranking.parquet at 5.15% with the same airport structure.

    uv run python scripts/experiment_mvt_precision.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

NEW = ["mvt_second", "mvt_sec_zero"]


def main() -> None:
    raw, X, cats = matrix.build(use_weather=True, use_sequencing=True, use_arrivals=True)
    t = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    X = X.copy()
    X["mvt_second"] = t.dt.second.to_numpy("int16")
    X["mvt_sec_zero"] = (t.dt.second == 0).to_numpy().astype("int8")

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
            print(f"  {name}: month {m:02d}", flush=True)
        preds[name] = pred

    def rmse(mask: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((y_np[mask] - pred[mask]) ** 2)))

    jj = np.isin(month, [1, 7])
    everything = np.ones(len(y_np), dtype=bool)
    coarse = X["mvt_sec_zero"].to_numpy().astype(bool)
    a, b = preds["without"], preds["with"]
    print(f"\n12-month  {rmse(everything, a):.4f} -> {rmse(everything, b):.4f}  "
          f"({rmse(everything, b) - rmse(everything, a):+.4f})")
    print(f"Jan+Jul   {rmse(jj, a):.4f} -> {rmse(jj, b):.4f}  ({rmse(jj, b) - rmse(jj, a):+.4f})")
    print(f"coarse rows (n={int((coarse & jj).sum()):,})  {rmse(coarse & jj, a):.2f} -> {rmse(coarse & jj, b):.2f}")
    print(f"fine rows   (n={int((~coarse & jj).sum()):,})  {rmse(~coarse & jj, a):.2f} -> {rmse(~coarse & jj, b):.2f}")

    for name, pred in (("mvtprec", b), ("mvtprec_base", a)):
        pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
            config.DATA_INTERIM / f"{name}_oof_predictions.parquet", index=False)
    print("wrote OOF artifacts")


if __name__ == "__main__":
    main()
