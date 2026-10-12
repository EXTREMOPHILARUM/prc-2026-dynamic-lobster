"""Paired 12-fold A/B of METAR weather features on the v7 mechanism model.

Only the ordinary (decomposition) component is refitted; the fallback gate probabilities
are reused from the frozen mechanism OOF, so the delta is attributable to weather alone.

    uv run python scripts/experiment_weather.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from prc import config, data, decomposition, evaluation, features, mechanism, sequencing, weather


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="mechanism_oof_predictions.parquet",
                    help="OOF artifact to compare against")
    ap.add_argument("--baseline-col", default="blended")
    ap.add_argument("--out", default="weather_oof")
    ap.add_argument("--sequencing", action="store_true",
                    help="also add departure-sequencing features")
    args = ap.parse_args()
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    raw = weather.attach(raw)
    extra = list(weather.FEATURES)
    if args.sequencing:
        raw = sequencing.add(raw)
        extra += sequencing.FEATURES
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    X_wx = X.join(raw[extra])
    missing = [c for c in extra if c not in X_wx.columns]
    if missing:
        raise SystemExit(f"weather features missing from X: {missing}")

    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / args.baseline)[[config.ID_COL, args.baseline_col]],
        on=config.ID_COL, how="left")[args.baseline_col].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    pred = np.full(len(raw), np.nan)
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        models = decomposition.fit_fixed(X_wx.loc[~valid], y.loc[~valid], cats, rounds=rounds)
        ordinary = decomposition.predict(models, X_wx.loc[valid])
        pred[valid] = mechanism.blend(gate_p[valid], proxy[valid], ordinary)
        print(f"  month {m:02d} done")

    sse_o = float(((y_np - baseline) ** 2).sum())
    sse_n = float(((y_np - pred) ** 2).sum())
    print(f"\nOOF RMSE  without weather {np.sqrt(sse_o/len(y_np)):.2f}  "
          f"with weather {np.sqrt(sse_n/len(y_np)):.2f}   SSE {100*(sse_n-sse_o)/sse_o:+.2f}%")

    snow = (raw["wx_snow"] == 1).to_numpy()
    tail = y_np > 3600
    rows = []
    for name, m in [("all", np.ones(len(y_np), bool)), ("bulk", ~tail), ("tail", tail),
                    ("snow", snow), ("no snow", ~snow),
                    ("freezing", (raw["wx_freezing"] == 1).to_numpy()),
                    ("low visibility", (raw["wx_low_visibility"] == 1).to_numpy())]:
        if m.sum() < 50:
            continue
        o = float(((y_np[m] - baseline[m]) ** 2).sum()); n = float(((y_np[m] - pred[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()), "rmse_old": np.sqrt(o/m.sum()),
                     "rmse_new": np.sqrt(n/m.sum()), "sse_delta_pct": 100*(n-o)/o})
    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False))

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y_np[v] - baseline[v]) ** 2).sum()); n = float(((y_np[v] - pred[v]) ** 2).sum())
        pm.append({"month": m, "rmse_old": np.sqrt(o/v.sum()), "rmse_new": np.sqrt(n/v.sum()),
                   "better": bool(n < o)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(2).to_string(index=False))
    print(f"\nmonths improved: {int(pmd.better.sum())}/12")

    config.DATA_INTERIM.joinpath(f"{args.out}_metrics.json").write_text(json.dumps({
        "rmse_without": float(np.sqrt(sse_o/len(y_np))),
        "rmse_with": float(np.sqrt(sse_n/len(y_np))),
        "per_month": pmd.to_dict("records"),
    }, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        config.DATA_INTERIM / f"{args.out}_predictions.parquet", index=False)
    print("wrote weather OOF artifacts")


if __name__ == "__main__":
    main()
