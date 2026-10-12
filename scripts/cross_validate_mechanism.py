"""Leave-one-month-out test of the schedule-fallback mixture.

Reuses the frozen decomposition OOF predictions as the "ordinary" component, so the
comparison is paired row-for-row against the current model. Only the gate is refitted per
fold, which is cheap.

    uv run python scripts/cross_validate_mechanism.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from prc import config, data, evaluation, features, mechanism

GATE_ROUNDS = 200


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32").to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    label = mechanism.fallback_label(raw).to_numpy()

    oof = pd.read_parquet(config.DATA_INTERIM / "decomposition_oof_predictions.parquet")
    merged = raw[[config.ID_COL]].merge(oof[[config.ID_COL, "prediction"]], on=config.ID_COL,
                                        how="left")
    ordinary = merged["prediction"].to_numpy()
    if not np.isfinite(ordinary).all():
        raise SystemExit("decomposition OOF predictions are missing rows; re-run the 12-fold OOF")

    gate_cols = mechanism.gate_features(X)
    gate_cats = [c for c in cats if c in gate_cols]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    probability = np.full(len(raw), np.nan)

    print(f"fallback rows: {label.sum():,} ({100 * label.mean():.3f}%)")
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        model = lgb.train(
            mechanism.GATE_PARAMS,
            lgb.Dataset(X.loc[~valid, gate_cols], label[~valid].astype("int8"),
                        categorical_feature=gate_cats),
            num_boost_round=GATE_ROUNDS,
        )
        probability[valid] = model.predict(X.loc[valid, gate_cols])
        print(f"  month {m:02d}: predicted mass {probability[valid].sum():8.0f}  "
              f"actual {label[valid].sum():6d}")

    blended = mechanism.blend(probability, proxy, ordinary)

    print("\n--- gate quality (out of fold) ---")
    print(f"  AUC        {roc_auc_score(label, probability):.4f}")
    print(f"  log loss   {log_loss(label, probability):.5f}")
    print(f"  Brier      {brier_score_loss(label, probability):.6f}")
    print(f"  mass       predicted {probability.sum():,.0f} vs actual {label.sum():,}")

    print("\n--- reliability ---")
    band = pd.cut(probability, [0, .01, .05, .2, .5, .8, 1.0], include_lowest=True)
    rel = pd.DataFrame({"p": probability, "y": label}).groupby(band, observed=True).agg(
        n=("y", "size"), mean_p=("p", "mean"), actual=("y", "mean"))
    print(rel.round(4).to_string())

    sse_old = float(((y - ordinary) ** 2).sum())
    sse_new = float(((y - blended) ** 2).sum())
    print("\n--- paired OOF result ---")
    print(f"  current   RMSE {np.sqrt(sse_old / len(y)):.2f}")
    print(f"  mechanism RMSE {np.sqrt(sse_new / len(y)):.2f}")
    print(f"  SSE change {100 * (sse_new - sse_old) / sse_old:+.2f}%")

    rows = []
    tail = y > 3600
    missing = raw["AOBT_3_flt"].isna().to_numpy()
    lirf = (raw["ADEP_mvt"] == "LIRF").to_numpy()
    for name, m in [("all", np.ones(len(y), bool)), ("bulk", ~tail), ("tail", tail),
                    ("LIRF", lirf), ("non-LIRF", ~lirf),
                    ("missing-clock", missing), ("matched-clock", ~missing),
                    ("fallback", label), ("non-fallback", ~label)]:
        o = float(((y[m] - ordinary[m]) ** 2).sum())
        n = float(((y[m] - blended[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()),
                     "rmse_old": np.sqrt(o / m.sum()), "rmse_new": np.sqrt(n / m.sum()),
                     "sse_delta_pct": 100 * (n - o) / o if o else 0.0})
    seg = pd.DataFrame(rows)
    print("\n" + seg.round(2).to_string(index=False))

    per_month = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y[v] - ordinary[v]) ** 2).sum())
        n = float(((y[v] - blended[v]) ** 2).sum())
        per_month.append({"month": m, "rmse_old": np.sqrt(o / v.sum()),
                          "rmse_new": np.sqrt(n / v.sum()),
                          "better": bool(n < o)})
    pm = pd.DataFrame(per_month)
    print("\n" + pm.round(2).to_string(index=False))
    print(f"\nmonths improved: {int(pm.better.sum())}/12")

    out = config.DATA_INTERIM / "mechanism_oof_metrics.json"
    out.write_text(json.dumps({
        "auc": float(roc_auc_score(label, probability)),
        "rmse_old": float(np.sqrt(sse_old / len(y))),
        "rmse_new": float(np.sqrt(sse_new / len(y))),
        "segments": seg.to_dict("records"),
        "per_month": pm.to_dict("records"),
    }, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "probability": probability,
                  "blended": blended}).to_parquet(
        config.DATA_INTERIM / "mechanism_oof_predictions.parquet", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
