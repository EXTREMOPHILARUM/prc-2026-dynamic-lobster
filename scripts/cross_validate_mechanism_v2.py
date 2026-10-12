"""Disjoint-component schedule-fallback mixture, leave-one-month-out.

The first mixture double-counted: `ordinary` estimated E[Y], which already contains the
fallback mass, so partially-gated rows had the schedule reconstruction added on top. Here
the ordinary component is fitted on NON-FALLBACK rows only, making it E[Y | not fallback, X]
so the two mixture terms are genuinely disjoint:

    E[Y|X] = P(fallback|X) * (MVT - SCHED) + (1 - P(fallback|X)) * E[Y | not fallback, X]

Round counts come from data/interim/component_rounds.json, selected on an inner holdout
disjoint from the ranking months.

    uv run python scripts/cross_validate_mechanism_v2.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prc import config, data, decomposition, evaluation, features, mechanism

GATE_ROUNDS = 200


def main() -> None:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    label = mechanism.fallback_label(raw).to_numpy()

    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    print(f"leak-free rounds: {rounds}")

    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "decomposition_oof_predictions.parquet")[
            [config.ID_COL, "prediction"]],
        on=config.ID_COL, how="left")["prediction"].to_numpy()

    gate_cols = mechanism.gate_features(X)
    gate_cats = [c for c in cats if c in gate_cols]
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    probability = np.full(len(raw), np.nan)
    ordinary = np.full(len(raw), np.nan)

    for m in range(1, 13):
        valid = (month == m).to_numpy()
        train = ~valid
        gate = lgb.train(
            mechanism.GATE_PARAMS,
            lgb.Dataset(X.loc[train, gate_cols], label[train].astype("int8"),
                        categorical_feature=gate_cats),
            num_boost_round=GATE_ROUNDS,
        )
        probability[valid] = gate.predict(X.loc[valid, gate_cols])

        clean = train & ~label  # E[Y | not fallback]
        models = decomposition.fit_fixed(X.loc[clean], y.loc[clean], cats, rounds=rounds)
        ordinary[valid] = decomposition.predict(models, X.loc[valid])
        print(f"  month {m:02d}: ordinary trained on {clean.sum():,} non-fallback rows")

    blended = mechanism.blend(probability, proxy, ordinary)

    def report(name: str, pred: np.ndarray) -> None:
        print(f"  {name:26} RMSE {np.sqrt(np.mean((y_np - pred) ** 2)):.2f}")

    print("\n--- overall ---")
    report("current deployed (baseline)", baseline)
    report("mechanism v2 (disjoint)", blended)
    sse_old = float(((y_np - baseline) ** 2).sum())
    sse_new = float(((y_np - blended) ** 2).sum())
    print(f"  SSE change {100 * (sse_new - sse_old) / sse_old:+.2f}%")
    print(f"  gate AUC {roc_auc_score(label, probability):.4f}  "
          f"mass {probability.sum():,.0f} vs {label.sum():,}")

    tail = y_np > 3600
    missing = raw["AOBT_3_flt"].isna().to_numpy()
    lirf = (raw["ADEP_mvt"] == "LIRF").to_numpy()
    rows = []
    for name, m in [("all", np.ones(len(y_np), bool)), ("bulk", ~tail), ("tail", tail),
                    ("LIRF", lirf), ("non-LIRF", ~lirf),
                    ("missing-clock", missing), ("matched-clock", ~missing),
                    ("fallback", label), ("non-fallback", ~label)]:
        o = float(((y_np[m] - baseline[m]) ** 2).sum())
        n = float(((y_np[m] - blended[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()),
                     "rmse_old": np.sqrt(o / m.sum()), "rmse_new": np.sqrt(n / m.sum()),
                     "sse_delta_pct": 100 * (n - o) / o})
    seg = pd.DataFrame(rows)
    print("\n" + seg.round(2).to_string(index=False))

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y_np[v] - baseline[v]) ** 2).sum())
        n = float(((y_np[v] - blended[v]) ** 2).sum())
        pm.append({"month": m, "rmse_old": np.sqrt(o / v.sum()),
                   "rmse_new": np.sqrt(n / v.sum()), "better": bool(n < o)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(2).to_string(index=False))
    print(f"\nmonths improved: {int(pmd.better.sum())}/12")

    bulk_delta = float(seg.loc[seg.segment == "bulk", "sse_delta_pct"].iloc[0])
    jan = bool(pmd.loc[pmd.month == 1, "better"].iloc[0])
    jul = bool(pmd.loc[pmd.month == 7, "better"].iloc[0])
    print("\n--- acceptance gate ---")
    print(f"  (a) total SSE lower        : {sse_new < sse_old}")
    print(f"  (b) Jan and Jul improved   : {jan and jul}")
    print(f"  (c) >=9/12 months improved : {int(pmd.better.sum()) >= 9}")
    print(f"  (d) bulk not materially worse (<= +1% SSE): {bulk_delta <= 1.0} ({bulk_delta:+.2f}%)")

    (config.DATA_INTERIM / "mechanism_v2_oof_metrics.json").write_text(json.dumps({
        "rmse_baseline": float(np.sqrt(sse_old / len(y_np))),
        "rmse_mechanism_v2": float(np.sqrt(sse_new / len(y_np))),
        "segments": seg.to_dict("records"), "per_month": pmd.to_dict("records"),
        "rounds": rounds,
    }, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "probability": probability,
                  "ordinary": ordinary, "prediction": blended}).to_parquet(
        config.DATA_INTERIM / "mechanism_v2_oof_predictions.parquet", index=False)
    print("wrote mechanism_v2 artifacts")


if __name__ == "__main__":
    main()
