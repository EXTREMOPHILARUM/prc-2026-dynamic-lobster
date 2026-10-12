"""Leave-one-month-out test of the unexplained-long-rows gate, layered on the current model.

    uv run python -u scripts/cross_validate_longrows.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prc import config, evaluation, longrows, matrix

GATE_ROUNDS = 200
SEVERITY_ROUNDS = 150


def main() -> None:
    raw, X, cats = matrix.build()
    y = raw[config.TARGET].astype("float64").to_numpy()
    month = evaluation.movement_month(raw)
    lab = longrows.label(raw, y)
    print(f"regime rows {lab.sum():,} ({100 * lab.mean():.3f}%)", flush=True)

    base = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet"),
        on=config.ID_COL, how="left")["prediction"].to_numpy()
    if not np.isfinite(base).all():
        raise SystemExit("baseline OOF predictions incomplete")

    gcols = longrows.gate_features(X)
    gcats = [c for c in cats if c in gcols]
    prob = np.full(len(y), np.nan)
    sev = np.full(len(y), np.nan)

    for m in range(1, 13):
        valid = (month == m).to_numpy()
        train = ~valid
        gate = lgb.train(longrows.GATE_PARAMS,
                         lgb.Dataset(X.loc[train, gcols], lab[train].astype("int8"),
                                     categorical_feature=gcats),
                         num_boost_round=GATE_ROUNDS)
        prob[valid] = gate.predict(X.loc[valid, gcols])
        fit = train & lab
        model = lgb.train(longrows.SEVERITY_PARAMS,
                          lgb.Dataset(X.loc[fit, gcols], y[fit], categorical_feature=gcats),
                          num_boost_round=SEVERITY_ROUNDS)
        sev[valid] = model.predict(X.loc[valid, gcols])
        print(f"  month {m:02d} fitted ({fit.sum():,} regime rows in train)", flush=True)

    new = longrows.blend(base, prob, sev)
    sse_o = float(((y - base) ** 2).sum())
    sse_n = float(((y - new) ** 2).sum())
    print(f"\nAUC {roc_auc_score(lab, prob):.4f}   predicted mass {prob.sum():,.0f} "
          f"vs actual {lab.sum():,}", flush=True)
    print(f"OOF RMSE  {np.sqrt(sse_o/len(y)):.2f} -> {np.sqrt(sse_n/len(y)):.2f}   "
          f"SSE {100*(sse_n-sse_o)/sse_o:+.2f}%", flush=True)

    rows = []
    for name, m in [("all", np.ones(len(y), bool)), ("regime rows", lab),
                    ("bulk", y <= 3600), ("gated p>0.05", prob > 0.05),
                    ("everything else", (prob <= 0.05) & ~lab)]:
        if m.sum() < 20:
            continue
        o = float(((y[m] - base[m]) ** 2).sum()); n = float(((y[m] - new[m]) ** 2).sum())
        rows.append({"segment": name, "n": int(m.sum()), "rmse_old": np.sqrt(o/m.sum()),
                     "rmse_new": np.sqrt(n/m.sum()), "sse_delta_pct": 100*(n-o)/o})
    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False), flush=True)

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((y[v] - base[v]) ** 2).sum()); n = float(((y[v] - new[v]) ** 2).sum())
        pm.append({"month": m, "old": np.sqrt(o/v.sum()), "new": np.sqrt(n/v.sum()),
                   "better": bool(n < o)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(2).to_string(index=False), flush=True)
    print(f"months improved: {int(pmd.better.sum())}/12", flush=True)

    (config.DATA_INTERIM / "longrows_oof_metrics.json").write_text(json.dumps({
        "auc": float(roc_auc_score(lab, prob)),
        "rmse_old": float(np.sqrt(sse_o/len(y))), "rmse_new": float(np.sqrt(sse_n/len(y))),
        "per_month": pmd.to_dict("records")}, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "probability": prob,
                  "severity": sev, "prediction": new}).to_parquet(
        config.DATA_INTERIM / "longrows_oof_predictions.parquet", index=False)
    print("wrote longrows artifacts", flush=True)


if __name__ == "__main__":
    main()
