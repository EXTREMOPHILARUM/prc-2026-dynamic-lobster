"""XGBoost as a diversity model, blended with the LightGBM ordinary component.

The 2025 runner-up ensembled three gradient-boosting libraries; our handoff listed
CatBoost/XGBoost diversity as untested. Two libraries make different splitting and
regularisation choices, so their errors decorrelate, and a non-negative blend is usually
better than either - a different kind of lever from adding another feature, which is
where we have hit diminishing returns.

The blend weight is fitted on out-of-fold predictions only, never on a submitted score.

    uv run python -u scripts/experiment_xgb_blend.py --device cuda
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from prc import config, decomposition, evaluation, matrix, mechanism

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 0,
    "grow_policy": "lossguide",
    "max_leaves": 255,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 20,
    "seed": 42,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()

    raw, X, cats = matrix.build()
    for c in cats:
        X[c] = X[c].astype("category")
    y = raw[config.TARGET].astype("float64")
    capped = y.clip(upper=decomposition.TAIL_THRESHOLD).to_numpy()
    month = evaluation.movement_month(raw)
    print(f"matrix {X.shape}, device={args.device}", flush=True)

    prior = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet"),
        on=config.ID_COL, how="left")
    lgb_pred = prior["prediction"].to_numpy()
    gate = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    xgb_base = np.full(len(raw), np.nan)
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        dtr = xgb.QuantileDMatrix(X[~valid], capped[~valid], enable_categorical=True)
        dva = xgb.QuantileDMatrix(X[valid], capped[valid], enable_categorical=True, ref=dtr)
        booster = xgb.train({**XGB_PARAMS, "device": args.device}, dtr,
                            num_boost_round=args.rounds)
        xgb_base[valid] = booster.predict(dva)
        print(f"  month {m:02d} fitted", flush=True)

    xgb_pred = mechanism.blend(gate, proxy, np.clip(xgb_base, 0, decomposition.TAIL_THRESHOLD))
    yv = y.to_numpy()

    def rmse(p):
        return float(np.sqrt(np.mean((yv - p) ** 2)))

    print(f"\nLightGBM alone {rmse(lgb_pred):.2f}   XGBoost alone {rmse(xgb_pred):.2f}", flush=True)
    print("\nblend weight (fitted on OOF only):", flush=True)
    best = (0.0, rmse(lgb_pred))
    for w in np.arange(0.0, 1.01, 0.1):
        r = rmse((1 - w) * lgb_pred + w * xgb_pred)
        print(f"  w_xgb={w:.1f}  {r:.2f}", flush=True)
        if r < best[1]:
            best = (float(w), r)
    print(f"\nbest w_xgb={best[0]:.1f} -> {best[1]:.2f} "
          f"(LightGBM alone {rmse(lgb_pred):.2f})", flush=True)

    blended = (1 - best[0]) * lgb_pred + best[0] * xgb_pred
    rows = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        o = float(((yv[v] - lgb_pred[v]) ** 2).sum())
        n = float(((yv[v] - blended[v]) ** 2).sum())
        rows.append({"month": m, "lgb": np.sqrt(o / v.sum()), "blend": np.sqrt(n / v.sum()),
                     "better": bool(n < o)})
    pm = pd.DataFrame(rows)
    print("\n" + pm.round(2).to_string(index=False), flush=True)
    print(f"months improved: {int(pm.better.sum())}/12", flush=True)

    (config.DATA_INTERIM / "xgb_blend_metrics.json").write_text(json.dumps({
        "rmse_lgb": rmse(lgb_pred), "rmse_xgb": rmse(xgb_pred),
        "best_weight": best[0], "rmse_blend": best[1],
        "per_month": pm.to_dict("records")}, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "xgb_base": xgb_base,
                  "xgb_pred": xgb_pred}).to_parquet(
        config.DATA_INTERIM / "xgb_oof_predictions.parquet", index=False)
    print("wrote xgb artifacts", flush=True)


if __name__ == "__main__":
    main()
