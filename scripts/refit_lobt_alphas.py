"""Out-of-fold refit of the LOBT blend weights.

The v17 alphas were fitted on training-row residuals, where the base is overfit and the
expert therefore looks too good (global 0.90). This recomputes the expert's out-of-fold
predictions per month, fits per-airport alphas on the ten non-scored months' OOF residuals
against the v17 context-only stage, and reports the Jan+Jul effect of the new alphas.

    uv run python -u scripts/refit_lobt_alphas.py --threads 8
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from prc import arrivals, config, evaluation, lirfbands, lobt, matrix


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=8); args = ap.parse_args()
    lobt.PARAMS["num_threads"] = args.threads; start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    airport = X["ADEP_mvt"].astype("string").to_numpy()
    oof = raw[[config.ID_COL]].merge(pd.read_parquet(config.DATA_INTERIM / "v17_oof_predictions.parquet"), on=config.ID_COL, how="left")
    base = oof["context_only"].to_numpy().copy(); v17 = oof["prediction"].to_numpy().copy()
    expert = np.full(len(y), np.nan)
    for m in range(1, 13):
        b = lobt.fit(X.loc[month != m], y[month != m], cats); expert[month == m] = lobt.predict(b, X.loc[month == m])
        print(f"  fold {m:02d} t={time.time()-start:.0f}s", flush=True)
    rows = lobt.scope(X) & ~np.isin(month, (1, 7))
    alphas = lobt.fit_alphas(y, base, expert, airport, rows)
    new = np.clip(lirfbands.lobt_clip(lobt.apply(base, expert, airport, alphas), X), 0, None)
    old = json.loads((config.MODELS / "lobt_expert.json").read_text())["alphas"]
    scored = np.isin(month, (1, 7))
    rep = {"alphas_oof": alphas, "alphas_v17": old,
           "scored_jan_jul": {"context_only": rmse(y[scored], base[scored]), "v17_alphas": rmse(y[scored], v17[scored]), "oof_alphas": rmse(y[scored], new[scored])},
           "all_months": {"context_only": rmse(y, base), "v17_alphas": rmse(y, v17), "oof_alphas": rmse(y, new)}}
    (config.DATA_INTERIM / "lobt_alphas_oof_metrics.json").write_text(json.dumps(rep, indent=2))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "expert": expert}).to_parquet(config.DATA_INTERIM / "lobt_expert_oof.parquet", index=False)
    print("alphas OOF", {k: round(v, 3) for k, v in alphas.items()}); print("alphas v17", {k: round(v, 3) for k, v in old.items()})
    print("Jan+Jul", {k: round(v, 3) for k, v in rep["scored_jan_jul"].items()}); print("12-month", {k: round(v, 3) for k, v in rep["all_months"].items()})


if __name__ == "__main__":
    main()
