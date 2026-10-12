"""Is the LIRF specialist's win just its round schedule, or the separate fit?

The specialist early-stops each component on LIRF alone and gets a very different
schedule from the global model (base_matched 148 vs 342, severity 36 vs 300). If the
schedule is the whole story, applying it to the *global* model should capture the gain
without a second model. This fits the global decomposition with the LIRF schedule on the
ten months disjoint from January and July, and reports on January+July.

Reporting only: the round schedules compared here were both selected on disjoint months.

    uv run python -u scripts/experiment_perairport_rounds.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

SCORED = (1, 7)
#: early-stopped on LIRF rows of months 4+10 by scripts/experiment_perairport.py
LIRF_ROUNDS = {"base_matched": 148, "base_missing": 52, "gate": 95, "severity": 36}


def main() -> None:
    start = time.time()
    raw, X, cats = matrix.build()
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    airport = raw["ADEP_mvt"].astype("string").to_numpy()
    missing = X[decomposition.ANCHOR_COL].isna().to_numpy()

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    valid = np.isin(month, SCORED)

    global_rounds = json.loads(
        (config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    for params in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS,
                   decomposition.SEVERITY_PARAMS):
        params["num_threads"] = 2

    rows = []
    for name, rounds in [("global schedule", global_rounds), ("LIRF schedule", LIRF_ROUNDS)]:
        models = decomposition.fit_fixed(X[~valid], y[~valid], cats, rounds=rounds)
        pred = mechanism.blend(gate_p[valid], proxy[valid],
                               decomposition.predict(models, X[valid]))
        lirf = airport[valid] == "LIRF"
        segments = {
            "all": np.ones(int(valid.sum()), dtype=bool),
            "lirf": lirf,
            "lirf_missing_aobt": lirf & missing[valid],
            "lirf_matched": lirf & ~missing[valid],
        }
        row = {"config": name}
        for segment, mask in segments.items():
            row[segment] = float(np.sqrt(((y_np[valid][mask] - pred[mask]) ** 2).mean()))
        rows.append(row)
        print(f"[{name}] t={time.time()-start:.0f}s {row}", flush=True)

    print("\nJanuary+July, trained on the ten disjoint months")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
