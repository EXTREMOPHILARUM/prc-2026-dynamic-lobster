"""Why does February regress? Test column position against feature sampling.

The production 12-fold improved 11/12 months; the review's throwaway script reported
12/12. The feature values are bit-for-bit identical between the two, so the difference
must come from how the columns are *arranged*: the review appended the arrival features
at the end of the matrix, production inserts them ahead of weather and sequencing, and
`feature_fraction=0.9` samples a different 90% depending on that order.

Four arms on the February fold alone (train = the other eleven months):

    A  ff=1.0, no arrival features      paired reference, no sampling
    B  ff=1.0, fixed-window arrivals    if B beats A, February is real and order was noise
    C  ff=0.9, arrivals appended last   the review's column order
    D  ff=0.9, no arrival features      paired reference at production sampling

If February stops regressing at ff=1.0 (B < A) or under the review's order (C < D), the
12/12 claim was an artifact of join order rather than a property of the features.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix, mechanism

FOLD_MONTH = 2


def main() -> None:
    raw, X, cats = matrix.build(use_arrivals=True)
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]],
        on=config.ID_COL, how="left", validate="one_to_one")["probability"].to_numpy()
    del raw
    gc.collect()

    rounds = json.loads(
        (config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    valid = month == FOLD_MONTH
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    keep = arrivals.FIXED_WINDOW_FEATURES
    drop_all = list(arrivals.FEATURES)
    base_cols = [c for c in X.columns if c not in drop_all]

    arms = {
        "A ff=1.0 no arrivals":      (base_cols, 1.0),
        "B ff=1.0 fixed arrivals":   (base_cols + keep, 1.0),
        "C ff=0.9 arrivals last":    (base_cols + keep, 0.9),
        "D ff=0.9 no arrivals":      (base_cols, 0.9),
    }

    original = dict(decomposition.BASE_PARAMS)
    results = {}
    for name, (cols, fraction) in arms.items():
        decomposition.BASE_PARAMS["feature_fraction"] = fraction
        Xa = X[cols]
        fold_cats = [c for c in cats if c in cols]
        print(f"\n=== {name}  ({len(cols)} cols) ===", flush=True)
        models = decomposition.fit_fixed(
            Xa.loc[~valid], y.loc[~valid], fold_cats, rounds=rounds)
        ordinary = decomposition.predict(models, Xa.loc[valid])
        pred = mechanism.blend(gate_p[valid], proxy[valid], ordinary)
        sse = float(np.square(y_np[valid] - pred).sum())
        rmse = float(np.sqrt(sse / valid.sum()))
        results[name] = {"rmse": rmse, "sse": sse}
        print(f"{name}: February RMSE {rmse:.4f}", flush=True)
        del models, Xa
        gc.collect()
    decomposition.BASE_PARAMS.update(original)

    print("\nFEBRUARY FOLD", flush=True)
    for name, value in results.items():
        print(f"  {name:28s} {value['rmse']:9.4f}", flush=True)
    for new, old, label in (
        ("B ff=1.0 fixed arrivals", "A ff=1.0 no arrivals", "arrivals at ff=1.0"),
        ("C ff=0.9 arrivals last", "D ff=0.9 no arrivals", "arrivals last at ff=0.9"),
    ):
        delta = 100 * (results[new]["sse"] - results[old]["sse"]) / results[old]["sse"]
        verdict = "IMPROVES" if delta < 0 else "REGRESSES"
        print(f"  {label:26s} {delta:+.4f}% SSE  {verdict}", flush=True)

    (config.DATA_INTERIM / "arrivals_february_probe.json").write_text(
        json.dumps({"fold_month": FOLD_MONTH, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
