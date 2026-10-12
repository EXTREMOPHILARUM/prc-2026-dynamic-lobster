"""Leave-one-month-out OOF for the deployed mechanism model, one seed at a time.

Writes the raw component predictions per seed so that seed averages can be assembled
afterwards without refitting (scripts/analyze_seedavg.py). Seed 1 reproduces the frozen
baseline exactly: it leaves every LightGBM seed at its library default, which is what
data/interim/sequencing_oof_predictions.parquet was built with.

    uv run python -u scripts/cross_validate_seedavg.py --seeds 1 --rounds-scale 1.0

Rounds come from data/interim/component_rounds.json (inner holdout April/October, disjoint
from the ranking months). --rounds-scale multiplies them for the capacity probe; it never
selects them on January or July.
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

GATE_ROUNDS = 200
OUT_DIR = config.DATA_INTERIM / "seedavg"


def seeded(params: dict, seed: int | None, threads: int) -> dict:
    out = dict(params) | {"num_threads": threads}
    if seed is not None:
        out |= {
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "extra_seed": seed,
        }
    return out


def _load_pinned(tag: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Read one cached matrix by its fingerprint, bypassing the fingerprint computation."""
    cache = matrix.CACHE_DIR
    cats = json.loads((cache / f"train_{tag}_meta.json").read_text())["categoricals"]
    raw = pd.read_parquet(cache / f"train_{tag}_raw.parquet")
    X = pd.read_parquet(cache / f"train_{tag}_X.parquet")
    for column in cats:
        X[column] = X[column].astype("category")
    print(f"pinned matrix {tag}: {len(X):,} rows, {len(X.columns)} features", flush=True)
    return raw, X, cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--rounds-scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--matrix-tag",
        default=None,
        help="load this cached matrix directly instead of fingerprinting the feature "
             "modules. Pin it when other work is editing features/weather/sequencing: a "
             "mid-run fingerprint change silently swaps the inputs under the experiment.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"r{args.rounds_scale:g}".replace(".", "p")

    raw, X, cats = (
        _load_pinned(args.matrix_tag) if args.matrix_tag else matrix.build()
    )
    y = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw)
    label = mechanism.fallback_label(raw).to_numpy()
    base_rounds = json.loads(
        (config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    rounds = {k: max(1, round(v * args.rounds_scale)) for k, v in base_rounds.items()}
    gate_rounds = max(1, round(GATE_ROUNDS * args.rounds_scale))
    print(f"rounds {rounds}  gate {gate_rounds}  seeds {args.seeds}", flush=True)

    gate_cols = mechanism.gate_features(X)
    gate_cats = [c for c in cats if c in gate_cols]

    for seed in args.seeds:
        out_path = OUT_DIR / f"oof_{tag}_seed{seed}.parquet"
        if out_path.exists():
            print(f"skip {out_path.name} (exists)", flush=True)
            continue
        # seed 1 means "library defaults", the configuration the frozen baseline used
        s = None if seed == 1 else seed
        cols = {
            name: np.full(len(raw), np.nan)
            for name in ("base", "tail_probability", "tail_excess", "tail_scale", "gate_prob")
        }
        for m in args.months:
            t0 = time.time()
            valid = (month == m).to_numpy()
            train = ~valid
            gate = lgb.train(
                seeded(mechanism.GATE_PARAMS, s, args.threads),
                lgb.Dataset(X.loc[train, gate_cols], label[train].astype("int8"),
                            categorical_feature=gate_cats),
                num_boost_round=gate_rounds,
            )
            cols["gate_prob"][valid] = gate.predict(X.loc[valid, gate_cols])

            models = _fit_decomposition(X.loc[train], y.loc[train], cats, rounds, s,
                                        args.threads)
            comp = decomposition.predict_components(models, X.loc[valid])
            for name in ("base", "tail_probability", "tail_excess", "tail_scale"):
                cols[name][valid] = comp[name]
            print(f"  seed {seed} month {m:02d} done in {time.time() - t0:.0f}s", flush=True)

        frame = pd.DataFrame({config.ID_COL: raw[config.ID_COL]} | cols)
        frame["proxy"] = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
        frame.to_parquet(out_path, index=False)
        print(f"wrote {out_path}", flush=True)


def _fit_decomposition(X, y, cats, rounds, seed, threads):
    """decomposition.fit_fixed with every LightGBM seed overridden."""
    matched = X[decomposition.ANCHOR_COL].notna()
    capped = y.clip(upper=decomposition.TAIL_THRESHOLD)
    anchor = X.loc[matched, decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD)
    tail = y > decomposition.TAIL_THRESHOLD
    tail_features = [c for c in decomposition.TAIL_FEATURES if c in X.columns]
    tail_cats = [c for c in cats if c in tail_features]

    def train(params, frame, target, categorical, name):
        return lgb.train(
            params,
            lgb.Dataset(frame, target, categorical_feature=categorical),
            num_boost_round=rounds[name],
        )

    return {
        "base_matched": train(seeded(decomposition.BASE_PARAMS, seed, threads),
                              X.loc[matched], capped.loc[matched] - anchor, cats,
                              "base_matched"),
        "base_missing": train(seeded(decomposition.BASE_PARAMS, seed, threads),
                              X.loc[~matched], capped.loc[~matched], cats, "base_missing"),
        "gate": train(seeded(decomposition.GATE_PARAMS, seed, threads), X[tail_features],
                      tail.astype("int8"), tail_cats, "gate"),
        "severity": train(seeded(decomposition.SEVERITY_PARAMS, seed, threads),
                          X.loc[tail, tail_features],
                          y.loc[tail] - decomposition.TAIL_THRESHOLD, tail_cats, "severity"),
    }


if __name__ == "__main__":
    main()
