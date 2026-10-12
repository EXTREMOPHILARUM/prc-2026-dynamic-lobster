"""Screen a two-stage correction for the matched early-AOBT reporting regime.

Hypothesis: when the NM AOBT anchor is late (``anchor - y < -600``), the v29
chain systematically underpredicts.  A gate for that reporting regime and a
conditional model of the **residual left by v29** can correct it without asking the base model
to sacrifice its much larger agree lane.  A companion normal-regime residual expert protects
rows that the gate rates as early but are actually ordinary.

Both models use only score-time columns already in the v29 chain.  They are fitted
leave-one-calendar-month-out on 2025.  The correction is the conditional mean
``P(early | X) * E[y - v29 | early, X] + P(normal | X) * E[y - v29 | normal, X]``;
its form and clipping below are fixed
before looking at this screen.  This is a screening run, not a submission candidate:
confirmation requires the paired bootstrap and robustness checks.

    uv run python -u scripts/experiment_early_anchor.py
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, lirfmoe, matrix

EARLY_CUTOFF = -600.0
MIN_CORRECTION = -3_600.0
MAX_CORRECTION = 3_600.0
CATEGORY_COLUMNS = [
    "ADEP_mvt",
    "ADES_mvt",
    "ADES_FILED_flt",
    "AIRCRAFT_OPERATOR_flt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_TYPE_flt",
    "MARKET_SEGMENT_flt",
    "RUNWAY_mvt",
    "STAND_mvt",
]
NUMERIC_COLUMNS = [
    "hour",
    "minute_of_day",
    "dow",
    "doy",
    "is_weekend",
    "adep_movements_hour",
    "nm_time_count",
    "offblock_delay",
    decomposition.ANCHOR_COL,
    "taxi_proxy_EOBT_1_flt",
    "taxi_proxy_IOBT_flt",
    "taxi_proxy_LOBT_flt",
    "taxi_proxy_SCHED_TIME_UTC_mvt",
]
PREFIXES = ("clk_", "ord_", "q_", "seq_", "wx_", "arrctx_")
GATE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbosity": -1,
    "seed": 20260920,
    "num_threads": 8,
}
SEVERITY_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbosity": -1,
    "seed": 20260920,
    "num_threads": 8,
}
ROUNDS = 400


def rmse(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))


def v29_oof(raw: pd.DataFrame, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The stored v29 chain and the live-chain signals aligned to training rows."""
    ids = raw[[config.ID_COL]]

    def col(filename: str, column: str) -> np.ndarray:
        frame = pd.read_parquet(config.DATA_INTERIM / filename)
        return ids.merge(frame, on=config.ID_COL, how="left")[column].to_numpy(dtype="float64")

    base = col("v21_oof_predictions.parquet", "prediction")
    d8 = col("catgpu_d8_oof_predictions.parquet", "member")
    lobt = col("catgpu_lobt_oof_predictions.parquet", "member")
    mixture = col("lirf_unmatched_moe_oof.parquet", "mixture")
    stack = col("stacker_mid_800_oof.parquet", "stack")
    stack_ref = col("stacker_mid_800_oof.parquet", "ref")
    ok = np.isfinite(d8) & np.isfinite(lobt)
    prediction = base.copy()
    prediction[ok] += 0.3 * (d8[ok] - base[ok]) + 0.3 * (lobt[ok] - base[ok])
    moe_rows = lirfmoe.target_rows(X, 8.0)
    prediction[moe_rows] = mixture[moe_rows]
    reference = np.clip(lirfbands.lobt_clip(prediction + 0.5 * (stack - stack_ref), X), 0, None)
    return reference, {
        "chain_v21": base,
        "chain_d8_member": d8,
        "chain_lobt_member": lobt,
        "chain_stack_delta": stack - stack_ref,
        "chain_stack_reference": stack_ref,
        "chain_moe": mixture,
        "chain_v29": reference,
    }


def main() -> None:
    raw, X, _ = matrix.build(
        "train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True
    )
    y = raw[config.TARGET].to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    anchor = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64")
    matched = np.isfinite(anchor)
    early = matched & (anchor - y < EARLY_CUTOFF)
    reference, chain_signals = v29_oof(raw, X)

    columns = [c for c in CATEGORY_COLUMNS + NUMERIC_COLUMNS if c in X]
    columns.extend(c for c in X if c.startswith(PREFIXES) and c not in columns)
    F = X[columns].copy()
    # These are all score-time values already computed by the production chain.  The
    # screen tests whether their disagreement identifies *residual* early-clock error;
    # it does not add an external source or a post-score correction.
    for name, value in chain_signals.items():
        F[name] = value
        columns.append(name)
    cats = [c for c in CATEGORY_COLUMNS if c in F]
    probability = np.full(len(y), np.nan, dtype="float64")
    conditional = np.full(len(y), np.nan, dtype="float64")
    normal_conditional = np.full(len(y), np.nan, dtype="float64")
    print(
        f"early-AOBT screen: {int(early.sum()):,} positives / {int(matched.sum()):,} matched rows; "
        f"{len(columns)} score-time features; fixed {ROUNDS} rounds per stage",
        flush=True,
    )

    for fold in range(1, 13):
        train = matched & (month != fold)
        valid = matched & (month == fold)
        early_train = train & early
        normal_train = train & ~early
        gate = lgb.train(
            GATE_PARAMS,
            lgb.Dataset(F.loc[train], early[train].astype("int8"), categorical_feature=cats),
            num_boost_round=ROUNDS,
        )
        severity = lgb.train(
            SEVERITY_PARAMS,
            lgb.Dataset(
                F.loc[early_train],
                (y[early_train] - reference[early_train]),
                categorical_feature=cats,
            ),
            num_boost_round=ROUNDS,
        )
        normal = lgb.train(
            {**SEVERITY_PARAMS, "min_data_in_leaf": 500},
            lgb.Dataset(
                F.loc[normal_train],
                (y[normal_train] - reference[normal_train]),
                categorical_feature=cats,
            ),
            num_boost_round=ROUNDS,
        )
        probability[valid] = gate.predict(F.loc[valid])
        conditional[valid] = severity.predict(F.loc[valid])
        normal_conditional[valid] = normal.predict(F.loc[valid])
        print(
            f"  fold {fold:02d}: train early {int(early_train.sum()):,}, "
            f"valid early {int((valid & early).sum()):,}",
            flush=True,
        )

    early_correction = np.clip(conditional, MIN_CORRECTION, MAX_CORRECTION)
    normal_correction = np.clip(normal_conditional, MIN_CORRECTION, MAX_CORRECTION)
    correction = probability * early_correction + (1.0 - probability) * normal_correction
    candidate = reference.copy()
    candidate[matched] += correction[matched]
    candidate = np.clip(lirfbands.lobt_clip(candidate, X), 0, None)
    scored = np.isin(month, (1, 7))
    late = matched & (anchor - y > 600)
    blocks = {
        "scored_jan_jul": scored,
        "jan": month == 1,
        "jul": month == 7,
        "all_months": np.ones(len(y), dtype=bool),
        "matched": scored & matched,
        "early": scored & early,
        "agree": scored & matched & ~early & ~late,
        "late": scored & late,
        "non_lirf_no_record": scored & ~matched & (raw["ADEP_mvt"].to_numpy() != "LIRF"),
    }
    report: dict[str, object] = {
        "hypothesis": "early/normal mixture of conditional v29-residual experts, matched rows only",
        "early_cutoff": EARLY_CUTOFF,
        "min_correction": MIN_CORRECTION,
        "max_correction": MAX_CORRECTION,
        "rounds": ROUNDS,
        "features": columns,
        "categories": cats,
        "blocks": {},
    }
    print("\nRMSE (candidate minus reference; negative is better)")
    for name, mask in blocks.items():
        base = rmse(y, reference, mask)
        new = rmse(y, candidate, mask)
        sse_delta = float(((y[mask] - candidate[mask]) ** 2).sum() - ((y[mask] - reference[mask]) ** 2).sum())
        report["blocks"][name] = {"n": int(mask.sum()), "reference": base, "candidate": new, "delta": new - base, "sse_delta": sse_delta}
        print(f"  {name:22s} {base:8.3f} -> {new:8.3f}  {new - base:+8.3f}s  SSE {sse_delta / 1e6:+.0f}M")

    pd.DataFrame(
        {
            config.ID_COL: raw[config.ID_COL].to_numpy(),
            "prediction": candidate,
            "reference": reference,
            "p_early": probability,
            "conditional_early_correction": conditional,
            "conditional_normal_correction": normal_conditional,
            "correction": correction,
        }
    ).to_parquet(config.DATA_INTERIM / "early_anchor_oof_predictions.parquet", index=False)
    (config.DATA_INTERIM / "early_anchor_oof_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
