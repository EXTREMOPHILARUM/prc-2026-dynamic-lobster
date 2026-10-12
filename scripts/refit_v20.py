"""Refit v20/v21 production: target encodings on all 2025, three-seed decomposition at the
lr 0.02 rounds (optionally uncapped matched base), gate refit with context v2, wrapper with
Rome rules. Encodings are stored in the ordinary model's JSON and applied at scoring time.

    uv run python -u scripts/refit_v20.py --out mechanism_v20 [--uncapped] [--rome-rules]
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from prc import arrivals, config, decomposition, features, lirfbands, matrix, mechanism, tencode

SEEDS = (100, 101, 102)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="mechanism_v20"); ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--rome-rules", action="store_true"); ap.add_argument("--rounds-file", default="rounds_v20_lr0.02.json"); ap.add_argument("--atfm", action="store_true")
    ap.add_argument("--base-params", default=None, help="JSON merged into decomposition.BASE_PARAMS (log section 63)")
    ap.add_argument("--standrwy", action="store_true", help="add prc.standrwy stand-to-runway distances (log section 63)")
    ap.add_argument("--cellspread", action="store_true", help="tencode spread tables (cell p90 / std, log section 63)")
    ap.add_argument("--rwystate", action="store_true", help="add prc.rwystate features to the base (log section 63)")
    ap.add_argument("--inner-from", default=None, help="copy this inner model JSON's serving flags (e.g. mechanism_mechanism_v26_inner)")
    args = ap.parse_args()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True, use_atfm=args.atfm)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    if args.rwystate:
        from prc import rwystate
        X = X.join(rwystate.compute(raw).set_index(X.index))
    if args.standrwy:
        from prc import standrwy
        X = X.join(standrwy.compute(raw).set_index(X.index))
    y = raw[config.TARGET].to_numpy(dtype="float64"); ys = raw[config.TARGET].astype("float32")
    anchor_raw = X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"); matched = np.isfinite(anchor_raw)
    enc = tencode.fit(raw, y, np.where(matched, anchor_raw - y, np.nan), np.ones(len(y), bool), spread=args.cellspread)
    Xte = X.join(tencode.transform(raw, enc))
    sel = json.loads((config.DATA_INTERIM / args.rounds_file).read_text())
    rounds = json.loads((config.DATA_INTERIM / "component_rounds_ctx.json").read_text())["rounds"] | {"base_matched": sel["best_iter"]}
    decomposition.BASE_PARAMS["learning_rate"] = sel["lr"]
    if args.base_params:
        decomposition.BASE_PARAMS.update(json.loads(args.base_params))
    ctx = json.loads((config.MODELS / "decomposition_ctx.json").read_text())
    meta = ctx | {"use_atfm": bool(args.atfm),"features": list(Xte.columns), "category_levels": features.category_levels(Xte, cats), "rounds": rounds,
                  "learning_rate": sel["lr"], "target_encoding": enc, "matched_uncapped": args.uncapped}
    stems = []
    for s in SEEDS:
        for p in (decomposition.BASE_PARAMS, decomposition.SEVERITY_PARAMS):
            p["seed"] = s
        models = decomposition.fit_fixed(Xte, ys, cats, rounds=rounds, uncapped_matched=args.uncapped)
        stem = f"decomposition_{args.out}_s{s}"; decomposition.save(models, meta | {"seed": s}, config.MODELS / stem); stems.append(stem); print("saved", stem, flush=True)
    label = mechanism.fallback_label(raw)
    gate_cols = mechanism.gate_features(Xte) + [c for c in Xte.columns if c.startswith(("nbr_apt", "q_apt", "arrctx_", "clk_", "ord_"))]
    gate = lgb.train({**mechanism.GATE_PARAMS, "seed": 3}, lgb.Dataset(Xte[gate_cols], label.astype("int8"), categorical_feature=[c for c in cats if c in gate_cols]), num_boost_round=200)
    inner = f"mechanism_{args.out}_inner"; gate.save_model(str(config.MODELS / f"{inner}.gate.txt"))
    mech = json.loads((config.MODELS / f"{args.inner_from or 'mechanism_ctx19'}.json").read_text())
    mech |= {"ordinary_stem": stems[0], "seed_stems": stems, "gate_features": gate_cols, "features": list(Xte.columns),
             "category_levels": features.category_levels(Xte, cats), "target_encoding": enc, "matched_uncapped": args.uncapped, "use_atfm": bool(args.atfm),
             "base_params": dict(decomposition.BASE_PARAMS), "use_rwystate": bool(args.rwystate),
             "use_standrwy": bool(args.standrwy)}
    (config.MODELS / f"{inner}.json").write_text(json.dumps(mech, indent=2))
    v19 = json.loads((config.MODELS / "mechanism_v19.json").read_text())
    v19 |= {"inner_stem": inner, "band_table": lirfbands.fit(X, y), "band_edges_h": [e if e != float("inf") else None for e in lirfbands.EDGES_H],
            "rome_rules": bool(args.rome_rules)}
    (config.MODELS / f"{args.out}.json").write_text(json.dumps(v19, indent=2))
    print(f"saved {inner}.*, {args.out}.json")


if __name__ == "__main__":
    main()
