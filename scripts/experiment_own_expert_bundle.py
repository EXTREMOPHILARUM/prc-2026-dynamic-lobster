"""v35 bundle screen for the own-flight expert: capacity, seed bagging, identity encodings.

The v34 expert (`ownexpert.PARAMS`, 300 rounds, one seed, v5 features) has kept the settings fixed
at the section 46 screen, when it had five inputs. Each config below is run twelve-fold exactly as
`experiment_own_expert.py` (fitted on the other eleven months of the v29 reconstruction's OOF
residual, applied to the held-out month), and compared **paired against the v5 baseline config**.

Selection is on the ten non-scored months only (SSE summed over months 2-6, 8-12); January and
July are reported as calendar-matched diagnostics, never used to choose.

Encodings (`--encode`): the residual the chain leaves, averaged by (airport, stand), (airport,
operator) and (airport, aircraft type), smoothed toward zero with K pseudo-rows. Nested: for a
held-out month m the encoding of a training row in month t uses the training months other than
t, and the held-out rows use all training months, so no row sees its own residual. The residual
is clipped to +/-CLIP before averaging (fixed before the first run: one day-slip row must not set
a stand's mean).

    uv run python -u scripts/experiment_own_expert_bundle.py base r600 r1200_lr025 l63_r600 enc seeds5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, lirfbands, matrix, ownexpert

SCORED = (1, 7)
ENC_KEYS = {"enc_stand": "STAND_mvt", "enc_operator": "AIRCRAFT_OPERATOR_flt", "enc_type": "AIRCRAFT_TYPE_flt"}
K = 100.0
CLIP = 1800.0
CONFIGS = {
    "base": {},
    "r600": {"rounds": 600},
    "r1200_lr025": {"rounds": 1200, "params": {"learning_rate": 0.025}},
    "l63_r600": {"rounds": 600, "params": {"num_leaves": 63}},
    "l63_r1200_lr025": {"rounds": 1200, "params": {"num_leaves": 63, "learning_rate": 0.025}},
    "md100_r600": {"rounds": 600, "params": {"min_data_in_leaf": 100}},
    "l127_r1200_lr025": {"rounds": 1200, "params": {"num_leaves": 127, "learning_rate": 0.025}},
    "l255_r1200_lr025": {"rounds": 1200, "params": {"num_leaves": 255, "learning_rate": 0.025}},
    "l127_r2400_lr025": {"rounds": 2400, "params": {"num_leaves": 127, "learning_rate": 0.025}},
    "enc": {"encode": True},
    "seeds5": {"seeds": 5},
}


def _load(name):
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def key_codes(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer code of (airport, identity) per row for each encoding."""
    return {name: pd.factorize(raw["ADEP_mvt"].astype(str) + "|" + raw[col].astype(str), use_na_sentinel=False)[0]
            for name, col in ENC_KEYS.items()}


def encode(codes: dict, resid: np.ndarray, month: np.ndarray, rows: np.ndarray, train_months, target_rows):
    """Smoothed mean of clipped `resid` over scope rows of `train_months`, per key, for `target_rows`."""
    src = rows & np.isin(month, list(train_months))
    r = np.clip(resid[src], -CLIP, CLIP)
    out = {}
    for name, c in codes.items():
        n = int(c.max()) + 1
        s = np.bincount(c[src], weights=r, minlength=n); cnt = np.bincount(c[src], minlength=n)
        out[name] = (s / (cnt + K))[c[target_rows]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=sorted(CONFIGS))
    ap.add_argument("--src", default="adsb_own_2025_v5.parquet")
    ap.add_argument("--features", type=int, default=5, choices=[5, 6, 7], help="ownexpert feature version")
    ap.add_argument("--drop-late", action="store_true",
                    help="v6 without the late take-off rows (own_to_late == 1): the callsign fix alone (section 55)")
    ap.add_argument("--tag", default="", help="suffix for the OOF file")
    ap.add_argument("--save", default=None, metavar="STEM",
                    help="also fit the (single) config on all twelve months and write models/STEM.txt")
    ap.add_argument("--combine", default=None, help="JSON dict merged into every config (e.g. the chosen capacity)")
    args = ap.parse_args()
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    p = _load("analyse_error_anatomy").v29_oof(raw, X)
    month = evaluation.movement_month(raw).to_numpy()
    apt = raw["ADEP_mvt"].astype(str).to_numpy()
    anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan, X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"))
    src = pd.read_parquet(config.DATA_INTERIM / args.src)
    if args.drop_late:
        src = src[src.own_to_late != 1]
    own = raw[[config.ID_COL]].merge(src, on=config.ID_COL, how="left")
    f = ownexpert.frame(apt, raw["MVT_TIME_UTC_mvt"], p, anchor, own)
    rows = ownexpert.scope(f)
    codes = key_codes(raw)
    resid = y - p
    base_feats = list({5: ownexpert.FEATURES_V5, 6: ownexpert.FEATURES_V6, 7: ownexpert.FEATURES_V7}[args.features])
    sc = np.isin(month, SCORED)
    print(f"{len(f):,} rows; scope {int(rows.sum()):,}; source {args.src}; features v{args.features}"
          f"{' without late take-offs' if args.drop_late else ''}", flush=True)

    results, oofs = {}, {}
    extra = json.loads(args.combine) if args.combine else {}
    for name in args.configs:
        cfg = {**CONFIGS[name], **extra}
        params = {**ownexpert.PARAMS, **cfg.get("params", {}), **extra.get("params", {})}
        rounds = cfg.get("rounds", ownexpert.ROUNDS); seeds = cfg.get("seeds", 1); enc = cfg.get("encode", False)
        feats = base_feats + (list(ENC_KEYS) if enc else [])
        q = p.copy()
        for m in range(1, 13):
            tr_months = [t for t in range(1, 13) if t != m]
            tr = (month != m) & rows; va = (month == m) & rows
            F = f.copy()
            if enc:
                for c in ENC_KEYS:
                    F[c] = np.nan
                for t in tr_months:     # nested: a training row's encoding never includes its own month
                    tgt = (month == t) & rows
                    for c, v in encode(codes, resid, month, rows, [s for s in tr_months if s != t], tgt).items():
                        F.loc[tgt, c] = v
                for c, v in encode(codes, resid, month, rows, tr_months, va).items():
                    F.loc[va, c] = v
            pred = np.zeros(int(va.sum()))
            for s in range(seeds):
                b = lgb.train({**params, "seed": params["seed"] + s}, lgb.Dataset(F.loc[tr, feats], resid[tr],
                              categorical_feature=["apt"]), rounds)
                pred += b.predict(F.loc[va, feats]) / seeds
            q[va] = np.clip(p[va] + pred, 0, None)
        e0, e1 = y - p, y - q
        per = {m: (rmse(e1[month == m]) - rmse(e0[month == m])) for m in range(1, 13)}
        ns = ~sc
        results[name] = {"cfg": cfg, "twelve": rmse(e1) - rmse(e0), "janjul": rmse(e1[sc]) - rmse(e0[sc]),
                         "nonscored_sse": float((e1[ns] ** 2).sum()), "per_month": per}
        oofs[name] = q
        print(f"{name:18s} twelve {results[name]['twelve']:+.3f}  Jan+Jul {results[name]['janjul']:+.3f}  "
              f"non-scored SSE {results[name]['nonscored_sse']/1e6:,.1f}M", flush=True)

    ref = "base" if "base" in oofs else None
    if ref:
        qb = oofs[ref]; ns = ~sc
        print(f"\npaired against {ref} (v5 expert settings):")
        for name, q in oofs.items():
            if name == ref:
                continue
            d = [rmse(y[month == m] - q[month == m]) - rmse(y[month == m] - qb[month == m]) for m in range(1, 13)]
            wins_ns = sum(d[m - 1] < 0 for m in range(1, 13) if m not in SCORED)
            dsse = float(((y - q) ** 2)[ns].sum() - ((y - qb) ** 2)[ns].sum())
            gain = (y - qb) ** 2 - (y - q) ** 2
            o = np.argsort(-gain[sc]); n = int(sc.sum()); bs = float(((y[sc] - qb[sc]) ** 2).sum())
            drops = [np.sqrt((bs - (gain[sc].sum() - gain[sc][o[:k]].sum())) / n) - np.sqrt(bs / n) for k in (1, 10, 50)]
            print(f"  {name:18s} non-scored dSSE {dsse/1e6:+8.1f}M, wins {wins_ns}/10 non-scored; "
                  f"twelve {rmse(y-q)-rmse(y-qb):+.3f}  Jan+Jul {rmse(y[sc]-q[sc])-rmse(y[sc]-qb[sc]):+.3f}  "
                  f"drop-1/10/50 {drops[0]:+.3f}/{drops[1]:+.3f}/{drops[2]:+.3f}", flush=True)
    out = config.DATA_INTERIM / "own_expert_bundle_results.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update({k: {**v, "per_month": {str(m): x for m, x in v["per_month"].items()}} for k, v in results.items()})
    out.write_text(json.dumps(prev, indent=1, default=str))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "v29": p, **{f"own_{k}": v for k, v in oofs.items()}}
                 ).to_parquet(config.DATA_INTERIM / f"own_expert_bundle_{'_'.join(args.configs)}{args.tag}_oof.parquet", index=False)


    if args.save:
        if len(args.configs) != 1 or CONFIGS[args.configs[0]].get("encode"):
            raise SystemExit("--save takes exactly one config without --encode")
        cfg = {**CONFIGS[args.configs[0]], **extra}
        params = {**ownexpert.PARAMS, **cfg.get("params", {}), **extra.get("params", {})}
        seeds = cfg.get("seeds", 1)
        if seeds != 1:
            raise SystemExit("--save serves one booster; seed bagging is not supported by prc.submit")
        b = lgb.train(params, lgb.Dataset(f.loc[rows, base_feats], resid[rows], categorical_feature=["apt"]),
                      cfg.get("rounds", ownexpert.ROUNDS))
        out = config.MODELS / f"{args.save}.txt"
        b.save_model(str(out)); print(f"saved {out}: {args.configs[0]}, features {b.feature_name()}")


if __name__ == "__main__":
    main()
