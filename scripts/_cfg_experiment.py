"""Paired 12-fold A/B of airport configuration-state features on top of v11 (weather+sequencing)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cfgstate as cfgmod
import _cfgstate2 as cfgmod2

from prc import config, data, decomposition, evaluation, features, mechanism, sequencing, weather

SUBSETS = {
    "core": ["cfg_n_dep_rwy_30m", "cfg_n_arr_rwy_30m", "cfg_this_rwy_share_30m",
             "cfg_this_differs_from_mode", "cfg_dep_changed_30m",
             "cfg_time_since_dep_change_s", "cfg_dep_set_jaccard_30m"],
    "all": list(cfgmod.FEATURES),
    "reversal": list(cfgmod2.FEATURES),
    "union": list(cfgmod.FEATURES) + list(cfgmod2.FEATURES),
}

def _limit_threads() -> None:
    for d in (decomposition.BASE_PARAMS, decomposition.GATE_PARAMS, decomposition.SEVERITY_PARAMS):
        d["num_threads"] = 2


def main() -> None:
    _limit_threads()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all", choices=list(SUBSETS))
    ap.add_argument("--out", default="cfgstate_oof")
    args = ap.parse_args()
    cfg_cols = SUBSETS[args.subset]

    mv = data.load_training()
    cfg_feat = cfgmod.add(mv)
    if any(c in SUBSETS[args.subset] for c in cfgmod2.FEATURES):
        cfg_feat = cfg_feat.merge(cfgmod2.add(mv), on=config.ID_COL, how="left")
    print("cfg features built", cfg_feat.shape, flush=True)

    raw = features.departures_with_target(mv).reset_index(drop=True)
    del mv
    raw = weather.attach(raw)
    raw = sequencing.add(raw)
    raw = raw.merge(cfg_feat, on=config.ID_COL, how="left")
    extra = list(weather.FEATURES) + list(sequencing.FEATURES)

    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    X_base = X.join(raw[extra])
    X_new = X_base.join(raw[cfg_cols])

    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw)
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    bl = pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet")
    bcol = "blended" if "blended" in bl.columns else next(c for c in bl.columns if c != config.ID_COL)
    baseline = raw[[config.ID_COL]].merge(bl[[config.ID_COL, bcol]], on=config.ID_COL,
                                          how="left")[bcol].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    pred = np.full(len(raw), np.nan)
    for m in range(1, 13):
        valid = (month == m).to_numpy()
        models = decomposition.fit_fixed(X_new.loc[~valid], y.loc[~valid], cats, rounds=rounds)
        ordinary = decomposition.predict(models, X_new.loc[valid])
        pred[valid] = mechanism.blend(gate_p[valid], proxy[valid], ordinary)
        print(f"  month {m:02d} done", flush=True)

    def sse(mask, p):
        return float(((y_np[mask] - p[mask]) ** 2).sum())

    allm = np.ones(len(y_np), bool)
    o, n = sse(allm, baseline), sse(allm, pred)
    print(f"\nOOF RMSE baseline {np.sqrt(o/len(y_np)):.4f}  with cfg {np.sqrt(n/len(y_np)):.4f}  "
          f"SSE {100*(n-o)/o:+.3f}%")

    tail = y_np > 3600
    aobt_missing = raw["AOBT_3_flt"].isna().to_numpy()
    segs = [("all", allm), ("bulk", ~tail), ("tail", tail),
            ("aobt_present", ~aobt_missing), ("aobt_missing", aobt_missing)]
    ap_arr = raw["ADEP_mvt"].astype(str).to_numpy()
    for a in sorted(set(ap_arr)):
        segs.append((f"apt {a}", ap_arr == a))
    rows = []
    for name, m in segs:
        if m.sum() < 50:
            continue
        oo, nn = sse(m, baseline), sse(m, pred)
        rows.append({"segment": name, "n": int(m.sum()), "rmse_old": np.sqrt(oo/m.sum()),
                     "rmse_new": np.sqrt(nn/m.sum()), "sse_delta_pct": 100*(nn-oo)/oo,
                     "sse_share_of_total_pct": 100*oo/o})
    seg_df = pd.DataFrame(rows)
    print("\n" + seg_df.round(3).to_string(index=False))

    pm = []
    for m in range(1, 13):
        v = (month == m).to_numpy()
        oo, nn = sse(v, baseline), sse(v, pred)
        pm.append({"month": m, "rmse_old": np.sqrt(oo/v.sum()), "rmse_new": np.sqrt(nn/v.sum()),
                   "better": bool(nn < oo)})
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(3).to_string(index=False))
    print(f"\nmonths improved: {int(pmd.better.sum())}/12")

    out = config.DATA_INTERIM / f"{args.out}_{args.subset}"
    Path(str(out) + "_metrics.json").write_text(json.dumps({
        "subset": args.subset, "features": cfg_cols,
        "rmse_baseline": float(np.sqrt(o/len(y_np))), "rmse_cfg": float(np.sqrt(n/len(y_np))),
        "sse_delta_pct": 100*(n-o)/o,
        "segments": seg_df.to_dict("records"), "per_month": pmd.to_dict("records"),
    }, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        str(out) + "_predictions.parquet", index=False)
    print("wrote", out)

if __name__ == "__main__":
    main()
