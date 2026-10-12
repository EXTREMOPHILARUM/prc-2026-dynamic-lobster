"""Screen EUROCONTROL daily per-airport ATFM features on the matched base component.

Joins `data/external/eurocontrol/daily_atfm.parquet` on (airport, UTC date) and refits
`base_matched` with and without the five `ec_*` columns on the given holdout months,
reporting capped-residual RMSE. Component-level screen, same protocol as
`scripts/experiment_highcard.py`.

    uv run python -u scripts/experiment_atfm_daily.py --months 2 10 1 7
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import arrivals, config, decomposition, evaluation, matrix

EC = ["ec_reg_share", "ec_out_late_share", "ec_arr_atfm_min_per_flt", "ec_arr_atfm_weather_min", "ec_daily_movements"]


def attach(raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    ec = pd.read_parquet(config.ROOT / "data/external/eurocontrol/daily_atfm.parquet")
    ec["date"] = pd.to_datetime(ec["date"])
    key = pd.DataFrame({"airport": raw["ADEP_mvt"].astype("string").to_numpy(),
                        "date": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None).dt.normalize().to_numpy()})
    joined = key.merge(ec[["airport", "date"] + EC], on=["airport", "date"], how="left")
    return X.join(joined[EC].set_index(X.index))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--months", type=int, nargs="+", default=[2, 10, 1, 7]); ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args(); start = time.time()
    raw, X, cats = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    X = X.drop(columns=[c for c in arrivals.FEATURES if "during_taxi" in c], errors="ignore")
    Xe = attach(raw, X)
    print("ec coverage on training rows:", {c: round(float(Xe[c].notna().mean()), 4) for c in EC}, flush=True)
    y = raw[config.TARGET].to_numpy(dtype="float64"); month = evaluation.movement_month(raw).to_numpy()
    matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
    resid = np.minimum(y, decomposition.TAIL_THRESHOLD) - X[decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD).to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]["base_matched"]
    cols0 = [c for c in X.columns if c not in decomposition.BASE_EXCLUDED]; cols1 = cols0 + EC; vcats = [c for c in cats if c in cols0]
    params = {**decomposition.BASE_PARAMS, "num_threads": args.threads, "seed": 42}
    out = {}
    for m in args.months:
        tr = matched & (month != m); va = matched & (month == m)
        r = {}
        for name, cols in (("without", cols0), ("with_ec", cols1)):
            b = lgb.train(params, lgb.Dataset(Xe.loc[tr, cols], resid[tr], categorical_feature=vcats), num_boost_round=rounds)
            r[name] = float(np.sqrt(np.mean((resid[va] - b.predict(Xe.loc[va, cols])) ** 2)))
        out[str(m)] = r; print(f"month {m:02d} without {r['without']:.3f} with_ec {r['with_ec']:.3f} ({r['with_ec']-r['without']:+.3f}) t={time.time()-start:.0f}s", flush=True)
    (config.DATA_INTERIM / "atfm_daily_screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
