"""Kill screen for ADS-B surface-congestion features on the no-record ordinary-taxi lane (section 40).

The lane: rows with no NM record (`nm_record_missing`), excluding schedule copies (|y - gap| <=
COPY_TOL) and day offsets (y > 12h), the two broken-clock classes that are closed. The quantity
screened is the v29 chain's leave-one-month-out residual, not the target (README.md: features that
correlate with the target but not the residual are redundant).

Decision is taken on the ten non-scored 2025 months only; January and July are reported but not
used, so they stay available for confirmation. Test: a shallow LightGBM residual corrector fitted
leave-one-month-out *within* the ten months, once with and once without the sx_* features, all
else equal. Criteria fixed before the first run:

    PASS if, pooled over the ten folds, the with-sx corrector cuts lane SSE by >= 5% more than the
    without-sx corrector, the with-sx one wins in >= 7 of 10 folds, and the gain keeps its sign
    after dropping the 10 and the 50 largest-|residual| rows.

5% of this lane is ~0.5% of total fold SSE, ~0.7s of fold RMSE: just above the 0.3s resolution
limit after the shrinkage a real model will show against an in-screen fit.

    uv run python -u scripts/adsb/screen_surface.py
"""
from __future__ import annotations

import importlib.util
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, evaluation, lirfbands, lirfmoe, matrix, mechanism

SX = ["sx_cov", "sx_dep_prev_n", "sx_dep_prev_taxi", "sx_dep_prev_hold", "sx_out_active", "sx_in_active", "sx_out_gs"]
SCORED = (1, 7)
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7}
ROUNDS = 300
# built on a GPU host, where the traces live; copied here because the matrix cache is local
FEATURES_PATH = "data/external/adsb_surface_features.parquet"


def _anatomy():
    path = pathlib.Path(__file__).resolve().parents[1] / "analyse_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("analyse_error_anatomy", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def corrector_oof(frame: pd.DataFrame, cols: list[str], months: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frame))
    for m in np.unique(months):
        tr, va = months != m, months == m
        ds = lgb.Dataset(frame.loc[tr, cols], frame.loc[tr, "resid"], categorical_feature=["apt"])
        out[va] = lgb.train(PARAMS, ds, ROUNDS).predict(frame.loc[va, cols])
    return out


def main():
    raw, X, _ = matrix.build("train", use_weather=True, use_sequencing=True, use_arrivals=True, use_context=True)
    y = raw[config.TARGET].to_numpy(dtype="float64")
    p = _anatomy().v29_oof(raw, X)
    month = evaluation.movement_month(raw).to_numpy()
    gap = X[mechanism.PROXY_COL].to_numpy(dtype="float64")
    missing = X[lirfbands.MISSING_COL].to_numpy() == 1
    lane = missing & ~(np.abs(y - gap) <= lirfmoe.COPY_TOL) & (y <= 43200)

    F = pd.read_parquet(FEATURES_PATH)
    t = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
    d = pd.DataFrame({config.ID_COL: raw[config.ID_COL].to_numpy(), "apt": raw["ADEP_mvt"].astype(str).to_numpy(),
                      "hour": t.dt.hour.to_numpy(), "dow": t.dt.dayofweek.to_numpy(), "gap": gap, "pred": p,
                      "resid": y - p, "month": month})
    d = d.merge(F, on=config.ID_COL, how="left")
    d = d[lane].reset_index(drop=True)
    d["apt"] = d["apt"].astype("category")
    covered = d.sx_dep_prev_n.notna()
    print(f"lane rows {len(d):,} (all months); with sx features {int(covered.sum()):,} ({covered.mean()*100:.1f}%)")
    print(d.assign(c=covered).groupby("apt", observed=True).agg(rows=("resid", "size"), covered=("c", "mean"),
          rmse=("resid", lambda r: float(np.sqrt((r ** 2).mean())))).round(3).to_string())

    ns = ~d.month.isin(SCORED).to_numpy()
    print("\nunivariate, non-scored months, covered rows: Spearman with v29 residual (and with |residual|)")
    c = d[ns & covered]
    for f in SX:
        ok = c[f].notna()
        print(f"   {f:18s} n={int(ok.sum()):6,}  rho {c.loc[ok, f].corr(c.loc[ok, 'resid'], method='spearman'):+.3f}"
              f"  |r| {c.loc[ok, f].corr(c.loc[ok, 'resid'].abs(), method='spearman'):+.3f}")

    base_cols = ["apt", "hour", "dow", "gap", "pred"]
    e = d[ns].reset_index(drop=True); mo = e.month.to_numpy()
    c0 = corrector_oof(e, base_cols, mo); c1 = corrector_oof(e, base_cols + SX, mo)
    r = e.resid.to_numpy()
    sse = {"v29": (r ** 2), "no-sx": (r - c0) ** 2, "with-sx": (r - c1) ** 2}
    print("\nresidual correctors, LOMO within the ten non-scored months")
    print(f"   {'month':>5} {'rows':>6} {'v29':>9} {'no-sx':>9} {'with-sx':>9}  with-sx vs no-sx")
    wins = 0
    for m in np.unique(mo):
        k = mo == m; s = {n: float(v[k].sum()) for n, v in sse.items()}
        wins += s["with-sx"] < s["no-sx"]
        print(f"   {m:>5} {int(k.sum()):>6} " + " ".join(f"{np.sqrt(s[n]/k.sum()):9.1f}" for n in sse)
              + f"  {100*(s['with-sx']-s['no-sx'])/s['no-sx']:+6.2f}%")
    tot = {n: float(v.sum()) for n, v in sse.items()}
    gain = (tot["no-sx"] - tot["with-sx"]) / tot["no-sx"]
    print("   pooled RMSE: " + "  ".join(f"{n} {np.sqrt(v/len(e)):.1f}" for n, v in tot.items()))
    print(f"   with-sx cuts lane SSE by {100*gain:+.2f}% vs no-sx; wins {wins}/10 folds")
    order = np.argsort(-np.abs(r)); drops = {}
    for n_drop in (10, 50):
        keep = np.ones(len(e), bool); keep[order[:n_drop]] = False
        drops[n_drop] = (sse["no-sx"][keep].sum() - sse["with-sx"][keep].sum()) / sse["no-sx"][keep].sum()
        print(f"   drop {n_drop:>2} largest |residual|: with-sx vs no-sx {100*drops[n_drop]:+.2f}%")
    passed = gain >= 0.05 and wins >= 7 and all(v > 0 for v in drops.values())
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'} (criteria: >= 5% pooled, >= 7/10 folds, positive after drop-10/50)")


if __name__ == "__main__":
    main()
