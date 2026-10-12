"""CatBoost as a drop-in for the decomposition base (ordinary) component.

Motivation: the whole stack is LightGBM and bulk hyperparameters are saturated
(log section 4). CatBoost uses ordered target statistics for categoricals, a genuinely
different inductive bias on a categorical-heavy dataset - not another GBM tuning pass.

Only the two *base* experts (matched-clock, missing-clock) are swapped. The tail gate,
tail severity and the mechanism fallback gate are reused from the LightGBM fit of the same
fold, so the delta is attributable to the categorical treatment alone.

    uv run python -u scripts/experiment_catboost.py --stage rounds
    uv run python -u scripts/experiment_catboost.py --stage cv
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc import config, data, decomposition, evaluation, features, mechanism, sequencing, weather

INNER_MONTHS = (4, 10)  # disjoint from the ranking months, as component_rounds.json uses
CB_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.1,   # 2x LightGBM's rate: symmetric trees are slow per round here
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "max_ctr_complexity": 1,  # categorical *combinations* cost far more than they return
    "border_count": 128,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": 100,
    "allow_writing_files": False,
}
ROUNDS_PATH = config.DATA_INTERIM / "catboost_rounds.json"


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = features.departures_with_target(data.load_training()).reset_index(drop=True)
    raw = weather.attach(raw)
    extra = list(weather.FEATURES)
    raw = sequencing.add(raw)
    extra += sequencing.FEATURES
    X, cats = features.build(raw)
    X = X.drop(columns=["month"])
    X = X.join(raw[extra])
    return raw, X, cats


def cb_frame(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    """CatBoost needs categoricals as non-null strings; numerics keep their NaNs."""
    out = X.copy()
    for c in cats:
        out[c] = out[c].astype("string").fillna("__NA__").astype(str)
    return out


def fit_cb(Xc: pd.DataFrame, target: pd.Series, cats: list[str], iterations: int,
           valid: tuple[pd.DataFrame, pd.Series] | None = None) -> CatBoostRegressor:
    model = CatBoostRegressor(iterations=iterations, **CB_PARAMS)
    pool = Pool(Xc, target, cat_features=cats)
    eval_pool = Pool(valid[0], valid[1], cat_features=cats) if valid else None
    model.fit(pool, eval_set=eval_pool, use_best_model=valid is not None,
              early_stopping_rounds=40 if valid else None)
    return model


def base_targets(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, pd.Series, pd.Series]:
    matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
    capped = y.clip(upper=decomposition.TAIL_THRESHOLD)
    anchor = X.loc[matched, decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD)
    return matched, capped[matched] - anchor, capped[~matched]


def cb_base(models: dict[str, CatBoostRegressor], Xc: pd.DataFrame, X: pd.DataFrame,
            cats: list[str]) -> np.ndarray:
    matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
    base = np.empty(len(X), dtype="float64")
    if matched.any():
        base[matched] = (
            X.loc[matched, decomposition.ANCHOR_COL].clip(0, decomposition.TAIL_THRESHOLD)
            + models["base_matched"].predict(Pool(Xc.loc[matched], cat_features=cats))
        )
    if (~matched).any():
        base[~matched] = models["base_missing"].predict(Pool(Xc.loc[~matched], cat_features=cats))
    return np.clip(base, 0, decomposition.TAIL_THRESHOLD)


def ordinary_from_base(base: np.ndarray, lgb_models: dict, X: pd.DataFrame) -> np.ndarray:
    """Recombine a base prediction with the frozen LightGBM tail gate and severity."""
    gate, sev = lgb_models["gate"], lgb_models["severity"]
    p = gate.predict(X[gate.feature_name()])
    excess = np.maximum(sev.predict(X[sev.feature_name()]), 0)
    scale = np.where(X["ADEP_mvt"].astype("string").eq("LIRF"),
                     decomposition.LIRF_TAIL_SCALE, decomposition.OTHER_TAIL_SCALE)
    return np.clip(base + scale * p * excess, 0, None)


def stage_rounds() -> None:
    raw, X, cats = load_frame()
    y = raw[config.TARGET].astype("float32")
    month = evaluation.movement_month(raw).to_numpy()
    Xc = cb_frame(X, cats)
    inner = np.isin(month, INNER_MONTHS)
    out = {}
    for name in ("base_matched", "base_missing"):
        want_matched = name == "base_matched"
        rounds = {}
        for part, mask in (("tr", ~inner), ("va", inner)):
            m, t_match, t_miss = base_targets(X.loc[mask], y.loc[mask])
            sel = X.index[mask][m if want_matched else ~m]
            rounds[part] = (sel, t_match if want_matched else t_miss)
        t0 = time.time()
        model = fit_cb(Xc.loc[rounds["tr"][0]], rounds["tr"][1], cats, 600,
                       valid=(Xc.loc[rounds["va"][0]], rounds["va"][1]))
        out[name] = int(model.get_best_iteration()) + 1
        print(f"{name}: best_iter={out[name]} rows={len(rounds['tr'][0]):,} "
              f"wall={time.time()-t0:.0f}s")
    ROUNDS_PATH.write_text(json.dumps({"inner_months": list(INNER_MONTHS), "rounds": out}, indent=2))
    print("wrote", ROUNDS_PATH)


def stage_cv(folds: list[int]) -> None:
    raw, X, cats = load_frame()
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    month = evaluation.movement_month(raw).to_numpy()
    Xc = cb_frame(X, cats)
    lgb_rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]
    cb_rounds = json.loads(ROUNDS_PATH.read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet")[
            [config.ID_COL, "prediction"]], on=config.ID_COL, how="left")["prediction"].to_numpy()
    ord_lgb = np.full(len(raw), np.nan)
    ord_cb = np.full(len(raw), np.nan)
    for m in folds:
        valid = month == m
        tr = ~valid
        t0 = time.time()
        lgb_models = decomposition.fit_fixed(X.loc[tr], y.loc[tr], cats, rounds=lgb_rounds)
        t_lgb = time.time() - t0
        ord_lgb[valid] = decomposition.predict(lgb_models, X.loc[valid])

        t0 = time.time()
        matched, t_match, t_miss = base_targets(X.loc[tr], y.loc[tr])
        idx = X.index[tr]
        cb_models = {
            "base_matched": fit_cb(Xc.loc[idx[matched]], t_match, cats,
                                   cb_rounds["base_matched"]),
            "base_missing": fit_cb(Xc.loc[idx[~matched]], t_miss, cats,
                                   cb_rounds["base_missing"]),
        }
        t_cb = time.time() - t0
        base = cb_base(cb_models, Xc.loc[valid], X.loc[valid], cats)
        ord_cb[valid] = ordinary_from_base(base, lgb_models, X.loc[valid])
        b = mechanism.blend(gate_p[valid], proxy[valid], ord_lgb[valid])
        c = mechanism.blend(gate_p[valid], proxy[valid], ord_cb[valid])
        yv = y_np[valid]

        def r(p: np.ndarray, yv: np.ndarray = yv) -> float:
            return float(np.sqrt(np.mean((yv - p) ** 2)))

        print(f"  month {m:02d} done  lgb={t_lgb:.0f}s cb={t_cb:.0f}s  "
              f"rmse baseline={r(baseline[valid]):.2f} lgb={r(b):.2f} cb={r(c):.2f}", flush=True)
        pd.DataFrame({config.ID_COL: raw[config.ID_COL], "ordinary_lgb": ord_lgb,
                      "ordinary_cb": ord_cb}).to_parquet(
            config.DATA_INTERIM / "catboost_oof_ordinary.parquet", index=False)

    scored = np.isin(month, folds)
    report(raw[scored], y_np[scored], month[scored], gate_p[scored], proxy[scored],
           ord_lgb[scored], ord_cb[scored])


def report(raw, y_np, month, gate_p, proxy, ord_lgb, ord_cb) -> None:
    baseline = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet")[
            [config.ID_COL, "prediction"]], on=config.ID_COL, how="left")["prediction"].to_numpy()

    def blended(o):
        return mechanism.blend(gate_p, proxy, o)

    p_lgb, p_cb = blended(ord_lgb), blended(ord_cb)
    rmse = lambda p, m=None: float(np.sqrt(np.mean((y_np - p)[m if m is not None else slice(None)] ** 2)))
    print(f"\nbaseline artifact RMSE {rmse(baseline):.4f}   "
          f"lgb refit {rmse(p_lgb):.4f}   catboost {rmse(p_cb):.4f}")

    r_lgb, r_cb = y_np - p_lgb, y_np - p_cb
    print(f"residual correlation lgb vs cb: full {np.corrcoef(r_lgb, r_cb)[0,1]:.4f}  "
          f"bulk {np.corrcoef(r_lgb[y_np<=3600], r_cb[y_np<=3600])[0,1]:.4f}")

    # blend weight fitted OUT of the scored months
    fitmask = ~np.isin(month, (1, 7))
    if fitmask.sum() == 0:
        raise SystemExit("no folds outside the ranking months: cannot fit a blend weight")
    grid = np.arange(0, 1.01, 0.05)
    sses = [float(((y_np[fitmask] - blended((1-w)*ord_lgb + w*ord_cb)[fitmask]) ** 2).sum())
            for w in grid]
    w = float(grid[int(np.argmin(sses))])
    print("\nblend weight scan (fitted on months != 1,7):")
    for g, s in zip(grid, sses):
        print(f"  w={g:.2f}  rmse={np.sqrt(s/fitmask.sum()):.4f}")
    print(f"chosen w={w:.2f}")
    p_bl = blended((1 - w) * ord_lgb + w * ord_cb)

    rows = []
    tail = y_np > 3600
    segs = [("all", np.ones(len(y_np), bool)), ("bulk", ~tail), ("tail", tail),
            ("jan", month == 1), ("jul", month == 7)]
    segs = [(n, m) for n, m in segs if m.sum() > 0]
    for name, m in segs:
        o = float(((y_np[m] - baseline[m]) ** 2).sum())
        for lbl, p in (("lgb", p_lgb), ("cb", p_cb), (f"blend w={w:.2f}", p_bl)):
            n = float(((y_np[m] - p[m]) ** 2).sum())
            rows.append({"segment": name, "model": lbl, "n": int(m.sum()),
                         "rmse": np.sqrt(n / m.sum()), "vs_baseline_s": np.sqrt(n/m.sum()) - np.sqrt(o/m.sum()),
                         "sse_pct": 100 * (n - o) / o})
    print("\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    pm = []
    for mm in sorted(set(month.tolist())):
        v = month == mm
        o = float(((y_np[v] - baseline[v]) ** 2).sum())
        rec = {"month": mm, "baseline": np.sqrt(o / v.sum())}
        for lbl, p in (("lgb", p_lgb), ("cb", p_cb), ("blend", p_bl)):
            rec[lbl] = np.sqrt(float(((y_np[v] - p[v]) ** 2).sum()) / v.sum())
        pm.append(rec)
    pmd = pd.DataFrame(pm)
    print("\n" + pmd.round(2).to_string(index=False))
    for lbl in ("lgb", "cb", "blend"):
        print(f"months improved vs baseline, {lbl}: {int((pmd[lbl] < pmd.baseline).sum())}/12")
    config.DATA_INTERIM.joinpath("catboost_oof_metrics.json").write_text(json.dumps(
        {"weight": w, "segments": rows, "per_month": pmd.to_dict("records")}, indent=2, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rounds", "cv"], required=True)
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10,11,12")
    args = ap.parse_args()
    if args.stage == "rounds":
        stage_rounds()
    else:
        stage_cv([int(f) for f in args.folds.split(",")])


if __name__ == "__main__":
    main()
