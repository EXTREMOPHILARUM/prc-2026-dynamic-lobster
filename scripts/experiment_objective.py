"""Paired 12-fold A/B of objective / target-parameterisation choices for the ordinary
(decomposition base) component of the mechanism model.

RMSE's optimal prediction is the conditional MEAN.  The base models currently fit L2 on a
winsorised target.  This script swaps only the two *base* learners - the tail gate, the
tail severity model and the frozen mechanism gate are shared across variants, so any delta
is attributable to the objective / parameterisation alone.

Robust and transformed fits do not estimate a mean, so each is paired with an explicit
mean correction (Duan smearing for the log fits, a binned mean-residual map for all of
them) applied on the original second scale.  The uncorrected version is reported too, so
the correction's contribution is visible.

    OMP_NUM_THREADS=2 uv run python -u scripts/experiment_objective.py --variants all
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import config, decomposition, evaluation, matrix, mechanism

CAP = decomposition.TAIL_THRESHOLD
ANCHOR = decomposition.ANCHOR_COL
NUM_THREADS = 2
N_BINS = 40


def _params(base: dict, **over) -> dict:
    p = dict(base)
    p["num_threads"] = NUM_THREADS
    p.update(over)
    return p


def _train(params, X, target, cats, rounds):
    return lgb.train(
        params, lgb.Dataset(X, target, categorical_feature=cats), num_boost_round=rounds
    )


def _binned_correction(pred_tr: np.ndarray, y_tr: np.ndarray, pred_va: np.ndarray) -> np.ndarray:
    """Additive mean-residual map, estimated on the training fold, in quantile bins of the
    prediction and interpolated between bin means so the correction is continuous."""
    good = np.isfinite(pred_tr) & np.isfinite(y_tr)
    pred_tr, y_tr = pred_tr[good], y_tr[good]
    if len(pred_tr) < 500:
        return pred_va
    edges = np.quantile(pred_tr, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pred_va + float((y_tr - pred_tr).mean())
    idx = np.clip(np.searchsorted(edges, pred_tr, side="right") - 1, 0, len(edges) - 2)
    resid = y_tr - pred_tr
    nb = len(edges) - 1
    counts = np.bincount(idx, minlength=nb).astype("float64")
    sums = np.bincount(idx, weights=resid, minlength=nb)
    centres = np.bincount(idx, weights=pred_tr, minlength=nb)
    ok = counts >= 200
    if ok.sum() < 2:
        return pred_va + float(resid.mean())
    xs = centres[ok] / counts[ok]
    ys = sums[ok] / counts[ok]
    order = np.argsort(xs)
    return pred_va + np.interp(pred_va, xs[order], ys[order])


# --- variants -------------------------------------------------------------------------
# each returns {name: prediction-on-capped-scale} for the validation rows of one subset


def v_baseline(Xtr, ytr, Xva, cats, rounds, anchored):
    """L2 on the winsorised target; residual-to-anchor parameterisation where available."""
    p = _params(decomposition.BASE_PARAMS)
    if anchored:
        a_tr = Xtr[ANCHOR].clip(0, CAP)
        a_va = Xva[ANCHOR].clip(0, CAP).to_numpy()
        m = _train(p, Xtr, ytr - a_tr, cats, rounds)
        return {"baseline": a_va + m.predict(Xva)}
    m = _train(p, Xtr, ytr, cats, rounds)
    return {"baseline": m.predict(Xva)}


def v_l2_direct(Xtr, ytr, Xva, cats, rounds, anchored):
    """Control: L2 on the winsorised target with no anchor subtraction, so the transform
    variants (which cannot subtract an anchor) have a like-for-like comparator."""
    m = _train(_params(decomposition.BASE_PARAMS), Xtr, ytr, cats, rounds)
    pred = m.predict(Xva)
    return {"l2_direct": pred}


def _robust(objective, extra, tag):
    def fn(Xtr, ytr, Xva, cats, rounds, anchored):
        p = _params(decomposition.BASE_PARAMS, objective=objective, **extra)
        if anchored:
            a_tr = Xtr[ANCHOR].clip(0, CAP)
            a_va = Xva[ANCHOR].clip(0, CAP).to_numpy()
            m = _train(p, Xtr, ytr - a_tr, cats, rounds)
            raw = a_va + m.predict(Xva)
            tr = a_tr.to_numpy() + m.predict(Xtr)
        else:
            m = _train(p, Xtr, ytr, cats, rounds)
            raw = m.predict(Xva)
            tr = m.predict(Xtr)
        return {tag: raw, f"{tag}_corr": _binned_correction(tr, ytr.to_numpy(), raw)}
    return fn


def v_logratio(Xtr, ytr, Xva, cats, rounds, anchored):
    """log1p target (log-ratio to the anchor where available), back-transformed to the
    MEAN by Duan smearing, and separately by the binned map."""
    p = _params(decomposition.BASE_PARAMS)
    off_tr = np.log1p(Xtr[ANCHOR].clip(0, CAP).to_numpy()) if anchored else 0.0
    off_va = np.log1p(Xva[ANCHOR].clip(0, CAP).to_numpy()) if anchored else 0.0
    z = np.log1p(np.maximum(ytr.to_numpy(), 0.0)) - off_tr
    m = _train(p, Xtr, pd.Series(z, index=Xtr.index), cats, rounds)
    zhat_tr = m.predict(Xtr)
    zhat_va = m.predict(Xva)
    smear = float(np.mean(np.exp(z - zhat_tr)))
    naive = np.expm1(zhat_va + off_va)
    duan = np.exp(zhat_va + off_va) * smear - 1.0
    tr_naive = np.expm1(zhat_tr + off_tr)
    tag = "logratio" if anchored else "log1p"
    return {
        f"{tag}_naive": np.clip(naive, 0, None),
        f"{tag}_duan": np.clip(duan, 0, None),
        f"{tag}_corr": np.clip(_binned_correction(tr_naive, ytr.to_numpy(), naive), 0, None),
    }


def v_sqrt(Xtr, ytr, Xva, cats, rounds, anchored):
    """sqrt target, squared back and then mean-corrected by the binned map."""
    p = _params(decomposition.BASE_PARAMS)
    z = np.sqrt(np.maximum(ytr.to_numpy(), 0.0))
    m = _train(p, Xtr, pd.Series(z, index=Xtr.index), cats, rounds)
    naive = np.square(m.predict(Xva))
    tr = np.square(m.predict(Xtr))
    return {
        "sqrt_naive": naive,
        "sqrt_corr": np.clip(_binned_correction(tr, ytr.to_numpy(), naive), 0, None),
    }


def v_quantiles(Xtr, ytr, Xva, cats, rounds, anchored):
    """Multi-quantile regression integrated to a mean by the trapezoid rule over the
    quantile levels, with the tails of the level grid handled by the end quantiles."""
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    preds = []
    a_tr = Xtr[ANCHOR].clip(0, CAP) if anchored else None
    a_va = Xva[ANCHOR].clip(0, CAP).to_numpy() if anchored else 0.0
    tr_preds = []
    for q in levels:
        p = _params(decomposition.BASE_PARAMS, objective="quantile", alpha=q, metric="quantile")
        target = (ytr - a_tr) if anchored else ytr
        m = _train(p, Xtr, target, cats, rounds)
        preds.append(a_va + m.predict(Xva))
        tr_preds.append((a_tr.to_numpy() if anchored else 0.0) + m.predict(Xtr))
    Q = np.sort(np.vstack(preds), axis=0)
    Qtr = np.sort(np.vstack(tr_preds), axis=0)

    def integrate(M):
        lv = np.array(levels)
        # trapezoid over the interior plus rectangles for the [0,l0] and [lK,1] ends
        inner = np.trapezoid(M, lv, axis=0)
        return M[0] * lv[0] + inner + M[-1] * (1 - lv[-1])

    mean_va, mean_tr = integrate(Q), integrate(Qtr)
    return {
        "quantile_mix": np.clip(mean_va, 0, None),
        "quantile_mix_corr": np.clip(_binned_correction(mean_tr, ytr.to_numpy(), mean_va), 0, None),
    }


VARIANTS = {
    "baseline": v_baseline,
    "l2_direct": v_l2_direct,
    "huber": _robust("huber", {"alpha": 300.0}, "huber300"),
    "huber900": _robust("huber", {"alpha": 900.0}, "huber900"),
    "fair": _robust("fair", {"fair_c": 300.0}, "fair300"),
    "logratio": v_logratio,
    "sqrt": v_sqrt,
    "quantile": v_quantiles,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline,l2_direct,huber,logratio,sqrt")
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--out", default="objective")
    args = ap.parse_args()
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    folds = [int(f) for f in args.folds.split(",")]
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n}")

    raw, X, cats = matrix.build("train")
    y = raw[config.TARGET].astype("float32")
    y_np = y.to_numpy(dtype="float64")
    capped = y.clip(upper=CAP)
    month = evaluation.movement_month(raw).to_numpy()
    rounds = json.loads((config.DATA_INTERIM / "component_rounds.json").read_text())["rounds"]

    gate_p = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "mechanism_oof_predictions.parquet")[
            [config.ID_COL, "probability"]], on=config.ID_COL, how="left")["probability"].to_numpy()
    base_pred = raw[[config.ID_COL]].merge(
        pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet")[
            [config.ID_COL, "prediction"]], on=config.ID_COL, how="left")["prediction"].to_numpy()
    proxy = X[mechanism.PROXY_COL].to_numpy(dtype="float64")

    matched = X[ANCHOR].notna().to_numpy()
    tail_features = [c for c in decomposition.TAIL_FEATURES if c in X.columns]
    tail_cats = [c for c in cats if c in tail_features]
    tail = (y > CAP).to_numpy()

    out: dict[str, np.ndarray] = {}
    for m in folds:
        t0 = time.time()
        va = (month == m)
        tr = ~va
        # --- shared components: identical across variants -----------------------------
        gate = _train(_params(decomposition.GATE_PARAMS), X.loc[tr, tail_features],
                      pd.Series(tail[tr].astype("int8"), index=X.index[tr]),
                      tail_cats, rounds["gate"])
        sev = _train(_params(decomposition.SEVERITY_PARAMS), X.loc[tr & tail, tail_features],
                     y.loc[tr & tail] - CAP, tail_cats, rounds["severity"])
        prob = gate.predict(X.loc[va, tail_features])
        excess = np.maximum(sev.predict(X.loc[va, tail_features]), 0)
        scale = np.where(X.loc[va, "ADEP_mvt"].astype("string").eq("LIRF"),
                         decomposition.LIRF_TAIL_SCALE, decomposition.OTHER_TAIL_SCALE)

        for name in names:
            fn = VARIANTS[name]
            parts: dict[str, np.ndarray] = {}
            for anchored, subset in ((True, matched), (False, ~matched)):
                trs, vas = tr & subset, va & subset
                if vas.sum() == 0:
                    continue
                key = "base_matched" if anchored else "base_missing"
                res = fn(X.loc[trs], capped.loc[trs], X.loc[vas], cats, rounds[key], anchored)
                for k, v in res.items():
                    k = k.replace("logratio", "log").replace("log1p", "log")
                    parts.setdefault(k, np.full(int(va.sum()), np.nan))
                    parts[k][subset[va]] = v
            for k, base in parts.items():
                full = f"{name}:{k}"
                if full not in out:
                    out[full] = np.full(len(raw), np.nan)
                b = np.clip(np.nan_to_num(base, nan=0.0), 0, CAP)
                ordinary = np.clip(b + scale * prob * excess, 0, None)
                out[full][va] = mechanism.blend(gate_p[va], proxy[va], ordinary)
        print(f"  fold {m:02d} done in {time.time()-t0:.0f}s", flush=True)

    # --- report -----------------------------------------------------------------------
    scored = np.zeros(len(raw), dtype=bool)
    for f in folds:
        scored |= (month == f)
    bands = [("0-10m", 0, 600), ("10-20m", 600, 1200), ("20-30m", 1200, 1800),
             ("30-60m", 1800, 3600), (">60m", 3600, np.inf)]
    rows, band_rows, month_rows = [], [], []
    base_sse = float(((y_np[scored] - base_pred[scored]) ** 2).sum())
    n = int(scored.sum())
    print(f"\nbaseline OOF RMSE {np.sqrt(base_sse/n):.4f} on {n} rows\n")
    for name, pred in sorted(out.items()):
        ok = scored & np.isfinite(pred)
        sse = float(((y_np[ok] - pred[ok]) ** 2).sum())
        bs = float(((y_np[ok] - base_pred[ok]) ** 2).sum())
        bulk, tl = ok & ~tail, ok & tail
        def rm(msk, p):
            return float(np.sqrt(((y_np[msk] - p[msk]) ** 2).mean()))
        improved = sum(
            float(((y_np[(month == f)] - pred[(month == f)]) ** 2).sum())
            < float(((y_np[(month == f)] - base_pred[(month == f)]) ** 2).sum())
            for f in folds)
        rows.append({
            "variant": name, "rmse": np.sqrt(sse / ok.sum()),
            "delta": np.sqrt(sse / ok.sum()) - np.sqrt(bs / ok.sum()),
            "sse_pct": 100 * (sse - bs) / bs,
            "bulk": rm(bulk, pred), "bulk_base": rm(bulk, base_pred),
            "tail": rm(tl, pred), "tail_base": rm(tl, base_pred),
            "jan": rm(ok & (month == 1), pred) if (ok & (month == 1)).any() else np.nan,
            "jan_base": rm(ok & (month == 1), base_pred) if (ok & (month == 1)).any() else np.nan,
            "jul": rm(ok & (month == 7), pred) if (ok & (month == 7)).any() else np.nan,
            "jul_base": rm(ok & (month == 7), base_pred) if (ok & (month == 7)).any() else np.nan,
            "months_better": f"{improved}/{len(folds)}",
        })
        for label, lo, hi in bands:
            b = ok & (y_np >= lo) & (y_np < hi)
            if b.sum() < 50:
                continue
            band_rows.append({"variant": name, "band": label, "n": int(b.sum()),
                              "rmse": rm(b, pred), "base": rm(b, base_pred),
                              "bias": float((pred[b] - y_np[b]).mean()),
                              "bias_base": float((base_pred[b] - y_np[b]).mean())})
        for f in folds:
            v = ok & (month == f)
            month_rows.append({"variant": name, "month": f, "rmse": rm(v, pred),
                               "base": rm(v, base_pred)})
    df = pd.DataFrame(rows).sort_values("rmse")
    print(df.round(3).to_string(index=False))
    print("\n--- by true-taxi band ---")
    print(pd.DataFrame(band_rows).round(2).to_string(index=False))
    print("\n--- by month ---")
    print(pd.DataFrame(month_rows).round(2).to_string(index=False))
    config.DATA_INTERIM.joinpath(f"{args.out}_metrics.json").write_text(json.dumps(
        {"summary": rows, "bands": band_rows, "months": month_rows,
         "baseline_rmse": float(np.sqrt(base_sse / n))}, indent=2, default=float))
    pd.DataFrame({config.ID_COL: raw[config.ID_COL],
                  **{k.replace(":", "__"): v for k, v in out.items()}}).to_parquet(
        config.DATA_INTERIM / f"{args.out}_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
