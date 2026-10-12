"""Leave-worst-row-out robustness test for the two-stage gate, and OOF artifact export.

The test the LIRF specialist failed: recompute the January+July delta after deleting the
rows the candidate helped most. A gain that lives in one or ten movements is not a gain.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from prc import config, evaluation, matrix, mechanism

COMPONENTS = config.DATA_INTERIM / "seedavg_contaminated" / "oof_r1_seed1.parquet"
GAMMA = 2.0

raw, X, cats = matrix.build()
y = raw[config.TARGET].astype("float64").to_numpy()
label = mechanism.fallback_label(raw).to_numpy()
month = evaluation.movement_month(raw).to_numpy()
janjul = np.isin(month, (1, 7))

comp = raw[[config.ID_COL]].merge(pd.read_parquet(COMPONENTS), on=config.ID_COL, how="left")
ordinary = np.clip(comp["base"].to_numpy() + comp["tail_scale"].to_numpy()
                   * comp["tail_probability"].to_numpy() * comp["tail_excess"].to_numpy(), 0, None)
p_old, proxy = comp["gate_prob"].to_numpy(), comp["proxy"].to_numpy()
p_new = raw[[config.ID_COL]].merge(
    pd.read_parquet(config.DATA_INTERIM / "gate_oof_probability.parquet"),
    on=config.ID_COL, how="left")["probability"].to_numpy()
q = p_new**GAMMA / (p_new**GAMMA + (1 - p_new)**GAMMA)

frozen = raw[[config.ID_COL]].merge(
    pd.read_parquet(config.DATA_INTERIM / "sequencing_oof_predictions.parquet"),
    on=config.ID_COL, how="left")["prediction"].to_numpy()
rebuilt = mechanism.blend(p_old, proxy, ordinary)
cand_g1 = mechanism.blend(p_new, proxy, ordinary)
cand_g2 = mechanism.blend(q, proxy, ordinary)


def loo(base, cand, name, mask=janjul, drops=(0, 1, 10, 50)):
    """Delta after deleting the k rows the candidate helped most."""
    idx = np.where(mask)[0]
    helped = (y[idx] - base[idx])**2 - (y[idx] - cand[idx])**2   # >0 where candidate wins
    order = np.argsort(-helped)
    out = []
    for k in drops:
        keep = idx[order[k:]]
        d = (np.sqrt(((y[keep]-cand[keep])**2).mean()) - np.sqrt(((y[keep]-base[keep])**2).mean()))
        out.append(d)
    print(f"  {name:34} " + "  ".join(f"-{k} rows {d:+7.3f}" for k, d in zip(drops, out)))
    return dict(zip([f"drop_{k}" for k in drops], map(float, out)))


print("=== leave-most-helped-row-out, January+July (2025 OOF) ===")
res = {}
for nm, base, cand in [("gamma=1 vs frozen", frozen, cand_g1),
                       ("gamma=2 vs frozen", frozen, cand_g2),
                       ("gamma=1 vs rebuilt (same ordinary)", rebuilt, cand_g1),
                       ("gamma=2 vs rebuilt (same ordinary)", rebuilt, cand_g2)]:
    res[nm] = loo(base, cand, nm)

print("\n=== concentration: where the January+July gain sits ===")
for nm, cand in [("gamma=1", cand_g1), ("gamma=2", cand_g2)]:
    idx = np.where(janjul)[0]
    helped = (y[idx]-frozen[idx])**2 - (y[idx]-cand[idx])**2
    tot = helped.sum()
    order = np.argsort(-helped)
    print(f"  {nm}: net SSE gain {tot:,.0f}; top-1 row {100*helped[order[0]]/tot:.1f}%, "
          f"top-10 {100*helped[order[:10]].sum()/tot:.1f}%, "
          f"top-100 {100*helped[order[:100]].sum()/tot:.1f}%; "
          f"rows improved {100*(helped>0).mean():.1f}%")

print("\n=== full-year, for completeness ===")
for nm, base, cand in [("gamma=1 vs frozen", frozen, cand_g1), ("gamma=2 vs frozen", frozen, cand_g2)]:
    res[nm + " (all months)"] = loo(base, cand, nm, mask=np.ones(len(y), bool))

# ---- artifacts -------------------------------------------------------------
for stem, pred in [("gate_oof_predictions", cand_g1), ("gate_gamma2_oof_predictions", cand_g2)]:
    pd.DataFrame({config.ID_COL: raw[config.ID_COL], "prediction": pred}).to_parquet(
        config.DATA_INTERIM / f"{stem}.parquet", index=False)
pd.DataFrame({config.ID_COL: raw[config.ID_COL], "probability": p_new}).to_parquet(
    config.DATA_INTERIM / "gate_oof_probability.parquet", index=False)

def seg(base, cand):
    tail, lirf = y > 3600, (raw["ADEP_mvt"].astype(str) == "LIRF").to_numpy()
    o = {}
    for nm, m in [("all", np.ones(len(y), bool)), ("bulk", ~tail), ("tail", tail),
                  ("LIRF", lirf), ("non_LIRF", ~lirf), ("fallback", label),
                  ("non_fallback", ~label), ("jan", month == 1), ("jul", month == 7)]:
        a = float(((y[m]-base[m])**2).sum()); b = float(((y[m]-cand[m])**2).sum())
        o[nm] = {"n": int(m.sum()), "rmse_old": float(np.sqrt(a/m.sum())),
                 "rmse_new": float(np.sqrt(b/m.sum())), "sse_delta_pct": 100*(b-a)/a}
    return o

metrics = {
    "candidate": "two-stage schedule-fallback gate",
    "baseline": "sequencing_oof_predictions.parquet (285.6514954823304)",
    "gamma": GAMMA,
    "gamma_selected_on_months": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
    "components_source": str(COMPONENTS),
    "components_caveat": (
        "seedavg quarantined this file as 'contaminated' after it was produced. Both arms "
        "of every paired comparison use the SAME ordinary from it, so component error "
        "largely cancels in the delta; the gate_prob column is the baseline arm and does "
        "not cancel. Treat absolute RMSE as provisional."),
    "gate": {
        nm: {"auc": float(roc_auc_score(label, p)), "ap": float(average_precision_score(label, p)),
             "logloss": float(log_loss(label, p)), "brier": float(brier_score_loss(label, p)),
             "mass": float(p.sum())}
        for nm, p in [("current", p_old), ("two_stage", p_new)]},
    "label_positives": int(label.sum()),
    "rmse": {"frozen": float(np.sqrt(((y-frozen)**2).mean())),
             "rebuilt": float(np.sqrt(((y-rebuilt)**2).mean())),
             "gamma1": float(np.sqrt(((y-cand_g1)**2).mean())),
             "gamma2": float(np.sqrt(((y-cand_g2)**2).mean()))},
    "segments_gamma1_vs_frozen": seg(frozen, cand_g1),
    "segments_gamma2_vs_frozen": seg(frozen, cand_g2),
    "leave_most_helped_out_janjul": res,
}
(config.DATA_INTERIM / "gate_oof_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print("\nwrote gate_oof_predictions.parquet, gate_gamma2_oof_predictions.parquet, "
      "gate_oof_probability.parquet, gate_oof_metrics.json")
