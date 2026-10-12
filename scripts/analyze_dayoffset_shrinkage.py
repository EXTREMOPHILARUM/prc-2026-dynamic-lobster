"""Is the day-offset shrinkage the thing costing us, or is it a severity problem?

Fits one additive constant per gated band on months disjoint from January and July,
then reports what it does to January and July. Never selects on the scored months.
"""
import numpy as np
import pandas as pd

from prc import config, dayoffset, evaluation, matrix

raw, X, cats = matrix.build()
y = raw[config.TARGET].astype("float64").to_numpy()
month = evaluation.movement_month(raw).to_numpy()
pred = raw[[config.ID_COL]].merge(
    pd.read_parquet(f"{config.DATA_INTERIM}/sequencing_oof_predictions.parquet"),
    on=config.ID_COL, how="left")["prediction"].to_numpy()
bands = dayoffset.band(raw, X).to_numpy()
dlab = dayoffset.day_offset_label(y)
janjul = np.isin(month, (1, 7))
resid = y - pred

print("--- residual the deployed model leaves inside each gated band ---")
for b in dayoffset.GATED_BANDS:
    m = bands == b
    print(f"  {b:12} n={m.sum():3d}  resid mean {resid[m].mean():9.0f}  median {np.median(resid[m]):9.0f}"
          f"  min {resid[m].min():8.0f}  max {resid[m].max():8.0f}  sd {resid[m].std():8.0f}")
    out = m & ~janjul
    print(f"               out-of-Jan/Jul n={out.sum():3d}  mean {resid[out].mean():9.0f}")

def sse(p, m):
    return float(((y[m] - p[m]) ** 2).sum())

print("\n--- one additive constant per band, fitted on months 2-6,8-12 ---")
fit = ~janjul
variants = {}
# (a) what the shipped dayoffset does
rates_all = dayoffset.fit_rates(pd.Series(bands), dlab)
variants["shipped (prior 30, all-month rates)"] = {
    b: rates_all.get(b, 0.0) * dayoffset.DAY for b in dayoffset.GATED_BANDS}
# (b) unshrunk empirical rate x 86400, fitted off Jan/Jul
raw_rate = {b: dlab[(bands == b) & fit].mean() for b in dayoffset.GATED_BANDS}
variants["unshrunk rate x 86400"] = {b: raw_rate[b] * dayoffset.DAY for b in dayoffset.GATED_BANDS}
# (c) the L2-optimal constant: mean residual in the band, fitted off Jan/Jul
variants["mean residual (L2-optimal)"] = {
    b: resid[(bands == b) & fit].mean() for b in dayoffset.GATED_BANDS}
# (d) zero - switch the regime off entirely
variants["no day-offset term"] = {b: 0.0 for b in dayoffset.GATED_BANDS}

base_all, base_jan, base_jul = sse(pred, np.ones(len(y), bool)), sse(pred, month == 1), sse(pred, month == 7)
n_all, n_jan, n_jul = len(y), int((month == 1).sum()), int((month == 7).sum())
print(f"  {'variant':36} {'LIRF/12-20h':>12} {'LIRF/20h+':>11} "
      f"{'dRMSE all':>10} {'dRMSE Jan':>10} {'dRMSE Jul':>10}")
for name, add in variants.items():
    p = pred.copy()
    for b, c in add.items():
        p[bands == b] = np.clip(p[bands == b] + c, 0, None)
    d_all = np.sqrt(sse(p, np.ones(len(y), bool))/n_all) - np.sqrt(base_all/n_all)
    d_jan = np.sqrt(sse(p, month == 1)/n_jan) - np.sqrt(base_jan/n_jan)
    d_jul = np.sqrt(sse(p, month == 7)/n_jul) - np.sqrt(base_jul/n_jul)
    print(f"  {name:36} {add['LIRF/12-20h']:12.0f} {add['LIRF/20h+']:11.0f} "
          f"{d_all:+10.3f} {d_jan:+10.3f} {d_jul:+10.3f}")

print("\n--- ceiling: perfect prediction of the 29 rows in the two gated bands ---")
p = pred.copy()
gated = np.isin(bands, dayoffset.GATED_BANDS)
p[gated] = y[gated]
print(f"  dRMSE all {np.sqrt(sse(p, np.ones(len(y),bool))/n_all)-np.sqrt(base_all/n_all):+.3f}"
      f"  Jan {np.sqrt(sse(p, month==1)/n_jan)-np.sqrt(base_jan/n_jan):+.3f}"
      f"  Jul {np.sqrt(sse(p, month==7)/n_jul)-np.sqrt(base_jul/n_jul):+.3f}")
print("\n--- ceiling: perfect prediction of ALL >12h rows, including the 5 undetectable ---")
p = pred.copy(); big = y > 43200
p[big] = y[big]
print(f"  dRMSE all {np.sqrt(sse(p, np.ones(len(y),bool))/n_all)-np.sqrt(base_all/n_all):+.3f}"
      f"  Jan {np.sqrt(sse(p, month==1)/n_jan)-np.sqrt(base_jan/n_jan):+.3f}"
      f"  Jul {np.sqrt(sse(p, month==7)/n_jul)-np.sqrt(base_jul/n_jul):+.3f}")
