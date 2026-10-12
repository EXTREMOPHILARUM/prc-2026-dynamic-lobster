"""Are the >12h rows the day-offset regime already detected but shrunk away?"""
import numpy as np
import pandas as pd

from prc import config, dayoffset, evaluation, matrix

pd.set_option("display.width", 200)
raw, X, cats = matrix.build()
y = raw[config.TARGET].astype("float64").to_numpy()
month = evaluation.movement_month(raw).to_numpy()
janjul = np.isin(month, (1, 7))
big = y > 43200.0

# the frozen deployed OOF - the same baseline every other agent is measured against
pred = raw[[config.ID_COL]].merge(
    pd.read_parquet(f"{config.DATA_INTERIM}/sequencing_oof_predictions.parquet"),
    on=config.ID_COL, how="left")["prediction"].to_numpy()
ordinary = pred

bands = dayoffset.band(raw, X)
dlab = dayoffset.day_offset_label(y)
print(f"total rows {len(y):,}   >12h {big.sum()}   >12h in Jan+Jul {int((big & janjul).sum())}")
print(f"day-offset label (y>=79200) {dlab.sum()}   in Jan+Jul {int((dlab & janjul).sum())}")

print("\n--- SSE share, Jan+Jul only ---")
sse = (y - pred) ** 2
tot = sse[janjul].sum()
for nm, m in [(">12h", big & janjul), ("y>=79200", dlab & janjul),
              ("LIRF", (raw['ADEP_mvt'].astype(str) == 'LIRF').to_numpy() & janjul)]:
    print(f"  {nm:10} n={int(m.sum()):5d}  {100*sse[m].sum()/tot:5.1f}% of Jan+Jul SSE")

print("\n--- every >12h row (all 12 months) ---")
sel = np.where(big)[0]
T = lambda c: pd.to_datetime(raw[c], utc=True, errors="coerce")
tab = pd.DataFrame({
    "month": month[sel], "ADEP": raw["ADEP_mvt"].astype(str).to_numpy()[sel],
    "y_h": (y[sel] / 3600).round(2),
    "aobt_missing": raw["AOBT_3_flt"].isna().to_numpy()[sel],
    "mvt_minus_sched_h": (X[dayoffset.PROXY_COL].to_numpy()[sel] / 3600).round(2),
    "band": bands.to_numpy()[sel],
    "pred": pred[sel].round(0),
})
print(tab.sort_values(["month", "y_h"]).to_string(index=False))

print("\n--- band composition of the >12h rows ---")
print(tab.band.value_counts().to_string())
print("\n--- what the day-offset gate would do, rates fitted on all 2025 ---")
rates = dayoffset.fit_rates(bands, dlab)
print("  fitted band rates:", {k: round(v, 5) for k, v in rates.items()})
p_day = dayoffset.probability(bands, rates)
print(f"  P(day offset) on the >12h rows: {np.round(p_day[sel], 5)}")
print(f"  seconds added by dayoffset.apply: {np.round(p_day[sel]*86400, 0)}")
print(f"  seconds NEEDED (y - pred):       {np.round(y[sel]-pred[sel], 0)}")

print("\n--- band population and purity over all 2025 ---")
bt = pd.DataFrame({"band": bands.to_numpy(), "day": dlab, "big": big})
print(bt.groupby("band").agg(n=("day", "size"), day_positives=("day", "sum"),
                             over12h=("big", "sum")).to_string())
