# Experiment log

Everything tried, with the measurement that settled it. Negative results are kept because
they are the expensive part of the knowledge here: four confident ideas died on contact
with the residual test, and three tail estimates were over-optimistic by 10-200x.

Status at time of writing: best submitted **344.21** (v10); **v12** built and awaiting the
daily quota. Prize cutoff 247.01, leader 245.02, field median 321.18 (leaderboard,
2026-09-15).

---

## 1. Submissions

| v | model | local OOF | scored | note |
| --- | --- | ---: | ---: | --- |
| 1 | LightGBM, time + crude congestion | 605.8 | 602.06 | holdout was filtered and read 346 |
| 2 | + NM off-block proxy features | 458.2 | **411.55** | biggest single gain, -190.5s |
| 3 | two-stage tail/bulk mixture | 406.5 | 438.42 | better holdout, worse score |
| 4-5 | decomposition, all-month refit | 388.9 | 403.05 | v5 a duplicate infrastructure check |
| 6 | schedule-fallback mechanism | 290.6 | — | rejected: daily limit; version spent |
| 7 | identical file, new version number | 290.6 | **349.22** | -53.8s |
| 8 | + day-offset regime gate | 287.0 | 348.97 | -0.25s |
| 9 | + METAR weather (10 features) | 287.2 | **345.39** | -3.58s, first external data |
| 10 | + crosswind, trends, accumulation | 286.9 | **344.21** | -1.17s |
| 11 | + sequencing + tail capacity | 285.7 | — | rejected: UTC/local date confusion |
| 12 | + sequencing + tail capacity | 285.7 | **342.86** | -1.36s, uploaded 2026-09-16 |
| 13 | + arrival proximity (4 fixed-window feats) | 285.0 | **342.22** | -0.64s, predicted -0.56 |

**v12 and v13 both transferred above prediction** - 109% and 113%:

| v | predicted (Jan+Jul OOF) | delivered (scored) | transfer |
| --- | ---: | ---: | ---: |
| 12 | -1.25 | **-1.36** | 109% |
| 13 | -0.56 | **-0.64** | 113% |

Both were bulk features that passed the leave-one-row-out gate in section 8e. Every change
that *over*-claimed this project's history was tail-concentrated. The robustness gate is not
merely filtering noise - it is selecting for what transfers. Consistent with the pattern established 2026-09-16, that
bulk-type gains transfer nearly intact while tail gains evaporate (section 2.3: 54s
predicted, 0.25s delivered).

**Correction to the note below**: the bucket shows **v6 Succeeded at 349.2152**, identical
to v7. Only *one* version number (v11) was actually lost to a quota rejection, not two - v7
was a duplicate upload of an already-scored file.

Two version numbers (v6, v11) were burned on quota rejections. A rejected version cannot
be re-scored by re-uploading the same filename; it must go up under a new number.
`prc.submit` now queries the bucket for the current **UTC** day and refuses to upload past
the limit.

---

## 2. Mechanisms discovered in the data

This is where almost all the score came from. **96.6% of total improvement came from
understanding the data; 3.4% from modelling choices.**

### 2.1 NM off-block times survive in the ranking set (-190.5s)

`BLOCK_TIME_UTC_mvt` is blanked for departures, but the Network Manager's own
`AOBT_3_flt` is present at 98.5%. `MVT_TIME - AOBT_3_flt` is a direct noisy reading of the
target: corr 0.54, median absolute error 175s, standalone RMSE 384.9.

### 2.2 Schedule fallback (-53.8s)

When an airport reports no actual off-block, the block time appears to be copied from the
*scheduled* time, so the recorded "taxi" is really the flight's lateness. 199,634 rows
(9.6%). On those rows `MVT - SCHED` reconstructs the target at **RMSE 36.6s, 100% within
a minute** - the problem is detecting the regime, not estimating the value. A calibrated
gate (AUC 0.873, mass within 0.3%) blended softly gives the gain.

Independently reported on the challenge Discord by two other teams, so not unique to us.

**The mechanism account above is WRONG (measured 2026-09-16). The gain is real; the
explanation is not.** This is not a reporting fallback:

- **198,199 of the 199,634 labelled rows HAVE NM's `AOBT_3_flt`.** "The airport reports no
  actual off-block" describes 1,435 rows, not 199,634.
- At 1-second resolution `BLOCK - SCHED` has spikes of near-equal height at **every**
  multiple of 60s (0, +/-60, +/-120 ... +/-300), each ~12s wide - minute-resolution
  timestamps, not copies. At 0s tolerance the rate is 0.39%; d=0 holds 8,123 rows against
  5,528 at d=+60 and 4,665 at d=+300. That is a ~50% mode, not a copy spike.
- The event is **on-time pushback**: `d = (AOBT - SCHED) + (BLOCK - AOBT)`, the second term
  with robust sigma ~226s against a +/-60s window. P(label) peaks at 0.27 even when
  `|AOBT - SCHED| < 10s`, so most of the event is irreducible.
- An unregularised lookup on (airport, 20s bin of `AOBT - SCHED`) reaches AUC 0.8355,
  ~96% of the deployed gate's discrimination from two variables.

It works because `MVT - SCHED` reconstructs the target exactly whenever pushback happened to
be near schedule - which is common - not because a clock was copied. The -53.8s stands.

**Correction (2026-09-16): the day-offset gate is not in the production model.** v9 dropped
it "to isolate the weather effect" and estimated recombining at ~0.25s; it was never
restored, so v10, v11 and v12 all ship without it. `models/mechanism_wx.json` has
`kind: mechanism`, not `mechanism_day`. This is an unintentional omission, not a decision.

### 2.3 Day offset, ~24h (+0.25s)

16 rows in 2025 where the block time refers to the previous day, carrying ~19% of total
variance. Signature: LIRF + AOBT missing + `MVT-SCHED` over 12h (41% positive), over 20h
(5 of 5). Gated with heavy shrinkage (prior 30) because 16 positives cannot support a
confident rate. Expected ~54s, delivered 0.25s.

**The stated cause of that miss is wrong.** "The regime is far rarer in 2026 than 2025" is
falsified by counting the at-risk band directly in `ranking.parquet` - both of its inputs
(`AOBT_3_flt` nullity and `MVT - SCHED`) survive there, so this is a label-free read of the
ranking *covariate* distribution and is explicitly permitted.

| | 2025 Jan+Jul | 2026 ranking |
| --- | ---: | ---: |
| DEP rows | 344,419 | 344,841 |
| band (LIRF & AOBT missing & gap >= 12h) | **11** | **11** |
| per 100k departures | **3.19** | **3.19** |
| 12-20h / 20h+ | 8 / 3 | 9 / 2 |
| AOBT-missing cohort | 1.560% | 1.534% |

**Prevalence is identical to three significant figures.** Whatever explains the 0.25s, it is
not that the regime is rarer.

### 2.4 Unexplained long rows (NEGATIVE)

2,362 rows long but neither a schedule copy nor a day offset, 7.8% of variance. Airport
reports over an hour where NM sees an ordinary taxi; block sits a median 47 min before
AOBT. Gate works (AUC 0.923, calibrated mass) but the mixture costs **+23.3% SSE, 0/12
months**. The severity model is fitted on ~2,100 rows and extrapolates catastrophically
when applied to a whole held-out month. Sample size, not structure - the fallback regime
works because it has 199,634 examples.

---

## 3. Features: the residual test

The decisive filter. Correlation with the *target* is meaningless if the model already
extracts that signal; what matters is correlation with the **residual the current model
leaves**. Measured within airport, to avoid between-airport variance inflating everything.

| feature | vs residual | vs target | verdict |
| --- | ---: | ---: | --- |
| `seq_heavy_ahead_15m` (wake mix ahead) | **+0.048** | +0.149 | **kept** |
| weather gust | +0.043 | +0.057 | **kept** |
| weather visibility | -0.042 | -0.078 | **kept** |
| `seq_heavy_share_30m` | +0.023 | +0.129 | kept |
| neighbour departure taxi proxy (30m median) | +0.020 | +0.236 | weak, not built |
| arrival taxi-in aggregates | +0.018 | +0.049 | weak, not built |
| `seq_departures_30m_runway` | +0.017 | +0.080 | kept |
| `arr_land_prev_15m` (arrivals landed, past 15m) | +0.012 | +0.015 | rejected |
| `arr_land_next_20m_samerwy` (same-runway arrivals due) | +0.011 | +0.008 | rejected |
| `arr_land_next_10m` (arrivals due to land) | +0.004 | +0.007 | rejected |
| `arr_land_next_20m_heavy` | +0.003 | -0.018 | rejected |
| runway takeoffs last 30m | +0.003 | +0.101 | rejected |
| NASA-style surface queue counts | **-0.003** | +0.059 | rejected |
| turnaround gap (stand occupancy) | +0.002 | -0.003 | rejected |
| `seq_prev_gap_s` | +0.003 | +0.002 | dropped |
| stand x runway reference encoding | ~0 | (39.5% of variance) | rejected |

> ## ⚠ THE RULE IN THIS SECTION IS UNSAFE — REFUTED 2026-09-16
>
> The residual-correlation threshold used to reject the features below **discards real
> signal**, confirmed three times independently on this project's own acceptance standard
> (paired 12-fold, whole-month holdout, unfiltered):
>
> | evidence | screen said | folds said |
> | --- | --- | --- |
> | the three originally-rejected arrival features | +0.004 to +0.012, rejected | **-0.111% SSE, 11/12 months** |
> | `neighbour` + arrival taxi-in | +0.020 / +0.018, "weak, not built" | **-0.530 Jan+Jul, 12/12 months** |
> | airport configuration state | **all \|r\| <= 0.009**, far below any threshold | **-0.295% SSE, 11/12 months** |
>
> The configuration-state case is decisive: its *best* feature screened at -0.0089, less
> than half the weakest thing ever rejected here, and it still delivered. A Pearson
> correlation is a linear statistic and the model is a gradient-boosted tree that exploits
> interactions; the screen cannot see what the model can use.
>
> **Use the screen to prioritise, never to reject.** Anything with a plausible mechanism
> deserves a 2-3 fold smoke test before it is discarded. The features listed below as
> "rejected" or "weak, not built" should be treated as *untested*, not as closed.

**The pattern:** everything describing *how busy it is* is redundant, because `MVT - AOBT`
already encodes congestion directly. Everything describing *conditions the clock cannot
see* - weather, wake separation - carries signal.

This also explains why published work disagrees with us. The SHAP study
(pmc.ncbi.nlm.nih.gov/articles/PMC13022428) ranks arrival and departure queues as 36% of
model influence; their model has no equivalent of an actual off-block time and must infer
congestion. We get it directly.

**Arrival intersections (NEGATIVE, 2026-09-16).** Item 2 of the untested list: arrivals
due to land in the next 10-20 min force departures to hold at runway crossings. Arrival
rows survive in the ranking set completely intact - landing time, runway and even
`BLOCK_TIME` are present at 100%, because only *departures* are blanked - so the feature
would have been free to score. Six variants built over all 2,082,750 arrival movements
and tested against the v11 residual.

The forward-looking variants, which are the actual hypothesis, are the weakest of the six
(+0.004 for the next 10 min, +0.002 for the next 20). What little signal exists is
*backward*-looking (`arr_land_prev_15m`, +0.012) - which is to say it is congestion again,
and `MVT - AOBT` already carries it. Same-runway targeting helps marginally (+0.011) but
stays below the +0.018 to +0.020 band at which `arrival taxi-in aggregates` and
`neighbour departure taxi proxy` were already judged too weak to build. Not taken to a
12-fold confirm: the screen is below the bar of things this project has already rejected.

This is the sixth feature killed by the residual test, and for the fifth time the reason
is that the feature describes how busy the airport is.

**Units, again.** The first run of this screen reported window counts averaging 4,118
arrivals per 10 minutes. Raw timestamps are `datetime64[us]`, so `.astype('int64') // 10**9`
yields milli-scale numbers and every window spanned about 19 years. Identical to the bug in
section 5.5. Screening scripts should use `.dt.total_seconds()` like production does.

---

## 4. Modelling experiments

**False positives have negative cost - the number behind the disjoint-component failure.**
Measured on the 12-fold OOF against an oracle gate firing exactly on the label:

| | RMSE |
| --- | ---: |
| ordinary only, no gate | 308.97 |
| current gate | 285.53 |
| perfect gate (oracle) | 274.70 |

Of the 1.266e10 gate-error SSE gap, **false negatives are +138.1% and false positives are
-38.1%**. Removing FN cost alone gives 270.45 - *better than a perfect gate*. Removing FP
cost alone gives 289.56 - *worse than doing nothing*.

Blending the proxy into rows where `|BLOCK - SCHED| > 60s` beats a gate that refuses to. The
soft mixture is not a hedge against gate error; it extracts real signal from rows the label
calls negative, because `MVT - SCHED` is informative well beyond the +/-60s window. Making
the components disjoint removed exactly this beneficial over-blending, which is why that
"fix" cost 14 seconds. **On this gate, chase recall; never chase precision.**

| experiment | result |
| --- | --- |
| Two-stage tail/bulk (v3) | holdout -52s, **score +27s**. Bulk trained on `y<=3600` only, biasing it low. |
| Disjoint-component fix | "Fixed" the double-count and got **worse** (304.88 vs 290.61, 6/12 months). **Explained 2026-09-16, see below.** |
| Stand x runway reference encoding | +4.09 full / -0.09 bulk. Top feature by gain (24.2%), converges in 286 rounds vs 384 - same information, better packaged, no new signal. |
| Global tail-probability shrinkage | Refuted. Leave-one-month-out says the classifier is well calibrated (ratios near 1.0, fitted shrinkage 1.2356 i.e. mild *under*-prediction) and RMSE is flat in the factor. |
| Bulk hyperparameters | **Saturated.** Five configs within 0.7s (lr .02-.05, 255-1023 leaves). Not under-fit. |
| Tail severity capacity | 8.1s on a single Feb+Oct split, **0.43s across 12 folds** - overstated 19x. Kept (July -7.6s) but January got worse. |
| `other_tail_scale` = 0.25 | Re-tested and near-optimal (0.0 -> 290.42, 0.25 -> 289.69, 0.5 -> 290.82). The one parameter I suspected was fine. |
| Seed averaging | **1.0s on bulk**, saturating by 3 seeds (221.40 -> 220.67 -> 220.40). Worth taking at production time. |
| XGBoost on CUDA | **3x slower than the laptop CPU** (299s vs 109s per fit) and 15s worse standalone. Abandoned at 5/12 folds. |

---

## 5. Validation: what we got wrong and fixed

1. **Filtered holdout.** Excluding taxi > 2h made the holdout read 346 when the true score
   was 602. Those 0.2% of rows carry ~50% of squared error. Validation is now unfiltered.
2. **Round-selection leak.** Component rounds were early-stopped on January+July, then
   reused by every OOF fold *including* folds 1 and 7 - the two scored months. Rounds are
   now selected on an inner holdout disjoint from the ranking months (April/October).
3. **Single splits mislead.** Tail capacity looked worth 8.1s on one split and 0.43s across
   twelve. Screening on one split is fine; submitting on one split is not.
4. **Variance explained is not headroom.** Stand x runway explains 39.5% of taxi variance
   and adds nothing, because the model already had it.
5. **Units.** An `astype('int64')` on microsecond timestamps made every time 1000x too
   small and nearly buried a real feature under a false negative. Production code uses
   `.dt.total_seconds()` throughout; only throwaway analysis was affected.

**Instrument quality now:** v9 predicted 3.43s and delivered 3.58s; v10 predicted 0.30s
and delivered 1.17s. Weather transfers *better* than the holdout shows, because the scored
January 2026 has 5,480 snow departures against 1,170 in January 2025.

---

## 6. External data

| source | licence | status |
| --- | --- | --- |
| Iowa State IEM METAR | **public domain** | **in use** (v9, v10) |
| OurAirports runway geometry | **public domain** | **in use** (crosswind, v10) |
| OpenStreetMap | ODbL (share-alike) | permitted, untested; attribution + derived-database care needed |
| OPDI flight events | open, DOI | permitted but "milestones do not cover off-block" (organiser) |
| OpenSky state vectors / Trino | — | **forbidden** (organiser ruling, 2026-09-03) |
| Meteostat | partly CC-BY-NC | avoid - non-commercial fails eligibility |

Governing ruling, `espinielli`, 2026-09-08: *"Open source data are allowed if declared in
the documentation for the solution/repo."*

---

## 7. Infrastructure built

- **Submission rehearsal** (`scripts/rehearse_submission.py`) - rebuilds a stand-in ranking
  set from 2025 Jan+Jul blanked exactly as the organisers blank theirs, runs the genuine
  scoring path, and scores against held-back truth. Includes a **leakage probe**: the same
  rows scored with `BLOCK_TIME` present and absent must give identical predictions. They do.
- **Tests 4 -> 13**, each encoding a failure that actually happened.
- **Feature matrix cache** - 7.7s cold, 0.2s warm, fingerprinted on feature source and raw
  parquet so a stale cache cannot be reused.
- **Daily quota guard** - queries the bucket for the current UTC day before uploading.
- **the GPU host offload** - repo, data, models and venv on a 16-core box; useful for not
  blocking the laptop, not useful for GPU acceleration.

---

## 8. Untested, in priority order

1. **Airport configuration state** - active runway mix in a rolling window, and changes to
   it. Derivable from provided data. Nobody on the leaderboard mentions it.
2. ~~**Impending arrival landings**~~ - screened 2026-09-16 and rejected, see section 3.
3. **Cloud ceiling and operational LVP thresholds** - `skyc1` is parsed but ceiling height
   is not; our low-visibility flag uses <1 mile rather than the operational <3 miles/<1000ft.
4. **Thunderstorm and shower codes** - measured +126.2s residual bias on 6,137 rows
   (+132.1s in July, a scored month). Small exposure (~0.07s) but free.
5. **Propulsion type** - turboprop vs jet, derivable from aircraft type.
6. **Seed averaging at production time** - measured 1.0s on bulk.
7. **OSM taxi distance** - permitted; but taxi distance was only 6.2% of influence in the
   published study and our stand x runway encoding already failed, so expect redundancy.
8. **NOTAMs** - permitted, unexplored, large build.

## 8a. The at-risk cohort

Departures with no NM flight record - `LOBT_flt` null, which is **exactly** the same
22,470 rows as `AOBT_3_flt` null (zero rows differ either way, so this is not a new
feature) - are 1.078% of training and carry:

- **39.0% of all-year squared error**
- **56.3% of January+July squared error**
- **33 of the 34 rows over 12 hours**, a ~3,000x enrichment
- a >1h rate of 4.84%, against 0.147% outside the cohort

**Correction: there is no enrichment in the ranking set.** This section first reported the
cohort as 1.423x more prevalent in 2026. That was a seasonal artifact - it compared an
**all-year** training baseline (1.078%) against a ranking set that is **January and July
only**. The cohort is strongly seasonal, running 0.62% in April and 2.07% in July:

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| % | 0.93 | 0.77 | 0.66 | 0.62 | 0.87 | 1.37 | **2.07** | 1.22 | 1.34 | 1.03 | 0.84 | 1.00 |

Like for like, 2025 Jan+Jul is **1.560%** against the ranking set's **1.534%** - a ratio of
**0.983**, i.e. no change. This is the third time in one session that comparing a 12-month
statistic to a Jan+Jul one has inverted a conclusion; always match the population.

**The real asymmetry is between the two scored months.** January 2026 carries 1.72x the
cohort of January 2025, while July carries 0.72x:

| | 2025 | 2026 | ratio |
| --- | ---: | ---: | ---: |
| January | 0.934% | 1.606% | **1.72x** |
| July | 2.065% | 1.477% | **0.72x** |

So expect a relatively heavier January and lighter July than a 2025-based rehearsal shows.
Per airport, on a matched Jan+Jul basis:

| airport | 2025 Jan+Jul % | 2026 ranking % | ratio |
| --- | ---: | ---: | ---: |
| EHAM | 1.97 | 3.16 | 1.60x |
| LSZH | 2.34 | 2.85 | 1.21x |
| LFPG | 2.22 | 1.84 | 0.83x |
| LTFM | 1.41 | 1.52 | 1.07x |
| LIRF | 1.50 | 1.42 | 0.95x |
| EDDF | 1.63 | 1.16 | 0.71x |
| EGLL | 1.16 | 0.62 | 0.54x |

EHAM at 1.60x is the one genuine mover.

**Extending the gate beyond LIRF: NEGATIVE.** The `other/6-12h`, `other/12-20h` and
`other/20h+` bands contain **zero** positives across twelve months, so gating them changes
the result by 0.00s. Adding `<6h` bands to reach the LFPG rows is *worse* (-9.73 against
-10.25 on Jan+Jul). The gate stays LIRF-only. The asymmetry this section originally
flagged is real in the covariates but carries no label signal.

---

## 8b. Where the error actually is

Measured on the v11/v12 OOF (RMSE 285.65, 2,085,047 rows, 2026-09-16).

By true taxi time:

| band | n | % rows | % of SSE | RMSE | bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0-10 min | 257,784 | 12.4 | 5.7 | 194 | +117 |
| 10-20 min | 1,337,468 | 64.2 | 27.9 | 188 | +40 |
| 20-30 min | 414,800 | 19.9 | 16.0 | 256 | -86 |
| 30-60 min | 70,494 | 3.4 | 14.3 | 586 | -374 |
| 1-2 h | 3,548 | 0.17 | 8.4 | 2,006 | -1,632 |
| 2-12 h | 550 | 0.03 | 5.3 | 4,044 | -2,366 |
| **> 12 h** | **34** | **0.002** | **22.4** | **33,484** | **-21,111** |

Concentration: the worst **208 rows (0.01%) carry 32% of all squared error**; 0.2% carry
48%; 1% carry 59%.

By airport, **LIRF alone is 37.1% of total SSE** on 7.7% of rows (RMSE 627, against 165-247
for EHAM/EDDM/EDDF). Splitting LIRF by the regime flag:

| LIRF subset | n | % of *total* SSE | RMSE |
| --- | ---: | ---: | ---: |
| AOBT missing | 1,488 | **24.2** | 5,258 |
| AOBT present | 159,216 | 12.9 | 371 |

So 1,488 rows - 0.07% of the training set - hold a quarter of the entire error, and 30 LIRF
rows with taxi over 12 h hold 12% of it on their own. 88% of the worst 208 rows are
AOBT-missing, against a 1.1% base rate, and they cluster in June-August (118 of 208).

**What this says about where to work.** The error is not in the bulk; the bulk is
saturated and hyperparameter work confirmed it. It is in the missing-AOBT regime at one
airport, and it is the *magnitude* that is wrong, not the detection - the same wall
section 2.4 hit. Three of the four gates built so far detect their regime well (AUC 0.87
to 0.92) and lose on severity. Adding features that describe congestion cannot touch any
of this, which is consistent with six consecutive negative residual screens.

**Correction (2026-09-16).** The bias column above was originally read as evidence that the
model hedges badly under an uncertain mixture, and that reading was wrong. The table
conditions on **true** `y`, and a monotone bias ladder is *mathematically forced* on any
conditional mean under that conditioning - it is regression to the mean, not a defect.
Conditioning on the **prediction** instead, which is the only thing available at scoring
time, the model is well calibrated: bias runs -2 s to -21 s across prediction deciles, an
order of magnitude smaller. See section 8d. The bias column is retained because it
describes the error's shape, but it is not a lead.

---

## 8d. Recalibration (NEGATIVE)

Tested 2026-09-16, prompted by the bias ladder in 8b. **Definitively closed; do not
re-test.**

Is `E[y | yhat] != yhat`? Essentially no. Across prediction deciles the bias is -2.0 s
(decile 0) to -20.8 s (decile 9), monotone but an order of magnitude below the ladder in
8b, which conditions on true `y` and therefore forces that shape by attenuation.

Leave-one-month-out isotonic recalibration applied where `yhat < 1800` - 97.6% of rows,
the entire 10-30 minute band that section 8c redirected effort toward:

- overall 285.6515 -> 285.5623 (**-0.089**), SSE -0.054%, 8/12 months
- on the `yhat < 1800` subset alone: 228.8957 -> 228.7817 (-0.114)

**The 10-30 minute band is calibrated to within a tenth of a second of RMSE.** The
hypothesis that its 44% of squared error is recoverable by recalibration is refuted.

Unrestricted isotonic gives -1.17 s, but decomposing by prediction region shows the gain
comes entirely from **928 rows with `yhat > 5000`** (0.045% of rows, 17.4% of SSE, 918 of
them LIRF). That is the tail-severity problem of sections 2.4 / 4 / 8c wearing a different
hat, not a calibration fix. Shrinking those predictions with a weight fitted on months
disjoint from Jan/Jul gives -1.87 s and 11/12 months, but **dropping the single worst row
from each month flips the sign in 5 of 12 months**. It is carried by a few hundred LIRF
rows - the same sample-size wall that killed the long-rows gate. Per-airport isotonic
(-2.08) and an AOBT-missing split (+3.30, worse) add nothing reliable; global linear
recalibration is -0.05.

If ever revisited, it should be framed as capping the mechanism component's severity on
LIRF and judged against the 8c ceiling, not as a calibration fix. Given the 1-20% delivery
rate of past tail estimates, expect well under 1 s scored.

---

## 8a-bis. Pattern recognition on the broken-clock rows (2026-09-16)

**Detection is finished; severity is where the value is.**

The signature is **"entire NM flight record absent"** - zero of the six NM timestamps
present. That is 22,470 rows (1.078%) containing **33 of the 34** rows over 12h. Narrowed to
`LIRF & NM-absent & MVT-SCHED >= 12h` it is **29 rows, 100% pure, zero false positives in
2.08M**. No further detection work is warranted.

**The five that escape have no signature.** Two LFPG and one LSZH with `MVT - SCHED` of
0.26-0.57h, plus one LIRF at 3.25h and one with AOBT present. Tested and negative:

- **Runway closure: no.** The 34 rows fall on **30 distinct airport-days** with ordinary
  runway mixes. LIRF runway 16R has a high rate (6.5% of its cohort rows) but every such row
  is already inside the band, so it adds nothing.
- **NM-feed degradation: no.** The rate of NM-absent rows in a +/-3h airport window is flat
  across quintiles within the cohort (1555 / 1557 / 1783 / 667 / 1783 per million), and the
  gated 29 sit *below* the cohort average (2.94% vs 4.39%).
- **Flight number** shows a 1,000x enrichment (16,360 vs 16 per million) and four numbers
  recur among the 34, but it was **not taken to an out-of-fold test** and remains untested,
  not confirmed.

**What is new: the band holds two mechanisms, and the shipped gate models neither.**

| regime | n | value |
| --- | ---: | --- |
| schedule copy | 18 of 29 | `y == MVT - SCHED` exactly |
| day offset | 11 of 29 | `y == 86400 + an ordinary taxi` (mean extra 1,136s) |

Inside the band `y` correlates **+0.827 with `MVT - SCHED`**. The shipped gate adds
`P x 86400 ~= 38,400s` when the mean shortfall is **13,924s** - it **overshoots by roughly
3x**, because it assumes a day offset for rows that are mostly schedule copies.

Severity fitted leave-one-month-out, applied to the combined model:

| severity model | 12-month | Jan+Jul | -1 row | -5 rows |
| --- | ---: | ---: | ---: | ---: |
| linear in `MVT - SCHED` | **-7.395** | **-26.636** | -12.847 | +0.823 |
| explicit two-regime mixture | -6.139 | -25.157 | -11.551 | +3.789 |
| additive shift | -4.341 | -14.334 | - | - |
| `P x 86400` as shipped | +6.147 | -1.325 | - | - |

9 of 12 months improve and 7 of 11 scored rows improve, but **one row is 50.9% of the scored
gain** and the effect reverses on dropping five. With 29 training rows and 11 scored rows any
estimate is unstable; four previous tail estimates delivered 1-20% of their promise, so
expect ~1-3s scored rather than 26. **The overshoot is worth fixing on correctness grounds
regardless of what the severity model turns out to be worth.**

---

## 8b-bis. COMPLIANCE: do not invert submitted scores

A technique was developed on 2026-09-16 that **breaches this project's rules** and is
recorded here so nobody re-derives it innocently. The v7 -> v8 score delta isolates a gate
touching exactly 11 known rows with known increments, so one scalar identifies the total
under-prediction on those rows and yields a direct estimate of 2026 severity.

`docs/rules-and-compliance.md` prohibits "uploading controlled variants to infer hidden
labels, segment errors, **tail prevalence**, or any other property of the ranking targets",
"treating a leaderboard result as an additional validation observation", and protocol step
9's "do not use it to modify the same hypothesis or tune a related variant". The day-offset
gate is literally the hypothesis v8 tested.

**No parameter, threshold or priority may be set from it.** Counting the at-risk band in
`ranking.parquet` is fine and stays (section 2.3) - that reads covariates, not targets.
Inferring severity from a score does not.

---

## 8c. The ceiling: how much is left in the tail at all

Measured 2026-09-16 on the v11/v12 OOF (285.65).

**The target is an exact timestamp difference.** `TAXITIME_SEC_mvt == MVT_TIME - BLOCK_TIME`
on 100.0000% of training rows, max deviation 0.000s. `MVT_TIME` survives in the ranking
set, so the problem is not duration regression - it is reconstructing the blanked
`BLOCK_TIME`. That matters because a broken clock that was *copied* from a surviving
timestamp can be recovered exactly, instead of fitted by a severity model on 2,000 rows -
the wall that killed sections 2.4, the long-rows gate, and the tail-capacity work.

**But the copies are not there.** A first pass reported that 31.1% of rows have
`BLOCK_TIME` within 60s of another timestamp. That was an artifact: the `BLOCK - AOBT_3`
delta is smooth and wide (quartiles -122s / +183s, median +51s), so a +/-60s window
catches a fifth of the distribution by construction. At **0s tolerance the copy rate is
1.18%**. It is measurement disagreement, not a copy.

| segment | n | exact copy from any of 5 sources |
| --- | ---: | ---: |
| all | 2,085,047 | 1.18% |
| taxi 30-60 min | 70,041 | 1.11% |
| taxi 1-12 h | 4,092 | 3.54% (SCHED alone 3.52%) |
| taxi > 12 h | 34 | 2.94% |

The only concentration is SCHED in the 1-12h band, which *is* the schedule-fallback
mechanism already shipped in v7. There is no second mechanism of the same kind underneath
it. Sources tested: `AOBT_3_flt`, `EOBT_1_flt`, `IOBT_flt`, `LOBT_flt`,
`SCHED_TIME_UTC_mvt`, `ARVT_1_flt`, `ARVT_3_flt`.

**Oracle bounds.** CORRECTED 2026-09-16 after an adversarial audit. The original version
of this section computed every bound on the **all-12-month** OOF (285.65) and compared the
result against the **scored** gap. That is a population error and it inverted the
conclusion. Local OOF over 12 months is 285.65; the same model over **January and July
only is 369.46**, and the scored set is 344.21. Jan+Jul is the population that gets scored;
the 12-month average is not.

Tail share of squared error is not stable across that split - it is 36.1% over 12 months
but **49.4% on Jan+Jul**, and 64.7% in January alone.

| population | n | RMSE | tail % SSE | perfect tail -> | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| all 12 months | 2,085,047 | 285.65 | 36.1% | 228.38 | -57.3 |
| January | 153,706 | 369.57 | **64.7%** | 219.69 | -149.9 |
| July | 190,713 | 369.38 | 37.1% | 292.89 | -76.5 |
| **Jan+Jul (scored population)** | 344,419 | **369.46** | **49.4%** | **262.75** | **-106.7** |
| other 10 months | 1,740,628 | 265.96 | 31.0% | 220.95 | -45.0 |

Scaling by the observed local->scored factor (344.21 / 369.46 = 0.932), perfect tail
prediction implies **~244.8 scored - below the 247.01 prize cutoff**. The tail, fully
solved, *does* close the gap. The original conclusion was the exact opposite and
understated the scored-population ceiling by about 1.9x.

Bounds that survive unchanged: perfect classification of every exact copy -> 284.27
(-1.4s, still worthless); cheating best-of-timestamps on tail rows -> 272.39, against
271.56 as originally logged, an immaterial definitional difference.

**Where the headroom is, on the population that actually gets scored (Jan+Jul):**

| band | n | % rows | % SSE | perfect -> | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| **> 12 h** | **14** | 0.004% | **38.8%** | 289.13 | **-80.3** |
| 1-12 h | 968 | 0.28% | 10.7% | ~354 | -15 |
| 30-60 min | 14,733 | 4.3% | 12.1% | 346.37 | -23.1 |
| 10-30 min | 285,832 | 83.0% | 34.4% | 299.25 | -70.2 |

**Fourteen rows carry 38.8% of the scored-month error, and fixing only those is worth more
(-80.3) than perfectly solving the entire 10-30 minute band (-70.2).** LIRF is 40.4% of
Jan+Jul SSE, up from 37.1% all-year.

**On the copy hypothesis, the audit refined it and the answer got worse, not better.** The
1.18% exact-copy rate is an *over*count. `BLOCK_TIME` carries second resolution (6.4% land
on `:00`) while the NM sources are minute-rounded (`AOBT_3` 96.7% on `:00`, `SCHED` 100%),
so exact matches are structurally confined to the 133,377 rows where `BLOCK` seconds are
zero. The right test is spike-versus-background, and **there is no spike**: per-airport
1-second histograms of `BLOCK - AOBT_3` run smooth through zero (LIRF 1,317 at delta=0
against neighbours 1,374 / 1,348 / 1,329 / 1,314). The genuine copy excess is **~0.15% per
source**. Three further candidate sources were tested and all failed: same-`FLIGHT_ID`
arrival on-block (0 matches), same-`FLIGHT_ID` arrival landing (0 matches), prior arrival
at the same ADEP+STAND (17 matches, 0.0008%), previous-day same flight number +24h (excess
~350 rows). **No second copy mechanism exists.**

**Caveat that now matters more, not less.** Fourteen rows is a very small base and the 2026
scored set will have its own count, not 14. Whether "perfect tail => 244.8" is a central
estimate or a lucky draw depends on the month-to-month variance of that count, which is
being measured. Treat the ceiling as real but the path to it as unproven: three previous
tail estimates delivered 1-20% of their promise.

---

## 8e. Parallel agent sweep, 2026-09-16

Eleven investigations run concurrently. Every number below was independently reproduced
from saved OOF artifacts before being recorded. Baselines: 12-month OOF **285.6515**,
**Jan+Jul 369.462** (n=344,419) - the scored population, and the headline metric.

### Confirmed positive

Measured in isolation against the same baseline, so **these do not simply add** - several
touch overlapping signal and a combined run is required before any total is claimed.

| change | Jan+Jul | 12-month | months | note |
| --- | ---: | ---: | ---: | --- |
| neighbour + arrival taxi-in (Pearson-rejected) | **-0.530** | -0.962 | **12/12** | the most stable result of the sweep |
| arrival features, AOBT-anchored fixed-window | -0.563 | -0.627 | 11/12 | needs a scoring-path change; leakage probe passed |
| runway configuration state | -0.326 | -0.422 | 11/12 | untested item 1; see note |
| storm codes (`wx_thunder` et al) | -0.085 | -0.126 | 10/12 | -5.4% SSE on 1,100 thunder rows |
| restore the day-offset gate | ~+0.25 (v8 result) | - | - | correctness fix, not a new gain |

**The LIRF specialist is WITHDRAWN. Its entire gain is one row.**

It refits all four decomposition components on LIRF's 160,704 rows alone. On Jan+Jul it
appeared to gain **+11.0s on 397 LIRF missing-AOBT rows** while paying back 4.3s across the
other 344,022, netting -6.74s. The decisive test is leave-one-row-out:

| Jan+Jul | baseline | specialist | delta |
| --- | ---: | ---: | ---: |
| all rows | 369.462 | 362.717 | **-6.744** |
| **minus `MVT_ID 192615553`** | 355.518 | 355.963 | **+0.445** |

**Removing a single movement turns the result net harmful.** That row has a true taxi of
87,361s (24.3h) - a day-offset broken clock. The baseline predicts 28,355 and the
specialist 46,472; both are hopeless, the specialist merely lands less far away. Within the
397-row segment, **215 rows (54.2%) get worse**, those regressions give back 46.3% of the
gross gain, and the single best row is **65.6%** of the net.

Bootstrap over 20,000 draws, agreeing across four resampling schemes (segment rows, all
LIRF rows, all scored rows, and day-clustered): **P(specialist is worse) = 0.23**, median
-5.96 against an observed -6.74, so the point estimate is itself optimistic.

This is the fourth tail over-claim in the project's history and it has the same signature as
sections 2.3 and 4: a large headline carried by a handful of extreme rows that no model can
actually predict. It was caught only because the gain was decomposed row-by-row rather than
accepted at 10/12 months. **10/12 months and a clean mechanism story were both true and both
misleading.**

Two solid negatives survive from the same investigation and are worth keeping (see below):
capacity cannot reproduce the effect, and neither can per-airport rounds.

**Runway configuration: the hypothesis was half wrong.** The *change* features are
worthless - `cfg_dep_changed_30m` and `cfg_this_differs_from_mode` score ~0.000% of
LightGBM gain, and a dedicated reversal family (true heading reversals, 3.7% of rows) gained
only -0.079% and made July worse. A wind-shift transient is not a usable signal at these
airports. What pays is the *level*: `cfg_n_rwy_30m` and `cfg_this_rwy_share_30m` dominate -
"which runway am I on relative to what the airport is currently using, and how many strips
are live". That is a mixed-mode versus segregated-operation descriptor, not a transition.
It is a bulk feature (bulk -0.454%, tail -0.014%) and barely touches LIRF (-0.086%).

**Why no operational feature reaches the scored error.** Config-transition rows were tested
directly, since that was the original hypothesis. The effect is real but lives in the wrong
population: on `cfg_reversal_60m == 1` rows the gain is -1.81% SSE on Jan+Jul, the only place
the scored months beat the annual average - but those 10,168 rows average **RMSE 258 against
369 pooled**. Rows near a configuration change carry *below*-average error. A reversal happens
during calm reconfiguration, not during the meltdowns that make the tail.

That generalises. Congestion features are redundant because `MVT - AOBT` already measures
congestion (section 3). Configuration features describe orderly operations. Weather reaches a
little further because it is a condition the clock cannot see. But the scored error is
concentrated in **broken clocks at LIRF and LFPG** - those two airports hold 68.6% of Jan+Jul
squared error - and config state moves LIRF by +0.007% and LFPG by -0.06%, i.e. not at all.
**Every feature family that describes how the airport is operating is looking in the wrong
place.** That is now five negative families with one explanation.

**Gate improvement (WITHDRAWN).** A two-stage gate raised 12-fold AUC 0.8732 -> 0.8879 and
appeared to give -4.32s (12/12 months). It fails the one-row test the same way the LIRF
specialist did: Jan+Jul -6.080 -> -2.098 (-1 row) -> **+1.412** (-10 rows). One row is 50.9%
of the net gain; the top 10 are 106.9%, so everything outside them is net negative.
Sharpening made it worse, not better - gamma=2 had the best headline and the worst -10.

What survives is a genuine **distributed bulk effect**: restricted to `y <= 3600` (99.8% of
rows) it is -1.577 over 12 months, still -1.261 after dropping the ten most-helped bulk rows,
11/12 months, with a row-level win rate of 52.1-52.7% uniform across months. But **July bulk
is +1.603**, so Jan+Jul bulk is +0.454 and it is not submittable. Why July regresses was
deliberately not investigated - that would be selecting on a scored month.

### Confirmed negative - do not re-test

| idea | result |
| --- | --- |
| Recalibrating the bias ladder | 10-30 min band is calibrated to within **0.1s**; see 8d |
| Extending `GATED_BANDS` beyond LIRF | `other/*` bands hold **zero** positives in 12 months; 0.00s |
| Cloud ceiling / LVP thresholds | **costs** 0.009s, and makes low-ceiling rows themselves worse (+0.27% SSE). The model already prices ceilings via visibility and fog. Kept as `weather.CEILING_FEATURES`, excluded from `FEATURES` |
| Per-airport rounds inside the global model | LIRF miss-AOBT 4922 -> 4990, **worse**. The specialist's schedule is a *consequence* of fitting on LIRF, not the cause of its gain |
| Raising global capacity to find airport interactions | LIRF error **flat across a 4x leaf range** (439.89 / 439.99 / 440.34). Confirms section 4's saturation finding and means the specialist gain is not tunable into the global model |

### The acceptance gate, and what it shows

`scripts/robustness_check.py` removes the rows a candidate helped most and recomputes.
Applied to every artifact, on the **ten selection-legal months** (Jan and Jul excluded):

| candidate | delta | -1 row | -10 | -50 | top row's share |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate, two-stage + gamma=2 | -3.886 | -3.114 | -1.636 | **-0.968** | 6.0% |
| LIRF specialist | -3.211 | -2.706 | -0.946 | +1.141 | 6.1% |
| neighbour + arrival taxi-in | -1.092 | -1.113 | -1.060 | **-0.878** | 0.1% |
| arrivals, fixed-window | -0.652 | -0.647 | -0.626 | **-0.567** | 0.1% |
| runway configuration state | -0.454 | -0.444 | -0.420 | **-0.346** | 0.2% |

The same candidates on **January+July**, where four of the five collapse:

| candidate | delta | -10 | verdict |
| --- | ---: | ---: | --- |
| LIRF routed | -11.008 | **+5.155** | fragile (35.2% in one row) |
| LIRF specialist | -6.744 | +9.604 | fragile (30.6%) |
| gate gamma=2 | -6.080 | **+1.412** | fragile (21.5%) |
| storm codes | -0.085 | +0.053 | fragile |
| neighbour | -0.530 | -0.339 | robust (0.5%) |
| arrivals fixed | -0.563 | -0.495 | robust (0.2%) |
| runway config | -0.326 | -0.254 | robust (0.2%) |

**Read those two tables together.** The gate and the specialist are *not* pure artifacts -
each has a real distributed component worth around a second that survives removing fifty
rows on clean months. What collapsed is specifically their **scored-month** figure, because
Jan+Jul contains a handful of extreme LIRF rows and both models happened to land closer to
them. A candidate can have genuine structure and still have a headline that is a lottery
ticket.

**The single most diagnostic column is the top row's share of the gross gain**: 17-35% for
every candidate that failed, 0.1-0.5% for every candidate that held.

### Method note

The Pearson residual screen that governs section 3 was shown to be **unsafe**: features
rejected at +0.004 to +0.012 went on to improve 11/12 months in a real paired fold test.
Screen with correlation to prioritise; never reject on it alone.

---

## 8g. Competition mechanics, checked at source 2026-09-16

From the official 2026 site, correcting assumptions this log has carried:

- **Prizes go to the top 3 only** - "the combined prize amount for the first 3 teams is
  5000 EUR". Earlier entries in this log treated a top-10 placing as the target. The real
  bar is **3rd place, 245.02** as of 2026-09-16, so the gap from our 342.22 is **97.2s**,
  not the ~79s implied by a 10th-place reading.
- **Deadline 11 October 2026, 23:59:59 CET.**
- **Final ranking is on the same Jan+Jul 2026 set as the public board.** There is *no*
  announced final submission set. This differs from the 2024 challenge, where final ranking
  used `final_submission_set.csv` posted a week before the deadline containing **52,190
  flights nobody had submitted against**. The organisers floated a phase 2 on Discord
  (section 8f) but have not announced one, and the site reserves the right to change rules.
- The 2024 challenge additionally required, for prize eligibility, that source be published
  under **GPLv3** with "sufficient documentation to understand and reproduce the results",
  and its Selection Committee weighed **"the originality of the proposed model, the open
  availability of the code and the relevant documentation"** alongside the score. If 2026
  follows suit, the documentation in this repo is an asset, not overhead.

**Strategic consequence.** With no announced hidden final set, a public-board score is the
ranking. That raises the value of the probing concern in 8f and lowers the expected payoff
of further honest modelling: 97.2s is not reachable from a base where the strong teams
themselves report everything after the off-block anchor is worth under a second each.

---

## 8f. What the challenge Discord says (read 2026-09-16)

Channel `#prc-data-competition` in the OpenSky Community server. Four things that change
the strategic picture.

**1. The strong teams independently derived our validation rules.** Piyush Patil,
2026-09-12, unprompted:

- hold out whole months, and **require January and July to improve separately** - a pooled
  number hides month-specific gains
- **split the error by lane**, rows with an NM off-block against rows without, "because the
  lane without it holds a third of the squared error in a few hundred rows and behaves
  differently in 2026"
- **"Reject any gain concentrated in under a hundred rows"**
- compare the 2025 and 2026 distribution of every new feature before training

That is section 8e's robustness gate, arrived at independently by another team.

**2. They report the same ceiling we hit.** "Anchoring the target on takeoff minus the NM
off-block time was the only large step for us; everything since has been model diversity
worth under a second each." **We are not missing a large feature. The strong teams are in
the same place.**

**3. There is an open allegation that the top of the board is being reverse-engineered.**
fazlurnu, 2026-09-12: a few hundred Rome departures without an NM off-block have labels
that are either a normal taxi or a schedule default of 40,000-110,000s; each is worth 1-3s
of RMSE alone, so one submission changing one row reveals its label from the score
difference, and five uploads a day resolves them within days - "flat plateaus, then one
large step nobody learning from 2025 can follow". Luis backed the arithmetic; Piyush
pushed back that it is not easy. **This is an allegation, not an established fact.**

The organiser's reply (`espinielli`): *"Thanks for the 'fair' notification... eventually we
can always set a phase 2: we have plenty of data... but it is a pity to play for the win
rather than for the real goal of the challenge of building a useful model."*

If some top scores come from probing, it reconciles what section 8c could not: a score of
245 requires both a near-perfect bulk *and* 75-85% of the tail removed, and the tail rows
are not predictable from the files. It is also exactly the technique section 8b-bis
quarantines. **A phase 2 on hidden data is explicitly on the table**, and it would be
decided by a model that generalises.

**4. Another team's failure list, which matches ours almost item for item.** JavierIOM,
2026-09-12/13, all measured against the board: dropping seasonal columns (CV said +43s on
July at p<0.001, board said **-7.6s**); fitting blend weights on a holdout lost 4 times out
of 4; hyperparameter tuning not significant for LightGBM, CatBoost *or* XGBoost; weighting
training rows toward the scored (airport, month) cells did nothing; XGBoost took zero weight
as a third blend member. They also checked all 52 numeric features, 2025 Jan+Jul against the
2026 ranking set, and **none shifted more than 0.25 sd**.

**The dew-point tip was tested and is NEGATIVE** (see below). **The one actionable tip:** *"Temperature and dewpoint spread beat a binary snow flag by a
distance, because most de-icing happens on clear frosty mornings that carry no precipitation
code at all."* `dwpf` was already being downloaded by `scripts/fetch_metar.py` and had never
been made into a feature. Residual check confirms signal: below -5C the model
**under-predicts by +18.6s** (n=2,774), and frost rows run +7.4s against the bulk average.
Tested in `scripts/experiment_dewpoint.py`, paired 12-fold against the v13 model:

| | 12-month | Jan+Jul | cold rows (n=48,895) | months | robustness |
| --- | ---: | ---: | ---: | ---: | --- |
| dew-point spread + frost risk | -0.157 | **-0.032** | **451.42 -> 451.56 (worse)** | 7/12 | FRAGILE, reverses at -10 |

**Rejected.** It makes cold rows *worse* - precisely where the mechanism was supposed to
act. `wx_freezing` already fires below 3C and captures the frost signal under a different
name. The features are computed and kept as `weather.DEWPOINT_FEATURES`, excluded from
`weather.FEATURES`, in the same way as the ceiling family.

**The lesson about importing other teams' numbers:** their 3.4s came from recovering a
weather block that had been silently null for nineteen submissions because of a
timezone-aware/naive join bug. It was a bug fix measured against a broken baseline, not a
discovery about dew point. **A gain measured against someone else's baseline says nothing
about ours.**

Caveat on their numbers: JavierIOM's gains are measured **against the leaderboard**, which
this project does not permit as a validation observation. Treat their directions as hints to
test locally, never as measurements to adopt.

---

## 8g. The 630 "neither" rows - tested and closed (2026-09-16)

Raised on the Discord by `romano`, who split the 981 heavy Jan+Jul rows into 345 schedule
copies, 7 day-offsets and **630 that are neither**, and noted they are not a Rome story.
Reproduced exactly (we get 636) and tested properly.

**What they are.** `BLOCK_TIME` precedes NM's `AOBT_3` by a median of **3,055s**: the airport
clocks off-block at first pushback, NM clocks it when the aircraft actually started moving.
So the label is a **genuine 71-minute taxi**, not a corrupted clock. Median `MVT - AOBT` is
1,264s, i.e. NM sees an ordinary taxi throughout.

Scored-month profile: EGLL 234, LFPG 134, LIRF 127, EHAM 54; July 375, January 261; median
1.18h, p90 1.86h; daytime peak at 10-11am; 174 distinct airport-days with the top ten
holding 36%. Weather-associated: fog 17.8% (5x base), low visibility 11.5% (11x), freezing
32.5% (4x), thunder 3.0% (10x).

All-year the snow association is dramatic (20.4% against a 0.33% base, **62x**) but that is
driven by a single February snow event at Istanbul - LTFM has 217 such rows on 2025-02-23
and 166 on 2025-02-19, 65% of all February cases in three airport-days. **February is not a
scored month**, and in Jan+Jul only 14 of 636 rows have `wx_snow`.

**The prize is enormous and unreachable.** These 636 rows are **44.4% of Jan+Jul squared
error on 0.185% of rows**. Perfect prediction moves Jan+Jul 369.00 -> 275.15, about
**-87s scored** - the whole gap to the leaders.

**They are not detectable.** A LightGBM classifier, out-of-fold on whole-month holdouts:

| | |
| --- | ---: |
| AUC, all-year / Jan+Jul | 0.7692 / 0.8207 |
| average precision | 0.0076 (base rate 0.00117) |
| precision, top 1,000 rows | **3.80%** |
| precision at 50% recall | 1.22% |

Expected-value gating with `P(neither) x E[extra]`, leave-one-month-out:

| scale | 12-month | Jan+Jul |
| --- | ---: | ---: |
| 1.00 | +306.3 | **+184.7** |
| 0.50 | +100.1 | +53.3 |
| 0.25 | +28.0 | +13.7 |

AUC 0.82 sounds usable but precision of 1-4% means catching half of them requires flagging
100,000 innocent rows. This is section 2.4's wall, now measured with a classifier rather
than a hand-built band, and it is the same conclusion: **detection is the binding
constraint, and it is not solvable from the available fields.**

---

## 8h. The whole-month CV is pessimistic by ~11-12s

Measured 2026-09-16. **Leave-one-month-out withholds the entire month, but production trains
on all twelve months of 2025 and predicts a *different year's* January and July - so the
production model has seen January and July, and the CV model has not.**

Scoring the same rows both ways:

| | trained without any of that month | trained with the month's first half | CV penalty |
| --- | ---: | ---: | ---: |
| July, 2nd half (n=98,756) | 435.40 | 422.99 | **+12.41** |
| January, 2nd half (n=78,771) | 492.15 | 480.93 | **+11.22** |

Caveat: the second arm trains on the first half of the same month, which carries genuine
within-month leakage (weather systems span a split), so 11-12s is an **upper bound** on the
artifact.

**Consequence.** Jan+Jul OOF reads 369.46 while the scored set returns 342.22. Roughly
11-12s of that 27s gap is this artifact rather than the 2026 population being lighter.
Section 8c's local-to-scored reasoning should be read with that in mind.

**What to do about it.** Keep leave-one-month-out for *selection* - both arms of a paired
test carry the same penalty, so the delta stays valid, and it errs conservative. But do not
compare its absolute numbers to a scored result. A fold that truly mirrors production needs
a second year of data, which we do not have; a half-month split only trades one bias for
another.

**Where it changes a decision:** seasonal features are systematically undervalued, because
whole-month CV can never have the same season in training. This is the likely explanation
for the divergence another team reported on the Discord - their CV said dropping seasonal
columns cost +43s on July at p<0.001, the leaderboard said it gained 7.6s. `matrix.build()`
already drops `month` (though `doy` survives), so we happen to sit on the right side of that
particular question by accident rather than by measurement.

---

## 8i. MVT_TIME precision (NEGATIVE)

The data page documents `MVT_TIME_UTC_mvt` as the **"(best available) movement time"** - the
only field carrying a hedge - which suggests several upstream sources of differing quality.
The seconds field confirms two populations: five airports report ~8.3% of movement times on
an exact minute boundary against ~1.7% (chance level) at the other five.

| ~8.3% on `:00` | ~1.7% on `:00` |
| --- | --- |
| EDDM, EGLL, LFPG, LIRF, LTFM | EDDF, EHAM, LEBL, LEMD, LSZH |

Those rows carry **832 RMSE against 325** on the scored months and a >1h rate of 30.3 per
10k against 19.2. The marker survives in `ranking.parquet` at 5.15% with the identical
airport structure, and `features.py` derives `hour` and `minute_of_day` but has never used
the seconds field - so the model could not see it.

Paired 12-fold with `mvt_second` and `mvt_sec_zero` added:

| | without | with | delta |
| --- | ---: | ---: | ---: |
| 12-month | 285.0826 | 285.2169 | **+0.1343** |
| Jan+Jul | 369.0021 | 369.1129 | **+0.1107** |
| coarse rows (n=17,817) | 832.00 | 832.20 | +0.20 |
| fine rows (n=326,602) | 325.31 | 325.42 | +0.11 |

**Rejected.** Worse everywhere, including on the coarse rows themselves.

**Why, and it generalises.** Under squared loss the optimal prediction is `E[y|x]`. A
coarser measurement inflates the *variance* of `y` but does not shift its *mean*, so knowing
a row is noisy gives the model nothing to act on. The 2.6x error difference is irreducible
noise in the label, not recoverable signal. Same lesson as the bias ladder in section 8d: a
real, measurable asymmetry that offers no purchase because the conditional mean is already
correct.

**Keep the observation though.** It explains part of why five airports are harder than the
other five, and it means any future per-airport error comparison should account for the
source difference rather than attributing it to operations.

---

## 9. Honest assessment

We are at 344.21 against a prize cutoff of 247.01. The gap is ~97 seconds and nothing
currently in flight closes it. The tail holds 36% of squared error and three consecutive
attempts to convert that into score were over-estimates, for a consistent reason: those
regimes have 16, 2,362 and 4,126 training rows, which is not enough to fit a magnitude
model that generalises. The one tail mechanism that worked had 199,634.

An earlier version of this section claimed section 8c had retired the tail. **That was
wrong and is retracted.** It computed the ceiling on the 12-month average and compared it
to the scored gap - different populations. On January and July, the months that are
actually scored, the tail is 49.4% of squared error and solving it perfectly implies
~244.8 scored, *below* the 247.01 cutoff. The tail is back as the main line of attack, and
within it **fourteen rows over 12 hours carry 38.8% of scored-month error**.

The 10-30 minute band is also NOT recoverable by recalibration: it is calibrated to within
0.1s of RMSE (section 8d). It remains 34.4% of scored SSE, but that error is genuine
conditional variance, not correctable bias.

The field is also moving. The prize cutoff tightened 15 seconds in two days while we
improved 54, and our rank went from 89 to 90.

---

## 10. Continuous schedule-proxy correction screen (2026-09-16)

`scripts/experiment_continuous_blend.py` tests whether the deployed mechanism blend
leaves a smooth residual along `MVT - SCHED - prediction`. It fits a slope in three
predeclared absolute-gap bands (under 30 minutes, 30 minutes to 2 hours, over 2 hours)
on the ten months other than January and July. The January and July labels are never
used to fit a slope; each selection month is evaluated with that month left out.
The baseline is the saved arrival-fixed OOF model that underlies v13.

| population | baseline | corrected | delta |
| --- | ---: | ---: | ---: |
| January + July | 368.8987 | 368.7657 | **-0.1330** |
| January + July bulk | 262.2972 | 262.0373 | **-0.2598** |
| ten selection months | 265.3041 | 265.1987 | **-0.1054** |

The correction is applied only where NM's `AOBT_3_flt` is present. It survives
removing the 1, 10, and 50 most-helped January+July rows: -0.1322, -0.1284,
and **-0.1184**. The global extra blend slope is only -0.001 on the selection
months, so there is no evidence for a large missing global offset.
Against the experimental combined OOF baseline the same procedure is -0.1366
on January+July and -0.1224 after removing 50 rows.

An initial six-group version also corrected AOBT-missing rows. Its January+July
headline was -0.9451, but turned **+0.0363 after removing 50 rows**. Its apparent
gain is tail-concentrated and is rejected by the robustness gate.

This is a *screen*, not a deployable improvement. Its second-stage slopes are
trained on saved first-stage OOF predictions whose underlying fold models were
not nested with the second stage; a strict nested-month confirmation is needed
before a submission. The surviving effect is small relative to the prize gap.

## 11. Flight identifiers do not transfer to July 2026 (2026-09-16)

`FLIGHT_mvt` and `CALLSIGN_flt` were native LightGBM categoricals in both base models,
holding 11% of the matched base's split gain (6.0% + 5.3%). Alphanumeric ATC callsigns
(62% of rows) rotate, so their level sets differ between years and seasons. Whole-month CV
can never show this because every holdout month has its season-mates in training:

| population | unseen `FLIGHT_mvt` | unseen `CALLSIGN_flt` |
| --- | ---: | ---: |
| any whole-month CV holdout | 1.3-1.7% | 1.2-1.6% |
| January 2026 ranking (vs all 2025) | 2.0% | 1.9% |
| **July 2026 ranking (vs all 2025)** | **36.3%** | **42.3%** |

July 2026 low-cost rows are 78% unseen; alphanumeric callsigns 58%, numeric ones 17%.
An unseen level reaches the trees as missing and takes the default branch, so 36-42% of
the July rows were being routed by a feature the model trusted for 11% of its fit. This
is the mechanism behind JavierIOM's Discord report that dropping year-specific columns
lost on CV and gained on the board.

**Component ablation** (`scripts/experiment_highcard.py`, matched base only, 342 rounds,
capped-residual RMSE): dropping both columns and adding categorical regularisation
(`cat_smooth 50, cat_l2 50, min_data_per_group 500, max_cat_threshold 16`) wins on every
fold with identifiers intact - Feb 214.24 -> 211.37, Oct 213.07 -> 210.90, Jan 207.93 ->
205.56, Jul 243.37 -> 241.01. With 40% of the holdout's identifiers masked to simulate
July 2026 (`scripts/experiment_unseen_categories.py`), the current model degrades a
further 0.6-2.1s (July 243.37 -> 245.49) while the identifier-free model is unchanged.

**Paired 12-fold** (`decomposition.BASE_EXCLUDED` + regularised `BASE_PARAMS`, gate
frozen, identifiers intact in the holdouts so this understates the July effect):

| configuration | baseline | Jan+Jul | 12-month | bulk | months | -1 / -10 / -50 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v12 (weather + sequencing) | `sequencing_oof` | -0.310 | **-0.909** | -0.914 | 12/12 | -0.09 / +0.08 / +0.40 |
| combined four families | `combined_oof` | -0.605 | **-1.172** | -1.184 | 12/12 | -0.45 / -0.23 / n.a. |
| v13 (arrivals, storm codes excluded) | `arrivals_fixed_oof` | -0.101 | **-0.755** | n.a. | 12/12 | +0.16 / +0.44 / +0.79 |

The scored-month figure is small and fails the -10 test in the v12 and v13
configurations; the 12-month figure is broad (every month, -0.75 to -1.2s). The case for the
change is not the CV delta, which structurally cannot contain the July effect, but the
masking simulation and the removal of arbitrary routing on 36-42% of the scored July rows.

Refit twice: `models/mechanism_noid` (`scripts/refit_noid.py`, the v12 63-feature list)
and `models/mechanism_noid_arr` (`scripts/refit_noid_arr.py`, the v13 71-feature list
including its storm codes); both scripts assert the feature list equals the shipped model,
so the base exclusion is the only change. Leakage probe PASS on both; rehearsal 311.74 and
310.99 (optimistic, full-data models). v14 is built from `mechanism_noid_arr` and
**scored 340.35 against v13's 342.22 (-1.88s)**, consistent with the masking simulation.

Also checked and closed in the same session: `doy` is 0.08% of gain (no seasonal-column
risk beyond identifiers); no ID overlap or numeric covariate drift between training and
ranking; per-airport bias within 8s except LIRF at -47s (worth under 2s); garbled LIRF
flight strings (`ITYTY680`, `RYRR4852`, 0.7% of rows, 26% of target variance) are already
captured through the missing-NM-record flag.

## 12. LIRF broken-clock bands: predict them, do not hedge (2026-09-16)

Three public repos (zestful-fountain 273.6, OpenAir 277.0, elegant-alligator 268.6) each have
exactly one step above 10s on the board, and it is the same step: LIRF departures with no NM
record and a long `MVT - SCHED` gap are predicted at the gap (schedule copy) or at a day plus
a normal taxi (day offset), rather than shrunk toward an ordinary taxi. Our v8 day-offset gate
shrank the same rows to a ~13% probability and the tail severity model landed between 33k and
76k on rows whose truth is 55k-95k; in 2025's Jan+Jul five day-offset rows alone were 14.9%
of total squared error.

`prc.lirfbands`: scope = LIRF, `nm_record_missing`, gap > 4h; bands (4,8], (8,12], (12,20],
20h+; prediction = p_copy * gap + p_day * (86400 + mean extra) + p_normal * mean normal, rates
shrunk toward the pooled scope rate with 3 pseudo-rows. Training 2025: 127 rows; (8,12] is
21/21 copies, (12,20] 17 copies + 7 day offsets, 20h+ 4 day offsets + 1 copy. Also
`lobt_clip`: the target never departs from `MVT - LOBT` by more than 3606s (0 violations in
2,062,577 rows; found in the elegant-alligator repo, verified here), and our OOF violated it
on 236 rows.

Leave-one-month-out table on the `noid_arr` OOF (`scripts/cross_validate_lirfbands.py`):

| population | v14 OOF | bands | bands + LOBT clip |
| --- | ---: | ---: | ---: |
| Jan+Jul | 368.80 | 344.04 | **341.93** |
| January | 368.98 | 354.24 | 352.85 |
| July | 368.66 | 335.60 | 332.87 |
| 12-month | 284.27 | 277.38 | 275.96 |
| ten other months | 264.36 | 262.19 | 260.93 |

10/12 months improve; February is +8.5 (one in-scope row that was a normal taxi). Robustness:
-14.2 after removing the single most-helped row, +0.1 after five, +1.6 after ten. **The gain
is concentrated by construction**: these rows are the error. Section 8e's gate was the right
tool for ordinary features and the wrong tool here; the direction is confirmed externally by
three independent teams' board results, not by our CV alone.

Shipped as `models/mechanism_v15.json`, kind `mechanism_bands`, wrapping `mechanism_noid_arr`
with the table fitted on all 12 months (`scripts/fit_lirfbands.py`). Leakage probe PASS;
rehearsal 310.42. v15 changes 162 ranking rows against v14: 43 in the band scope (the
12-20h rows move from 26k-66k to 57k-68k, the 24h row from 76k to 84k), 119 by the LOBT clip.
**Scored 310.58 against v14's 340.35 (-29.8s)**, matching the -26.9s CV estimate.

## 13. Unmatched lanes: lane two flat, LIRF under-4h rows -12s (2026-09-16)

Lane split of the v15 OOF on Jan+Jul 2025: matched 339,046 rows RMSE 241 (49% of SSE),
unmatched non-LIRF 4,976 rows RMSE 1,559 (30%), unmatched LIRF 397 rows RMSE 4,620 (21%).

**Lane two (unmatched, non-LIRF): NEGATIVE.** An expert trained on all 2.08M departures
with 47 NM-free features (`scripts/experiment_missing_expert.py`), four cap/weight variants,
moves the lane from 1,559 to 1,548 at best: -0.7s on Jan+Jul. The shipped `base_missing`
was not undertrained; the lane's error is genuine long taxis (the 130 LFPG "neither" rows
alone are 8% of scored SSE) and no public leader claims to predict them either.

**Lane three, gap under 4h (LIRF unmatched outside the v15 bands): the same expert wins.**
The old path (base_missing + tail gate x severity) over-predicts these 352 scored rows
(median truth 1,324s, median prediction ~5,000s). Replacing them with the NM-free expert:

| variant | Jan+Jul | 12-month | -1 / -10 / -50 rows | months |
| --- | ---: | ---: | --- | ---: |
| cap 12,000, unmatched weight 8 | 341.93 -> **329.64** | 275.95 -> 272.63 | -11.5 / -7.9 / -1.6 | 8/12 |
| cap 12,000, weight 1 | 331.94 | 273.46 | -9.2 / -5.8 / +0.1 | 7/12 |
| no cap, weight 8 | 330.85 | 273.45 | -10.3 / -6.5 / 0.0 | 7/12 |
| cap 3,600, weight 8 | 332.42 | 280.88 | -8.5 / -4.0 / +4.6 | 3/12 |
| OpenAir-style T = D - G, G learned on all LIRF rows (`scripts/experiment_lirf_gatehold.py`) | 333.02 | 273.16 | -8.4 / -5.0 / +0.9 | 7/12 |

Unlike the bands this survives the -10 test: it is distributed over the 352 rows, not a
handful. Shipped as `models/lirf_short_expert.txt` (`scripts/fit_lirf_short_expert.py`, all 12
months) inside `models/mechanism_v16.json` via `lirfbands.apply_short_expert`; v15's bands and
LOBT clip unchanged. Leakage probe PASS; rehearsal 294.61. v16 changes 340 ranking rows
against v15, median prediction on them 4,994 -> 2,875. **Scored 301.14 against v15's 310.58 (-9.4s).**

## 14. v17 bundle: ranking-month context + LOBT residual expert (2026-09-16/17)

Built from the roadmap in `docs/roadmap-to-268.md`. Paired 12-fold against the v16 OOF
(`scripts/cross_validate_v17.py`; bands, short expert and LOBT clip re-applied per fold):

| stage | Jan+Jul | Jan | Jul | 12-month | other 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16 | 329.64 | 349.87 | 312.38 | 272.63 | 259.87 |
| + `prc.context` (61 features) | 324.26 | 346.42 | 305.22 | 266.42 | 253.41 |
| + `prc.lobt` expert, per-airport alpha | **321.27** | **343.64** | **302.04** | **263.46** | **250.44** |

12/12 months (-6.2 to -18.8); robustness -8.2 / -7.6 / -6.3 after removing 1 / 10 / 50 rows,
so the gain is distributed. Alphas are the mean of twelve fold-fitted per-airport weights
(global 0.90, EHAM 0.59 to LIRF 0.99); they were fitted on training-row residuals and are
therefore inflated, but the OOF delta is measured with them as used.

Context features (`src/prc/context.py`, all label-free, identical with departure block times
blanked): neighbour NM taxi `MVT - AOBT_3` of other departures, past `[t-w, t)` and future
`(t, t+w]` for 15/60 min by airport/runway/stand, own-minus-window, last/next proxy; aircraft
taxiing at own AOBT and take-off by airport/runway; arrival taxi-in mean/median/count for the
same (airport, month, stand) and (airport, month, runway); previous arrival at the stand
(age, taxi-in, type match); minute and second of every NM timestamp; remaining pairwise
clock deltas. Sources: zestful-fountain neighbour/following (-4.9 official), elegant-alligator
arrival context (-1.9 local) and LOBT expert (-2.0 official).

Screened against the v17 core and **not shipped**:

| candidate | result |
| --- | --- |
| Rome matched regime: LIRF-scoped identity gate, fallback-rate encoders, isotonic calibration (`scripts/experiment_lirf_regime.py`) | Jan+Jul -0.95 (Jul -1.8, Jan 0.0), 7/12, AUC 0.856; small, kept in reserve |
| EDDF/EGLL/EHAM specialists 50/50 (`scripts/experiment_members.py --member specialists`) | Feb -0.6, Oct -0.2, Jan -0.1, Jul +0.04; EGLL worse in 3 of 4 |
| CatBoost residual member 50/50, depth 8, 1500 it, identifiers as CTR cats | Feb +0.1, Oct +1.3: NEGATIVE |

Shipped: `models/decomposition_ctx.*` (132 features), `models/mechanism_ctx.json` (gate
reused), `models/lobt_expert.*`, wrapper `models/mechanism_v17.json`. Leakage probe PASS;
rehearsal 281.84 (optimistic). v17 changes 341,319 ranking rows against v16, RMS 93s.

## 15. v18 screens: all four remaining roadmap items fail (2026-09-17)

Paired against the v17 OOF (321.27 Jan+Jul), `scripts/experiment_v18.py` and
`scripts/experiment_lirf_regime.py --disruption`:

| candidate | Jan+Jul | 12-month | notes |
| --- | ---: | ---: | --- |
| faulty-AOBT tail gate (47,544 rows, correction model at alpha 0.5) | +0.09 | -0.03 | no signal left in those rows after v17 |
| gate-hold `T = D - G` blended into *matched* LIRF rows | **+7.06** | +5.30 | worse in all 12 months even with a fitted alpha; the matched anchor beats the schedule anchor everywhere |
| `linear_tree` + signed-log clock deltas, matched base | +5.9 / +48.1 / -0.6 / +5.0 on Feb / Oct / Jan / Jul | | unstable; kind-mango's gain came from other fixes in the same upload |
| Rome regime gate + 60-min disruption meters | -0.91 | -0.32 | identical to the regime without meters (-0.95); meters add nothing |

No v18 bundle reaches the per-version bar. The public-repo method inventory is exhausted:
everything measurable from the five repos is now either shipped (v14-v17) or shown not to
transfer to this pipeline. The Rome regime (-0.9) remains the only positive reserve item.

## 16. v18: out-of-fold LOBT alphas + Rome regime gate; ATFM daily data (2026-09-17)

**LOBT alphas refit on OOF residuals** (`scripts/refit_lobt_alphas.py`): the v17 weights
were fitted on training-row residuals and inflated (global 0.90); fitted on the ten
non-scored months' OOF they drop to global 0.64 (EDDF 0.45 .. LIRF 0.74). Jan+Jul
321.27 -> 320.36 (-0.91), 12-month 263.46 -> 262.84. Alphas are selected on months disjoint
from January and July.

**Rome matched regime** (`prc.lirfregime`, section 15's -0.91 candidate) fitted on all 2025:
encoders on all LIRF rows, gate on months outside {4,10}, isotonic calibration on {4,10}.

Both shipped in `models/mechanism_v18.json` (wraps `mechanism_ctx` with regime, OOF alphas,
bands, short expert, LOBT clip). Not jointly cross-validated; the two touch different rows
(regime: LIRF matched; alphas: all matched rows) so the sum, about -1.8s, is the estimate.
Leakage probe PASS; rehearsal 282.41. v18 differs from v17 on 322,559 rows, RMS 46s.

**EUROCONTROL daily ATFM data** (`scripts/fetch_eurocontrol_daily.py`): ATFM slot adherence,
airport arrival ATFM delay and airport traffic cover both ranking months at daily x airport
granularity; the two pre-departure delay files stop at June 2026 and are unusable. Licence
(ansperformance.eu/about/disclaimer): "may be copied ... provided that EUROCONTROL is
mentioned as the source and it is not used for commercial purposes" - a non-commercial
clause, the same eligibility problem recorded for Meteostat. Screened for information only;
result below. Not shipped.

ATFM daily screen (`scripts/experiment_atfm_daily.py`, matched base, capped-residual RMSE):
see `data/interim/atfm_daily_screen.json`. Result recorded in the line below.
2: 202.43 -> 202.55 (+0.12) | 10: 202.27 -> 201.92 (-0.35) | 1: 199.09 -> 198.86 (-0.23) | 7: 232.58 -> 230.46 (-2.12) 

## 17. v19: context round counts, three seeds, gate refit, finer Rome bands (2026-09-17)

`scripts/select_rounds_ctx.py` on inner months 4 and 10 with the 132-feature set:
base_matched 342 -> **1310**, base_missing 70 -> 149, gate 73 -> 56, severity 300 -> 38. The
base had been under-fitted since the context features arrived. Paired 12-fold against a
reconstructed v18 OOF (`scripts/cross_validate_v19.py`):

| stage | Jan+Jul | 12-month | other 10 |
| --- | ---: | ---: | ---: |
| v18 | 320.09 | 263.10 | 250.30 |
| + new rounds, 3-seed average | 319.25 | 261.99 | 249.11 |
| + schedule-fallback gate refit with context features (AUC 0.873 -> 0.876) | **319.13** | **261.67** | **248.74** |

12/12 months; -0.72 after removing 50 rows. Separately, finer Rome band edges at 40k/50k/
60k/70k/90k seconds (`lirfbands.EDGES_H`, tables now carry their own edges): Jan+Jul
329.64 -> 327.78 on the v16 OOF, every month group better; in 2025 the 40-50k band is 9/9
schedule copies, which the old 12-20h band had been blending with day offsets on five
ranking rows. v19 total estimate about -2.9s over v18.

Shipped: `decomposition_ctx19_s{100,101,102}` averaged through `seed_stems`,
`mechanism_ctx19` (63-column gate), wrapper `mechanism_v19.json` (`scripts/refit_v19.py`).
Leakage probe PASS; rehearsal 271.95. v19 differs from v18 on 328,034 rows, RMS 45s.

## 18. v20: target encodings, clock differences, take-off order, lr 0.02 (2026-09-17)

Ported from merry-quicksand (260.4, public) and kind-mango: 24 smoothed in-fold target
encodings of y and of the clock delta (`prc.tencode`, K=50, incl. airport x runway x stand);
`LOBT-SCHED`, `EOBT-SCHED`, `EOBT-LOBT`, `IOBT-SCHED`; take-off order against plan
(`ord_*`); matched base at lr 0.02 with 3,175 trees selected on months 4/10
(`scripts/select_rounds_v20.py`, inner 193 -> 191.06). Paired 12-fold vs v19
(`scripts/cross_validate_v20.py`, single seed):

| population | v19 | v20 |
| --- | ---: | ---: |
| Jan+Jul | 319.13 | **315.59** |
| Jan / Jul | 342.92 / 298.59 | 341.78 / 292.78 |
| 12-month | 261.67 | 259.05 |
| matched, scored | 226.99 | 225.31 |

12/12 months; robustness -2.9 / -0.9 / -0.5 after 1 / 10 / 50 rows.

Also measured tonight: weather drop test on the v20 base (+1.7 / +0.2 / +2.1 / +0.04 on
Feb/Oct/Jan/Jul when dropped: weather stays); uncapped matched base in the full pipeline
(`scripts/experiment_uncapped_pipeline.py`: matched-row RMSE -1.9 / -0.8 / -0.7 / -0.8 on
Feb/Oct/Jan/Jul vs v19; winsorising the training target discards learnable tail, as
merry-quicksand reported); deterministic Rome rules (`lirfbands.rome_rules`: schedule floor
and same-day exact schedule on LIRF no-record rows with 24000 <= gap < 86400: -0.9 Jan+Jul,
all month groups better, 14/14 same-day rows are copies). Lane accounting of v19 + rules:
matched 227.0 (theirs 222.6), Rome unmatched July 3,672 (theirs 3,926), July all rows 297.1
(theirs ~301.6 before ADS-B and Rome rules). The remaining gap to 260 is, by their ledger,
ADS-B at three airports.

## 19. v21: v20 + uncapped matched base + Rome rules (2026-09-17)

Paired 12-fold vs v19 (`scripts/cross_validate_v20.py --uncapped --rome-rules`): Jan+Jul
319.13 -> **314.43** (v20 alone 315.59), Jan 341.42, Jul 290.85, 12-month 258.47, matched
scored 225.21. 12/12 months; robustness -4.0 / -1.3 / -0.6 after 1 / 10 / 50 rows. Shipped
as `models/mechanism_v21.json` (`scripts/refit_v20.py --uncapped --rome-rules`): three-seed
decomposition with the matched base on the full residual, tail term kept for unmatched rows
only, encodings fitted on all 2025 and stored in the model JSON, gate refit with context v2,
fine bands, Rome rules, regime, OOF alphas, short expert, LOBT clip. Leakage probe PASS;
rehearsal 270.24. v20 was not built as a file; v21 supersedes it. **Scored 280.80 against v16's 301.14 (-20.3s); the fold said -15.2.**

## 20. CatBoost GPU member with native identity categoricals (2026-09-17)

The CPU screen (section 14: depth 8, 1,500 it, 50/50) was negative. On the RTX 4070 Ti
(`scripts/experiment_catboost_gpu.py`, 5,000 it, depth 8, lr 0.06, 145 features incl. 23
native categoricals: FLIGHT_mvt, CALLSIGN_flt, airport x stand, stand x runway, airport x
operator, runway x type, prefixes; target = uncapped residual to the AOBT anchor; in-fold
target encodings), one fold takes ~2 min. Member alone on matched rows beats v21's final
prediction in 11 of 12 months by 1-3s and loses in Jan (204.2 vs 202.4) and Oct (206.6 vs
205.8): equally good, different errors, which is what a blend needs.

First pass blended the raw member after the LOBT bound and July went 290.9 -> 304.9 on one
row (a matched schedule copy at 87,002s the member put at 12,140s). With the bound re-applied
after blending:

| blend weight | Jan+Jul | Jan | Jul | 12-month | months | -1 / -10 / -50 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 (v21) | 314.43 | 341.42 | 290.85 | 258.47 | | |
| 0.5 | **312.37** | 340.03 | 288.15 | 255.94 | 12/12 | -2.0 / -1.9 / -1.5 |
| 0.575 (fit on non-scored months) | 312.44 | 340.12 | 288.21 | 255.92 | 12/12 | -1.9 / -1.8 / -1.4 |

Also: LOBT blend alphas were stale (fitted vs the v17 base); refit vs the v21 base they fall
from global 0.64 to 0.33 and are worth -0.3s (`data/interim/lobt_alphas_v21.json`). Seasonal
alphas (winter/summer sets) add a further -0.3s, July only; not shipped.

Shipped: `models/catmember.cbm` (fit on all 2025 on GPU, `scripts/fit_catmember.py`),
`prc.catmember`, wrapper `mechanism_v22.json` = v21 + member at weight 0.5 (before the LOBT
clip) + refit alphas. **Scored 276.46 against v21's 280.80 (-4.3s; fold said -2.4).**

## 21. Member variants (2026-09-17, GPU)

| variant | result vs v21 + d8 member |
| --- | --- |
| CatBoost depth 10 (5,000 it, lr 0.05, l2 20) added to depth 8 | Jan+Jul 312.45 -> 312.33: +0.1s, same errors as depth 8 |
| CatBoost on the schedule-relative target `BLOCK - SCHED` (merry-quicksand's untested second reference clock) | member alone 467 on matched rows vs 214; fitted weight 0.011; NEGATIVE, closed |
| `base_missing` uncapped for unmatched non-LIRF rows (`scripts/experiment_missing_uncapped.py`, 12 folds) | lane 1,545 -> 1,602, Jan+Jul +4.1s, 1/12 months: NEGATIVE. The matched tail is learnable (section 19); the unmatched tail is not. Capped refit with encodings: -0.36s, 10/12 (small, kept for the next build) |

## 22. LIRF unmatched mixture of experts (2026-09-17)

The 2026 ranking set has 2.2x the 2025 monthly rate of LIRF no-record departures in July
(276 rows) and 2.4x at EHAM in January (805), so the unmatched lanes weigh more on the board
than on the folds. elegant-alligator's missing-row MoE form, applied to LIRF no-record rows
under the band scope (`prc.lirfmoe`, `scripts/experiment_lirf_unmatched_moe.py`): calibrated
copy classifier (NM-free features + encodings, unmatched weighted 8x, isotonic on months 4/10,
AUC 0.928, mass 608 vs 621) mixed with a copy-free regressor (y < 12000):

| population | v21/v23 OOF | mixture |
| --- | ---: | ---: |
| Jan+Jul | 314.43 | **310.15** |
| Jan / Jul | 341.42 / 290.85 | 338.96 / 284.82 |
| lane, scored (352 rows) | 2,828 | 2,320 |
| 12-month | 258.47 | 254.71 |

11/12 months; robustness -4.05 / -3.37 / -2.71 / +0.49 after 1 / 5 / 10 / 50 rows (the lane
has 352 scored rows, so -50 removes 14% of it). Replaces the v16 single expert on those rows.
Shipped as `models/lirf_moe.*` inside `mechanism_v24.json`; leakage probe PASS; rehearsal
269.29. **Scored 275.99 -> 271.88 (-4.1s), matching the fold estimate.**

The mixture scope extended to 8h (`--max-gap-h 8`, 1,438 target rows; the 4-8h band was 62
copies / 15 ordinary at RMSE 6,481 under the band table): Jan+Jul 310.15 -> **309.41**,
10/12 months (Mar +2.8, Jun +0.3), AUC 0.927. Shipped as `lirf_moe_8h` in `mechanism_v25.json`,
applied after the band table so it overrides the 4-8h band; 18 ranking rows change vs v24.
**Scored 271.88 -> 269.82 (-2.1s), nearly three times the fold's -0.74: the 4-8h band is
denser in 2026.**

## 23. EUROCONTROL daily ATFM features in the full pipeline (2026-09-18)

Permitted by the organiser's ruling of 2026-09-17. `prc.atfm` joins five daily airport
aggregates (slot-regulated share, late-out share, arrival ATFM minutes per flight, weather
ATFM minutes, daily movements). Paired 12-fold, v21 configuration with and without
(`scripts/cross_validate_v20.py --atfm`, tag `v26_atfm`): Jan+Jul 314.43 -> 314.11 (-0.32;
Jul -0.57, Jan -0.06), 12-month 258.47 -> 258.27, 12/12 months, robustness -4.3 / -1.6 / -1.0.
Second CatBoost seed (10k it, seed 7) blended 0.25 + 0.25 with the first: 312.30 -> 312.17.
Both go into v26.

## 24. ADS-B (ADSB.lol, ODbL, permitted 2026-09-17): NEGATIVE on our pipeline (2026-09-18)

Pilot on 2025-07-15, 2025-01-15, 2026-07-15 plus 8 further July-2025 days (`scripts/adsb/`):
per-aircraft readsb traces, departure ground segments within 8 km of the ten airports,
callsign join within +/-20 min of take-off, off-block estimated from the last parked point /
first motion of the final taxi. Coverage 2026-07-15: EDDM 97%, LEBL 97%, EGLL 95%, EHAM 95%,
LSZH 92%, EDDF 43%, LEMD 33%, LFPG 8%, LIRF 0%, LTFM 0% (2025: LIRF 92%, EDDM 0-99% by month).
Seen parked before the final taxi: 35-50% of departures at covered airports.

Value: none measurable. On matched rows seen parked (10,419 rows), corr(v21 residual,
trace taxi - NM taxi) is within +/-0.11 at every airport and an in-sample linear correction
gains < 0.5s. Raw trace off-block vs BLOCK_TIME is 4-10x worse than NM's AOBT on the same
rows (median lag 5-20 min: the aircraft appears after it starts moving). On unmatched rows
the callsign join fails (NM callsign absent; FLIGHT_mvt is not broadcast) and estimates are
10-30x worse than the model. merry-quicksand's stacker gain (~5,100 fold units) is not
reproducible from trace timing at this resolution; a stand-geometry gate would be a multi-day
build against a signal correlating at ~0.05 with the error. Ingest stopped after 11 days;
features kept in data/external/adsb/features.parquet on the server. Closed.

## 25. Rome mixture variants (2026-09-18)

Against v21 OOF (shipped v25 mixture: 8h scope, LightGBM regressor, 309.41 Jan+Jul):

| variant | Jan+Jul | months better |
| --- | ---: | ---: |
| scope 12h | 309.26 | 10/12 |
| scope 20h | 311.58 | 8/12 (the regressor is capped at 12,000 and cannot produce day offsets; the band table is better there) |
| 8h scope, CatBoost GPU regressor (3,000 it, depth 8) | 309.09 | 10/12 |

12h + CatBoost combined running; scope stays below 12h.
12h + CatBoost: 309.094, identical to 8h + CatBoost (309.09). Ship 8h scope with the CatBoost
regressor as `lirf_moe_8h_cat` in v27 (-0.32 vs v25/v26 on Jan+Jul).

## 26. Where the remaining error is; second-stage stacker (2026-09-18)

Decomposition of the Jan+Jul 2025 OOF of the shipped chain (v21 base, CatBoost member at 0.5
total, 8h Rome mixture, LOBT clip; 308.35), `scratchpad/decompose.py` logic:

| slice | rows | share of scored SSE |
| --- | ---: | ---: |
| LFPG no-record rows | 880 | 31.7% - **two January rows** (y = 84,240 and 58,206; block time a day before MVT, no NM record, no other field) are 98% of the lane |
| LIRF no-record rows | 397 | 12.5% |
| LIRF matched | 26,131 | 11.1% |
| all other matched rows | ~313k | ~40% (LTFM 8.3, LFPG 6.9, EGLL 6.4, LEBL 4.1 ...) |
| rows with abs error < 1,200s | 343k | 40.7% |

Without the two CDG rows the fold score is ~255 against a board of 269: the ranking set carries
its own unpredictable day-offset rows and the whole-month CV was never pessimistic for the
reason section 8h assumed. 2026 no-record counts per airport are within 10% of 2025 (LIRF 384
vs 400), so the Rome layers keep their weight. The remaining error is spread thinly over
ordinary taxis at every airport (matched, y < 1,800s: RMSE 177 on 324k rows); no lane can give
10s any more.

**Second-stage stacker** (`scripts/experiment_stacker.py`), leave-one-month-out over the stored
component OOFs (v21 prediction / ordinary / gate, CatBoost d8 member, Rome mixture and
P(copy), LOBT expert, LIRF identity probability, four clock proxies, airport, lane); target =
residual to the chain:

| variant | Jan+Jul | months | -1 / -10 / -50 rows |
| --- | ---: | ---: | --- |
| per-lane linear least squares | 308.35 -> 306.57 | 7/12 | -0.9 / +1.6 / +3.4 - a handful of rows, NEGATIVE |
| LightGBM 300 rounds, all rows | 308.16 | 10/12 | +0.2 / +1.5 / +2.6 - NEGATIVE on robustness |
| LightGBM 800 rounds, all rows | 308.09 | 9/12 | +0.2 / +1.8 / +2.9 - NEGATIVE |
| LightGBM 300, applied to matched rows only | 307.68 | 12/12 | -0.65 / -0.57 / -0.39 |
| LightGBM 800, applied to matched rows only | **307.57 (-0.78)** | 12/12 | -0.76 / -0.67 / -0.46 |

The stacker's whole value is on matched rows; on the no-record lanes it chases the tail and
fails the drop-10 test. Reduced-input variant (only what the live chain exposes) below.

Reduced-input stacker (only what the live chain exposes without refactoring: pre-member
prediction, CatBoost member, blend, four clock proxies, airport; matched rows): 300 rounds
-0.14, 800 rounds -0.14, 12/12 but drop-50 flips positive. The value of the full stacker is in
the intermediate signals (ordinary prediction, gate probability, LOBT expert, Rome
probabilities).

**Flight identity in the Rome copy classifier (elegant-alligator's V6 winter correction):
NEGATIVE** (`scripts/experiment_lirf_identity_clf.py`, classifier refit LOMO with FLIGHT_mvt as a
categorical, stored regressor). All-season swap +0.90 on Jan+Jul, 4/12 months, AUC 0.927 ->
0.925; winter-only swap (Nov-Feb) +0.21, 2/12, winter AUC 0.9615 -> 0.9622 with no RMSE gain.
Same reason as section 11: 36-42% of July identities are unseen and the classifier already has
the operator, destination and stand encodings that carry the identity signal. Closed.

Mid-input stacker (reduced inputs + ordinary base, gate probability, LOBT expert; all of
which the live chain computes): 300 rounds -0.69, **800 rounds -0.82**, 12/12, drop
-1/-10/-50 = -0.81/-0.73/-0.53. Weight sweep on the OOF: 0.25 -0.43, 0.5 -0.70, 0.75 -0.84,
1.0 -0.82. Shipped as `prc.stacker` (`models/stacker_mid`, fitted on all 12 months of the
stored OOFs, matched rows) at **weight 0.5**: the live inputs are stronger models than the OOF
ones it was fitted on and its live shifts run ~30% wider (RMS 32.5s vs 24.6s OOF), so half
weight keeps -0.70 of the gain with half the exposure. In v28.

## 27. Second reference clock: CatBoost member on the LOBT residual (2026-09-18, GPU)

`scripts/experiment_catboost_gpu.py --target lobt`: same member as section 20 (5,000 it,
depth 8, lr 0.06, native identity categoricals) but target = `y - clip(MVT - LOBT, -3600,
7200)`, served on matched rows with LOBT. Member alone beats the v21 chain on matched rows in
12/12 months by 1-3s; raw blend blows up July (+14.6) on the single 87,002s matched copy row,
as in section 20, and the LOBT clip removes it. With the clip (`scripts/blend_members.py`):

| members | weights | Jan+Jul | 12-month | months | -1 / -10 / -50 |
| --- | --- | ---: | ---: | ---: | --- |
| AOBT d8 | 0.5 | 314.43 -> 312.45 | 255.97 | 12/12 | -1.9 / -1.8 / -1.4 |
| AOBT d8 + LOBT | 0.25 + 0.25 | **312.03** | 255.69 | 12/12 | -2.4 / -2.2 / -1.9 |
| AOBT d8 + LOBT | lstsq 0.27 + 0.37 | 312.01 | 255.57 | 12/12 | -2.4 / -2.2 / -1.8 |

-0.45 over the shipped member; the different reference clock gives different errors, which
is what the depth-10 and second-seed variants (section 21) did not. Full-data fit running as
`models/catmember_lobt` (`scripts/fit_catmember.py --anchor lobt`) for v29.

Board: v28 (stacker at 0.5) 269.25 -> 268.33 (-0.92; fold -0.70); v29 (+ LOBT member) 268.33 ->
267.87 (-0.46; fold -0.45). Both within the fold protocol's usual 1-3s tracking.

## 28. EOBT/IOBT clock members and the stacker refit: at the resolution limit (2026-09-18)

`--target eobt` (same recipe as section 27). Member alone beats the chain on matched rows in
12/12 months; blended with the LOBT clip (`scripts/blend_members.py`):

| members | Jan+Jul | months |
| --- | ---: | ---: |
| AOBT | 312.45 | 12/12 |
| AOBT + LOBT | 312.01 | 12/12 |
| AOBT + LOBT + EOBT | 311.90 | 12/12 |

The third clock adds 0.11; the second added 0.45. Refitting the stacker against the
three-member reference (`--members catgpu_d8 catgpu_lobt catgpu_eobt`, weight 0.75) adds a
further 0.10. Full v30 candidate vs the v29 chain on identical footing: **Jan+Jul 307.27 ->
307.05 (-0.22), 11/12 months, drop -0.21/-0.16/-0.07**.

**Board: v30 scored 267.96 against v29's 267.87, i.e. +0.09 where the fold said -0.22.** The
first sign reversal of the week. v28 (-0.70 fold / -0.92 board) and v29 (-0.45 / -0.46) tracked;
a -0.22 change does not. Treat ~0.3s as the resolution limit of the fold-to-board mapping and
stop shipping changes below it: they are coin flips that consume upload slots. Best score
remains v29 267.87 (the board keeps the best per team, so v30 costs nothing but the slot).

IOBT was cancelled mid-run (4 of 12 folds) to free the GPU after a concurrent-job OOM killed
the EOBT member fit; expected value ~0.05 on the above trend, not worth rerunning.

## 29. The broken-clock detector: diagnosis, and two dead branches (2026-09-18)

Motivated by section 26: the 14 day-offset rows (y > 12h) in scored months hold 35% of squared
error. Question asked: can a cross-airport, both-lane detector find them?

**Day offsets outside LIRF are undetectable. CLOSED.** All 34 in 2025: LIRF no-record 30, LIRF
matched 1, LFPG no-record 2, LSZH no-record 1. Base rate per no-record row: LIRF 2.02%, LFPG
0.053%, LSZH 0.044%, the other seven airports exactly 0. The LIRF ones sit at large schedule
gaps (median 15.1h) which is what the band table keys on; **the three non-LIRF ones have gaps of
0.26, 0.48 and 0.57h, i.e. they look like ordinary on-time flights on every field we hold.**
Hedging cannot pay either: predicting the mean over a candidate set saves only the fraction p of
the day-offset error, and p is 3 rows in ~21,000 non-LIRF no-record rows. Nothing to build.

**Beware the `copy` label off LIRF.** y == MVT - SCHED within 60s holds for 7-10% of *matched*
rows at every airport, because a flight that pushes back on time genuinely has that taxi. It is
only a broken clock when the gap is too wide for an on-time departure to explain it.

**Where the error actually is** (scored Jan+Jul, on the v29 chain, total SSE 32,519M):

| block | rows | scored SSE | share | RMSE if solved |
| --- | ---: | ---: | ---: | ---: |
| non-LIRF no-record | 4,976 | 11,880M | 36.5% | 244.80 (but 10,184M of it is the 2 LFPG rows above) |
| LIRF no-record, gap <= 8h (the mixture) | 381 | 2,821M | 8.7% | 293.64 |
| LIRF matched | 26,131 | 3,506M | 10.8% | 290.24 |
| LIRF no-record, gap > 8h (the band table) | 16 | 1,276M | 3.9% | 301.18 |

**The mixture is the real target.** Perfect copy/not-copy classification on its 381 scored rows
takes their SSE from 2,821M to 301M, worth ~12s of fold RMSE. Its error is split evenly between
copies called normal (123 rows, mean p 0.74) and normals called copies (258 rows, mean p 0.18).

**Retraining the classifier: all four variants NEGATIVE** (`scripts/experiment_copy_classifier.py`,
target SSE, shipped = 2,619M): no-record rows only 3,343M (matched LIRF rows are worth more as
data than they cost as label noise); weighting by the squared error at stake `(gap-reg)^2` 3,538M
and AUC 0.927 -> 0.899 (the weight spans six orders of magnitude and collapses the effective
sample); both together 2,982M; cross-fitted calibration months 3,538M. The ranking cannot be
improved this way. Do not retry weighted training without a bounded weight.

**Calibration is the live branch.** Out of fold the classifier is well calibrated at narrow gaps
and badly under-confident at wide ones, which is where the cost is:

| gap | mean p | actual copy rate | SSE at stake |
| --- | ---: | ---: | ---: |
| < 1h | 0.393 | 0.416 | 109M |
| 1-2h | 0.367 | 0.373 | 1,610M |
| 2-4h | 0.578 | 0.590 | 3,799M |
| 4-6h | 0.671 | **0.781** | 2,313M |
| 6-8h | 0.726 | **0.923** | 639M |

One isotonic fit on two months, pooled over all gaps, drags the rare wide-gap rows toward the
narrow-gap average. If p were the true conditional probability the expected SSE would be 2,380M
against 2,619M observed, so ~240M (about 1s of fold RMSE) is miscalibration rather than
irreducible Bernoulli variance. `scripts/experiment_copy_calibration.py` fits the isotonic per
gap stratum on cross-fitted scores from every training month.

**Calibration: NEGATIVE in every form. The detector direction is CLOSED.**
(`scripts/experiment_copy_calibration.py`, and a stored-OOF recalibration screen.) Against the
shipped target SSE of 2,619M: isotonic per gap stratum on cross-fitted scores, two strata
2,650M and three strata 2,722M (AUC does rise, 0.9273 -> 0.9296, but SSE and months both fall,
4/12, drop-10 +2.4); Platt on the shipped probability 2,684M; Platt + gap 2,710M; Platt + gap +
interaction 2,708M; a floor of 0.9x the band's own copy rate 4,416M (+8.3s). Even the
two-parameter minimum-variance form loses. **The apparent wide-gap under-confidence is sampling
noise, not bias**: it is measured on 64 and 13 rows a year, and correcting it out of fold costs
more than it removes. The shipped isotonic is at the information limit for these rows, which
carry only a movement time, a schedule time, a stand and an aircraft. The 12s oracle needs a
sharper *ranking*, and nothing available sharpens it.

Summary of the whole direction: 3 branches, 11 variants, all negative. Do not reopen without a
genuinely new observable for LIRF no-record rows.

## 30. LIRF matched rows: broad error, and the anchor is the reason (2026-09-18)

The one block whose error is *not* concentrated. Scored LIRF matched rows: 26,131 rows, RMSE
366.3, 10.8% of scored SSE; removing its five worst rows moves it only to 361.4. By target band:
y <= 30m 41.3% of the block's SSE (RMSE 248), 30-60m 28.7% (RMSE 670), 1-2h 28.7% (RMSE 1,842).

RMSE of the raw AOBT anchor (`MVT - AOBT_3`) against the target, scored matched rows, vs ours
(`scripts/analyse_error_anatomy.py`, part 6 - all ten airports):

| airport | anchor | ours | ours/anchor |
| --- | ---: | ---: | ---: |
| EDDM | 396.3 | 165.1 | 0.42 |
| EGLL | 514.6 | 228.7 | 0.44 |
| LTFM | 531.4 | 242.5 | 0.46 |
| EHAM | 351.4 | 167.9 | 0.48 |
| LSZH | 311.1 | 172.3 | 0.55 |
| LFPG | 428.3 | 240.1 | 0.56 |
| **LIRF** | **647.5** | **366.3** | **0.57** |
| EDDF | 294.7 | 177.8 | 0.60 |
| LEMD | 285.7 | 172.0 | 0.60 |
| LEBL | 341.4 | 216.3 | 0.63 |

**Correction (2026-09-19).** An earlier draft of this section compared LIRF's ratio against only
EDDM/EGLL/LTFM and concluded "we extract the least from Rome's anchor". That was cherry-picked
and is **wrong**: across all ten airports LIRF's 0.57 is mid-pack, between LFPG 0.56 and EDDF
0.60. What is true is narrower: **LIRF's anchor is the noisiest of the ten (647.5 against
285-531), and our error there is proportionate to it.** On the 312 scored LIRF matched rows with
y > 3,600 the anchor is wrong by a median of 3,358s (it reads ~1,000s where the truth is
3,600-6,700s), i.e. LIRF's AOBT sits systematically late against the block time the target is
built from; our chain already pulls RMSE 3,823 -> 1,835 on them. Not the `TAIL_THRESHOLD` clip:
only 11 LIRF matched rows in 2025 have an anchor above 3,600.

So there is **no evidence of Rome-specific modelling headroom** - only evidence that Rome's input
clock is worse. Combined with the old negatives (a LIRF specialist at -3.2s headline but
-0.9/+1.1 on drop-10/50, per-airport rounds and 4x capacity both flat), the prior on this block
should be low. The one thing that is new and still true is that its error is *broad*, so a
genuine gain would survive the drop gate.

Note against re-testing: a LIRF specialist was measured long ago at -3.2s headline but
-0.9 / +1.1 on the drop-10 / drop-50 gate, and per-airport rounds and 4x capacity were both flat
on LIRF (section 8 tables). What is new here is that the *residual* block is broad rather than
concentrated, so a genuine improvement would now pass the gate. No such improvement is in hand.

## 31. Independent δ audit and measurement infrastructure (2026-09-20)

The v29 OOF was rebuilt without fitting anything and cut by signed NM-anchor disagreement,
`δ = (MVT - AOBT_3) - y`, on matched January/July rows. This directly tests the claim in the
handoff that an early-anchor regime (`δ < -600s`) was hidden inside the airport cuts.

| δ regime | rows | SSE | share of matched SSE | share of all SSE |
| --- | ---: | ---: | ---: | ---: |
| early (< -600s) | 14,898 | 6,656M | **40.2%** | 20.5% |
| agree (within 600s) | 313,148 | 8,719M | 52.7% | 26.8% |
| late (> +600s) | 11,000 | 1,167M | 7.1% | 3.6% |

(Corrected 2026-09-20: this table first carried a run made with the schedule proxy rather than
`MVT - AOBT_3`, which put early at 6,743 rows / 2.1% and late at 151,108 rows / 55.0% and
contradicted the prose below it. The figures above are `scripts/analyse_error_anatomy.py`
part 1a on the shipped v29 OOF and match the per-airport split that follows.)

The rival's headline is reproduced almost exactly (14,898 rows / 40.2% versus 14,789 / 42%).
This is a cross-airport modelling/observability block, not a Rome-only block: LIRF carries
2,185M early SSE (62.3% of its matched SSE), LTFM 1,052M (39.0%), LFPG 934M (41.8%), and
LEBL 669M (50.2%). A perfect solution would remove 6,656M fold SSE, or about 33 RMSE seconds.
The prize gap requires about 4,041M board SSE, roughly 61% of this early block if its
prevalence and scale transfer.

This invalidates the old ADS-B closure's pooled-correlation rationale but not its coverage
constraint. The three largest early cells (LIRF, LFPG, LTFM) hold 4,171M / 6,656M = 63% of early
SSE and have 2026 ADS-B coverage of 0%, 8%, and 0% in the pilot. Thus a covered-airports-only
ADSB stack cannot plausibly reach the prize by itself; its remaining early SSE ceiling is 2,485M
(about 12 fold RMSE seconds before imperfect recovery). It may still be a 2--4 second optional
improvement, but is not evidence for a prize path.

`scripts/paired_bootstrap.py` now supplies a reproducible, paired row bootstrap for stored OOF
predictions. It samples the same rows from baseline and candidate predictions, reports pooled,
January, and July percentile intervals, and calls a gain measurable only when the 95% interval
excludes zero. A 500-replicate check on the shipped matched-only stacker (`stack` versus `ref` in
`stacker_mid_800_oof.parquet`) gives -0.817s [-1.150, -0.596] pooled; January -0.565
[-1.007, -0.340]; July -1.067 [-1.371, -0.753]. This validates the apparatus, not a new model.

The historical transfer ledger is in `docs/submissions.md`. It is descriptive only: organiser
scores are never used to choose a feature, threshold, or blend.

**Conditional proposal, not built.** Treat the early regime as a matched-lane reporting
mechanism rather than adding capacity: train a cross-fitted all-airport gate for
`P(AOBT_3 proxy - y < -600 | X)`, then a separate conditional residual model for
`E[y - AOBT_3 proxy | early, X]`, and add their product only on matched rows. Its inputs must be
limited to score-time observables already in the chain: the four clock proxies and their
disagreements, base/member/stacker component predictions, airport, schedule deviation, context,
weather and sequencing. Select every cap, calibration, and blend using only the ten training
months of each outer fold. Evaluate unfiltered leave-one-month-out OOF, January and July
separately, report early/agree/late SSE, run the paired bootstrap and drop-10/drop-50 checks.
This directly tests whether the current clock members/stacker have information that the rival's
pooled single LightGBM did not; per-airport rounds and more capacity are explicitly not part of
the proposal.

**ADS-B decision.** Do not spend multiple days reopening it for the prize objective. The old
pooled-correlation screen was insufficient, but 63% of the early error is at airports with
effectively no 2026 coverage. The covered subset cannot remove the required 4,041M SSE even with
an oracle, and the rival's reported recovery is conditional on coverage. At most, a strictly
time-boxed one-day feasibility screen is optional after the internal early-regime mechanism; it
is not the main path to 245.02.

## 32. Early-anchor conditional residual experts (2026-09-20): NEGATIVE

The conditional proposal in section 31 was implemented in
`scripts/experiment_early_anchor.py`, all twelve leave-one-month-out folds, scored on every
departure.  The target was the residual to the reconstructed v29 chain, not the raw anchor:

`P(early | X) E[y - v29 | early, X] + (1 - P(early | X)) E[y - v29 | normal, X]`.

The gate uses score-time categorical, clock, context, weather and sequencing features.  It
finds the regime well (stored screen AUC 0.961), but this is not enough for a net correction.
Two fixed screens were run; none used organiser scores, and all output artifacts are retained
under `data/interim/early_anchor_*`.

| screen | extra inputs to residual experts | Jan+Jul v29 -> candidate | early | agree | paired bootstrap |
| --- | --- | ---: | ---: | ---: | --- |
| residual mixture | source features only | 307.274 -> 308.276 (+1.002s) | -2.58s | +2.83s | +1.002, [ +0.732, +1.342 ] |
| residual mixture + chain signals | also v21, d8/LOBT member, stack delta/reference, MoE and v29 prediction | 307.274 -> 308.538 (+1.264s) | -9.39s | +4.70s | +1.264, [ +0.892, +1.732 ] |

Both losses are measurable in January and July separately.  The richer screen does make the
early rows better (SSE -186M) and late rows better (-45M), but it harms the much larger agree
population (+498M), so it cannot pass the unconditional RMSE objective.  The prior raw-anchor
severity formulation was much worse (+60.4s) because it double-counted an existing v29
correction; it is retained only as a diagnostic artifact.

**Decision: close post-hoc early-anchor residual experts with the current observables.** The
remaining early SSE is a genuine and large error block, but a well-ranked gate does not supply a
usable severity estimate.  Do not tune its probability threshold, cap, blend weight or airport
exceptions on these scored-month results.  Reopen only with a new score-time observable with
material coverage in LIRF, LFPG and LTFM (for example a verified trajectory source), not another
variant of the same gate.

## 33. Missing-record identity persistence (2026-09-20): real pattern, no robust gain

`scripts/analyse_missing_identity.py` separates two questions.  First, flight identity strongly
predicts whether the Network Manager join will be absent: among exact flight designators with at
least ten 2025 observations and two ranking observations, the 2025-versus-2026 missing-rate
correlation is **0.740**.  Of the 5,290 ranking departures with no NM record, **60.9%** use an
exact designator that was also missing at least once in 2025.  Aircraft type and airline show
similar coverage patterns.  This is operational/source structure, not random row loss.

That does not by itself help prediction because absence is directly observable at score time:
LOBT, IOBT, EOBT and AOBT are null together.  The useful question is whether identity predicts
the target *conditional on absence*.  A fixed leave-one-month-out screen used a normalized
`airline + numeric flight number + ADEP + ADES` key, K=50 residual shrinkage (the existing target
encoding prior), and applied the correction only to no-record rows:

| population/check | v29 -> candidate / delta |
| --- | ---: |
| scored January+July, all rows | 307.274 -> 307.207 (**-0.066s**) |
| scored no-record lane | 1,724.400 -> 1,723.642 (-0.758s) |
| remove most-helped 1 / 10 / 50 rows | **+0.009 / +0.047 / +0.081s** |

The gain is smaller than the 0.3-second resolution limit and reverses after removing one row.
Applying the same correction to all rows loses 0.238s and improves only 1/12 months.  Transfer
coverage is also seasonal: normalized airline+number covers 94.1% of ranking rows, but adding
route drops coverage to 97.0% in January and 69.8% in July.

The illustrative LFPG outlier confirms the limitation rather than creating the conclusion:
EJU983W appears 47 times in 2025; 46 rows have an NM record and ordinary targets of 359--1,317s,
while its sole no-record row carries the 84,240s previous-day label.  EJU has 492 no-record rows
and A319 has 879; this is the only strict near-24h case in either group.  Broad airline and
aircraft patterns are already represented by v29's smoothed encodings.  Exact flight/callsign
features remain excluded because their July unseen rates are 36.3%/42.3% (section 11).

**Decision:** identity patterns are useful for expected ordinary taxi duration and NM coverage,
but not for robustly detecting the costly day-offset/reporting mechanism.  Do not add a direct
flight-number correction to v29.

## 34. The stand→runway physical floor as a late-AOBT detector: NEGATIVE (2026-09-20)

Idea (owner's): the taxi from a given stand to a given runway has a physical minimum. If the
anchor-implied taxi `MVT - AOBT_3` falls below that minimum, the AOBT must have been recorded
late — a hard constraint rather than the statistical gate that failed in section 32. Inputs are
all score-time: `MVT_TIME_UTC_mvt`, `STAND_mvt`, `RUNWAY_mvt` (all 100% populated), with the
floor itself learned from training months only.

Floor = 2nd percentile of true `y` per `(ADEP, STAND, RUNWAY)`, groups with n >= 30, computed
leave-one-month-out so no row sees its own month. 330,516 of the scored matched rows get one.

**As a detector it is strong.** In-sample over all 12 months, 34,769 rows (1.72%) violate:

| | |
| --- | ---: |
| base rate P(early) | 0.0367 |
| P(early \| violation) | **0.6136** (16.7x lift) |
| P(early \| no violation) | 0.0266 |
| share of all early rows caught | 28.8% (21,334 of 74,042) |
| corr(violation margin, anchor shortfall `y - anchor`) | **0.658** |

**Against the residual v29 already leaves, it is worth nothing.** Scored months, out-of-fold
floors:

| cohort | rows | mean residual | RMSE | SSE |
| --- | ---: | ---: | ---: | ---: |
| violating | 6,434 | **+14.9s** | 359.2 | 830M |
| not violating | 324,082 | -3.9s | 215.3 | 15,021M |

`corr(violation margin, v29 residual) = 0.008`, and the residual by margin band is flat
(+11, +33, +16, +8, +1, +54 across bands from 0-60s to >1200s). The violating rows are higher
*variance* but essentially unbiased: the chain already prices the floor, so there is no
conditional-mean correction left to apply and RMSE is already at its minimum for this signal.

This is section 8's "stand x runway explains 39.5% of taxi variance and adds nothing" in a new
form. That note concerned the conditional *mean* and did not logically exclude a *floor* test;
the floor test is now run and the model turns out to have absorbed that too. Closed. Do not
re-test with a different quantile, distance-based geometry, or an OurAirports runway-threshold
path length: the failure is not that the floor is imprecise, it is that the residual it would
correct is already zero in the mean.

## 35. Stand->runway geometry and takeoff-time queues: bounded and not worth building (2026-09-20)

Owner's proposal: build the departure queue from takeoff times, and add stand->runway distance
from a public geometric source (OSM `aeroway` is ODbL and would be permitted, like ADSB.lol),
optionally with ADS-B speed vectors.

**Queues are built and their count form is already closed.** `prc.sequencing` reconstructs the
sequence ahead of each departure from take-off times, and `q_apt_at_takeoff` / `q_rwy_at_takeoff`
are in the matrix. Its docstring records the measurement: raw counts of nearby departures explain
essentially none of the model residual because `MVT - AOBT` already encodes congestion, and
`seq_prev_gap_s` was dropped at +0.003. Only the *mix* ahead (wake categories) survived.

**Geometry is bounded by how few pairs are thin.** Out-of-fold support per `(ADEP, STAND, RUNWAY)`
on scored matched rows; a distance proxy can only beat the learned per-pair history where that
history is thin:

| out-of-fold support | rows | % rows | % matched SSE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| unseen | 430 | 0.1% | 0.3% | 358.9 |
| 1-4 | 1,039 | 0.3% | 0.6% | 317.1 |
| 5-9 | 1,165 | 0.3% | 0.6% | 283.3 |
| 10-29 | 5,896 | 1.7% | 2.6% | 272.4 |
| 30-99 | 23,466 | 6.9% | 9.1% | 253.5 |
| 100+ | 307,050 | 90.6% | 86.7% | 216.1 |

Pairs with support < 30 are 8,530 rows, 2.52% of scored matched rows, carrying 4.2% of matched
SSE. **An oracle on every one of them** gives matched RMSE 220.9 -> 216.2, about 3.3s of fold,
~264.6 on the board at the ~1x matched transfer — still 19.6s short of the prize. A distance
feature captures some fraction of an oracle, plausibly ~20%, i.e. 0.3-0.7s: at or below the
resolution limit, against a multi-day build whose hard part is joining NM stand identifiers
(`B41`) to OSM `ref` tags across ten airports — least reliable precisely on the rare pairs it
would serve. ADS-B speed vectors are the same coverage problem as section 24/31.

Not closed on evidence of failure, closed on a bound: the ceiling is too low to justify the
build inside the remaining deadline. Revisit only if the deadline moves.

## 36. The ordinary matched bulk: the residual is clock noise, not unmodelled physics (2026-09-20)

Owner's question, correctly pressed: forget the outliers, can physics improve the *ordinary*
flight across the board? This bounds it. The agree cohort (|anchor - y| <= 600s) is 313,148
rows, 92.4% of scored matched rows, RMSE 166.86, 8,719M SSE = 26.8% of all fold SSE. It is the
largest single block by size, and a 10% RMSE reduction there would be worth ~8s of fold — more
than any concentrated block. It deserved the check.

**The chain is already doing substantial work here:** raw anchor `MVT - AOBT_3` alone scores
234.07 on these rows against v29's 166.86, i.e. 67s of correction, SSE 17,157M -> 8,719M.

**What remains is the anchor's own recording error,** not taxi physics: `y - (MVT - AOBT_3)`
has mean -36.0s, sd 231.3s, median absolute 147s, and only 21.0% of rows within 60s. For a
matched row the physical taxi is *already observed* by the two timestamps; the unknown is how
badly the off-block stamp misses the true one. Physical descriptors re-describe the observed
quantity and are flat against the residual: `q_rwy_at_takeoff` +0.024, `q_apt_at_takeoff`
+0.034, `seq_departures_30m_runway` +0.015, the anchor itself +0.064.

**And the residual is conditionally unbiased,** so there is no recording-entity correction
either. Mean residual -15.03s overall; per-group mean bias with n >= 50, and the SSE that
subtracting each group's own mean would remove *in sample*:

| grouping | groups | median \|bias\| | p95 | max removable |
| --- | ---: | ---: | ---: | ---: |
| airport | 10 | 14.7s | 52.4s | 185M (2.1%) |
| airport x hour | 211 | 15.6s | 77.5s | 222M (2.5%) |
| operator | 199 | 18.0s | 63.7s | 154M (1.8%) |
| airport x operator | 628 | 19.9s | 85.6s | 250M (2.9%) |
| airport x stand | 1,413 | 17.2s | 71.5s | 265M (3.0%) |

The best is 3.0% of cohort SSE = 0.8% of total = ~1.2s of fold as an *in-sample* oracle, before
out-of-fold shrinkage and before the 24 existing target encodings that already capture most of
it. Closing 23s from this cohort would need 166.86 -> ~137, an 18% reduction; the largest
conditional structure anyone can find in it is 3%, with hindsight.

Conclusion: the bulk residual is measurement noise in the NM off-block stamp, irreducible from
our inputs. This is the general form of the four specific nulls (stand x runway means, section
8; queue counts and `seq_prev_gap_s`, `prc.sequencing`; the physical floor, section 34;
geometry, section 35) — all of them re-describe a taxi we already observe directly. Physics
would matter on the no-record lane, which has no off-block at all.

**Correction (same day, section 40): the clause that originally ended this paragraph — "and
lacks reliable stand/runway besides" — is false.** `STAND_mvt` and `RUNWAY_mvt` are 100% present
on no-record rows. That error was the stated reason for dismissing physics on the one lane where
it applies; see section 40 for what the lane is actually worth.

## 37. Correction: the 2026 ADS-B coverage figures are a single day and do not support section 24's closure (2026-09-20)

Section 24 closed ADS-B partly on 2026 coverage "EDDM 97%, LEBL 97%, EGLL 95%, EHAM 95%,
LSZH 92%, EDDF 43%, LEMD 33%, LFPG 8%, LIRF 0%, LTFM 0%". Sections 31 and 35 then reused
"LIRF 0%, LFPG 8%, LTFM 0%" as the reason the early block is unreachable. **That figure is one
day.** `data/external/adsb/2026-07-15/pilot_report.csv` is the only 2026 date ever ingested.

Set beside the 2025 report for the same calendar day, the instability is decisive:

| airport | 2025-07-15 joined % | 2026-07-15 joined % |
| --- | ---: | ---: |
| EDDM | 0.0 | 97.0 |
| EGLL | 19.6 | 95.2 |
| EDDF | 81.5 | 42.8 |
| LEMD | 23.5 | 32.7 |
| LFPG | 2.7 | 8.3 |
| **LIRF** | **91.8** | **0.0** |
| LTFM | 0.0 | 0.1 |

Rome at 91.8% one year and 0.0% the next, Munich 0.0% then 97.0%, Heathrow 19.6% then 95.2%:
these are feed/ingest artefacts (which receivers were in the ADSB.lol aggregation, which archive
part was fetched), not changes in ground coverage at major European airports. Section 24's own
parenthesis already said "EDDM 0-99% by month"; the single-day headline was quoted and the
variance ignored. merry-quicksand, measuring over the whole 2026 Jan+Jul table rather than one
day, reports LIRF 0.20 and LFPG 0.12 — disagreeing with ours at the airport that matters most.

**What survives of section 24:** trace-derived off-block is far worse than NM's AOBT on the same
rows (2025-07-15: LIRF 1,864s vs AOBT 791s; EGLL 3,128 vs 433; EDDF 1,841 vs 221). Using the
trace *as a clock* is settled negative.

**What does not survive:** (a) the pooled-correlation screen, already shown in the 2026-09-20
handoff to be the wrong instrument — the rival's gain comes from stand geometry on the early
subset, which a pooled correlation over all seen-parked rows dilutes to ~0.05; and (b) the
coverage argument, which rests on n = 1.

ADS-B is therefore **not closed**. It is unresolved, and the cheap decisive step is a coverage
census over a sample of 2026 January and July dates before any feature build. Sections 31 and 35
should be read with their coverage claims struck.

## 38. ADS-B is a systematically late off-block observer: the transponder does not start at pushback (2026-09-20)

Owner's point: the transponder is often not transmitting at off-block — crews switch it on at
pushback or on entering the manoeuvring area — so ADS-B cannot see the moment the aircraft
leaves the stand. Measured on 4,953 joined 2025 departures where we hold the true
`BLOCK_TIME_UTC_mvt`, against the *true* off-block rather than against `AOBT` as section 24 did:

| off-block estimator | median error | RMSE | share after the true off-block |
| --- | ---: | ---: | ---: |
| ADS-B first motion of final taxi | **+495s** | 4,826s | 79.2% |
| ADS-B last stationary point | +600s | 2,966s | 91.5% |
| ADS-B restricted to `seen_parked` rows | +520s | 1,658s | — |
| **NM `AOBT_3`** | **-35s** | **449s** | — |

ADS-B first observes motion a median of eight minutes *after* the aircraft actually left the
stand, in one direction (79-92% late). Even on rows we believe were observed parked, the
estimate is 3.7x worse than the NM clock; on the rest, 13x. Stand shadowing at gates compounds
the transponder-timing effect. This is a physical floor that coverage does not fix, and it is
independent of both the licence and the prediction-time questions.

**Correction (same day, from section 39): these RMSE figures are contaminated.** About 20% of
rows have zero detected parked points, and on those the "first motion" picked up is the aircraft's
*inbound* taxi from the previous leg, not the departure. That inflates the RMSE columns badly
(the 4,826s figure especially); the medians are the robust statistic and the directional finding —
ADS-B sees motion well after the true off-block — survives, because it rests on the 80% of rows
where the dwell was detected.

**Caveat on our own extraction.** `scripts/adsb/pilot_day.py` estimates off-block with a simple
last-parked/first-motion heuristic. merry-quicksand's unpublished v2 used stand-polygon geometry
and dwell-to-centroid distance and reported dwell ending +24s from `AOBT_3` on well-covered rows,
far better than our +520s. Part of the gap is our extractor. But the transponder floor binds any
extractor: where nothing is transmitted from the stand, better geometry has nothing to read.

**What this leaves.** ADS-B cannot be a clock. It could still be a *covariate* — e.g. "already
moving 200m from the stand while NM records the aircraft as parked" as evidence that the record
is late — which is a far weaker claim than the ~9s the rival's ledger attributes to it, and it
would still have to earn its place against the 40.2% early block (section 31) under the
prediction-time ruling (section 37, unresolved).


## 39. The ADS-B lower bound on taxi time: sound logic, untestable with our extractor (2026-09-20)

Owner's proposal, and the best form of the ADS-B idea so far: never mind the off-block. If the
aircraft is *seen moving* at time T, then it was off-block at or before T, so
`y = MVT - BLOCK_TIME >= MVT - T`. A lower bound, immune to the transponder-lateness of section
38 — a late transponder only loosens the bound, it cannot invalidate it. And it bites exactly
where we need it: if `T < AOBT_3`, the aircraft was moving while NM recorded it parked, which
*proves* the anchor is late without inferring anything.

Tested on 4,661 joined 2025 rows carrying `AOBT_3`, an ADS-B motion time and a v29 OOF
prediction. It fails, on our extractor:

| | |
| --- | ---: |
| rows where `t_first_motion < aobt_s` ("proves late") | 1,007 (21.6%) |
| of those, bound actually valid (`bound <= y`) | **6.5%** |
| implied taxi on those rows, mean / median | 7,943s / 5,498s |
| true `y` on those rows, mean / median | 960s / 893s |
| `max(v29, bound)` on all joined rows | RMSE 220.2 -> 4,762.7 |

**Cause, confirmed not assumed.** The firing rows have `seen_parked` 7.1% and median `n_parked`
of **0** — no stationary period was ever detected — against 45.2% and 2 for the rest. Median
trace span from first ground point to take-off is 1.61h in the firing subset and 1.35h overall:
our segments cover the entire turnaround, so "first motion" is the *inbound* taxi of the previous
leg. The implied 5,498s median is a turnaround, not a taxi.

**Status: untested, not refuted.** The bound is logically valid; what it needs is the motion that
*ends the stand dwell*, which requires stand-polygon / dwell segmentation (merry-quicksand's
`dwell_to_centroid_m` machinery, unpublished). That is the multi-day extractor build, and it sits
behind the unresolved prediction-time ruling of section 37. Do not quote section 38's RMSE
columns; they carry the same contamination.


## 40. The ordinary-taxi no-record block: the one lane where physics applies, and it is worth 15s (2026-09-20)

Sections 34-36 established that physical features cannot help *matched* rows, because
`MVT - AOBT` already observes the taxi and what remains is clock-recording noise. That argument
does not extend to no-record rows, which carry no off-block at all: there the model predicts an
actual taxi from scratch, and congestion is *not* already encoded. Sizing that lane on scored
months:

| class | rows | SSE | % of all fold SSE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| day-offset | 13 | 11,460M | 35.2% | 29,691 |
| schedule-copy | 253 | 1,417M | 4.4% | 2,367 |
| **ordinary taxi** | **5,107** | **3,099M** | **9.5%** | **779** |

The broken-clock classes are closed (sections 29, 31). The ordinary-taxi remainder is 5,107 rows
across all ten airports — broad, so a gain would pass the drop-10/50 gate:

- **oracle: fold 307.27 -> 292.26 (-15.0s)**
- **a 20% RMSE improvement: -> 301.96 (-5.3s)**

By airport: LIRF 1,412M (258 rows, RMSE 2,339 — concentrated), EGLL 569M, EHAM 242M, LTFM 206M,
LFPG 201M, LSZH 195M, EDDF 127M, EDDM 87M, LEBL 44M, LEMD 17M.

**Availability on these rows, measured, correcting section 24 and section 36:**

| field | no-record | matched |
| --- | ---: | ---: |
| `FLIGHT_mvt` | **99.7%** | 100% |
| `STAND_mvt`, `RUNWAY_mvt` | **100%** | 100% |
| `CALLSIGN_flt`, `AIRCRAFT_OPERATOR_flt` | 0.0% | 98.9% |

Section 24 closed ADS-B on unmatched rows because "the callsign join fails (NM callsign absent)".
True as stated, but not a wall: `FLIGHT_mvt` is present, and the flight-number -> callsign mapping
is learnable from the 2,062,454 matched rows that carry both. 95.0% of the 44,861 distinct flight
numbers map to exactly one callsign, and **74.6%** of no-record flight numbers resolve through it.

**Two routes, with different rule exposure:**

1. *Target-flight ADS-B* (trace of the scored aircraft, via reconstructed callsign) — blocked by
   the unresolved prediction-time question, section 37.
2. **Surrounding-traffic surface congestion** — how many aircraft are moving on the surface, and
   holding near the runway, at this flight's take-off time, built from *other* aircraft's traces.
   **This requires no join to the scored flight**, so it is airport-day context of the kind
   already permitted (cf. the EUROCONTROL ATFM ruling) and corresponds to option 2 of the
   section 37 question. It is also exactly the quantity these rows lack.

Route 2 is the strongest untested direction in this project: it is large (9.5% of SSE), broad,
mechanistically motivated, applies where the closed physics arguments do not, and is the least
exposed to the pending ruling. Not yet built; nothing here is a measured gain.

## 41. January 2026 ADS-B coverage census, all 31 days: section 24's figure was an artefact (2026-09-20)

Section 37 flagged that the 2026 coverage numbers rested on a single day. All 31 days of January
2026 are now ingested (152,719 departures). Joined percentage per airport:

| airport | mean | median | min-max | sd | days at zero |
| --- | ---: | ---: | --- | ---: | ---: |
| EDDM | 97.0 | 97.6 | 83.0-98.9 | 2.8 | 0 |
| LEMD | 96.1 | 96.8 | 76.0-97.9 | 3.8 | 0 |
| LSZH | 94.0 | 95.0 | 81.8-97.7 | 3.9 | 0 |
| LEBL | 92.1 | 95.3 | 61.1-98.2 | 8.3 | 0 |
| EHAM | 90.9 | 96.3 | 57.7-98.0 | 10.9 | 0 |
| EGLL | 76.1 | 74.4 | 46.4-92.6 | 11.2 | 0 |
| **LIRF** | **66.0** | 67.3 | 48.6-83.0 | 9.4 | 0 |
| EDDF | 56.7 | 60.9 | 9.3-73.4 | 15.9 | 0 |
| LFPG | 6.9 | 5.1 | 1.8-17.5 | 4.6 | 0 |
| LTFM | 0.0 | 0.0 | 0.0-0.0 | 0.0 | **31** |

Weighted mean 61.9%.

**Rome is covered at 66% and never falls below 48.6% on any day of the month.** Section 24's
"LIRF 0%", from 2026-07-15 alone, is an artefact. LFPG (6.9% against the quoted 8.3%) and LTFM
(0.0% on all 31 days — genuine absence, not a feed glitch) were quoted correctly.

**Reachable share of the early block recomputed** (section 31's 6,656M, weighted by these
coverage rates): LIRF 1,442M, LEBL 616M, EGLL 446M, EDDM 287M, LEMD 270M, EDDF 204M, LSZH 203M,
EHAM 70M, LFPG 64M, LTFM 0 = **~3,602M, 54% reachable**, against the 37% implied by the claim in
sections 31 and 35. Those two sections' coverage arguments are now superseded by this one.

**Caveat, resolvable in hours:** this is January. The anomalous single day was in *July*, and
July 2026 is the next 31 days of the ingest. If Rome is genuinely dark in July, half the scored
set is uncovered there. Do not treat the 54% as settled until the July census is in.

Coverage is necessary, not sufficient: section 38 showed the trace is a poor off-block clock, so
any value must come from geometry and surface congestion rather than timing, and the
prediction-time ruling (section 37) is still unresolved for target-flight features.

## 42. ADS-B archive availability for 2025: 22 days missing, five of them in January (2026-09-21)

Enumerated the `adsblol/globe_history_2025` releases directly (the repo publishes three variants
per day — `prod-0`, `staging-0`, `mlatonly-0`; only `prod-0` is fetched by
`scripts/adsb/fetch_day.sh`). **343 of 365 prod-0 releases carry assets.** Missing:

| month | available | missing days |
| --- | --- | --- |
| January | **26/31** | 01, 11, 18, 19, 21 |
| May | 27/31 | 28-31 |
| June | 19/30 | 01-11 |
| December | 29/31 | 25, 31 |
| **July** | **31/31** | — |
| Feb, Mar, Apr, Aug, Sep, Oct, Nov | complete | — |

2025-05-28 to 2025-06-11 is a genuine 15-day outage in the upstream archive, confirmed by direct
tag lookup returning "Not Found" with the API rate limit unexhausted (54/60 remaining). The
ingest's `fetch failed -> continue` path handled all 22 correctly.

**Correction to an earlier figure in this session.** A first survey reported "359 of 365, seven
missing days". Its regex was unanchored and matched the `-staging-0` and `-mlatonly-0` variants
as though they were `prod-0`, inflating availability. The corrected figure is 343/365 and 22
missing days, verified both by anchored enumeration and empirically by the ingest run itself.

**Consequence for validation.** Of the two calendar-matched scored months, July 2025 is complete
(31/31) and January 2025 is 26/31 — so January's ADS-B features will be built on 84% of its days
while July's use 100%. Any January-versus-July comparison of an ADS-B-derived feature must
account for that asymmetry before attributing a difference to the feature. The 15-day May/June
outage reduces LOMO fold coverage in two non-scored months. 2026 is unaffected: all 62 ranking
days are present with traces.

## 43. ADS-B ingest complete, and two defects that made failures look like successes (2026-09-22)

**Data state: complete.** 2026 January and July at **62/62** days; 2025 at **343/343** of the
days that exist upstream (22 days are genuinely absent, section 42). Scored-month asymmetry
stands: January 2025 has 26 days, July 2025 has 31. ~1.3TB of raw traces retained on the HDD at
`~/Data/adsb`, symlinked from `data/external/adsb`, so a different feature extraction can be
rebuilt without re-downloading.

**Throughput: the ingest was 8x slower than necessary for two compounding reasons.** GitHub's
CDN throttles per connection to ~11 MB/s, and the run used a single connection (5 MB/s under
contention). Parallelising to six workers then hit a second wall immediately: the unauthenticated
GitHub API allows **60 requests/hour** and `fetch_day.sh` spends one per day discovering asset
names, so six workers exhausted it in seconds and *every* day reported "no assets found" — the
same message a genuinely missing day produces. The single worker never hit it because 10 min/day
is 6 calls/hour. Fixed by dropping the API entirely: asset URLs are constructible from the date
(`v<Y>.<M>.<D>-planes-readsb-prod-0.tar.{aa,ab,...}`), so `/tmp/fetch_direct.sh` builds them and
stops when a part 404s, which also detects missing days correctly in 0.33s. Result: **5 -> 42
MB/s**, and ~30 hours of remaining work finished in about three.

**Two defects that reported success on failure, both worth keeping fixed:**

1. `scripts/adsb/pilot_day.py` caught `(OSError, ValueError, KeyError)` around the per-file gzip
   read, intending to skip unreadable trace files. **`zlib.error` does not inherit from
   `OSError`**, so a single corrupt gzip member inside an upstream archive propagated out of the
   `ProcessPoolExecutor` and discarded the whole day. 2025-10-15 failed this way twice, on two
   independent downloads, so the corruption is upstream. Fixed by adding `zlib.error` to the
   tuple; the day then yielded 2,976 segments.
2. `scripts/adsb/ingest_days.sh` printed "segments kept" unconditionally after extraction,
   whether or not a parquet was produced. Three days logged success with no segments file and
   were only caught by inspecting directories. Fixed with an explicit existence check.

**Operational note added to the hard-won list:** stopping a worker by killing its shell leaves
its `fetch_day`/`curl` children running, and those orphans corrupt day directories that a
restarted worker then treats as present. Kill the process *group* (`setsid` at launch, record the
pgid, `kill -- -PGID`). And `pkill -f` inside an `ssh` command matches the ssh invocation itself
and kills the session — this happened twice more today despite already being documented.

## 44. July 2026 census: Rome is dark in July; a dwell-segmenting extractor replaces pilot_day (2026-09-24)

**July 2026 coverage, all 31 days** (`report_day.py --no-labels`, same join as section 41):

| airport | Jan 2026 mean | **Jul 2026 mean** | Jul min-max | Jul days at zero |
| --- | ---: | ---: | --- | ---: |
| EGLL | 76.1 | **96.8** | 92.2-99.0 | 0 |
| EDDM | 97.0 | **96.5** | 92.1-98.4 | 0 |
| EHAM | 90.9 | **95.8** | 93.0-97.3 | 0 |
| LEBL | 92.1 | **94.7** | 86.3-98.8 | 0 |
| LSZH | 94.0 | **93.6** | 90.3-96.6 | 0 |
| LEMD | 96.1 | **59.8** | 32.7-83.2 | 0 |
| EDDF | 56.7 | **42.0** | 19.2-52.7 | 0 |
| LFPG | 6.9 | **6.4** | 0.4-13.4 | 2 |
| **LIRF** | 66.0 | **0.1** | 0.0-2.0 | **29** |
| LTFM | 0.0 | **0.0** | 0.0-0.2 | 31 |

Weighted mean 55.6% (January 61.9%). Section 41's caveat is realised: **Rome is covered in January
2026 and absent in July 2026**, where July 2025 read 91.8%. The 2026-07-15 reading that section 24
rested on was representative of July after all; what it could not show was January. So Rome's
ADS-B reach is half the scored Rome rows, and any Rome feature trained on 2025 (covered in July)
meets NaN at serve time in July 2026. Section 41's "54% of the early block reachable" should be
read with LIRF halved: ~2,880M, ~43%.

**Extractor, stage A (`scripts/adsb/ground_points.py`).** Keeps every ground point near the ten
airports (thinned to 10s) and segments each ground run on its *longest* stop of at least 300s
(silent gaps with < 60m displacement count as stationary; the stop must stay within 120m overall,
so a creeping queue is not taken for the stand). Points before the dwell on a run that began
airborne are taxi-in, after it on a run that ends airborne are taxi-out. Checked against labels
on 2025-07-15, 925 departures joined with a dwell:

| | median | RMSE | within 2 min |
| --- | ---: | ---: | ---: |
| dwell end - BLOCK_TIME | **+27s** | 1,683 | 57.8% |
| (section 38, pilot_day first motion) | +495s | — | — |
| AOBT - BLOCK_TIME, same rows | -58s | 392 | — |

The inbound-taxi contamination of section 39 is gone at the median; the tail is still heavy
(EDDF RMSE 4,190: some runs segment on the wrong stop). Taxi measured from the dwell end is
<= y + 60s on 94.5% of rows, so the lower-bound reading of section 39 holds on the new extractor.
ADS-B take-off (last ground point) is a median 32s before MVT (5-95%: -214s to +1s). 68s/day,
~40s on winter days; all 405 days run by `scripts/adsb/extract_ground_all.sh`.

**Stage B (`scripts/adsb/surface_features.py`).** Seven airport-surface features per departure from
*other* aircraft only: the scored flight is excluded by time (runs taking off in [MVT-420s,
MVT+60s] are dropped from that row's features), never joined by identity, so this sits under
option 2 of the section 37 question. Counts are divided by airport-day coverage (ADS-B take-offs
/ NM departures) and left NaN below 10%. Screened by `scripts/adsb/screen_surface.py` against the
v29 residual on the no-record ordinary lane, decision on the ten non-scored months, criteria
fixed in its docstring before the first run. Result: section 45.

`scripts/adsb/fetch_direct.sh` (the API-free fetcher of section 43) is now in the repo; it had
lived only in `/tmp` on a GPU host.

## 45. ADS-B surface congestion on the no-record ordinary lane: FAIL (2026-09-24)

`scripts/adsb/screen_surface.py`, all 405 days extracted, criteria as fixed in section 44 before
the first run. Lane 21,020 rows over 2025; sx features present on 72% outside LTFM (LTFM 0%,
LFPG 33%). Decision on the ten non-scored months:

| month | rows | v29 | corrector, no sx | corrector, with sx | with vs without |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1,016 | 1,448.3 | 1,756.3 | 1,714.5 | -4.70% |
| 3 | 978 | 520.7 | 511.2 | 509.1 | -0.83% |
| 4 | 974 | 501.2 | 412.9 | 428.2 | +7.56% |
| 5 | 1,491 | 868.1 | 839.9 | 837.9 | -0.49% |
| 6 | 2,330 | 600.0 | 519.4 | 531.1 | +4.57% |
| 8 | 2,177 | 741.1 | 697.6 | 705.7 | +2.33% |
| 9 | 2,334 | 595.8 | 511.0 | 502.3 | -3.39% |
| 10 | 1,805 | 617.6 | 523.9 | 511.7 | -4.57% |
| 11 | 1,268 | 508.9 | 507.2 | 485.2 | -8.46% |
| 12 | 1,540 | 630.3 | 635.5 | 611.2 | -7.51% |
| pooled | 15,913 | 720.3 | 724.3 | 715.3 | **-2.47%** |

Wins 7/10; drop-10 +3.47%, drop-50 +3.89% (positive). **Fails the 5% pooled criterion.** Against
v29 itself the with-sx corrector cuts lane SSE by 1.4%; scaled to the scored months' lane
(3,099M of 32,519M) that is ~0.2s of fold RMSE, under the 0.3s resolution limit.

Univariate Spearman with the residual, all |rho| <= 0.10 (largest `sx_dep_prev_hold` +0.102,
`sx_out_gs` -0.089) — the band of the queue-count and physics features already closed.

**Noise floor, measured in passing.** An early run with November and December not yet extracted
(sx NaN there) still moved those folds by +4.97% and -9.58% from refitting alone. Single-fold
swings of 5-10% on this lane are noise; only the pooled figure means anything.

**Why it does not transfer, as far as can be told:** the context stack already carries NM-side
congestion for these rows (`nbr_*` neighbour taxi, `q_*` taxiing counts), and ADS-B re-measures
the same congestion with ~60% coverage. What ADS-B adds that NM cannot (hold time, ground speed)
is the part with the largest rho, and it is still ~0.1.

**Closed:** route 2 of section 40 (surrounding-traffic congestion). **Still open:** only route 1,
the scored flight's own trace (dwell end, lower bound; section 39, section 44 validation), which
waits on the section 37 ruling. `ground.parquet` for all 405 days (~1 GB) holds the dwell
segmentation it would need; the raw traces are no longer required for either route.

## 46. The scored flight's own ADS-B stand dwell: PASS on 2025, conditional on the ruling (2026-09-24)

`scripts/adsb/screen_own_dwell.py`. **2025 training rows only** — using this on ranking rows is
the section 37 question and stays paused. Each 2025 departure joined to its own take-off run in
`ground.parquet` (callsign + airport, ADS-B take-off within 600s of MVT): 1,011,357 joined,
417,848 with a stand dwell seen. Matched lane: 48.8% joined, 20.2% dwell seen; no-record lane
18.7% / 7.4%. Features: `own_dwell_taxi` (MVT - dwell end), dwell length, take-off offset, and
`own_minus_anchor` (own dwell taxi - `MVT - AOBT_3`).

Criteria fixed before the run: with-own LOMO residual correctors must cut total non-scored SSE
by >= 0.5% more than without-own, win >= 7/10 months, stay positive after drop-10/50.

| | result |
| --- | --- |
| matched lane RMSE, non-scored months | v29 197.9, corrector without own 197.9, **with own 192.0** |
| no-record lane | 720.3 / 705.8 / 708.4 — no gain there (7% dwell seen) |
| total non-scored SSE | **-3.96%**, ~6.1s of fold; **10/10 months** (-2.4% to -8.6%); drop-10 -3.97%, drop-50 -3.99% |

**PASS.** Confirmation on January + July 2025, which took no part in the decision (correctors
fitted on the ten other months, matched lane):

| | fold RMSE |
| --- | ---: |
| v29 | 307.27 |
| with own dwell, 2025 coverage | **303.93 (-3.34s)** |
| re-weighted to 2026 joined share per airport-month | **304.32 (-2.95s)** |

Largest cells: EHAM Jul 163.6 -> 142.4, EDDM Jan 164.0 -> 140.0, LSZH Jul 174.6 -> 157.0, EHAM
Jan 173.1 -> 158.1. Every covered airport-month improves; uncovered ones (LTFM, LFPG, LEMD Jan,
LIRF Jan in 2025) move by noise only. The re-weighting only scales *down* (gain x min(1, 2026
share / 2025 share)); it credits nothing where 2026 is better covered than 2025 (LEMD Jan 0% ->
96%, LIRF Jan 0% -> 66%, EGLL Jul 7% -> 97%), so -2.95s is a floor on the coverage effect, not
an estimate of it.

**Corrects section 36.** The matched bulk is irreducible *from NM data*; an independent
observation of the physical pushback reduces it. On early-anchor rows (anchor > 600s short of y,
77,797 rows) the own dwell is seen on 20.1% and is a median 402s short of y where the anchor is
897s short — it recovers about half the gap on the rows it sees.

**What remains before this is a submission:** (1) the organisers' answer on own-flight external
data (`docs/discord-question-prediction-time.md`, rewritten today, also asks about ATFM timing);
(2) 2026 own-dwell features from the 62 ranking days' `ground.parquet` (already extracted);
(3) integration into the chain rather than a post-hoc corrector — a feature of the matched base
or a CatBoost member, confirmed on 12 folds with the section 31 infrastructure; (4) the
rehearsal leakage probe. None of (2)-(4) starts before (1).

## 47. Own-flight expert: first-seen lower bound, join ceiling, and the twelve-fold result (2026-09-24)

**Join ceiling.** On 24 sampled Apr/Jul 2025 days (95,772 departures), widening the ADS-B
take-off window from 600 to 1,200s adds nothing (46.4% both) and keying on `FLIGHT_mvt` as well
adds 0.3 points. 87% of unjoined departures have no ADS-B take-off within [-120, +30]s of MVT at
all; the ceiling is coverage (EGLL July 2025 4.8%, against 96.8% in July 2026), not the join.

**A lower bound for runs seen mid-taxi.** Only 41% of joined runs show a stand dwell. Every run
with a taxi-out phase has a first taxi-out point, and the aircraft left the stand before it, so
`own_first_taxi = MVT - t_first_out` bounds y from below. On 606,845 2025 rows it holds within
60s on 94.3% (87.9% where no dwell was seen; LIRF lowest at 85.0%). Where it says the NM anchor
is more than 120s short (72,390 rows), 77.1% are truly early.

Built by `scripts/adsb/own_flight_features.py` (in the repo now; 2026 refuses to run without
`--allow-ranking`). Screen re-run with the bound (`screen_own_dwell.py`, same criteria):
matched lane 197.9 -> **190.9** (dwell only: 192.0); non-scored SSE -4.74% (dwell only -3.96%);
10/10 months; Jan+Jul confirmation **-4.10s**, -3.60s re-weighted to 2026 coverage.

**Twelve folds (`src/prc/ownexpert.py`, `scripts/experiment_own_expert.py`).** Residual expert on
the v29 reconstruction's OOF residual, fitted per held-out month on the other eleven, applied
only to matched own-joined rows (1,007,156, 48.3%); parameters fixed at the section 46 screen,
not tuned since.

| month | scope rows | v29 | + own expert | change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56,984 | 337.13 | 333.88 | -3.24 |
| 2 | 65,382 | 251.94 | 247.26 | -4.68 |
| 3 | 84,991 | 204.73 | 198.37 | -6.35 |
| 4 | 87,032 | 190.99 | 185.63 | -5.36 |
| 5 | 86,470 | 291.66 | 287.74 | -3.92 |
| 6 | 64,073 | 307.94 | 304.72 | -3.22 |
| 7 | 82,214 | 280.92 | 276.26 | -4.66 |
| 8 | 90,725 | 231.73 | 226.29 | -5.45 |
| 9 | 94,593 | 233.43 | 226.59 | -6.84 |
| 10 | 102,535 | 213.93 | 206.11 | -7.82 |
| 11 | 96,971 | 211.70 | 203.27 | -8.43 |
| 12 | 95,186 | 204.71 | 194.98 | -9.73 |

Twelve months 250.71 -> 245.29 (**-5.42s, 12/12**). **Jan+Jul fold 307.27 -> 303.33 (-3.95s)**;
drop-1/10/50 of the most-gaining rows -3.93 / -3.85 / -3.62; re-weighted to 2026 coverage
(scaled down only) **-3.63s**. The gain is broad bulk, the kind that has transferred to the
board near 1x (v28, v29). It is an order of magnitude over the 0.3s resolution limit.

Caveats: measured on the `v29_oof` reconstruction, not the shipped chain (audit 2026-09-20);
a post-chain residual stage, so the stacker's input set is unchanged. `models/ownexpert.txt`
is fitted on all twelve months and **not served**: using it needs (1) the organisers' answer on
own-flight external data, (2) `own_flight_features.py 2026 --allow-ranking`, (3) wiring into
`prc.submit.predict_frame` before `lirfbands.lobt_clip`, and (4) the rehearsal leakage probe.

## 48. Queue model on 2025, the ruling, and v31 (2026-09-24)

**Queue model (`scripts/queue_model.py`), analysis only.** 2,080,533 ordinary 2025 departures
(0 < y <= 3600s). Traffic described at stand exit (true off-block): departures already taxiing
on the same runway (`q_ahead`), and take-offs / landings in the fixed 10 minutes after. The
first version counted take-offs *during the flight's own taxi*; that is circular (a longer taxi
contains more take-offs by construction; its coefficient came out at the ~90s inter-departure
spacing and it reached RMSE 170 by leaking y) and was discarded. Unimpeded time = median taxi of
flights with nobody ahead, per (airport, stand, runway), fitted leave-one-month-out.

| airport | median taxi | unimpeded | additional share of taxi time | s per aircraft ahead |
| --- | ---: | ---: | ---: | ---: |
| EDDF | 837 | 837 | 12% | +41.9 |
| EDDM | 774 | 662 | 13% | +81.5 |
| EGLL | 1,319 | 1,207 | 14% | +52.3 |
| EHAM | 742 | 643 | 11% | +27.4 |
| LEBL | 906 | 849 | 13% | +52.8 |
| LEMD | 985 | 1,058 | 3% | +38.3 |
| LFPG | 954 | 845 | 18% | +80.5 |
| LIRF | 1,022 | 960 | 13% | +63.7 |
| LSZH | 712 | 654 | 11% | +65.2 |
| LTFM | 963 | 834 | 20% | +74.5 |

Given the queue ahead, more take-offs in the next 10 minutes *reduce* the additional time (-22 to
-90s each): a runway that is serving faster clears the queue, as a queueing model says. LOMO RMSE
over these rows: airport median 350.6, unimpeded alone 363.2, unimpeded + queue state 297.1.
ADS-B check (407,488 flights with own dwell): holding and moving time each correlate ~0.36 with
the additional time, so traffic costs about as much in slow taxiing as in standing still.

**Not a score lever.** The queue state here uses the flight's true off-block. Matched rows carry
the queue in `MVT - AOBT` already (sections 36, 45); no-record rows would need the off-block
solved for, and the same lane gave -0.7s (section 13) and ~-0.2s (section 45) to closely related
features. Kept as the per-flight "no traffic vs traffic" decomposition for the write-up
(`data/interim/queue_scenarios_2025.parquet`), which is the post-ops use the organisers name.

**The ruling.** `espinielli`, 2026-09-19, replying to another team's proposal to forbid
post-pushback ADS-B: "there are no such restrictions: if you can find open trajectory data with
good ground coverage and extract off-block times, we are ok", and "the model is for post-ops,
not for tactical use" (recorded in `docs/rules-and-compliance.md`). That unblocks the own-flight
expert and makes the ATFM previous-day hedge moot; its paired 12-fold (`--atfm-lag 1`) was
stopped at fold 2 and the `lag_days` option left in `prc.atfm` with default 0.

**v31** (`docs/submissions.md`): v29 + own-flight expert; leakage probe PASS, rehearsal
258.43 vs 262.05, 53.6% of ranking rows changed.

## 49. Post-v31 error anatomy, and the silent-gap rule: -5.94s on twelve folds (2026-09-25)

**Where the Jan+Jul fold error sits after the v31 expert** (303.33, 31,689M; `own_expert_oof.parquet`):

| block | rows | share | RMSE v29 -> v31 | solved = |
| --- | ---: | ---: | ---: | ---: |
| day offsets (y > 12h) | 14 | 36.2% | 28,627 | -61.1s |
| matched, own trace not joined (other airports) | 115,639 | 17.2% | 217 -> 217 | -27.3s |
| matched, LTFM / LFPG (no ADS-B) | 84,208 | 15.4% | 241 -> 241 | -24.4s |
| matched, joined, no dwell seen | 81,230 | 12.2% | 223 -> 218 | -19.1s |
| no-record, not a copy | 5,107 | 9.8% | 779 | -15.2s |
| matched, own dwell seen | 57,968 | 4.7% | **193 -> 161** | -7.3s |
| no-record schedule copies | 253 | 4.5% | 2,367 | -6.9s |

The dwell is what carries the gain (-17% RMSE where seen, -2% where only joined).

**The silent-gap rule** (`own_flight_features.py`, `--tag _v2`). 404,105 of 1,011,357 joined 2025
runs had no feature at all: they landed and took off in one ground run with no accepted dwell.
On 2025-07-15, 1,160 such runs: the largest silent gap is a median 5,005s (p10 3,287s), and the
displacement across it a median 416m (the last point is still taxiing in, the first already
moving), so the 60m stationary rule rejected the turnaround. Rule: after the last ground gap of
>= 1,200s, the first moving point bounds the taxi, and a stationary point before it is a dwell
end. 400,919 rows gain features (3,388 remain featureless); the bound holds within 60s on
**99.4%** (median slack 385s: transponders come on ~6 min after pushback); 17% also get a dwell
end, weaker than ground_points.py's (|e| < 120s 33% vs 53%).

Twelve folds, `experiment_own_expert.py --v2` (same parameters, one extra flag feature):

| | v31 expert | v2 expert |
| --- | ---: | ---: |
| twelve months | 250.71 -> 245.29 (-5.42) | 250.71 -> **242.76 (-7.94)**, 12/12 |
| Jan+Jul fold | -3.95 | **-5.94** |
| drop-1 / 10 / 50 | -3.93 / -3.85 / -3.62 | -5.93 / -5.84 / -5.60 |
| re-weighted to 2026 coverage (floor) | -3.63 | **-5.11** |

v31 scored -7.81 on a -3.95 fold. `models/ownexpert_v2.txt`; wrapper `models/mechanism_v32.json`.

**Stand coordinates (lead from the rank-3 team).** X-Plane Scenery Gateway airport data is GPL v2
or later (copyright notice kept, derivatives under the same licence), compatible with the GPLv3
release; declare it. The Gateway API serves the recommended scenery per airport: EHAM has 224
start locations (`1300` rows with lat/lon/heading/name, e.g. `P10`) and 619 taxi-network nodes.
Not yet used.

## 50. Stand coordinates (X-Plane Scenery Gateway): the sharpest off-block clock yet, -8.38s (2026-09-25)

`scripts/fetch_xplane_stands.py` pulls each airport's recommended Gateway scenery (GPL v2+) and
keeps its `1300` startup locations: 2,838 stands (EDDF 227, EDDM 285, EGLL 284, EHAM 224, LEBL
234, LEMD 380, LFPG 513, LIRF 151, LSZH 189, LTFM 351). NM `STAND_mvt` names meet apt.dat names
after normalisation (`stand_exit.stand_key`) on 90.5% of joined departures; share of Jan+Jul
departures matched by full name or last token: EGLL 97.7%, LIRF 99.4%, LEMD 96.6%, LEBL 98.5%,
EDDF 92.2%, LFPG 90.5%, EDDM/EHAM ~69% and LSZH 64% before zero-stripping, LTFM 0% (no ADS-B
there anyway).

**Stand exit** (`scripts/adsb/stand_exit.py`): the last ADS-B point within R of the NM-recorded
stand, after the run's last >= 1,200s gap. Radius chosen on **2025-04-15 + 2025-10-15**
(non-scored; 5,313 joined departures, stand located for 90.5%):

| estimator | rows with it | |e| < 60s | |e| < 120s | RMSE |
| --- | ---: | ---: | ---: | ---: |
| NM AOBT - BLOCK, same rows | 100% | 24.4% | 46.9% | — |
| stand exit, 15m | 17.3% | 78.3% | 90.5% | 105 |
| stand exit, 25m | 22.4% | 79.9% | 92.9% | 89 |
| **stand exit, 40m (chosen)** | **27.2%** | 78.3% | 92.9% | 89 |
| stand exit, 60m | 31.2% | 71.0% | 89.6% | 93 |
| stand exit, 90m | 37.3% | 42.5% | 66.9% | 163 |

Where it exists it is within a minute of BLOCK_TIME four times as often as NM's own AOBT.
2025: 281,018 departures (27.8% of joined) carry it (`--stands --tag _v3`).

Twelve folds, `experiment_own_expert.py --version 3` (v2 features + `own_stand_exit_taxi`,
`own_stand_dmin`, `own_stand_minus_anchor`; parameters unchanged):

| | v2 | **v3** |
| --- | ---: | ---: |
| twelve months | -7.94 | **250.71 -> 239.15 (-11.56)**, 12/12 |
| Jan+Jul fold | -5.94 | **307.27 -> 298.89 (-8.38)** |
| drop-1 / 10 / 50 | -5.93 / -5.84 / -5.60 | -8.37 / -8.26 / -7.96 |
| re-weighted to 2026 coverage (floor) | -5.11 | **-7.18** |

Board transfer so far: v31 2.0x (coverage gain in 2026), v32 1.2x (bound features on rows already
covered). v3's new rows are like v32's, so expect ~1-1.2x of the -2.44 increment over v2.
`models/ownexpert_v3.txt`; wrapper `models/mechanism_v33.json`.

## 51. Stand location v2: more names, and a name-free position fallback (2026-09-25)

Unmatched NM stand names after section 50's normaliser: EDDM 29.7% of departures (NM `117`, apt.dat
only `117A` / `117B`), EDDF 9% (duplicate apt.dat names such as `B22` dropped by the 30m
ambiguity rule; remote `V1xx` stands absent), LSZH 16% (`I0x`, GA areas), EHAM 4% (`A81-A85`
absent), LFPG 7% (`U`, `E` series absent). Three changes (`stand_exit.stand_table`,
`nearest_stand`; builder flag `--stands-v2`, files `_v4`): sub-positions also filed under their
base number, ambiguity threshold 30m -> 60m (2-D), and where the NM stand is still not located,
the apt.dat stand within 40m of the first point after the last long gap.

Non-scored screen (2025-04-15 + 2025-10-15, 5,313 joined): stand exit seen 24.6% -> **29.1%** of
joined, accuracy unchanged (|e| < 60s 78.1%, RMSE 89). Where both the name and the position
locate a stand, the two exits agree within 30s on 99.6% (1,414 rows). By airport: EDDM 36.4 ->
53.9%, LSZH 23.4 -> 32.4%, others within ~1.5 points. 2025 stand-exit rows 281,018 -> 317,351.

Twelve folds (`--version 4`, same features as v3): Jan+Jul -8.62s (v3 -8.38), twelve months
-11.89 (v3 -11.56). **Paired against v3: 12/12 months better (-0.20 to -0.53s), Jan+Jul -0.242s,
twelve months -0.328s, drop-10 -0.204, drop-50 -0.152.** Consistent but at the 0.3s resolution
limit on the fold (v30's -0.22 came out +0.09 on the board), though coverage-driven changes have
transferred ~2x (v31, v33). Not shipped alone: kept as the base for the next bundle.
`models/ownexpert_v4.txt`.

## 52. Time-only join: precise, but only +1.8% of departures; not built (2026-09-25)

`scripts/adsb/time_join.py`. A departure the callsign join misses is paired with an unclaimed
ADS-B take-off run at the same airport in [MVT - 120s, MVT + 30s], unique both ways. Screen on
2025-04-15 + 2025-10-15 (11,838 departures, 5,313 callsign-joined):

- **precision**, callsigns of a tenth of the joined rows hidden at a time and re-joined by time
  among the runs no other departure claimed: 3,576 re-joined (67.3%), **99.9% to the right run**.
  The stand check (run within 100m of the NM stand) passes on 41.7% and does not separate right
  from wrong (100.0% vs 99.8%), so it is not needed as a filter.
- **yield** on the truly unjoined departures: **219 rows, +1.8% of departures, +4.1% on the joined
  set**, not section 47's +4.5% of departures (most unjoined departures have zero or several
  candidate take-offs, not one). Their stand check passes on only 11.6% against 41.7% above:
  mostly traces that start mid-taxi, and possibly take-offs of aircraft absent from NM.

Expected at most ~4% more expert rows, of lower quality than the average: ~0.1-0.3s of fold,
under the resolution limit. Not built into the features.

## 53. CatBoost member with own-flight features: NEGATIVE (2026-09-25)

`experiment_catboost_gpu.py --own-features adsb_own_2025_v4.parquet --tag catgpu_own` (5,000 it,
depth 8, lr 0.06, as the shipped AOBT member; RTX 4070 Ti, ~2 min/fold): the v4 own-flight columns
plus their differences to the anchor as numeric inputs next to the 156 matrix features and 23
categoricals. The member alone is much better than the old one on matched rows (January 194.5
against 204.2), but it does not improve the chain:

| composition, v29 reconstruction | Jan+Jul | twelve months |
| --- | ---: | ---: |
| v4 expert alone (section 51) | **-8.62** | -11.89 |
| member blended into v29 (w 0.877, non-scored months), then the expert re-fitted | -6.34 | -10.48 |
| v4 expert, then the member blended after it (w 0.202) | -8.55 (+0.07) | -12.01 (-0.12) |

At w 0.88 the member replaces most of v29's matched-row chain (stacker, LOBT member and
expert) and loses; after the expert it adds nothing. The small LightGBM on v29's residual already
extracts what the own-trace columns carry. The member's July (matched 287.8) is dominated by one
LIRF day-slip row (y 87,002s, anchor 87,181s) that its clipped-anchor target cannot represent; it
was found by inspecting July, a selection month. Scoping the member to `MVT - AOBT <= 3600s` is a
structural rule (the member is fitted on `clip(MVT - AOBT, 0, 3600)`), but it changes nothing
here because the final LOBT clip already restores that row (301.84 -> 301.78). Not shipped.

## 54. First-sighting distance from the stand (v5 features): small, consistent (2026-09-25)

On 2025-04-15 + 2025-10-15, 3,610 joined departures (about two thirds of the joined set) were
first seen already moving and away from their NM stand, with no stand exit. How late that first
sighting is grows with its distance from the stand (corr +0.568, slope 0.218 s/m, ~4.6 m/s), so
`first sighting - fitted(distance)` estimates the off-block with sd 172s against NM AOBT's 247s
(|e| < 60s 33.7% vs 24.6%). Added as `own_first_dist` (builder `--stands-v2 --tag _v5`, expert
`--version 5`).

Twelve folds: Jan+Jul 307.27 -> 298.54 (**-8.73s**), twelve months -11.99, 12/12, drop-50 -8.31.
Paired against v4: **12/12 months better, Jan+Jul -0.111s, twelve months -0.101s**. The expert
already had the first-sighting time and the stand geometry through other columns, so the
distance adds little on top. v4 + v5 together are -0.35s over v33's v3 expert (-8.73 vs -8.38),
12/12. Bundled as v34.

## 55. Coverage audit, ground extraction v2, and a retuned expert: -4.33s over v34's expert (2026-09-27..29)

**Expert capacity (v5 features).** The v34 expert kept the section 46 settings (31 leaves, 300 rounds,
lr 0.05, one seed) through five feature versions. `scripts/experiment_own_expert_bundle.py`, twelve folds on
the v29 reconstruction, selection on the ten non-scored months only, paired against those settings:

| config | non-scored dSSE | wins (non-scored) | twelve | Jan+Jul |
| --- | ---: | ---: | ---: | ---: |
| identity encodings (stand / operator / type, nested, clipped +/-1800) | +5.5M | 3/10 | +0.01 | +0.01 |
| 5 seeds | -42.7M | 10/10 | -0.05 | -0.02 |
| 600 rounds | -394M | 10/10 | -0.48 | -0.40 |
| 63 leaves, 1200 rounds, lr 0.025 | -667M | 10/10 | -0.80 | -0.62 |
| 127 leaves, 1200 rounds, lr 0.025 | -822M | 10/10 | -0.98 | -0.73 |
| **255 leaves, 1200 rounds, lr 0.025 (chosen)** | **-894M** | 10/10 | -1.05 | -0.73 |
| 127 leaves, 2400 rounds | -813M | 10/10 | -0.94 | -0.61 |

The encodings are closed: a screen on the expert's residual had stand explaining 1.8% of its variance,
but a refitted expert already reaches it.

**Coverage audit.** A raw-archive audit of 2026-07-16 (`prod-0` and `staging-0` replicas; they are
equivalent) classified every NM departure: 3,034 joined, 1,476 seen only airborne, 833 not seen,
**823 (13%) with ground points for the right aircraft but no join**. Two defects in
`ground_points.py`: (1) the run's callsign is the last one set *on the ground*, on a turnaround often
the inbound leg's (EDDF 176 rows: `DLH947` where NM has `DLH17F`); (2) ground reception lost mid-taxi
leaves more than AIR_GAP_S before the first airborne point, so the run is never a take-off (LEMD 203,
LFPG 120). The time-only join does not recover (1): 64% right icao on 101 unique candidates.

**Extraction v2** (`ground_points.py --v2`, `own_flight_features.py --ground-v2`, feature `own_to_late`):
the join uses the first callsign set within 900s after the run, and a run followed within 1,800s by an
airborne point within 15km of the airport counts as a take-off (`to_late`, joined on that point). On the
audit day joined departures 48.7% -> 57.7% (EDDF 45 -> 75%, LEMD 10 -> 40%, LFPG 3 -> 14%); rebuilding the
old extractor from the same traces reproduced v34's features exactly (max diff 0.0).

**Data.** The raw traces had been deleted after the first extraction; a read-only recovery from old btrfs
trees (no snapshots) found only Aug-Dec 2025, no day complete (5-96% of files readable), so all 426 days
were re-fetched (`scripts/adsb/reingest_v2.sh`, authenticated per-asset curl, resumable; `gh release
download` died with HTTP/2 PROTOCOL_ERROR on hour-long streams) and kept on the HDD. The fetcher had missed
**21 days section 42 called absent**: single-`.tar` days (2025-01-01/11/18/19/21, 2025-12-25) and the
2025-05-28..06-11 outage under `staging-0tmp`; only 2025-12-31 does not exist. 2025-10-15's `prod-0` tar is
corrupt (tar header error mid-archive; the old fetcher ignored tar's status), so it comes from `staging-0`.

Joined departures: 2025 1,011,357 -> **1,217,207 (+20.4%**, 82,869 late take-offs), 2026 188,817 -> **218,015
(+15.5%)**. On Jan+Jul 2025 matched rows (0 < y < 1h) the air-callsign rows are as good as the old ones
(bound holds 94.8% vs 95.8%, stand exit within 60s 81.1% vs 83.5%); the late take-off rows are weaker
(86.0%, 26.5%) at every gap to MVT.

**Twelve folds, 255 leaves / 1200 rounds for all, paired against v5 features:**

| | twelve | Jan+Jul | months | drop-1/10/50 | Jan+Jul 95% CI |
| --- | ---: | ---: | ---: | --- | --- |
| v6a (air callsign only) | -1.24 | -1.22 | 11/12 | -1.21 / -1.16 / -1.01 | [-1.58, -0.94] |
| **v6b (all of v6)** | **-3.24** | **-3.61** | 12/12 (10/10 non-scored) | -3.58 / -3.49 / -3.23 | [-4.57, -2.83] |

Against v34's expert (v5 features, old settings): **Jan+Jul 298.54 -> 294.21 (-4.33s, CI [-5.48, -3.40])**,
January -4.66, July -4.05, 12/12 months; twelve months -16.29 against v29 (v34 -11.99). The late take-off
rows carry most of the gain despite their weak stand exits; the expert reads them through `own_to_late`.
Measured on the v29 reconstruction, as sections 46-54. `models/ownexpert_v6.txt`; wrapper
`models/mechanism_v35.json`.

## 56. Late take-off stand exits: an airport-mix artefact, not an extraction bug (2026-09-29)

Handoff next step 1 hypothesised that late take-off runs (`to_late`, section 55) are often the *arrival*
(land, taxi in, park, lose reception), so their "stand exit" would be the arrival parking. Diagnostic
on the 15th of the ten non-scored 2025 months, matched rows with 0 < y < 1h joined on `ground_v2`
(33,814 rows, 2,025 late), paired with the v6b OOF (`own_l255_r1200_lr025`):

- **Hypothesis refuted.** Late exits are rarely early (0.9% more than 300s before BLOCK, 1.1% for
  normal joins). Their median error is +93s against +14s, and the whole gap is airport mix: EGLL is 44%
  of late rows (4% of normal) and carries a constant +135s exit offset for *both* groups (within 60s:
  5% normal, 4% late). Per airport, the late and normal rows are about equally accurate; after removing
  each (airport, late) median, 80% of late rows are within 60s against 89% of normal rows.
- **The expert's residual agrees.** Per airport, late RMSE is close to normal RMSE (EGLL 156 vs 174,
  LEMD 105 vs 126, EDDF 116 vs 100; LEBL 223 vs 121 on 97 rows is the exception). The block-level
  201 vs 135 in the handoff table is a composition effect, so the "~-2s" estimate for step 1 does not hold.
- **What is real is smaller.** The lower bound (first taxi-out point >= BLOCK - 60s) fails on 18% of late
  runs that did not begin with a landing (1.3% for runs that did). On those runs no dwell is found, or
  the longest stop is an earlier one (a tow or remote stand), so the first taxi-out point is the run's
  first point, about 6,000s early. Taking the first moving point after the run's last gap of 1,200s or more
  (`stand_exit.departure_start`) makes the bound hold on 95.0% instead of 91.0% of late rows and 98.1%
  instead of 96.5% of normal ones. But the failing late rows carry 7.9M of 566M expert SSE (1.4%), and the
  residual is uncorrelated with the correction there (0.004): the expert already reads those rows. On the
  1,103 normal rows where the bound fails (50.5M, RMSE 214), the corrected bound correlates 0.19 with the
  residual. An in-sample linear fit removes 6.8M (1.2% of the expert's SSE on these rows): an
  optimistic ceiling well below a second.

Not pursued as a standalone change. At most, a gap-aware first taxi-out feature to bundle with other v7 work.

**Addendum, day-offset rows re-examined with a new question (2026-09-30).** Of the 34 rows in 2025 with y > 12h, 32
are LIRF (the model already places all 12 in Jan+Jul between 44k and 85k; residual about 4% of fold SSE) and 3 are
elsewhere; the two Jan 2025 LFPG rows alone are ~34% of the Jan+Jul fold's squared error. Two new hypotheses for
the non-Rome rows, both refuted:
- *Block copied from the aircraft's inbound in-block.* No arrival at the same stand has an in-block matching the
  departure's recorded block time. The nearest arrivals at the airport (120s away for both LFPG rows) have other
  stands and types.
- *Block copied from a schedule.* True for LSZH NJE389D (block = schedule - 86,402s, a date slip), not for LFPG.
  EJU983W's block of 01-18 19:30:00 and EJU42AY's of 01-22 22:58:57 match no schedule of that flight on nearby days.
  Both are no-record rows (no `_flt` match) with the block on the previous evening, 16-23h before departure.
- The ADS-B trace joins 7 of 34; all show an ordinary taxi, so the trace cannot flag a bad record. The 3 non-Rome
  rows look like three different accidents in 20,982 non-LIRF no-record departures. Section 29 stands.

## 57. Garbled Rome flight strings: a schedule-copy rule in the 4-8h lane, -0.85s twelve (2026-09-30)

At LIRF some no-record departures carry a flight string with the operator prefix typed twice
(`ITYTY672`, `WMTT5232`, `RYRR90BN`, `QFAQFA6`; `lirfbands.garbled_flight`). Section 11 saw them
and judged them captured by `nm_record_missing`. Re-examined against the v6b OOF (the v35 chain on the v29
reconstruction), 2025, LIRF no-record departures:

- **Rows with y > 12h:** 21 of 169 garbled against 9 of 1,319 ordinary (12.4% vs 0.7%). But 29 of the 30 such rows
  sit in the band lane (gap > 8h), where garbled rows split evenly between schedule copies and day offsets at
  50-66k (the ambiguous range), so the flag does not help the band table.
- **Copy rate in the <= 8h lane, garbled vs ordinary:** 1-2h 65% vs 36%, 2-4h 79% vs 56%, 4-8h **100% (20/20)**
  vs 74%. At 1-4h the chain already reflects this (mean prediction within 3% of the mean truth for garbled rows).
  At 4-8h it does not: mean prediction 14.7k against a truth of 19.0k, and some rows are predicted at 4k where
  the truth is 15-17k.

**Rule** (`lirfbands.garbled_copy`, wrapper key `garbled_copy_rule`): LIRF, no NM record, garbled string, gap in
(4h, 8h] -> predict MVT - SCHED. The edges are the band table's first edge and the mixture's limit, not tuned.
Evidence: 20/20 exact copies across 10 months (17/17 in the ten non-scored months). Applied to the stored final OOF:

| | v6b OOF | + rule | delta |
| --- | ---: | ---: | ---: |
| twelve months | 234.42 | 233.57 | -0.85 |
| Jan+Jul | 294.21 | 292.89 | -1.32 |
| ten non-scored | 220.68 | 219.95 | -0.74 |

No month worsens (9 gain, 3 have no rows). The gain is concentrated by construction (about 8 badly
under-predicted rows carry most of it). It touches 5 ranking rows (3 in Jan, 2 in Jul); if one is a normal taxi,
that row costs about +1.6s. Applied last in `predict_frame`, after the own-flight expert (which never serves
no-record rows), so no later stage moves a pinned row. The served definition selects the same 20 training rows.
Wrapper `models/mechanism_v36.json` = v35 + `garbled_copy_rule`.

## 58. Rome no-record lane (gap <= 8h): arrival schedule copies as a feed-fault signal — NEGATIVE (2026-09-30)

**Lane anatomy** (v6b OOF, 2025, LIRF no-record, gap <= 8h, 1,438 rows, 16.0% of the year's squared error):
schedule copies 683 rows (under-predicted by 1,439s on average, 4.7% of SSE), normal taxis 696 (over by 1,221s,
2.9%), one day offset (ITY332, 6.5%), three rows with y 13-31k at gaps of ~25 min (1.3%), 55 long taxis under 12k
(0.6%). The addressable part is copy vs normal, 7.6%. The mixture is already the RMSE-optimal form, so only
discrimination can help. ADS-B cannot: the aircraft taxis normally in both cases.

**Hypothesis:** copies are a reporting-feed fault, and arrivals, whose block times the ranking set keeps, show it
too. LIRF arrivals with |BLOCK - SCHED| <= 60s: 12.1%. By quarters of the day's arrival copy rate, the lane's copy
rate is 40%, 41%, 49%, 62%, and the v6b residual is -39, +93, +401, +557s (corr with residual 0.08). The inbound
arrival at the same stand carries nothing (copy rate 48% vs 46%).

**Test:** `prc.arrcopy` (day rate, day count, same-operator day rate, +/-3h rate) added to the mixture's inputs,
`scripts/experiment_lirf_unmatched_moe.py --max-gap-h 8 [--arr-copy]`, leave-one-month-out, LightGBM regressor in
both arms, no other change. The lane delta (arrival-copy arm minus base arm) was carried onto the v6b OOF with the
v36 garbled rule applied to both:

| | base | + arrcopy | delta |
| --- | ---: | ---: | ---: |
| twelve | 233.571 | 233.904 | +0.333 |
| Jan+Jul | 292.890 | 294.194 | +1.304 |
| ten non-scored | 219.946 | 220.026 | +0.080 |

Won 4/10 non-scored months; Jan+Jul drop-1/10/50 +1.45 / +1.96 / +2.56. Lane AUC rose 0.927 -> 0.932, yet
squared error worsened. As in section 29, better ranking does not pay when the calibration on the large-gap rows
moves. Closed; the module stays for the record, unserved.

**Addendum 2, the two Paris day slips against raw ADS-B (2026-09-30).** A raw-trace search for both callsigns on
18/19 and 22/23 Jan 2025 found each flight's actual take-off from the air (EJU983W OE-LVK 19 Jan 18:55, EJU42AY
OE-IZN 23 Jan 15:10), both matching the recorded MVT, and no movement under either callsign on the previous day.
Paris has no adsb.lol ground coverage near the recorded blocks (0 moving ground points within 15 min). Both
flights' neighbouring days are all in NM (EJU42AY 16/18/20/23, EJU983W 17/19/20/24), so no record is missing.
EJU42AY's every-other-day pattern skips the 22nd, so an overnight delay is plausible but unobservable. The
NM-only signature "flight number absent the previous day" covers 7,185 non-LIRF no-record rows, 3 of which are
day slips (0.04%). A pilot airborne take-off extractor (7 days, ~2,500 take-offs a day, 95% matched to NM rows)
found no unmatched take-off followed by a next-day no-record or day-slip row. Nothing flags these rows.

## 59. Own-flight expert on no-record departures: -0.26s twelve, CI excludes zero (2026-09-30)

Handoff step 2. `scripts/experiment_own_norec.py`; scope `ownexpert.norec_scope`: `nm_record_missing` rows outside
LIRF whose own trace joined (4,783 in 2025, 1,157 in Jan+Jul). Base: the v6b OOF with the v36 garbled rule (the
v36 chain on the v29 reconstruction; the matched expert and the rule leave these rows untouched). Inputs
`ownexpert.FEATURES_NOREC`: airport, hour, weekday, the chain's prediction, the schedule gap, and the v6 trace
features (no anchor exists). Target: the base residual clipped to +/-1,800s, fixed before the first run. On
these rows the stand exit alone has RMSE 278 against the chain's 427 where it exists. The dwell end and the
first taxi-out point carry large outliers (RMSE 2,400+; an earlier stop or tow), so the trace goes in as inputs,
not as the prediction.

Twelve folds by month; selection on the ten non-scored months (dSSE):

| config | non-scored dSSE | won | twelve | Jan+Jul | drop-1/10/50 (Jan+Jul) |
| --- | ---: | ---: | ---: | ---: | --- |
| 7 leaves | -173.8M | 9/10 | -0.243 | -0.310 | -0.27 / -0.15 / -0.05 |
| 15 leaves | -183.7M | 9/10 | -0.254 | -0.315 | -0.27 / -0.17 / -0.05 |
| **31 leaves (chosen)** | **-186.3M** | 9/10 | **-0.258** | **-0.321** | -0.27 / -0.17 / -0.05 |
| 15 leaves, 600 rounds, lr 0.025 | -183.7M | 9/10 | -0.255 | -0.321 | -0.27 / -0.17 / -0.05 |
| 15 leaves, min_data 100 | -171.2M | 9/10 | -0.239 | -0.304 | -0.26 / -0.17 / -0.06 |
| 31 leaves, cap 3,600 | -174.4M | 9/10 | -0.259 | -0.383 | -0.33 / -0.16 / -0.02 |

May is the one losing month in every config. Paired bootstrap (31 leaves, cap 1,800) against the base: pooled
Jan+Jul **-0.321, 95% CI [-0.507, -0.186]**; January -0.209 [-0.485, -0.048], July -0.436 [-0.700, -0.251]. The
wider cap looks better on Jan+Jul but worse on the non-scored months, so it is not taken. Scope RMSE 1,352 -> 1,333:
the three non-Rome day-slip rows sit in scope and the cap keeps the expert off them. `models/ownexpert_norec_v1.txt`
(all twelve months), served by `predict_frame` under `own_norec_stem`; wrapper `models/mechanism_v37.json` = v36 +
the no-record expert.

## 60. v7 own-flight features: taxiway-network distance + gap-aware first taxi-out, -1.11s over v37 (2026-10-01)

Handoff step 3 plus the defect from section 56. `own_flight_features.py --v7` (with `--stands-v2 --ground-v2`) adds:

- `own_first_net_dist`: stand-to-first-moving-sighting distance along the X-Plane taxi-routing network
  (`scripts/adsb/taxi_network.py`: apt.dat 1201 nodes / 1202 edges, one connected graph per airport, all-pairs
  Dijkstra; each end snapped to its nearest node, NaN beyond 300m). Median 1.57x the straight line.
- `own_first_taxi_gap`: MVT minus the first moving point after the run's last gap >= 1,200s (`departure_start`),
  whatever the stand. Section 56 found the first taxi-out point is often an earlier stop or tow, about 6,000s early,
  on runs without a landing; the tail falls from 60,093s to 16,253s on the screen days.

Rebuilt `adsb_own_{2025,2026}_v7.parquet` on a GPU host: same rows as v6 (1,217,207 / 218,015), every v6 column
byte-identical, the new columns on 95-99% of rows. `ownexpert.FEATURES_V7` adds both plus `own_first_gap_minus_anchor`.

**Matched expert** (255 leaves, 1,200 rounds, lr 0.025, the section 55 settings, nothing retuned), twelve folds paired
against v6b: twelve 234.42 -> 233.40 (**-1.02**), Jan+Jul 294.21 -> 293.25 (**-0.96, CI [-1.26, -0.73]**), January
-1.02, July -0.90, **12/12 months**, drop-1/10/50 -0.94 / -0.86 / -0.71. A broad gain, not a few rows.

**No-record expert** (section 59, `FEATURES_NOREC_V7`, same config): non-scored dSSE -236.8M against -186.3M for v6
features; against the v6 no-record expert, pooled Jan+Jul -0.157, CI [-0.296, -0.049] (July alone not resolved).

**Combined** (disjoint rows), v38-equivalent against v37-equivalent OOF: twelve 233.31 -> 232.20 (**-1.11**), Jan+Jul
292.57 -> 291.45 (**-1.12, CI [-1.50, -0.84]**), non-scored -1.12, 12/12 months. Models `ownexpert_v7.txt`,
`ownexpert_norec_v2.txt`; wrapper `mechanism_v38.json`.

**Addendum (2026-10-01): push-back detection is moot on the rows it targets.** Of 1,211,707 matched joined 2025
rows, 607,071 see neither a stand exit nor a dwell ("bound only"). Their first moving sighting is a median 846m
along the taxiway network from the stand (p10 232m, p75 1,545m), about 423s after BLOCK (p10 117s, p90 825s);
only 6.4% are first seen within 100m of the stand. The push-back is not in these traces, so handoff-10-01 step 1
is dropped. What the trace offers there is the sighting time and distance, which the v7 expert already reads.

**After v38, where the scored-month traces are missing (2026 ranking, matched departures joined):** LTFM 0%,
LFPG 13-15%, **LIRF July 0.2%** (adsb.lol has 0-11 LIRF ground take-offs a day in July 2026, against 200-350 in
January; the feeder is gone, not our pipeline), LEMD July 67%, EDDF 72-77%. On the v38-equivalent Jan+Jul fold,
unjoined matched rows carry about a third of the error (LTFM 9.2%, LFPG/EHAM/EDDF 7.9%, other airports 11.8%).

## 61. Coverage and competitor sweep; published monthly taxi-out means as a calibration (2026-10-01)

**Other ADS-B sources (research agent).** No eligible archive fills LTFM, LFPG or LIRF July 2026. adsb.fi and
airplanes.live are non-commercial, ADS-B Exchange is commercial, ADSBHub has no history, and OPDI derives from
OpenSky. adsb.lol's `mlatonly-0` tag (ODbL) has **zero ground points** (checked on 2026-07-16: 39,364 traces).
The one lead is FlyItalyADSB (CC BY-SA 4.0), which keeps only about 20 days self-serve; older data is on request.

**Competitors and Discord since 09-26 (research agent).** No public code from any team above us. kind-mango (249.4)
moved to the same own-flight adsb.lol approach. GREKI warns that adsb.lol's receiver network changed between
2025 and 2026. Ruling 2026-09-27 (rules-and-compliance): the published Jan/Jul 2026 monthly taxi-out means may be
used as inputs.

**EUROCONTROL monthly taxi-out** (`data/external/ansperformance/Taxi-Out_Additional_Time.xlsx`, through 2026-07):
mean = (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) / TOTAL_REF_NB_FL. Against our 2025 target (y < 1h), corr 0.984
over 120 airport-months. In winter the file drops flights (VALID_FL / TF down to 0.69-0.87 at EDDM, LSZH and EDDF),
and its mean then sits 80-170s under ours while our model was unbiased (EDDM Jan 2025: published -166s, our
bias -3s). Where VALID_FL / TF >= 0.99 (89 airport-months), the published mean fixes the true mean of the rows
with prediction < 3600 to within an sd of 0.8-4.5s after a per-airport offset at EDDF, EDDM, EGLL, EHAM, LEBL,
LEMD and LSZH (LIRF 9.0, LTFM 8.6, LFPG 16.1). Our own airport-month bias sd was 4-16s.

Calibration rule: for airport-months with valid share >= VMIN, shift predictions < 3600 by
(published mean - our mean of those predictions) - the airport's 2025 offset. 2025 leave-one-month-out on the
v38-equivalent OOF: VMIN 0.97 twelve -0.051, Jan+Jul -0.238, non-scored -0.002 (4/12 months won); VMIN 0.99
-0.040 / -0.220 / +0.008 (8/12). Neutral on 2025, as expected when the model is already within ~7s of each mean.
On the 2026 ranking set the v38 shifts would be: LIRF Jul -57s, EHAM Jan about +76s (valid 0.989: only at VMIN 0.97),
LFPG Jul +20, LEBL Jul -20, EGLL Jul -10, LTFM Jul -11, the rest within 6s.
Undecided at the time of writing: the fold cannot show the 2026 benefit, so shipping rests on the 2025 accuracy
of the published means.

## 62. Does 2026 differ from 2025? Expert drift and the published-mean calibration revisited (2026-10-03)

**The own-flight experts carry over to 2026 unchanged.** v38 was rerun on the ranking set one stage at a time,
reproducing `predict_frame`'s tail (final predictions identical to `dynamic-lobster_v38.parquet`). The experts'
mean shift per airport-month (final minus chain, rows with prediction < 3600) is -7.7 to +8.0s in 2026, against
-8.4 to +6.1s on the 2025 Jan/Jul OOF. The trace inputs do change: EDDF stand exits fall (Jul 0.14 -> 0.04 of joined
rows) and its first sighting is further out (network distance 651 -> 1,628m); EGLL and EDDM Jul 2025 were poorly
joined (0.22, 0.37) and are 0.97 in 2026; EGLL 2025 take-offs sit about 200s before MVT, 2026 about 0. The expert
has seen these regimes in other 2025 months, and nothing at the mean level says it extrapolates.

**Published-mean calibration, threshold chosen on 2025 only.** Leave-one-month-out error of the implied true mean
(published minus the airport's offset, offset from the other 2025 months), by the file's valid share:

| valid share | months | implied-mean RMS error | our model's bias RMS |
| --- | ---: | ---: | ---: |
| <= 0.95 | 11 | 114.8 | 7.8 |
| 0.95-0.97 | 5 | 40.2 | 3.5 |
| 0.97-0.985 | 6 | 16.6-19.0 | 5.7-8.7 |
| 0.985-0.99 | 9 | 3.3 | 14.1 |
| > 0.99 | 89 | 7.2 | 8.8 |

A season offset instead of an all-year one changes nothing (7.1 vs 7.2). The 2025 bias-SSE change from shifting:
VMIN 0.97 -52.4M, 0.98 -67.2M, **0.985 -79.8M**, 0.99 -44.4M (63/98 months improved at 0.985). The 0.985 edge
rests on 3 months in the 0.98-0.985 cell; EHAM's valid share is 0.987-0.991 in every 2025 month, so 0.99 cuts it
in half arbitrarily.

**On the 2026 ranking set at VMIN 0.985** the implied bias of v38 is small except in two airport-months: EHAM Jan
-72.6s (published 982s, up 179s on Jan 2025; our prediction rose 111s) and LIRF Jul +59.5s (published 1,214s,
70s below Jul 2025; Rome's anchor is the constant ~900s clock and the July traces are gone, so nothing we hold
could see it). Then LEBL Jul +20, LFPG Jul -20, EGLL Jul +12; the rest under 10s. LTFM Jan (+125s) is excluded:
valid 0.952, and LTFM Feb 2025 at 0.889 read 248s under the truth. If the implied means carry their 2025 error
(7s RMS), the shift removes about 149M of squared error over 344,841 rows (MSE -432), 92% of it from EHAM Jan
and LIRF Jul. That is a projection resting on two published numbers, not a fold measurement. Not yet shipped.

**Outcome.** Shipped as v39 (`pubcal`, VMIN 0.985): scored 244.1917 against v38's 244.1913. The board is history,
not evidence (audit finding 5), and it says nothing about which airport-months gained or lost. The calibration is
closed: re-choosing VMIN, the offsets or the airport-months after this score would be tuning against it.

## 63. Retune sweep: the matched expert reads the chain's matrix, -0.96s twelve over v38 (2026-10-04)

Start of a sweep over every stage's settings (user goal 2026-10-04). Error anatomy of the v38-equivalent OOF
(twelve 232.20): day slips (34 rows, y > 12h) 26.7% of squared error, no-record rows 14.1%, matched rows joined to a
trace 21.1% (RMSE 140), matched rows not joined 38.1% (LTFM 12.4%, LIRF 7.7%, LFPG 9.2%, EGLL 5.5%).

**Matched expert inputs.** The shipped expert (section 60) reads the trace, the anchor, the chain's prediction,
airport, hour and weekday; never the matrix the chain was fitted on. `scripts/experiment_own_expert_rich.py` adds
it: every numeric matrix column (`ownexpert.rich_columns`, minus `doy` and the identity strings) and, for `cat`, the
stand, runway, type, operator, wake, flight type and market segment as LightGBM categoricals. Same protocol as
`experiment_own_expert_bundle.py` (v29-reconstruction residual, other months fitted, unclipped target, 255 leaves,
1,200 rounds, lr 0.025), paired against the v7 expert's stored OOF. Screen on months 3 and 9 (non-scored):

| config | month 3 | month 9 | non-scored dSSE |
| --- | ---: | ---: | ---: |
| v7 inputs, rerun | +0.01 | +0.00 | +0.4M |
| v7, target clipped +/-1,800 | +0.01 | +0.03 | +2.9M |
| + numeric matrix | -0.53 | -0.56 | -76.1M |
| + numeric + categoricals | -1.37 | -0.63 | -132.1M |
| same, seed 8 | -1.38 | | |
| same, 2,400 rounds | -1.42 | -0.54 | -127.9M |
| **same, feature_fraction 0.5** | **-1.44** | **-0.84** | **-152.7M** |
| same, feature_fraction 0.3 | -1.20 | -0.81 | -136.5M |
| same, all matched rows (joined or not) | +0.70 | +0.86 | +110.5M |
| same, all matched rows, clip 3,600 | +0.66 | +0.91 | +111.5M |
| CatBoost GPU, depth 8, 4,000 it | -0.41 | -0.22 | |

The categoricals carry most of it: where the trace sits relative to the off-block depends on the stand and runway.
Applied to unjoined matched rows the same residual model loses: the chain is at its limit there with these inputs.

**Twelve folds, feature_fraction 0.5** (months 1-6 on a GPU host, 7-12 on this Mac; `scripts/splice_rich_oof.py`
takes only the expert's scope rows from each part, so the GPU host's out-of-scope chain cannot leak in), against the
v38-equivalent: twelve 232.204 -> **231.248 (-0.956)**, Jan+Jul 291.449 -> 290.985 (**-0.463, CI [-0.734, -0.251]**,
January -0.486, July -0.446, both resolved), non-scored -1.094, **12/12 months** (-0.45 to -1.47), drop-1/10/50 -0.451 /
-0.374 / -0.175. `data/interim/v40_equiv_oof.parquet` is the new baseline.

The two taxi-length arrival windows (`arr_during_taxi_*`) are in the training matrix but not served (the inner model
ships `arrivals.FIXED_WINDOW_FEATURES`); the first rehearsal failed on them, so `RICH_EXCLUDE` drops them and the
production booster was refitted without them.

**Not levers.** The same inputs on the no-record expert: joined scope -227.2M non-scored against the v7 expert's
-236.8M (categoricals) and -237.0M (numeric); extending it to unjoined no-record rows outside Rome, -0.29 twelve but
all of it February (one month -3.6, Jan+Jul +0.03), not taken.

**Measurement note.** the GPU host's `lirf_unmatched_moe_oof.parquet` was the `lirf_moe_12h_cat` screen's OOF
(2026-09-18), this Mac's the older non-CatBoost mixture; the v29 reconstruction differs between the two machines on
Rome's no-record rows. The Mac copy was synced to the GPU host (the old one kept as `.the GPU host_old`).

**Shipped as v40: 244.19 -> 243.09.**

**Second round (same day).** Screens on months 3 and 9 (non-scored dSSE against the v7 expert; v40's config -155.2M
without the unserved columns): stronger categorical regularisation (cat_smooth 50, cat_l2 50, min_data_per_group 200)
-169.9M; 511 leaves -164.6M; both -193.2M; 1,023 leaves with it -196.9M (not taken); cat_smooth 200 -166.4M;
feature_fraction 0.3 -136.5M; `own_hold` as an input -157.4M.

**v8 trace features** (`own_flight_features.py --v8`, every v7 column byte-identical, same rows): over the departure
segment from its first moving point to take-off, the ground path length (`own_obs_path`), the ground speed at that
point (`own_first_gs`) and the number of stops (`own_stops`), with `own_hold`; `ownexpert.frame` derives the observed
speed and the time the unobserved stand-to-first-sighting stretch takes at it (`own_unobs_time`). Aimed at the rows
first seen mid-taxi, which carry 13.8% of all squared error under v40 (RMSE 159, against 93 where a stand exit is seen).
With the round-two settings: months 3/9 -276.1M. Twelve folds against the v40-equivalent: twelve 231.248 -> **230.485
(-0.763)**, Jan+Jul -0.600 (CI [-0.814, -0.441], both months resolved); against v38 12/12 months (-1.00 to -2.69),
drop-1/10/50 -1.05 / -0.96 / -0.76. `data/interim/v41_equiv_oof.parquet`. The no-record expert does not gain from v8
(paired against v7 twelve -0.009, 4/10 non-scored months). **Shipped as v41: 243.09 -> 242.30.**

**Smaller, not yet confirmed:** v9 reception features (`--v9`: points, largest gap, first gap in the observed segment;
derived points per minute and moving speed net of holds) -285.5M on months 3/9 against v8's -276.1M; a nested per-stand
first-sighting lag encoding (mean of `y - own_first_taxi_gap` per airport-stand and airport-stand-runway, smoothed with
50 pseudo-rows, clipped +/-1,800s) -292.2M.

**Checked and not taken (same day).** (1) The chain's blend weights on matched rows *without* a trace (38% of error),
re-weighted from the stored component OOFs: stacker weight 1.0 instead of 0.5 reads -114.5M non-scored, 10/10 months;
members 0.3/0.3 are already the best of the grid. Not taken: the stacker's OOF is not strictly nested (audit finding 2),
so a larger weight on it is exactly the direction that OOF overstates. (2) Joined rows with |residual| > 1,200s under
the v41-equivalent (1,756 rows, 28% of joined squared error): 925 LIRF, 354 LEBL; y ~3,000s where the anchor, the first
sighting, the stand exit and the dwell all read 500-1,000s (each within 300s of y on 2-8% of these rows), and 82% are
under-predictions. The recorded off-block precedes every physical clock by 30-40 minutes: the early-anchor reporting
pattern of section 31, which the trace cannot see.

**v9 + lag encodings, twelve folds** against the v41-equivalent: twelve -0.196, Jan+Jul -0.189 (CI [-0.288, -0.119]),
12/12 months, non-scored dSSE -150.6M. `data/interim/v42_equiv_oof.parquet` (twelve 230.289, Jan+Jul 290.196).
**Shipped as v42: 242.30 -> 242.03.**

**Third round, screens on months 3/9** (non-scored dSSE against the v7 expert; v42's config, seed 7: -304.2M): seed 8
-303.3M, seed 9 -292.1M (seed spread ~+/-6M); lr 0.0125 with 2,400 rounds -313.9M; lag encodings extended to
airport-operator and airport-type -302.6M (not taken); the mean of seeds 7-9 -318.9M, of seeds 7-9 and the lr 0.0125
fit -322.0M. Matched-row schedule copies (y = MVT - SCHED to the second on 4.4% of LIRF matched rows, 13.7% for one
operator) are already the `lirfregime` gate's, which the expert sees through `pred`; where they cost most (SCHED more
than 600s before AOBT_3) they are rare (LIRF 2.2%, LEBL 0.3%, LEMD 0.1%). EGLL's take-off-minus-MVT drifted through
2025 (-153s in April to -14s in December) and sits at -7/+5s in 2026, inside what the expert was fitted on.

**LIRF second stage: NEGATIVE.** Under the v42-equivalent, LIRF's joined matched rows (67,510; May-December 2025 only)
carry ~37% of joined squared error at RMSE 344, almost all on rows first seen mid-taxi. A small LightGBM on the residual
the v42-equivalent leaves there (`scripts/experiment_lirf_stage2.py`, twelve folds, the expert's inputs) loses in every
configuration: 15 leaves +0.668 twelve (0/10 non-scored months), clipped +/-1,800 +0.584, 63 leaves +0.916. What is left
on those rows is the early-record pattern, which a second stage only fits noise to.

**Seed bagging, twelve folds:** seeds 8 and 9 of the v42 expert (each alone twelve -1.89 / -1.84 against v38's expert,
12/12), averaged with v42's seed 7 on the scope rows: against the v42-equivalent twelve -0.112, Jan+Jul -0.110 (CI
[-0.166, -0.067]), 12/12 months. `data/interim/v43_equiv_oof.parquet`. **Shipped as v43: 242.03 -> 241.91.**

**Fourth round, screens on months 3/9** (v42's config, seed 7: -304.2M). The chain's components as expert inputs
(schedule gate, ordinary base, AOBT and LOBT members, stacker correction, as offsets from `pred`): -299.5M, not taken.
**v10 trace features** (`--v10`; v9 columns byte-identical): the first moving point's latitude and longitude (reception
zones), and MVT minus the run's own landing where the inbound leg is in the same ground run: -319.5M. LEBL's high RMSE
where the stand exit is seen (131 against 55-103 elsewhere) is a heavy two-sided tail: y is within 60s of the exit on
82% of those rows.

**v10 twelve folds: NOT taken.** Seeds 7 and 8 over twelve months, each against the same seed on v9 features: seed 7
twelve -0.040 (10/12, April +0.22), seed 8 -0.016 (8/12, April +0.26); the two-seed bags -0.029 twelve, Jan+Jul -0.012
(CI [-0.063, +0.037], not resolved). The months-3/9 screen (-15M) overstated it; v9 stays.

**Daily reception, twelve folds (seed 7): held, not shipped.** `ownexpert.daily_reception` (per airport-day over the
joined departures: median stand-to-first-sighting network distance, share with a stand exit seen, count; target-free)
against v42's seed 7: twelve -0.080, Jan+Jul -0.026 (CI [-0.096, +0.037], not resolved), 8/12 months; April -1.07 and
every other month within +/-0.09. April 2025 is an EHAM reception outage (median first sighting 524m against ~250m,
stand exits seen 13% against ~34%); EGLL had one June-October. The feature reads such regimes, and 2026 EDDF is one, but
the fold does not resolve it, so it is not shipped on its own. Serving support is in place (`own_expert_daily`).

**Neighbours' expert corrections for unjoined rows: closed.** For each unjoined matched row, the mean of the v43 expert's
correction over joined departures at the same airport within +/-15/30/60 minutes, against the row's v43-equivalent
residual: corr +0.015 / +0.012 / +0.007 overall, at most +0.08 at any airport. The chain's bias on untraced rows is not
shared by their traced neighbours (cf. section 45).

**Post-processing on the v43-equivalent (closed).** LOBT clip bound: +/-3,606 is right (+/-3,000 +485M non-scored);
scaling the expert's correction: alpha 1.0 is best (0.9 +195M, 1.1 +123M); flooring at the first moving point after the
last gap: catastrophic (+211,910M at 120s slack; that point is sometimes far too early).

**Blend with the lean expert: shipped as v44.** The v43-equivalent blended with the v38-equivalent (the v7-input expert,
255 leaves, feature_fraction 0.8, no matrix) on the expert's rows: non-scored dSSE minimum at weight 0.175 (-68.3M);
the v40 and v41 configurations add nothing (they are too close to v43). Against the v43-equivalent: twelve -0.100,
Jan+Jul -0.139 (CI [-0.198, -0.095]), 11/12 months. `data/interim/v44_equiv_oof.parquet`. 241.91 -> 241.79.

**Expert mix: shipped as v45.** Further twelve-fold expert OOFs: a lean expert on v9 + lag without the matrix (255
leaves, ff 0.8; alone -1.50 against the v7 expert, 12/12) and a CatBoost GPU expert on v9 + lag + matrix (depth 8, 6,000
it; -1.44, 12/12); a target anchored on the first moving point instead of the chain was dropped after two months (+1.34
in January alone). Non-negative least-squares weights on the non-scored months' scope rows over every stored expert OOF
(v9 seeds 7-9, v9 + daily, v10 seeds 7-8, lean v7, lean v9, CatBoost): with CatBoost, against the v44-equivalent twelve
-0.230, Jan+Jul -0.201 (CI [-0.293, -0.132]); without it (served as v45) -0.207 / -0.173 (CI [-0.256, -0.112]), 12/12.
`data/interim/v45_equiv_oof.parquet`. 241.79 -> 241.70.

**Correction (same day): the v45/v46 fold numbers.** The NNLS weights summed to 1.0008-1.0009 (the sum constraint is a
penalty), and the blended OOF was taken as `A @ w` over every row, so rows outside the expert's scope were scaled by
~1.0009. Serving normalises the weights and touches scope rows only. Rebuilt with normalised weights on scope rows
(`v45_equiv_oof.parquet`, `v46_equiv_oof.parquet` overwritten), against the v44-equivalent: v45 twelve -0.184, Jan+Jul
-0.146 (recorded above as -0.207 / -0.173); v46 twelve -0.207, Jan+Jul -0.175 (CI [-0.253, -0.118]; January -0.119, July
-0.232). Both remain resolved gains; the weights themselves were fitted on the non-scored months and are unchanged.

**Base model retune (the chain's decomposition).** The matched base had kept feature_fraction 0.9 and 255 leaves since
v20. `scripts/cross_validate_v20.py --uncapped --rome-rules --base-params ...` (now with `--months` and `--base-params`;
single seed, as the stored v21 OOF), months 3/9 against the stored v21 OOF, final chain-stage prediction: feature_fraction
0.5 -5.5M / -11.4M; feature_fraction 0.5 + 511 leaves -30.2M / -35.6M (ordinary base on matched unjoined rows -42M / -40M).
Twelve folds of the latter, rebuilt into the v29 reconstruction (same members, stored stacker correction, LOBT clip): the
chain alone twelve -0.178. Taken only on rows outside the experts' scopes (expert, no-record expert and garbled rows keep
v46's values), against the v46-equivalent: twelve -0.099, 12/12 months (10/10 non-scored), Jan+Jul -0.048 (CI [-0.123,
+0.048], not resolved; July -0.085 resolved, January -0.011). `data/interim/v47_equiv_oof.parquet`. Production refit:
`scripts/refit_v20.py --atfm --uncapped --rome-rules --base-params ... --inner-from mechanism_mechanism_v26_inner`.

**Shipped as v47: 241.62 -> 241.31.** Error map under the v47-equivalent (twelve 229.77): day slips 27.3%, LTFM matched
12.6%, joined matched outside Rome 12.3%, LFPG unjoined 9.4%, Rome no-record 9.1%, other unjoined matched 9.0%, Rome
matched unjoined 7.8% / joined 7.2%. At LTFM and LFPG 0.55-0.65% of rows (|residual| > 900s) carry 25-27% of the
airport's squared error, with the true off-block ~23-25 minutes before NM's AOBT: the section 31 early-anchor pattern,
which no observable flags there. The base model is what moves these blocks. Capacity screen, months 3/9 against the
stored v21 OOF: 511 leaves -30.2M / -35.6M, **1,023 leaves -41.0M / -50.9M** (matched unjoined -15.4M / -19.1M).

**Runway state and gate retention in the base (`prc.rwystate`; from team outgoing-boat's public repo).** From the
departure frame alone (no target): mean interval of the airport's last 20 departures, this runway's share of departures
within +/-60 min, distinct runways in the last hour and whether that set changed from the hour before; at the NM
off-block clock, departures past their EOBT not yet pushed (EOBT in the 2h before), and the count and mean AOBT_3 -
SCHED of the pushes in the 15 min before. On top of the v47 base, months 3/9: -50.2M / -69.6M (LTFM -14.7M / -13.1M).
Twelve folds (the GPU host), rows outside the experts' scopes, against the v47-equivalent: twelve -0.107, Jan+Jul -0.133
(CI [-0.241, -0.002]; July -0.252 resolved, January -0.016), 11/12 months (February +0.51); the chain alone against
the old v21 base -0.421. `data/interim/v48_equiv_oof.parquet`.

**Day slips re-examined (2026-10-05).** Of the 34 rows with y > 12h in 2025, 31 are LIRF, 33 have no NM record, and 19
have BLOCK equal to SCHED to the minute (schedule copies of flights that left 12-36h after schedule). Under the
v48-equivalent the copies cost 1.1% of squared error (12 of 19 predicted to the second by the Rome rules); the other 15
cost 26.2%: BLOCK ~24h before the real off-block (y 23.4-24.6h), at LFPG (2), LSZH (1) and one LIRF row with a normal
schedule gap, and 11 LIRF rows with gaps of 14-26h predicted near the gap. A blend p x gap + (1-p) x mean(non-copy y)
on LIRF no-record rows with large gaps, estimated leave-one-month-out (p 0.61-0.80, non-copy mean 24.3h): NEGATIVE in
every scope (gap > 8h +2.80 twelve; > 12h +1.58; 12-24h +2.03; 8-24h +3.14). The Rome rules already isolate the
near-certain copies and hedge the rest (mean prediction 1.05 x gap); a single blend throws the exact hits away. Closed.

**Correction, cause of the day slips (2026-10-06).** The paragraph above and section 58's addendum 2 describe these
rows as BLOCK "~24h before the real off-block". That is an interpretation the evidence does not settle. By airport:
- **LSZH (1):** BLOCK = SCHED - 86,402s. A date error: no real delay produces that.
- **LIRF copies (19):** BLOCK = SCHED and take-off 12-36h after schedule. The delay was real; the recorded off-block
  is the schedule, not when the aircraft actually pushed back.
- **LFPG (2): the aircraft was grounded at CDG across the recorded block.** Raw adsb.lol traces searched by
  registration (17-19 and 21-23 Jan 2025):
  - OE-LVK (440cfe): seven legs on 17 Jan including EJU983W at 18:39; on 18 Jan lands LFPG 10:14 as EJU68UL and
    does not fly again until the 19 Jan 18:55 EJU983W take-off. **Recorded block 18 Jan 19:30 falls in that 32.5h
    ground stay**, about an hour after EJU983W's usual slot; NM has no EJU983W on the 18th.
  - OE-IZN (440ca9): lands LFPG 21 Jan 22:13 as EJU76VR, no trace at all on 22 Jan, next movement the 23 Jan
    15:10 EJU42AY take-off. **Recorded block 22 Jan 22:58 falls in that 41h ground stay.**

  Not a typed date: each aircraft was out of service for a day and the block most likely records a real push-back
  of a flight that did not leave, kept when it flew the next day. The push-back itself is unobservable (no ground
  coverage at CDG).

The target is the recorded `MVT - BLOCK` either way (`data.html`). Section 29's "no signature in any field we hold"
stands for the supplied fields, but the departing aircraft's ground time before take-off (registration from the
trace join) is a candidate signature; measured below ("Aircraft ground time before take-off").

**Shipped as v48: 241.31 -> 240.51.** Stand-to-runway distance (`prc.standrwy`, X-Plane network and straight-line,
10,195 of 12,447 airport-stand-runway combinations located) with cell-spread encodings (p90 / std of y per cell,
`tencode.fit(spread=True)`), on top of the runway-state base, months 3/9: -10.0M / +2.7M. Not taken.

**Istanbul's NM AOBT is mostly synthetic (2026-10-06).** Label-free: at LTFM, `MVT - AOBT_3` is exactly 19, 20, 24 or
25 minutes (+/-5s) on 223k of 273k 2025 departures (82%), a standard taxi time that appears to follow the runway
configuration (19/20 in summer, 24/25 more often in winter). The mix is the same in Jan and Jul 2026 (Jan: 25m 38%,
20m 22%, 24m 18%, 19m 10%; Jul: 20m 54%, 19m 27%), so no change between 2025 and 2026 in how NM receives IST
times (EUROCONTROL LSSIP 2024 Türkiye: LTFM A-CDM "NM integration process is still ongoing"). On those rows the anchor
says nothing about the real off-block. The v48-equivalent is already unbiased there (mean residual -6 to +1s, RMSE
198-229) and they carry 62% of LTFM's squared error; LTFM rows with no AOBT (3,645) carry 14% at RMSE 771.
Not a lever by itself: an explanation of why LTFM matched rows are 13% of the error, with near-zero adsb.lol
ground coverage there.

**Aircraft ground time before take-off: NEGATIVE, closed (2026-10-06).** `scripts/adsb/air_legs.py` extracts every
airborne leg of every aircraft from the raw adsb.lol traces (364 days of 2025, `data/interim/air_legs/`, ~190k legs a
day); `scripts/experiment_ground_idle.py` joins each departure to its take-off leg (callsign, airport, first airborne
point in [MVT-120s, MVT+900s]) and takes ground time = MVT - the end of the aircraft's last leg ending >= 10 min
before MVT, kept only if that leg ended low within 20 km of ADEP. Join 83.6% of departures (no-record 49.9%), ground
time known 82.3%; LTFM 1,883 rows only (no coverage). Against the v48-equivalent:

| scope | ground > 24h flagged | day slips caught | hit rate | in-sample hedge saving | day-slip SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-record outside LIRF | 687 | 3 of 3 | 0.44% | 76.6M | 17,750M |
| no-record, all | 703 | 3 of 33 | 0.43% | 60.6M | 30,024M |
| matched outside LIRF | 43,511 | 0 | 0 | 0.4M | 0 |

Every non-Rome day slip is a long-grounded aircraft (LFPG 32.7h and 40.9h, LSZH NJE389D 76.1h, a business jet), but
so are 684 ordinary no-record departures a year, so the flag catches the rows without isolating them. The saving is
an in-sample upper bound and is worth < 0.1s of RMSE; any threshold from 8h to 30h gives the same picture.
General use is dead too: on 1.72M rows with y <= 12h the residual mean is within +/-8s in every ground-time and
turnaround-slack bin, corr(residual, log ground time) <= 0.02 at every airport. A physical clip
`y <= MVT - inbound landing` is +17,246M worse: the recorded target breaks it on 1,306 rows (285 LIRF no-record,
schedule copies written before the aircraft landed), and the target is the record.

**Outside-the-box screens (2026-10-06), all negative.**
- *Arrival recording quality as a departure noise flag.* Arrival in-blocks stay in the ranking set and come from the
  same airport system. Per departure, over arrivals landing at ADEP within +/-60 min: mean taxi-in anomaly against the
  (airport, stand) median of the other months, share with |anomaly| > 600s, share of in-blocks copied from schedule.
  A strong *variance* signal: v48 residual RMSE runs 98 -> 276 across deciles of the |anomaly| > 600s share
  (corr with |residual| 0.16), but the residual mean is flat (within +/-3s per decile). A month-fold LightGBM corrector
  on the v48-equivalent residual, paired against the same corrector without the arrival features: +70.7M, 5/12 months,
  +0.087 twelve. It says where the error is large, not which way. Closed.
- *ID-order and rounding artifacts.* Within (airport, day), rank-corr(MVT_ID, MVT) equals rank-corr(MVT_ID, BLOCK) to
  three decimals at every airport; block seconds are not rounded. No artifact (and not a legitimate input anyway).
- *Filed block time (ARVT_1 - EOBT_1) against its leave-one-month-out route/type median* (kind-mango, -1.2 on their
  scale): corr with the v48 residual 0.0004, at most 0.8M of 63,052M if every bin were debiased.
- *Towing / stand change via the tail chain.* `experiment_ground_idle.py --save` writes each departure's inbound leg
  (landing time, callsign); joined to the inbound's arrival row (callsign, airport, landing within 30 min) for
  1.69M of 2.09M departures. Departures whose stand differs from the inbound's (117k, 7%): v48 residual mean -0.43s,
  the same as the rest; no airport biased beyond +/-12s (LIRF, 4k rows). Recorded in-block -> take-off: flat except
  735 rows under 30 min (-53s, ~2M). Closed.
- *De-icing / snow (lead from the A-CDM research: remote-pad de-icing happens after off-block and lengthens the
  recorded taxi).* v48-equivalent residual in snow, cold months: unbiased everywhere except LTFM (+219.5s on 3,380
  snow-now rows, 163M of the 168M a per-airport debias could remove; EGLL -150 and LFPG -92 on <200 rows each).
  The LTFM bias is a few days, not snow: 19, 23 and 24 Feb 2025 +500 to +930s, while 11 Feb (heaviest snow) -15 and
  30 Dec -170; neither snow intensity, visibility, temperature nor the day's arrival taxi-in anomaly separates them
  (20 snow days). Leave-one-day-out flat shift on LTFM snow rows: -69.5M twelve (-0.073), but February -108M and
  December/March/April worse: one storm week, not shippable. Exposure noted: Jan 2026 holds 1,921 LTFM snow
  departures (12, 18 and 19 Jan heavy) against 0 in Jan 2025, so the ranking set carries this risk either way.
  Follow-up (Instinct: all five dates were news-confirmed disruption days; proposed a daily METAR SN-count flag): over
  all 2025 LTFM days, the six with >= 10 hourly SN reports split +531, +531, +176 (23, 19, 6 Feb) against -16, -23,
  -95 (11, 21, 22 Feb). 11 Feb had the most snow reports (19) and no bias; 21-22 Feb were ministry-restricted days
  (448 and 473 departures against ~620) and over-predicted. Disruption and snow do not fix the sign. Closed.
- *Base capacity, 1,023 leaves on the v48 base (runway state, feature_fraction 0.5).* Paired screen against the same
  config at 511 leaves, months 3/9, chain alone: -2.48 / -2.96 against -2.29 / -2.83 (diff -0.19 / -0.13). Under the
  experts the chain reaches ~42% of rows; by v48's own ratio (chain -0.42 -> -0.11 under the experts) the expected
  twelve-fold gain is a few hundredths. Twelve folds stopped at the user's call: not worth the compute.
  `scripts/compare_base_swap.py` (self-tested at exactly zero on v48 against itself) is the tool if a base swap is
  ever worth confirming.

**Where the top teams' gap came from, and our ADS-B coverage in the scored months (2026-10-06).** Full submission
history (`datacomp.opensky-network.org/api/competitions/<id>/leaderboard`, 4,608 rows): the leaders made single
jumps of 16-40 points between 24 Sep and 1 Oct (strong-crane 253 -> 236 in one upload on 26 Sep; charming-eel -24 on
27 Sep; versatile-watermelon -21 on 28 Sep; adventurous-radio 263 -> 226 on 30 Sep - 1 Oct; quick-boat -16 on 24 Sep),
after the ADS-B rulings (17-19 Sep) and the X-Plane stand post (24 Sep). Ours over the same window was -8 (v31). The
gap is most likely the same source (adsb.lol) used more effectively, not data we lack.
Own-flight join rate by airport (`own_joined`, v10 features), Jan/Jul 2025 -> Jan/Jul 2026: LIRF 0 / 94% -> 77% / **0.2%**;
LFPG 0 / 6% -> 15% / 13%; LTFM 0 throughout; EDDF 91% -> 71-76%; others 67-97%. The LIRF July 2026 hole is coverage, not
our extraction: the raw archive on 2026-07-15 has 0 ground points within 8 km of LIRF and 14k airborne points against
10k ground and 39k airborne on 2026-01-15 (the Rome-area ground receiver went offline). Every adsb.lol user shares it.
- *A second ADS-B source.* Instinct's licence survey: none legal (adsb.fi and airplanes.live non-commercial with no bulk
  history; ADS-B Exchange history paid; ADSBHub / ADS-B One / theairtraffic not openly licensed; OpenSky-derived sets
  forbidden). adsb.lol's own daily `mlatonly` releases exist for every day of 2025 and 2026 (~180 MB/day, same ODbL
  licence): 2026-07-15 holds 0 ground points within 8 km of LIRF, LTFM, LFPG, EHAM or EDDF, and 26 / 59 airborne points
  near LIRF / LFPG. MLAT does not see the surface. Closed: adsb.lol prod is the only usable trace source.

**Deep dive: traced vs untraced departures (2026-10-06), and why no multi-point lever is left.** v48-equivalent, 2025,
y < 12h (RMSE 195.8): matched + joined 1.21M rows RMSE 133 (26.8% of SSE); **matched + unjoined 0.85M rows RMSE 224
(53.3%)**; no-record unjoined 17k RMSE 759 (12.2%); no-record joined 5k RMSE 1,055 (7.7%).
- Where the trace joins, v48 already beats every single trace proxy on the same rows (with a per-airport median offset):
  EDDF stand-exit 81.6 vs v48 58.2; EHAM 85.7 vs 65.8; LSZH 96.8 vs 55.2; EDDM 111.0 vs 53.7. Our use of the trace is
  not the gap. The joined residual is concentrated: top 1% of rows carry 54% of its SSE.
- The unjoined matched rows are a coverage gap, not a join gap. Jan + Jul 2025, 163,006 unjoined matched departures:
  67% have an airborne take-off leg (`air_legs`), but only 2.0% (3,201) have >= 5 ground points of that aircraft at ADEP
  in the 3h before take-off. An aircraft-keyed join recovers nothing. LTFM: 1% even have an airborne leg.
- Surface traffic of other aircraft (section 45's `sx_*`, closed then on the no-record lane only) on the matched
  unjoined lane: present on 8% of rows, |Spearman| <= 0.02, paired month-fold corrector +38.5M, 4/12 months, +0.053.
Conclusion: with adsb.lol as the only legal trace source, the untraced half of the error has no observable left in
the data we can use. The leaders' larger late-September jumps are not explained by anything measurable here.
- *Delayed-flight schedule copies on matched rows.* `|y - (MVT - SCHED)| <= 60s` holds on 9.6% of matched 2025 rows (16.7%
  of matched SSE); harmless at schedule gaps of 5-20 min (on-time push-backs), costly when the flight was late: gap
  30-60 min 8,883 copies RMSE 661 (mean +445), gap > 60 min 1,060 copies RMSE 1,918 (mean +1,413). Copies carry 27.7% of
  LEBL's and 41.4% of LIRF's matched SSE. Leave-one-month-out copy propensity among delayed rows by operator, callsign
  prefix, flight number or stand (k = 20 shrinkage) ranks them (AUC 0.82-0.92) but the base rate is 1.4% (gap > 30 min)
  / 0.7% (> 45 min), and the mixture p*gap + (1-p)*pred is worse in every month for every key: +974M to +6,586M,
  +1.3 to +8.8 RMSE. Closed.
  Side fact: matched rows with a schedule gap > 30 min are 35% of rows and 67% of matched SSE (RMSE ~245 vs ~125).
- *Early-anchor rows, traced vs not (section 31 pattern, re-cut on the v48-equivalent).* Delayed matched rows (gap > 30
  min): 2.7% of rows with |res| > 600 carry 58% of their SSE, almost all with y - anchor > 600s. Where the trace has a stand
  exit on those rows it sides with the record 87% of the time (median |exit - y| 45s vs |exit - anchor| 784s): the aircraft
  left the stand early and waited; NM's AOBT marks a later moment. v48 by trace depth on early rows: stand exit seen 11.1k
  RMSE 342 (2.0% of matched SSE); dwell only 10.7k RMSE 399 (2.7%); first point only 18.4k RMSE 600, mean +285 (10.4%);
  unjoined 37.6k RMSE 652, mean +406 (25.0%). Prediction = stand-exit taxi + airport median offset on the stand-exit rows:
  360 vs v48 342 (v48 already better). Clipping the prediction up to the first trace point (`own_first_taxi - c`, first
  point <= 30/40/60 min, c = 60-300s) is worse in every variant and every month (+0.13 to +7.9): the recorded value falls
  below it on 12% of joined rows and the first point is not a usable bound. Closed.

**How Rome writes a day offset (2026-10-06).** All 11 LIRF no-record day offsets of 2025 (y = 86,400 x date difference
+ taxi) have the recorded block = **the schedule's local date + the actual push-back time of day** (e.g. EXSXS1GT:
sched 24 Jan 11:55, block 24 Jan 04:24, take-off 25 Jan 04:37). Schedule copies take both date and time from the
schedule; so both kinds carry the schedule's date (50/50 rows), and a day offset is impossible when the scheduled and
actual local dates agree (all 10 such rows are copies; `rome_rules` same-day exact already uses this). Among gap > 8h
no-record rows with different dates, by gap band: 8-14h 21 copies / 0 day offsets; 14-20h 7 / 7; 20-26h 0 / 4; > 26h
1 / 0. Garbled strings, operator and time of day fall on both sides in the 14-20h band; nothing separates it.
Ranking set: 22 LIRF no-record rows with gap > 8h (8 Jan, 14 Jul), 13 with different dates; the 8-14h ones are already
predicted as exact copies, and the open ones are three 14-20h rows (WMTT3508, SVII3006, ITYTY672) and one 31h row.
Expected SSE under v48 at the 2025 band rates ~0.7G (a perfect oracle on them: about -4 RMSE on the board), not the
26-point gap to the leaders.

**Discord read (23 Sep - 5 Oct) and the 2025 -> 2026 trace drift (2026-10-06).** Leaders disclosed nothing specific.
GREKI's methodology (27 Sep): validate January <-> July; check every input for 2025 -> 2026 drift ("features encoding
where an aircraft was heard didn't transfer; features describing its movement did"); keep record-contradicts-sensor
rows; correct a strong base and average a second model family. Other teams confirm the same coverage wall (LTFM, LIRF
July 2026, LFPG thin). OPDI (organiser 6 Sep: milestones do not cover off-block; licence unconfirmed) already assessed.
- *Drift is real.* Label-free classifier, joined rows, 2025 Jan/Jul vs 2026 Jan/Jul: AUC 0.903 on the v10 trace
  features (gain led by `own_first_lon/lat`, `own_to_minus_mvt`, `own_npts`). Univariate |2*AUC-1| up to 0.80
  (`own_to_minus_mvt` at LEMD), 0.68 (`own_npts` at LIRF); movement features (dwell, stand exit, since-land) <= 0.28.
- *Dropping the drifted inputs does not help.* `lean_v9_lag_mov` (= the shipped `lean_v9_lag` minus 12 features with
  drift >= 0.45 and their derivatives) against the stored `lean_v9_lag` 12-fold OOF: +1,885M, 0/12 months. Reweighted to
  2026 by the drift classifier's odds (`scripts/compare_drift_weighted.py`; weights clipped 0.05-20, effective n 236k):
  RMSE on scope 166.1 -> 172.6 (+6.5), worse at every airport, drop-50 +7.2. The reception features carry more, not
  less, under the 2026 distribution. Closed.

**Surface congestion of other aircraft in the no-record expert (2026-10-12).** Section 45's `sx_*` (taxi-out/in activity,
departures ahead, their hold and taxi, surface speed; coverage-normalised) added to the shipped no-record expert
(`experiment_own_norec.py l31 --v7 --sx`), paired twelve-fold against the same run without them. Joined lane: twelve
-0.342 -> -0.334 (+0.008), nothing. Extending the expert to every no-record row outside LIRF with `sx_*` and serving it on
the unjoined ones (shipped expert kept on joined rows): twelve +0.227, Jan+Jul +0.275, Feb+Jun +0.774 (February +1.93),
worse in 9/12 months, drop-50 +307M. Third negative for observed surface queues (sections 45, 63); closed on every lane.

**gracious-tornado's levers on our chain (2026-10-12; their `docs/METHOD.md`, board 232.27).** All against the
v48-equivalent OOF, twelve folds as stored.
- *Daily pre-departure delay* (`all_pre_departure_delays_<y>.csv`, DLY_ALL_PRE_2, airport x take-off day; covers Jan-Jul
  2026). Identity verified: sum of max(0, BLOCK - SCHED) over DEP rows keyed on MVT day equals it, corr 0.9996, median
  |gap| 2.8 min. `scripts/experiment_daily_predep.py`: a common day shift on ordinary rows (d <= 1h), loading fitted
  leave-one-month-out: twelve -0.188, 11/12, but Feb+Jun +0.470 (February +107M). Copy-vs-held posterior on LIRF long
  rows (exact convolution, Student-t(3), day noise sigma 5,273s at LIRF): d > 1h twelve -0.144 with drop-1 +537M; d > 2h
  +0.497; d > 4h +0.648. The February gain is one row each time; our LIRF chain already holds what the sum resolves.
  No organiser ruling names the file. Not shipped.
- *Cap y <= d where d > 2h*: +1.659 (our day-offset rows legitimately exceed d). *LEMD/LEBL <= 3604s*: -0.008, 6 rows.
  *60s floor*: -0.001.
- *Recovered orphans* (no-record departure -> same flight number and route arriving 10 min-6h later): 885 rows/yr.
  The LOBT +/-3606s window holds on 2.6% of them (they are the rows that failed the organisers' join), and the
  arrival-side AOBT_3 as an anchor (y within 120s on 35%) loses to the chain (RMSE 3,335 vs 3,068): blend w 0.3 +0.248,
  9/12 months worse.
- Monthly level (section 61, v39): already tested, no change.
