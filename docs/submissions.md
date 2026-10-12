# Submission log

Scores are RMSE in seconds on the organisers' hidden `truthing.parquet` (January +
July 2026), returned in `<file>_result.json` in the team bucket. "Holdout" is our own
month-disjoint validation on 2025 January + July.

| v | holdout | scored | model | note |
| --- | ---: | ---: | --- | --- |
| 1 | 346.3 (filtered) | **602.06** | LightGBM, time + congestion features | holdout was filtered to taxi < 2h and badly optimistic |
| 2 | 458.2 | **411.55** | + NM off-block proxy features, unfiltered validation | previous best |
| 3 | 406.5 | **438.42** | two-stage P(tail)·E[y\|tail] + (1−P)·E[y\|bulk] | better holdout, worse score |
| 4-5 | 388.9 | **403.05** | decomposition, all-month refit | v5 an identical duplicate |
| 6 | — | rejected | schedule-fallback mechanism | daily limit; **a rejected version number is spent** |
| 7 | 290.6 OOF | **349.22** | identical file to v6, new version number | current best |
| 4 | 388.85 | **403.046** | capped clock residual + missing-clock expert + tail excess | current best; refit on all 12 months |
| 5 | 388.85 | **403.046** | prediction-identical to v4 | infrastructure/provenance check only |

Unfiltered holdout for v1 was 605.8, which matches its 602.06 — the 346.3 came from
excluding the rows that dominate the metric. See `outlier-analysis.md`.

## What each submission taught us

**v1 → v2.** The NM off-block times (`AOBT_3_flt` and friends) survive in the ranking
set even though `BLOCK_TIME_UTC_mvt` is blanked, and `MVT_TIME − AOBT_3_flt` reads the
target almost directly. Worth 190s.

**v2 → v3.** A two-stage model that scores 52s better on the holdout lost 27s on the
real test set. Category-code mismatch was rejected because predictions were byte-identical.
The tail-calibration experiment rejected only a global scalar correction; it did not rule out
classifier instability or segment shift because its folds were not nested around the outer
holdout. The old bulk regressor's `y <= 3600` filtering was subsequently replaced in v4.
The remaining transfer error is concentrated in unstable missing-clock/reporting regimes.

**v3 → v4.** Replaced the bulk-only regressor with a capped-target model trained on every
row, anchored AOBT-present rows to `MVT - AOBT`, and added a separate missing-clock base.
The tail remained a probability-weighted excess model. Local January+July RMSE improved
17.6s and the hidden score improved 35.4s over v3, but only 8.5s over v2. Twelve-month OOF
was 312.86 full / 222.25 bulk, but the non-LIRF tail scale was selected on those same OOF
predictions, so the full figure is optimistic.

**v4 → v5.** No model change. The two submission frames are exactly equal and both scored
403.046. A second full refit followed by prediction produced an equal prediction frame, which
verified deterministic output for the current environment. It must not be treated as a second
model-selection observation.

## Measurement health warning

Tail prevalence swings 5× month to month (February 0.530%, March 0.095%). A two-month
holdout is a single draw from that distribution, so full-RMSE differences of ~25s — the
size of the v2/v3 gap — are not reliably distinguishable from noise. Bulk-only RMSE is
the stable signal; treat full RMSE as having a wide error bar. Before spending more
submissions on tail-sensitive changes, the validation needs to be leave-one-month-out
across all 12 months.

## Rules

Five submissions per day, 1 GB per bucket, ranked on best score across all submissions.
The organisers explicitly forbid learning from or exploiting the ranking process, so
parameters are never tuned by submitting variants and watching the score — the tail
shrinkage factor in `scripts/calibrate_tail.py` is fitted on internal data only.


## Operational lesson from v6/v7

A submission rejected for the daily limit cannot be recovered by re-uploading the same
filename, even days later: the scorer returns the same `DAILY_LIMIT_REACHED` message
regardless of the actual quota state. The identical file uploaded as the next version
number scored immediately. Check the bucket for the day's uploads before submitting -
the session's own tally is not the authority.

## v8 — day-offset regime gate

Sixteen 2025 departures carry a taxi-out of ~24h plus an ordinary taxi, because the
block time refers to the previous day. They hold ~19% of total variance. The gate adds
`P(day offset | band) * 86400` to the mechanism prediction, where the probability is an
empirical rate over observable bands (AOBT missing x reporting airport x MVT-SCHED),
shrunk hard (prior 30) because 16 positives cannot support a confident rate.

- 12-month OOF: 290.61 -> 287.03 (-2.45% SSE), both ranking months better, 2 months worse.
- **The shrinkage prior was chosen with OOF results visible**, so that figure is optimistic.
- Fires on 11 ranking rows, expected positives 1.45. Worst case if all 11 are normal:
  about +7s. Submitted score **348.97**, against 349.22 for v7 — a gain of 0.25s.

The 2025 band rate (41% in LIRF/12-20h) clearly overstates the 2026 scored set. Further
tuning of this gate against the submitted score is prohibited by the challenge rules and
by docs/rules-and-compliance.md; the gate stays as fitted from training data.

## v9 — METAR ground weather (first external data)

Source: Iowa Environmental Mesonet ASOS/METAR archive (Iowa State University),
**public domain** (https://mesonet.agron.iastate.edu/disclaimer.php). Retrieved by
`scripts/fetch_metar.py`; ten airports, Jan 2025 - Jul 2026, ~13,850 hourly observations
each. Permitted under the organiser ruling of 2026-09-08: "Open source data are allowed
if declared in the documentation for the solution/repo." (OpenSky state vectors are NOT
permitted - ruling of 2026-09-03.)

Ten features: temperature, wind, gust, visibility, humidity, and flags for snow,
freezing, fog, rain and low visibility. Matched backwards to each movement within 2h;
100% of ranking departures matched.

Paired 12-fold OOF, gate and blend held fixed so the delta is attributable:

| segment | n | without | with | SSE |
| --- | ---: | ---: | ---: | ---: |
| all | 2,085,047 | 290.61 | 287.18 | -2.35% |
| bulk | 2,080,921 | 234.42 | 229.10 | -4.48% |
| freezing | 165,340 | 376.86 | 357.27 | **-10.13%** |
| low visibility | 21,028 | 383.98 | 364.11 | **-10.08%** |
| snow | 6,842 | 853.64 | 827.79 | -5.96% |
| tail | 4,126 | 3868.30 | 3899.29 | +1.61% |

11/12 months improved. January -8.6s, February -19.3s, July unchanged.

**Submitted 348.97 -> 345.39, a gain of 3.58s against an OOF prediction of 3.43s.** The
closest holdout-to-hidden agreement of the project so far.

The driver is not snow. Freezing conditions (165k rows) and low visibility (21k rows)
each gave ~10% squared-error reductions, and bulk improved 4.48% - the first time any
change has moved broad accuracy rather than the tail. Note v9 omits the v8 day-offset
gate, to isolate the weather effect; recombining is worth ~0.25s.

## v10 — crosswind and weather trends

Extends v9's ten point-in-time observations to 21 features:

- **Runway geometry** (OurAirports, public domain, `scripts/fetch_ourairports.py`):
  crosswind and headwind resolved onto the actual departure runway, plus runway length.
  A 20kt wind matters completely differently down the runway versus across it.
- **Trends**: temperature delta 1h/3h, visibility delta 1h, wind delta 1h. One
  observation cannot distinguish clearing fog from thickening fog.
- **Accumulation**: snow-hours and freezing-hours over 6h, fog-hours over 3h, minimum
  visibility over 3h. De-icing demand follows accumulated cold, not instantaneous
  temperature.

Paired 12-fold OOF against v9: **287.18 -> 286.88 (-0.20% SSE)**, 8/12 months improved,
January -0.04s, July +0.10s. Snow -4.31%, freezing -1.42%, low visibility -1.03%.

**Submitted 345.39 -> 344.21, a gain of 1.17s against a predicted 0.30s.** The holdout
understated it, as it did for v9 - consistent with the scored January 2026 being far
snowier than January 2025 (5,480 snow departures vs 1,170), so weather features are worth
more on the hidden set than any 2025-based holdout can show.

Diminishing returns are clear: v9's ten crude features bought 3.58s, these eleven
additional ones bought 1.17s.

## v11 — burned by a UTC/local date confusion

v11 was rejected for the daily limit and is therefore spent. The quota resets at **00:00
UTC**, not local midnight: with IST at UTC+5:30, the local date reads one day ahead for
five and a half hours every evening. v6 through v10 all went up on 15 September UTC, and
v11 was the sixth.

This is the second version burned the same way (v6 was the first). `prc.submit` now
refuses to upload when the bucket already shows `DAILY_LIMIT` submissions for the current
UTC day, listing them, rather than relying on anyone's memory of what was sent.

## v12 — sequencing features (ready, awaiting quota)

Departure sequencing from the challenge's own data - no external source. Wake-turbulence
separation means a departure behind heavy traffic waits longer, so the features describe
the *mix* of traffic ahead rather than its volume:

- `seq_heavy_ahead_15m`, `seq_heavy_share_30m`, `seq_prev_wake_heavy`
- `seq_departures_30m_runway`, `seq_median_gap_30m`

`seq_heavy_ahead_15m` scores **+0.048 against the model residual** - the strongest of
anything tested in this project (weather gust +0.043, queue counts +0.003) - with a
monotone gradient: 0 heavy ahead gives a -18.9s residual, three ahead +11.0s.
`seq_prev_gap_s` measured +0.003 and was dropped.

Paired 12-fold OOF: **286.45 -> 285.65 (-0.56% SSE), 12/12 months improved**, the only
change in the project to improve every month. January and July both better.

Also folded in: the tail severity capacity raised from 15 leaves/57 rounds to 63/300.
That looked worth 8.1s on a single Feb+Oct split but only 0.43s across 12 folds, and made
January slightly worse - a reminder that one split overstated it by roughly 19x. Kept
because July improved 7.6s and the combined OOF is better, but not worth a slot alone.

## Unexplained-long-rows gate — NEGATIVE, not submitted

The 2,362 rows that are long but neither a schedule copy nor a day offset carry 7.8% of
variance and were the largest unmodelled block. A gate on scoring-time features reaches
**AUC 0.923** out of fold with well-calibrated mass (2,520 predicted vs 2,362 actual), so
detection works.

The mixture does not. Paired 12-fold OOF: **285.65 -> 317.15, +23.3% SSE, 0/12 months.**

The severity model is the fault. It is fitted on ~2,100 regime rows per fold and then
asked to predict for every row in the held-out month; on rows unlike its training set it
extrapolates to extreme values, and multiplying those by a non-trivial gate probability
destroyed the 4,768 gated rows (+443% SSE) while improving the 2,362 genuine regime rows
by only 10.8%.

The lesson is about sample size, not structure: the same mixture works for the
schedule-fallback regime because that has 199,634 training examples. Two thousand is not
enough to fit a magnitude model that will be applied out of distribution.

Estimated gain beforehand was ~2s. This is the third consecutive over-estimate on the
tail (24h gate: expected ~54s, got 0.25s; tail capacity: expected 8.1s, got 0.43s), which
is itself evidence that tail-directed work here is near exhausted.

## v12 and v13 — scored

v12 (sequencing features, `mechanism_wx`) scored **342.86**; v13 (+ four AOBT-anchored
arrival features, `mechanism_arr`) scored **342.22**, the current best. Both uploaded
2026-09-16.

## v14 — identifiers removed from the base models

`models/mechanism_noid_arr`: the v13 configuration (same 71 features) with `FLIGHT_mvt`
and `CALLSIGN_flt` excluded from both base models and stronger categorical regularisation. See experiment
log section 11: 36-42% of July 2026 rows carry identifier levels unseen in 2025, against
under 2% in any CV holdout, so the whole-month CV structurally understates this change.
Paired 12-fold: against v12 Jan+Jul -0.31 / 12-month -0.91; against v13 Jan+Jul -0.10 /
12-month -0.76; 12/12 months both times, scored-month delta fragile under the -10 test.
The submission rests on the identifier mismatch, not on the CV delta. Leakage probe PASS.
Predictions differ from v13 by 54s RMS, max 1,636s.

**Scored 342.22 -> 340.35, a gain of 1.88s**, inside the 1-2s expected from the masking
simulation and well beyond the -0.10s the whole-month CV showed against v13. Uploaded
2026-09-16 (third of the day's five). This is the first change whose board result was
predicted by a mechanism the CV structurally cannot measure; per policy it is a single
observation and nothing is tuned from it.

## v15 — LIRF broken-clock bands + LOBT clip

`models/mechanism_v15`: v14 plus `prc.lirfbands` (experiment log section 12). Leave-one-month-out
Jan+Jul 368.80 -> 341.93 (-26.9s), 12-month -8.3s, 10/12 months. The gain sits in a few dozen
rows by construction; three public leaders confirm the direction on the board. Leakage probe
PASS. 162 rows differ from v14.

**Scored 340.35 -> 310.58, a gain of 29.8s**, the largest single step of the project and
in line with the 26.9s leave-one-month-out estimate. Uploaded 2026-09-16 (fourth of five).
Single observation; nothing is tuned from it.

## v16 — NM-free expert for LIRF unmatched rows under 4h

`models/mechanism_v16`: v15 plus `lirf_short_expert` (experiment log section 13).
Leave-one-month-out Jan+Jul 341.93 -> 329.64 (-12.3s), 12-month -3.3s, survives removing the
ten most-helped rows (-7.9s), 8/12 months. Leakage probe PASS. 340 rows differ from v15.

**Scored 310.58 -> 301.14, a gain of 9.4s** against a 12.3s leave-one-month-out estimate.
Uploaded 2026-09-16 (fifth of five). Single observation; nothing is tuned from it.

## v17 — ranking-month context features + LOBT residual expert (built, never uploaded: superseded by v21)

`models/mechanism_v17`: v16 plus `prc.context` (61 label-free features from the ranking
month's own movements) and `prc.lobt` (second anchor on the last planned off-block time,
blended per airport). Experiment log section 14. Leave-one-month-out Jan+Jul 329.64 -> 321.27
(-8.4s), 12/12 months, -6.3s after removing the 50 most-helped rows. Leakage probe PASS.
Expected board: ~292 from 301.14.

## v18 — OOF LOBT alphas + Rome matched-regime gate (built, never uploaded: superseded by v21)

`models/mechanism_v18`: v17 with the LOBT blend weights refit on out-of-fold residuals
(-0.9s) and the LIRF schedule-identity gate on matched Rome rows (-0.9s). Experiment log
section 16. Combined estimate about -1.8s on Jan+Jul; leakage probe PASS. Order for the
morning: v17 first, then v18, so each is attributable.

## v19 — round re-selection, three seeds, gate refit, finer Rome bands (built, never uploaded: superseded by v21)

`models/mechanism_v19`: v18 with the base refit at 1,310 rounds averaged over three seeds,
the schedule-fallback gate refit with context features, and Rome band edges at 40k/50k/60k/
70k/90k s. Experiment log section 17. Estimate about -2.9s over v18. Leakage probe PASS.
Upload order for 2026-09-17: v17, v18, v19.

## v21 — target encodings, uncapped base, Rome rules, lr 0.02

`models/mechanism_v21`: v19 plus 24 in-fold target encodings, four clock differences,
take-off-order block, matched base at lr 0.02 / 3,175 trees on the uncapped residual, gate
refit, deterministic Rome rules. Experiment log sections 18-19. Leave-one-month-out Jan+Jul
319.13 -> 314.43 (-4.7s), 12/12 months. Leakage probe PASS.

**Scored 301.14 -> 280.80, a gain of 20.3s** (fold estimate vs v16: -15.2s; the board moved
more). Uploaded 2026-09-17 07:30 UTC, first of the day. Rank ~37 of 152 at upload. v17-v19
remain built but unscored; they are superseded and would only be uploaded to attribute the
steps.

## v22 — CatBoost GPU member (weight 0.5) + refit LOBT alphas

`models/mechanism_v22`: v21 plus a CatBoost member (5,000 it, depth 8, native identity
categoricals, fitted on the RTX 4070 Ti) blended 50/50 on matched rows before the LOBT clip,
and LOBT blend weights refit against the v21 base. Experiment log section 20. Leave-one-
month-out Jan+Jul 314.43 -> 312.37 (-2.1s), 12/12 months, -1.5s after removing 50 rows.
Leakage probe PASS (and it caught an index-alignment bug in the member's feature frame).

**Scored 280.80 -> 276.46, a gain of 4.3s.** Uploaded 2026-09-17 ~08:40 UTC, second of five.

## v23 — long CatBoost member, seasonal LOBT weights, base-only unmatched non-LIRF

`models/mechanism_v23`: v22 with the 10,000-iteration lr 0.03 CatBoost member in place of the
5,000-iteration one (-0.15), LOBT blend weights per season (winter set for January, summer
for July; -0.3), and unmatched non-LIRF rows served by the capped base without the shrunk
tail term (-0.36, 10/12 months). Experiment log sections 20-21. Estimate about -0.9s.
Leakage probe PASS. **Scored 276.46 -> 275.99 (-0.46s).** Uploaded 2026-09-17, third of five.

## v24 — LIRF unmatched mixture of experts

`models/mechanism_v24`: v23 with the copy-classifier x copy-free-regressor mixture on Rome
no-record rows under 4h (experiment log section 22). Fold Jan+Jul 314.43 -> 310.15.
**Scored 275.99 -> 271.88 (-4.1s).** Uploaded 2026-09-17, fourth of five. Rank ~20.

## v25 — Rome mixture scope extended to 8 hours

`models/mechanism_v25`: v24 with the mixture also serving the 4-8h gap band (18 ranking rows).
Fold Jan+Jul 310.15 -> 309.41. **Scored 271.88 -> 269.82 (-2.1s).** Uploaded 2026-09-17,
fifth of five. Rank ~20; elegant-alligator 268.56 is 1.3s ahead.

## v26 — EUROCONTROL ATFM daily features + second CatBoost seed

`models/mechanism_v26`: v25 with five daily airport ATFM aggregates (organiser-permitted) in
the base and gate, and two 10k-iteration CatBoost members at 0.25 each. Fold estimate about
-0.45s. Leakage probe PASS. Upload refused on 2026-09-17 (quota 5/5); uploaded 2026-09-18 00:02 UTC.
**Scored 269.82 -> 269.33 (-0.49s).**

## v27 — CatBoost regressor inside the Rome mixture

`models/mechanism_v27`: v26 with `lirf_moe_8h_cat` (CatBoost GPU regressor in the LIRF
no-record mixture). Fold Jan+Jul 309.41 -> 309.09 (-0.32). Uploaded 2026-09-18 00:15 UTC.
**Scored 269.33 -> 269.25 (-0.08s).** Rank ~21; elegant-alligator 268.56 is 0.7s ahead.

## v28 — second-stage stacker on matched rows

`models/mechanism_v28`: v27 with `prc.stacker` (`models/stacker_mid`, LightGBM on the chain's
component predictions: pre-member prediction, CatBoost member, blend, ordinary base, gate
probability, LOBT expert, four clock proxies, airport) adding half its residual correction on
matched rows before the LOBT clip. Fold Jan+Jul -0.70 at weight 0.5 (-0.82 at 1.0), 12/12
months, drop-50 robust (experiment log section 26). Leakage probe PASS; rehearsal 264.09 ->
260.81 (in-sample). Uploaded 2026-09-18 09:37 UTC, third of five. **Scored 269.25 -> 268.33
(-0.92s).** Passes elegant-alligator (268.56) and kind-earthquake (268.52): rank ~17.

## v29 — LOBT-anchored CatBoost member

`models/mechanism_v29`: v28 with a third CatBoost GPU member on the residual to `MVT - LOBT`
(`models/catmember_lobt`, `scripts/fit_catmember.py --anchor lobt`) at 0.3, the two AOBT
members reduced to 0.15 each; the stacker keeps the AOBT members as its input. Fold Jan+Jul
-0.45 over the AOBT member alone, 12/12 months, drop-50 robust (experiment log section 27).
Leakage probe PASS; rehearsal 262.05 (in-sample; the 5,000-iteration member fits training
rows less tightly than the 10,000-iteration ones, so this number is not comparable). Uploaded
2026-09-18 09:41 UTC, fourth of five. **Scored 268.33 -> 267.87 (-0.46s; fold said -0.45).**

## v30 — EOBT clock member + stacker refit (no gain)

`models/mechanism_v30`: v29 plus a third clock member on the EOBT residual
(`models/catmember_eobt`) and the stacker refit against the three-member reference
(`models/stacker_mid3`, weight 0.75, `stacker_member_mean: all`). Fold Jan+Jul -0.22, 11/12
months (experiment log section 28). Leakage probe PASS; rehearsal 260.78. Uploaded
2026-09-18 13:05 UTC, fourth of five. **Scored 267.87 -> 267.96 (+0.09): no gain, v29 remains
our best.** Below the fold-to-board resolution limit.

## Fold-to-board transfer ledger (2026-09-20)

This is a descriptive audit of submissions already made, not a tuning table. The organiser
returns only one whole-board RMSE, so a board delta can be assigned to a lane only when the
submitted change was confined to that lane; mixed bundles are explicitly marked and must not be
used as lane evidence. `transfer = board gain / Jan+July OOF gain`, with gains expressed as
positive seconds.

| version | dominant changed lane | OOF gain | board gain | transfer | attribution quality |
| --- | --- | ---: | ---: | ---: | --- |
| v16 | LIRF no-record short expert | 12.30 | 9.44 | 0.77x | lane-confined |
| v24 | LIRF no-record mixture | 4.28 | 4.11 | 0.96x | lane-confined |
| v25 | LIRF no-record, 4--8h scope | 0.74 | 2.06 | 2.78x | 18 board rows; unstable denominator |
| v27 | LIRF no-record CatBoost regressor | 0.32 | 0.08 | 0.25x | tiny gain; not resolved then |
| v28 | matched-only stacker | 0.70 | 0.92 | 1.31x | lane-confined |
| v29 | matched-only LOBT member | 0.45 | 0.46 | 1.02x | lane-confined |
| v30 | matched-only EOBT/stacker refit | 0.22 | -0.09 | -0.41x | below resolution; not a gain |

The clean matched observations are near one but only number two once the sub-resolution v30
result is excluded. Rome/no-record observations range from 0.25x to 2.78x because their row
counts are tiny; they do **not** establish a stable 0.15x lane multiplier. Future candidates
need a paired bootstrap interval from `scripts/paired_bootstrap.py` before they are described as
measurable, plus the existing drop-10/drop-50 check.

## v31 — own-flight ADS-B expert

`models/mechanism_v31`: v29 plus `prc.ownexpert` (`models/ownexpert.txt`), a residual expert on
matched rows whose own ADS-B ground run joins (stand-dwell end, first taxi-out point as a lower
bound; experiment log sections 46-47), applied after the LOBT clip and clipped again. Permitted
by the organiser ruling of 2026-09-19 (no prediction-time cutoff; open trajectory data of the
scored flight is fine). Twelve-fold on the v29 reconstruction: Jan+Jul **-3.95s**, 12/12 months,
drop-50 -3.62s, **-3.63s** re-weighted to 2026 coverage (a floor: 2026 scope is 54.4% of ranking
rows against 48.3% of 2025 rows). Leakage probe PASS; rehearsal 258.43 against v29's 262.05
(both in-sample). Submission file: 344,841 rows, 53.6% changed against v29, mean shift +0.4s,
LTFM / LFPG / LIRF-July / no-record rows untouched.
Uploaded 2026-09-25 through the quota-checked path (first of five that UTC day). **Scored 267.87
-> 260.06 (-7.81s; fold said -3.95, coverage-weighted floor -3.63).** Transfer ~2x: 2026 has
own-trace coverage at airport-months that 2025 lacked (LEMD Jan 0% -> 96%, LIRF Jan 0% -> 59%,
EGLL Jul 7% -> 97%), which the floor deliberately did not credit. Rank 18 of 190; 3rd place (the
last prize position) 234.14, leader 225.84.

## v32 — own-flight expert v2, silent-gap rule

`models/mechanism_v32`: v31 with `models/ownexpert_v2.txt` on the `_v2` feature files
(silent-gap rule, experiment log section 49; `own_feature_version: 2`). Twelve-fold Jan+Jul
-5.94s against v29 (v31: -3.95), 12/12 months, drop-50 -5.60, -5.11 re-weighted to 2026
coverage. Leakage probe PASS; rehearsal 256.76 (v31 258.43, v29 262.05; in-sample). File:
344,841 rows, 53.0% changed against v31, mean shift -0.0s, |shift| median 13s, p99 149s.
Uploaded 2026-09-25, second of five that UTC day. **Scored 260.06 -> 257.73 (-2.33s; fold said
-2.0 over v31).** Transfer ~1.2x, not the ~2x of v31: the silent-gap rows are bound-only features
whose coverage does not change much between 2025 and 2026. Rank 18 of 191, 0.16s behind
gentle-ladder (257.57) and 0.41s behind strong-turtle (257.32); 3rd place (last prize position) 234.14.

## v33 — own-flight expert v3, stand exit from X-Plane stand coordinates

`models/mechanism_v33`: v32 with `models/ownexpert_v3.txt` on the `_v3` feature files (stand
exit against the NM stand's X-Plane Scenery Gateway coordinates, GPL v2+; experiment log section
50; `own_feature_version: 3`). Twelve-fold Jan+Jul -8.38s against v29 (v32: -5.94), 12/12,
drop-50 -7.96, -7.18 re-weighted to 2026 coverage. 2026 stand-exit coverage 33.7% of joined
rows against 27.8% in 2025. Leakage probe PASS; rehearsal 254.79 (v32 256.76; in-sample).
File: 344,841 rows, 53.7% changed against v32, mean shift -1.5s, |shift| median 25s, p99 181s.
Uploaded 2026-09-25, third of five that UTC day. **Scored 257.73 -> 252.69 (-5.04s; fold said
-2.44 over v32).** Transfer ~2.1x: stand-exit coverage is higher in 2026 (33.7% of joined) than
in 2025 (27.8%), and 2026 joins more rows overall. Rank 14 of 191, passing merry-quicksand
(254.11); 4.1s behind humble-rocket (248.56); 3rd place (last prize position) 234.14.

## v34 — own-flight expert v5: wider stand location + first-sighting distance

`models/mechanism_v34`: v33 with `models/ownexpert_v5.txt` on the `_v5` files (stand location v2,
experiment log section 51; first-sighting distance, section 54). Twelve-fold Jan+Jul -8.73s
against the v29 reconstruction (v33's expert: -8.38), 12/12 months better than v33's expert in
the paired comparison (v4 -0.24, then v5 -0.11), drop-50 -8.31. Leakage probe PASS; rehearsal
254.51 (v33 254.79; in-sample). File: 344,841 rows, 52.3% changed against v33, mean -0.3s,
|shift| median 8s, p99 67s.
Uploaded 2026-09-26, first of five that UTC day. **Scored 252.69 -> 252.14 (-0.55s).** Board at
upload: v33 still listed as our best (the board lags the scorer); rank 16 of 194 on it, 3rd place
quick-boat 231.73, leader jolly-lobster 224.88.

## v35 — ground extraction v2 + retuned own-flight expert

`models/mechanism_v35`: v34 with `models/ownexpert_v6.txt` (255 leaves, 1200 rounds, lr 0.025) on the `_v6`
feature files (ground_v2 extraction: departing callsign taken once airborne, take-offs after lost ground
reception; experiment log section 55; `own_feature_version: 6`). Twelve-fold Jan+Jul **-4.33s against v34's
expert** (CI [-5.48, -3.40]), 12/12 months, drop-50 against the v5-feature retuned expert -3.23. Joined
2026 departures 188,817 -> 218,015. Leakage probe PASS; rehearsal 248.94 (v34 254.51; in-sample). File:
344,841 rows, ids identical to v34, 61.8% changed, mean shift +0.4s, |shift| median 19s, p99 213s; LTFM and
no-record rows untouched, LFPG 13.2% changed. Uploaded 2026-09-29 from the GPU host (this Mac's route to
s3.opensky-network.org timed out), first of five that UTC day, the rehearsed file byte-identical (sha256
82a4e770...). **Scored 252.14 -> 247.54 (-4.60s; fold said -4.33 over v34's expert).** Rank 22 of 202;
3rd place (last prize position) jolly-lobster 224.50, leader strong-crane 220.65.

## v36 — Rome garbled-string schedule-copy rule

`models/mechanism_v36`: v35 + `garbled_copy_rule` (`lirfbands.garbled_copy`, applied last in `predict_frame`):
LIRF no-record departures with a doubled-prefix flight string and a schedule gap in (4h, 8h] are predicted at
MVT - SCHED (20/20 schedule copies in 2025; experiment log section 57). On the v6b OOF: twelve -0.85, Jan+Jul
-1.32, non-scored -0.74, no month worse. Leakage probe PASS; rehearsal 248.93 (v35 248.94; in-sample). File:
344,841 rows, ids identical to v35, 3 rows changed (5 in scope, 2 already at the gap). Uploaded 2026-09-30 from
this Mac, first of five that UTC day (sha256 c69aba6f...). **Scored 247.54 -> 246.86 (-0.68s).**

## v37 — own-flight expert on no-record departures

`models/mechanism_v37`: v36 + `own_norec_stem: ownexpert_norec_v1` (`ownexpert.apply_norec`, after the matched
expert and the LOBT clip): no-record departures outside Rome whose own trace joined get a residual correction
from the trace times, the schedule gap and the chain's prediction (experiment log section 59). Twelve-fold on the
v36-equivalent OOF: twelve -0.26, Jan+Jul -0.32 (CI [-0.51, -0.19]), 9/10 non-scored months, drop-10 -0.17.
Leakage probe PASS; rehearsal 248.17 (v36 248.93; in-sample). File: 344,841 rows, ids identical to v36, 1,386 rows
changed (all no-record, none at LIRF; LEBL 433, EDDM 226, EGLL 206, LEMD 190, LSZH 182, LFPG 120), shift -757 to
+1,383s, median -55s. Uploaded 2026-09-30 from this Mac, second of five that UTC day (sha256 c5e344e2...).
**Scored 246.86 -> 246.01 (-0.85s).**

## v38 — v7 own-flight features (taxiway-network distance, gap-aware first taxi-out)

`models/mechanism_v38`: v37 with `ownexpert_v7.txt` (matched expert, section 55 settings) and `ownexpert_norec_v2.txt`
(no-record expert) on the `_v7` feature files (`own_feature_version: 7`, `own_norec_feature_version: 7`;
experiment log section 60). Twelve-fold against the v37 equivalent: twelve -1.11, Jan+Jul -1.12 (CI [-1.50, -0.84]),
12/12 months; matched part alone drop-50 -0.71. Leakage probe PASS; rehearsal 247.30 (v37 248.17; in-sample). File:
344,841 rows, ids identical to v37, 210,446 rows changed (61.0%, all joined to a trace, no LIRF no-record row),
mean shift +0.9s, |shift| median 9s, p99 173s. Uploaded 2026-10-01 from this Mac, first of five that UTC day
(sha256 94a48652...). **Scored 246.01 -> 244.19 (-1.82s).**

## v39 — airport-month means calibrated to EUROCONTROL's published monthly taxi-out

`models/mechanism_v39`: v38 plus `pubcal` (`pubcal_offsets`, `pubcal_vmin` 0.985; `scripts/fit_pubcal.py`; experiment
log sections 61-62). Last stage: in airport-months whose published valid share is >= 0.985, rows predicted under 3,600s
shift together to the published mean minus the airport's 2025 offset, then the LOBT clip on those rows. 2025
leave-one-month-out on the v38-equivalent OOF: twelve 232.204 -> 232.121, 9/12 months won (Jan -14.0M, Jul -33.3M).
The 2026 benefit cannot be measured on folds: it rests on the published means, chiefly EHAM Jan (+71.5s) and
LIRF Jul (-59.5s). Leakage probe PASS; rehearsal 247.43 (v38 247.30; in-sample, both models fitted on these months,
so the model is already unbiased there). File: 344,841 rows, ids identical to v38, 250,216 rows changed, 14 airport-months
(sha256 34e405f7...). Built 2026-10-03; uploaded 2026-10-04 from this Mac, first of five that UTC day. **Scored 244.1913 -> 244.1917 (+0.0004s): no change.** Do not iterate on its threshold or offsets: any variant would be chosen against this score.

## v40 — the matched own-flight expert reads the chain's matrix

`models/mechanism_v40`: v38 with `own_expert_stem` `ownexpert_v8rich` and `own_expert_rich` `cat` (experiment log section
63): the v7 expert's inputs plus every numeric matrix column and seven categoricals (stand, runway, type, operator,
wake, flight type, market segment), feature_fraction 0.5, otherwise the section 55 settings. Built on v38, not v39
(the calibration is closed as neutral). Twelve folds against the v38-equivalent: twelve -0.956, Jan+Jul -0.463 (CI
[-0.734, -0.251]), 12/12 months. Leakage probe PASS; rehearsal 242.19 (v38 247.30; in-sample). File: 344,841 rows, ids
identical to v38, 213,413 rows changed (61.9%), mean shift +0.4s, |shift| median 22s, p99 151s (sha256 0752bdae...).
Uploaded 2026-10-03 UTC (10-04 local) from this Mac, second of five that UTC day. **Scored 244.19 -> 243.09 (-1.10s).**

## v41 — v8 own-flight trace features, retuned rich expert

`models/mechanism_v41`: v40 with `ownexpert_v9rich8` on `adsb_own_{2025,2026}_v8.parquet` (`own_feature_version` 8;
experiment log section 63): the v8 trace features (observed ground path, speed at first sighting, stops, hold time,
and the derived observed speed and unobserved stand-to-first-sighting time), 511 leaves, cat_smooth 50 / cat_l2 50 /
min_data_per_group 200. Twelve folds against the v40-equivalent: twelve 231.248 -> 230.485 (-0.763), Jan+Jul -0.600
(CI [-0.814, -0.441]; January -0.520, July -0.688); against v38 12/12 months. Leakage probe PASS; rehearsal 240.25
(v40 242.19; in-sample). File: 344,841 rows, ids identical to v40, 212,140 rows changed (61.5%), mean shift +0.5s,
|shift| median 15s, p99 101s (sha256 6bc4b756...). Uploaded 2026-10-03 UTC (10-04 local) from this Mac, third of five that UTC day.
**Scored 243.09 -> 242.30 (-0.79s).**

## v42 — v9 reception features and per-stand first-sighting lag encodings

`models/mechanism_v42`: v41 with `ownexpert_v10lag` on `adsb_own_{2025,2026}_v9.parquet` (`own_feature_version` 9) and
`own_expert_lag` (tables `models/ownexpert_v10lag.lag.json`, fitted on all 2025 months; experiment log section 63).
Twelve folds against the v41-equivalent: twelve -0.196, Jan+Jul -0.189 (CI [-0.288, -0.119]; January -0.129, July
-0.251), 12/12 months, drop-1/10/50 -0.18 / -0.15 / -0.06. Leakage probe PASS; rehearsal 240.16 (v41 240.25; in-sample,
and the lag tables include the rehearsal months' own rows). File: 344,841 rows, ids identical to v41, 210,192 rows
changed (61.0%), mean shift -0.15s, |shift| median 10s, p99 64s (sha256 485c85bf...). Uploaded 2026-10-03 UTC (10-04 local) from this
Mac, fourth of five that UTC day. **Scored 242.30 -> 242.03 (-0.27s).**

## v43 — the v42 expert bagged over three seeds

`models/mechanism_v43`: v42 with `own_expert_stems` [`ownexpert_v10lag`, `_s8`, `_s9`] (seeds 7, 8, 9, same inputs and
lag tables), served as the mean of the three (`ownexpert.Bagged`). Twelve folds against the v42-equivalent: twelve
-0.112, Jan+Jul -0.110 (CI [-0.166, -0.067]), 12/12 months. Leakage probe PASS; rehearsal 240.08 (v42 240.16; in-sample).
File: 344,841 rows, ids identical to v42, 204,295 rows changed, mean shift -0.01s, |shift| median 6s, p99 37s (sha256
c2a364db...). Uploaded 2026-10-04 UTC from this Mac, first of five that UTC day. **Scored 242.03 -> 241.91 (-0.12s).**

## v44 — the bagged rich expert blended with the lean v7 expert

`models/mechanism_v44`: v43 plus `own_expert_blend` {`ownexpert_v7`, weight 0.175}: on the matched expert's scope rows the
prediction is 0.825 x the v43 bagged rich expert + 0.175 x v38's lean v7-input expert, both on the same chain output
(experiment log section 63). Weight chosen on the ten non-scored months (grid 0.05-0.30). Twelve folds against the
v43-equivalent: twelve -0.100, Jan+Jul -0.139 (CI [-0.198, -0.095]; January -0.059, July -0.221), 11/12 months.
Leakage probe PASS; rehearsal 240.77 (v43 240.08; in-sample, where the rich expert fits its own training months more
tightly). File: 344,841 rows, ids identical to v43, 200,573 rows changed, mean shift -0.12s, |shift| median 5s, p99 28s
(sha256 6a955cc5...). Uploaded 2026-10-04 UTC from this Mac, second of five that UTC day. **Scored 241.91 -> 241.79
(-0.12s).**

## v45 — a weighted mix of six own-flight experts

`models/mechanism_v45`: the matched expert is `own_expert_mix`, one frame (v10 trace files, the chain's matrix, daily
reception, lag encodings) read by six boosters: v9 rich seeds 7 and 8 (0.032, 0.063), v9 rich + daily reception (0.303),
v10 rich seeds 7 and 8 (0.134, 0.150), and a lean v9 + lag expert without the matrix (0.319). Weights by non-negative least
squares on the ten non-scored months' scope rows, normalised to one (experiment log section 63); the lean v7 expert and
a CatBoost GPU expert were candidates (v7 weight 0; CatBoost 0.135 when allowed, not served yet). Twelve folds against the
v44-equivalent: twelve -0.184, Jan+Jul -0.146 (corrected; first recorded as -0.207 / -0.173, see log section 63), 12/12 months.
Leakage probe PASS; v44 re-rehearsed through the new mix path reproduces exactly (240.7655); v45 rehearsal 241.19
(in-sample). File: 344,841 rows, ids identical to v44, 205,674 rows changed, mean shift -0.06s, |shift| median 6s, p99
40s (sha256 da292b9e...). Uploaded 2026-10-04 UTC from this Mac, third of five that UTC day. **Scored 241.79 -> 241.70
(-0.09s).**

## v46 — the expert mix with the CatBoost expert served

`models/mechanism_v46`: v45's mix with the CatBoost GPU expert added (`ownexpert_cb_v9.cbm`, served through
`ownexpert.CatExpert`), weights from the same non-negative least-squares fit including it (v9 s7 0.012, s8 0.039, daily
0.291, v10 s7 0.124, s8 0.135, CatBoost 0.135, lean v9 0.265; normalised). Twelve folds against the v44-equivalent: twelve
-0.230, Jan+Jul -0.201 (CI [-0.293, -0.132]); over v45's mix about -0.02 / -0.03. Leakage probe PASS; rehearsal 241.34
(in-sample). File: ids identical to v45, 185,564 rows changed, mean shift +0.03s, |shift| median 2s, p99 15s (sha256
fccd69dd...). Uploaded 2026-10-04 UTC, fourth of five that UTC day. **Scored 241.70 -> 241.62 (-0.08s).**

## v47 — retuned base model (feature_fraction 0.5, 511 leaves)

`models/mechanism_v47`: v46 with inner model `mechanism_mechanism_v47b_inner` (`refit_v20.py --atfm --uncapped --rome-rules
--base-params '{"feature_fraction": 0.5, "num_leaves": 511}' --inner-from mechanism_mechanism_v26_inner`; three seeds,
gate refit, v26's serving flags kept). Twelve folds, rows outside the experts' scopes, against the v46-equivalent:
twelve -0.099, 12/12 months (10/10 non-scored), Jan+Jul -0.048 (CI [-0.123, +0.048], not resolved; July -0.085
resolved); the chain alone -0.178 (experiment log section 63). The experts read the new chain output as `pred` in
serving; their OOFs were fitted on the old one. Leakage probe PASS; rehearsal 238.23 (v46 241.34; in-sample, the larger
base fits its training months more tightly). File: ids identical to v46, 297,280 rows changed, mean shift +0.03s, |shift|
median 3s, p99 45s (sha256 8fd56dbd...). Uploaded 2026-10-04 UTC, fifth of five that UTC day. **Scored 241.62 -> 241.31
(-0.31s).**

## v48 — runway state and gate retention in the base model

`models/mechanism_v48`: v47 with inner model `mechanism_mechanism_v48b_inner` (`refit_v20.py --atfm --uncapped
--rome-rules --rwystate --base-params '{"feature_fraction": 0.5, "num_leaves": 511}'`, run on a GPU host), serving the
`prc.rwystate` features (`use_rwystate`). the GPU host's copy of the v26 inner JSON lacked `unmatched_other_base_only`, so
the refit did not inherit it; set to true by hand to match v47 (the two inner JSONs then differ only in `use_rwystate`).
Twelve folds against the v47-equivalent, rows outside the experts' scopes: twelve -0.107, Jan+Jul -0.133 (CI [-0.241,
-0.002]), 11/12 months (experiment log section 63). Leakage probe PASS; rehearsal 237.91 (v47 238.23; in-sample). File:
ids identical to v47, 295,522 rows changed, mean shift -0.15s, |shift| median 3s, p99 49s (sha256 a1d08031...).
Uploaded 2026-10-05 UTC, first of five that UTC day. **Scored 241.31 -> 240.51 (-0.80s).**
