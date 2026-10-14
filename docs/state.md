# Where this entry stands

> **Audit note, 2026-09-20:** [`foundational-assumptions-audit-2026-09-20.md`](foundational-assumptions-audit-2026-09-20.md)
> re-checks older claims (the v29 OOF reconstruction, stacker nesting, January/July reuse, ATFM temporal
> availability, ADS-B conclusions, claims of irreducibility). Where it and an older statement disagree, the audit
> governs.

The five-minute summary of the solution. `docs/experiment-log.md` is the evidence behind every statement here, by
section; `docs/submissions.md` is the per-version record; `docs/reproduce.md` rebuilds the submission. The anatomy
section is the 2026-09-20 view at v29; later sections of the log refine it.

## Position

| | |
| --- | --- |
| Final submission | `dynamic-lobster_final.parquet`: v48 with the own-flight trace features rebuilt over the final phase's four months (Jan, Feb, Jun, Jul 2026) |
| Best public-board score | **v48, 240.51** (Jan + Jul 2026), 35th of 242 at the close |

Ladder: v13 342.22 (2026-09-16) -> v15 310.58 -> v16 301.14 -> v21 280.80 -> v22 276.46 ->
v24 271.88 -> v25 269.82 -> v26 269.33 -> v27 269.25 -> v28 268.33 -> v29 267.87 -> v30 267.96
(no gain) -> v31 260.06 (own-flight ADS-B expert, fold -3.95, board -7.81) -> v32 257.73 (silent-gap
rule, fold -2.0, board -2.33) -> v33 252.69 (stand exits from X-Plane coordinates) -> **v34 252.14** (stand location v2 +
first-sighting distance) -> v35 247.54 (ground extraction v2 + retuned expert, section 55) -> v36 246.86 (garbled-string copy rule, section 57) -> v37 246.01 (no-record own-flight expert, section 59) -> **v38 244.19** (v7 own-flight features, section 60) -> v39 244.19 (published-mean calibration, no change, section 62) -> v40 243.09 (matched expert reads the chain's matrix, section 63) -> v41 242.30 (v8 trace features, retuned expert, section 63) -> v42 242.03 (v9 reception features + per-stand lag encodings) -> v43 241.91 (expert bagged over three seeds) -> v44 241.79 (blended with the lean v7 expert) -> v45 241.70 (weighted mix of six experts) -> v46 241.62 (+ CatBoost expert in the mix) -> v47 241.31 (base model feature_fraction 0.5, 511 leaves) -> **v48 240.51** (runway state and gate retention in the base).

## The model

Described module-by-module in `README.md` under "Architecture". In one sentence: a mixture over
*reporting regimes* rather than a single regressor, because ~0.2% of departures have a recorded
taxi time that is a broken clock rather than a long taxi, and RMSE is dominated by them.

The submitted wrapper is `models/mechanism_v48_final.json`; `README.md` lists its stages and `docs/reproduce.md`
how each component was fitted.

## Anatomy of the remaining error

Leave-one-month-out on 2025, January and July only, the v29 chain: **RMSE 307.27** over 344,419
rows, total squared error **32,519M**. Where it sits (log section 26, 29):

| block | rows | SSE | share | RMSE if solved |
| --- | ---: | ---: | ---: | ---: |
| non-LIRF matched (ordinary taxis, nine airports) | ~313,000 | 13,036M | 40.1% | — |
| non-LIRF no-record | 4,976 | 11,880M | 36.5% | 244.80 |
| LIRF matched | 26,131 | 3,506M | 10.8% | 290.24 |
| LIRF no-record, gap ≤ 8h (the mixture) | 381 | 2,821M | 8.7% | 293.64 |
| LIRF no-record, gap > 8h (the band table) | 16 | 1,276M | 3.9% | 301.18 |

Reproduce with `uv run python -u scripts/analyse_error_anatomy.py`; every number in this
section and the next is one of its six parts.

Three facts that follow, and that govern what is worth trying:

1. **Two rows dominate, and they are unpredictable.** 10,184M of the non-LIRF no-record block is
   two January 2025 Paris departures whose recorded block time is ~a day before take-off. Both
   aircraft sat grounded at CDG for 32-41h across the recorded block (registration search of raw
   traces, log section 63, "cause of the day slips"): most likely a real push-back of a flight
   that did not leave, kept when it flew next day. Their schedule
   gap is 29 and 34 minutes, i.e. they look like ordinary on-time flights on every field we
   hold. Without them the fold reads ~255 against a board of 267.9, so **the hidden set carries
   its own unpredictable rows and our fold has never been "pessimistic" for the reason
   section 8h assumed.**
2. **One day-offset row is worth 43 seconds of RMSE.** A row with truth 86,400s predicted at
   1,050s contributes 21,125 to the mean squared error, against a total of 71,754. A 23-second
   difference in board RMSE is worth 0.55 of one such row: rankings at this level are decided by a handful of rows.
3. **The bulk is thin and level.** Matched rows with y < 30 minutes run RMSE 177 over 324,000
   rows. Closing 23 seconds from the bulk alone would need that to fall to about 137, a 23%
   reduction, which no feature work will deliver.

## What is closed

Do not re-test these. Each has a log section with the measurement that settled it.

**Broken-clock detection (section 29; 3 branches, 11 variants).** Day offsets outside Rome:
3 rows a year, no signature, and a base rate of 3 in ~21,000 so hedging cannot pay either.
Retraining the Rome copy classifier: no-record-only, squared-error-stake weighting, both,
cross-fitted calibration months — all worse; the stake weight spans six orders of magnitude and
collapses the effective sample (AUC 0.927 → 0.899). Recalibration: isotonic per gap stratum (two
and three strata), Platt, Platt + gap, Platt + gap + interaction, band floor — all worse,
including the two-parameter minimum-variance form. The wide-gap under-confidence that motivated
it (mean p 0.68 against 0.78 actual at 4–6h) is sampling noise on 64 rows a year.

**Model and feature work (sections 11–25, 28).** ADS-B (organiser-permitted; correlation with
our residual ~0.05, ingest stopped). CatBoost 50/50 at CPU settings. Airport specialists.
`linear_tree`. Faulty-AOBT tail gate. Matched gate-hold. Uncapped unmatched base. Schedule-target
CatBoost. Depth-10 and second-seed members. Schiphol hedge. Weather drop. Disruption meters.
Rome mixture scope above 12h. Flight identity in the Rome copy classifier, all-season and
winter-only (elegant-alligator's V6 correction). The stacker applied to no-record lanes. The
reduced-input stacker. EOBT as a third clock member (+0.11 fold, below the resolution limit).

**Closed 2026-09-29..10-01 (sections 56-58).** Late take-off stand exits: an airport-mix artefact
(EGLL carries a constant +135s offset for all rows), not an extraction bug. The day-slip rows outside
Rome: no signature in the supplied fields ("flight absent the previous day" is 3 in 7,185). All three
were aircraft grounded 32-76h before take-off, but so are 684 ordinary no-record departures a year
(ground time > 24h: hit rate 0.44%, in-sample saving < 0.1s; log section 63). Aircraft ground time
and turnaround slack as features, and the "no off-block before landing" clip: all negative.
Istanbul's NM AOBT is MVT minus a fixed 19/20/24/25 min on 82% of rows (section 63): explains LTFM's
share of the error, already unbiased, not a lever. Arrival schedule copies as a
copy-classifier input for Rome's <= 8h lane: AUC up, RMSE worse (+0.33s twelve).

**Not a lever:** the stand->runway physical floor as a late-AOBT detector (section 34) — a
strong detector against the raw anchor (16.7x lift, margin correlating 0.658 with the anchor's
shortfall) but worth nothing against v29's residual, which is already unbiased on the rows it
flags (corr 0.008). The `TAIL_THRESHOLD` anchor clip — only 11 LIRF matched rows in all of 2025
have `MVT − AOBT_3` above 3,600s.

## What is open

1. **LIRF matched rows (section 30) — open, but the prior is low.** The only block whose error
   is *broad* rather than concentrated: 26,131 scored rows at RMSE 366 against 165–243 at the
   other nine, and removing its worst five rows moves it only to 361. Because it is broad, a
   genuine gain here would survive the drop gate, unlike the old LIRF specialist (−3.2s headline,
   −0.9/+1.1 on drop-10/50). **But the cause looks like input noise, not model failure:** Rome's
   AOBT anchor is the noisiest of the ten (raw RMSE 647 against 285–531) and our ratio of error
   to anchor error, 0.57, is mid-pack across the ten, between Paris 0.56 and Frankfurt 0.60.
   On the 312 scored rows with y > 1h the anchor reads ~1,000s where the truth is 3,600–6,700s.
   Per-airport rounds and 4× capacity were both flat on LIRF (section 8). Try it only with a
   mechanism that is specific to the late-AOBT pattern, not more capacity.
2. **IOBT as a fourth clock member.** Cancelled at fold 4 of 12 to free the GPU. Expected ~0.05s
   on the EOBT trend, i.e. below the resolution limit. Listed for completeness, not recommended.
3. **ADS-B own-flight experts: served** since v31 (permitted, organiser ruling 2026-09-19). Matched
   expert v7 and a no-record expert since v37/v38 (sections 46-60). Still open on them: push-back
   detection from the reversed heading in the first trace points, and a refit on a true nested OOF
   of the shipped chain (audit finding 1). Coverage limits the rest: LTFM 0%, EHAM/EDDF/LFPG
   ground reception poor in adsb.lol; a second open-licensed ADS-B source is the only larger lever
   seen (not yet researched).

## How claims are stated (foundational audit, 2026-09-20)

Adopted 2026-09-25 after re-reading `docs/foundational-assumptions-audit-2026-09-20.md`:

- Every fold number names its baseline. "v29 OOF" is `analyse_error_anatomy.v29_oof`, a
  reconstruction of the shipped chain (audit finding 1), and all own-flight expert gains
  (sections 46-51) are measured on it.
- January/July 2025 are **calendar-matched selection diagnostics**, not untouched holdouts
  (finding 3). Screens decide on the ten other months; Jan/Jul is reported, not used as proof.
- Board results are history in `docs/submissions.md`, not evidence (finding 5). Do not derive
  acceptance thresholds, transfer multipliers or priorities from them, and do not scale a fold
  RMSE into a predicted board score (finding 11). The ~0.3s "resolution limit" and the v31/v33
  "~2x transfer" are both board-derived and must not decide what ships.
- Drop-row checks are a concentration diagnostic, not a veto (finding 6).
- Scorer responses are kept verbatim in `docs/reference/scorer-results/`; `prc.submit --upload`
  saves each new one (finding 13).

## Rules learned the hard way

- **The cross-airport early-anchor block is real, but is now closed with current inputs.** The
  v29 OOF δ audit (log section 31), correctly using `MVT - AOBT_3` rather than the schedule
  proxy, finds `anchor - y < -600s` on 14,898 matched Jan/Jul rows carrying 40.2% of matched SSE
  (20.5% overall). The all-fold conditional residual expert (log section 32) used the source
  features and then the full live-chain signals; both caused a measurable +1.00s / +1.26s loss
  despite improving the early rows. Do not threshold-tune it. LIRF, LTFM and LFPG hold 63% of
  the block's SSE but have 0%, 0% and 8% ADS-B pilot coverage, so reopening requires a genuinely
  new observable with coverage there, not another gate or blend.
- **Require a paired-bootstrap interval as well as the robustness check.**
  `scripts/paired_bootstrap.py` resamples aligned stored OOFs and reports pooled, January and
  July 95% intervals. A candidate is not a measurable gain unless its pooled interval excludes
  zero; the historical 0.3-second rule is a prior, not a substitute for this evidence.

- **The fold-to-board resolution limit is about 0.3 seconds** (section 28). v28 (−0.70 fold →
  −0.92 board) and v29 (−0.45 → −0.46) tracked; v30 (−0.22 fold) came out **+0.09**. Changes
  below ~0.3s are coin flips that consume upload slots. This supersedes any "every second
  counts" instinct: sub-resolution changes are not small gains, they are noise.
- **A worse submission cannot displace the team's best score**, so a fold-confirmed upload is
  free optionality — but only above the resolution limit, or it is just noise in the slot.
- **The drop-10/drop-50 gate rejects concentrated gains by construction**, and the highest-value
  rows in this competition are concentrated. That tension is real and was examined deliberately
  in section 29; the conclusion was that the concentrated rows are genuinely unpredictable, not
  that the gate is wrong. v30's reversal is independent evidence that the gate is protecting us.
- **Never tune against a submitted score.** Organiser rule and project policy.
- **Validation is unfiltered, whole months, Jan and Jul separately**, with the leakage probe in
  `scripts/rehearse_submission.py` before every upload.
