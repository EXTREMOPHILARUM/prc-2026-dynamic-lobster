# Challenge rules and compliance policy

This project follows the published PRC Data Challenge 2026 rules and the conservative
interpretation below. Official pages were checked on 2026-09-13:

- Overview: https://prc-data-challenge-2026.netlify.app/index.html
- Eligibility: https://prc-data-challenge-2026.netlify.app/eligibility.html
- Data: https://prc-data-challenge-2026.netlify.app/data.html
- Ranking: https://prc-data-challenge-2026.netlify.app/ranking.html

## Explicit rules

- Predict `TAXITIME_SEC_mvt` for every departure in the submission template.
- The score is root mean square error (RMSE).
- The ranking period contains January and July 2026 movements.
- Submission IDs must match `submitting.parquet` exactly, with no missing or extra rows.
- Files must be named `<team-name>_v<N>.parquet` using incremental version numbers.
- A team may submit at most five files per day; its bucket may contain at most 1 GB.
- Teams are ranked by their best submission.
- Attempts to learn from or exploit the ranking process are forbidden.
- The deadline is 2026-10-11 23:59:59 CET.

Prize eligibility additionally requires:

- Public source code on GitHub under GNU GPLv3.
- Openly accessible, usable, and documented external datasets.
- Additional datasets distributed under an open-source licence.
- Sufficient documentation to understand and reproduce the result.
- An original solution. Reused implementations require permission and significant
  algorithmic or model modifications.
- Compliance with the participation and sanctions restrictions on the eligibility page.

The organisers reserve the right to alter or stop the challenge.

## Project interpretation

The following are allowed:

- Training and selecting models using only 2025 labels.
- Using all fields supplied in training and ranking data, including NM timestamps,
  `MVT_TIME_UTC_mvt`, callsigns, schedules, and complete arrival rows.
- Constructing label-free features from the ranking covariate distribution.
- Using open external data when its licence, source, retrieval process, and checksums are
  recorded and another participant can reproduce access.
- Reading public research and repositories for general ideas, followed by an independent
  implementation that satisfies the originality rule.

The following are prohibited in this project:

- Selecting features, thresholds, calibration factors, blend weights, or row-level
  corrections from leaderboard score changes.
- Uploading controlled variants to infer hidden labels, segment errors, tail prevalence,
  or any other property of the ranking targets.
- Treating a leaderboard result as an additional validation observation.
- Using hidden departure block times or taxi times from any unauthorised source.
- Copying another team's implementation or exact fitted recipe without the required rights
  and significant original modification.
- Using commercial, access-restricted, or non-reproducible external datasets in a
  prize-eligible final solution.

The published rules do not define a prediction-time cutoff for external records, and the
organiser has confirmed there is none (ruling of 2026-09-19 below): the model is for
post-operations analysis, and open trajectory data of the scored flight itself, including an
off-block time extracted from it, may be used. This project had held off joining a ranking
flight to its own post-event record until that answer; the hold is lifted for ADSB.lol (ODbL).
Each other external source still needs its licence documented and declared before use.

## Experiment protocol

Every submission candidate must follow this sequence:

1. State the hypothesis, features, target, validation folds, and acceptance criteria before
   generating ranking predictions.
2. Select all parameters using training data only.
3. Generate temporal out-of-fold predictions. Use complete-month folds; never random row
   splits for final model selection.
4. Report additive squared error and RMSE for all rows, bulk, tail, LIRF, non-LIRF, NM-clock
   present/missing rows, and each calendar month.
5. Reject improvements driven by target leakage, one unstable month, or an unlicensed data
   source.
6. Freeze code, parameters, and model artifacts before creating a submission.
7. Refit the frozen model on all 12 training months.
8. Validate submission IDs, row count, target type, finite values, and provenance metadata.
9. Upload once. Record the score as an external result, but do not use it to modify the same
   hypothesis or tune a related variant.
10. Download or otherwise preserve the scorer result JSON with the local submission provenance.

Repeated submissions are acceptable only for independently motivated, locally selected
models. A duplicate file may be uploaded to verify infrastructure, as with v4/v5, but its
identical result provides no modelling information.

## Compliant next model

The next high-priority experiment models the timestamp-reporting mechanism rather than
guessing extreme taxi duration directly. It uses only supplied data.

For 2025 training departures, define observable mechanism labels such as:

- schedule fallback: `abs(BLOCK_TIME_UTC_mvt - SCHED_TIME_UTC_mvt) <= 60`;
- AOBT availability and disagreement regimes;
- stale-schedule/day-offset bands;
- ordinary airport-event reporting.

For missing-clock LIRF rows, estimate a calibrated conditional mixture:

```text
P(schedule fallback | X) * (MVT - SCHED)
+ P(stale-day regime | X) * stale-day conditional mean
+ P(ordinary regime | X) * ordinary taxi prediction
```

This is valid because `BLOCK_TIME_UTC_mvt` is used only to create training labels, while all
ranking predictors (`MVT`, `SCHED`, airport, stand, runway, service, operator, and missingness)
are supplied covariates. Gate probabilities and mixture parameters must be cross-fitted and
calibrated without leaderboard feedback.

For rows with `AOBT_3_flt`, retain the clock-residual formulation:

```text
(MVT - AOBT) + E[AOBT - BLOCK | X]
```

Candidate improvements include per-airport residual experts and an independently trained
CatBoost model. Any blend must be fitted from temporal out-of-fold predictions and frozen
before submission.

## External-data approval checklist

Before adding an external source, record:

- authoritative dataset URL and publisher;
- exact licence and whether prize use is permitted;
- historical coverage for all required 2025 and 2026 dates;
- retrieval commands, file versions, and checksums;
- whether each field existed before the prediction event;
- join logic, match coverage, and ambiguity rate;
- organiser approval when target-flight post-event information could reveal the outcome.

If any item is unresolved, the external source cannot be used in the final prize-eligible
pipeline.

## Organiser rulings received (Discord, 2026-09-17)

Asked by this team, answered by `espinielli`:

1. EUROCONTROL's daily per-airport performance files (ATFM slot adherence, arrival ATFM
   delay, airport traffic; attribution + non-commercial clause) "can be considered open data
   for the challenge". Permitted, with attribution and declaration.
2. ADSB.lol historical archives (ODbL 1.0, not OpenSky): "if their license applies as open
   data, ok". Permitted if the ODbL terms are met (attribution, share-alike on derived
   databases) and the source is declared. OpenSky state vectors remain excluded.

## Organiser ruling on prediction time (Discord, 2026-09-19)

Asked by another team (`fl340`, messages 1550191955269386283 and 1550466235555389440, export
`data/external/prc-discord-2026-09-24.json`), who proposed that the rules should forbid
post-pushback observations such as ADS-B off-block times and set a per-flight prediction
cutoff. Answered by `espinielli`:

1. "the model is for post-ops, not for tactical use." (reply to 1550191955269386283)
2. "there are no such restrictions: if you can find open trajectory data with good ground
   coverage and extract off-block times, we are ok. Practically speaking it won't be
   possible." (message 1550778441287208963, reply to 1550466235555389440)

Consequences for this project: the own-flight ADS-B expert (`prc.ownexpert`, experiment log
sections 46-47) is permitted on ranking rows, and same-day EUROCONTROL daily aggregates are
consistent with a post-operations model with no prediction cutoff. On 2026-09-24 the same
organiser added that the aim of the challenge is the model, its documentation and its
implementation, not a data source that hands over the answer; the write-up should present the
own-flight trace as one observation fused with the NM clocks, which is what it is.

Earlier (2026-09-16) the organiser also confirmed every field of the scored row other than
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, and every other row, is available to the model.

## Organiser ruling on published monthly taxi-out aggregates (Discord, 2026-09-27)

disclaimer8, 2026-09-27 18:19 UTC: "ansperformance.eu now publishes the monthly average taxi-out time per
airport (and per runway on the AIU airport dashboard) for January and July 2026. Is it OK to use those
published monthly figures as a model input, or should aggregates from the ranking period be treated as off
limits?" espinielli (organiser), 20:44 UTC, replying to that message: "I already answered above: yes".
Export: `data/external/prc-discord-2026-10-01.json`. So EUROCONTROL's `Taxi-Out_Additional_Time.xlsx`
(monthly, per airport, through 2026-07) may be used, declared in `external_data` like the other EUROCONTROL
performance files. This is not the board score: using it does not tune against the hidden set.
