# PRC Data Challenge 2026 — team `dynamic-lobster`

Taxi-out time prediction for departures from ten European airports (EUROCONTROL Performance Review Commission /
OpenSky Network data challenge 2026). Licensed under the GNU GPL v3 (`LICENSE`).

Final submission: `models/mechanism_v48_final` (v48, public board 240.51 on Jan+Jul 2026), predicting the final phase's
Jan, Feb, Jun and Jul 2026. `docs/reproduce.md` gives the end-to-end rebuild; the submitted predictions are attached to the
GitHub release `final`. Start with `docs/state.md`.

## What this is

An entry in the PRC Data Challenge 2026 (team `dynamic-lobster`): predict taxi-out time in
seconds for departures from 10 European airports, scored by **RMSE** on withheld January
and July 2026 movements. Deadline 2026-10-11.

**Read `docs/state.md` first** — current position, where the remaining error actually sits,
what is closed, what is open, and the methodology rules. `docs/experiment-log.md` is the
evidence behind it (2,500+ lines) and records every experiment with the measurement that
settled it, including the ideas that failed and why. Many obvious-sounding changes have
already been tested and rejected; a large majority of what looks promising here is noise.
`docs/foundational-assumptions-audit-2026-09-20.md` is the correction layer over older claims:
where it and an older document disagree, the audit governs.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"     # setup (lightgbm needs `brew install libomp` on macOS)
cp .env.example .env                       # then add S3 keys from s3-console.opensky-network.org

uv run pytest -q                           # all tests
uv run pytest tests/test_submission.py -q  # one file
uv run pytest -q -k scorer_rejects         # one test by name
uv run ruff check src scripts tests        # lint — CI-equivalent gate, must pass
uv run ruff check --fix -q <path>          # autofix

uv run python -m prc.s3 ls                 # inspect the competition bucket
uv run python -m prc.s3 pull               # download training/ranking parquet
uv run python -m prc.submit --version N --model models/mechanism_wx [--upload]
uv run python scripts/rehearse_submission.py --model models/mechanism_wx
uv run python -m prc.s3 results            # save every scorer result JSON to docs/reference/scorer-results

# own-flight ADS-B features (run on the GPU host, where the per-day ground.parquet files live)
uv run python -u scripts/adsb/own_flight_features.py 2025 --stands-v2 --ground-v2 --v7 --tag _v7
uv run python -u scripts/adsb/own_flight_features.py 2026 --allow-ranking --stands-v2 --ground-v2 --v7 --tag _v7
uv run python -u scripts/experiment_own_expert_bundle.py l255_r1200_lr025 --src adsb_own_2025_v7.parquet --features 7 [--save STEM]
uv run python -u scripts/experiment_own_norec.py l31 --v7 [--save STEM]      # no-record expert, twelve-fold
```

Long experiments (`scripts/cross_validate_*.py`, `scripts/experiment_*.py`) take 25–60
minutes for a 12-fold run. Launch them in the background with `python -u` — without it
stdout buffers and the log stays empty until exit, which has wasted whole runs here.

## Architecture

The model is a **mixture over reporting regimes**, not a single regressor: a small number of
departures whose recorded taxi time is a reporting anomaly (day slips, schedule copies) carry a
large share of the squared error, and RMSE is dominated by them. Their exact prevalence is not
established (audit finding 8). On matched rows the remaining error is mostly noise in NM's
recorded off-block, which the departing aircraft's **own ADS-B trace** observes independently.

Prediction flows through `prc.submit.predict_frame`, the single path used by both real
submissions and local rehearsals. The best-scoring model is `models/mechanism_v38.json` (kind
`mechanism_bands`): v29's chain plus the own-flight ADS-B experts (matched and no-record) on the
v7 trace features, and the Rome garbled-string copy rule. The wrapper wraps an inner
`mechanism` model and applies, in order:

1. `features.build` + `weather.attach` + `sequencing.add` + `arrivals.add` + `context.add`
   (61 ranking-month context features) + `atfm.attach` (EUROCONTROL daily, permitted) +
   `tencode.transform` (24 in-fold target encodings stored in the inner model JSON)
2. `decomposition.predict` averaged over `seed_stems` (3 seeds; matched base on the
   uncapped residual to `MVT - AOBT_3`, lr 0.02 / 3,175 trees; unmatched base capped)
3. `mechanism.blend` — schedule-fallback gate (refit with context features)
4. `lobt.apply` — LOBT residual expert, per-airport and per-season weights
5. `lirfregime.apply` — LIRF matched-row identity gate
6. `lirfbands.apply` — LIRF no-record rows with gap > 8h: band table (edges 4/8/11.1/13.9/
   16.7/19.4/25h), then `lirfmoe.apply` — copy classifier x copy-free regressor for gap <= 8h
   (`lirf_moe_8h_cat`), then `lirfbands.rome_rules` (schedule floor, same-day exact)
7. `catmember.apply` — CatBoost GPU members (native identity categoricals) blended on the rows
   each serves. v29: two on the AOBT residual at 0.15 each, one on the LOBT residual at 0.3.
   `catmember.anchor_and_rows` supports `aobt`/`lobt`/`eobt`/`iobt`; an alternative clock gives
   a member whose *errors* differ, which is what a blend needs (a second seed or depth does not)
8. `stacker.apply` — second-stage LightGBM on the components' predictions (pre-member
   prediction, member mean, blend, ordinary base, gate probability, LOBT expert, clock proxies,
   airport), applied to matched rows only at `stacker_weight`. `stacker_member_mean` selects
   which members feed its `cat` input and **must match the OOF set the stacker was fitted on**
9. `lirfbands.lobt_clip` — clip to `MVT - LOBT +/- 3606s` (never violated in training)
10. `ownexpert.apply` — residual expert on matched rows whose own ADS-B take-off run joins
    (callsign + airport): stand-dwell end, first taxi-out point (a lower bound), silent-gap
    turnaround rule, and the stand exit against the NM stand's X-Plane coordinates. Features
    from `data/interim/adsb_own_<year>_v<k>.parquet` (`own_features`, `own_feature_version` in
    the wrapper); the booster carries its own feature list. Then `lobt_clip` again. v7 features
    (`own_flight_features.py --v7`) add the stand-to-first-sighting distance along the X-Plane
    taxiway network (`scripts/adsb/taxi_network.py`) and the first moving point after the run's
    last long gap (log section 60)
11. `ownexpert.apply_norec` (`own_norec_stem`) — a separate residual expert on **no-record**
    rows outside LIRF whose trace joins: no anchor, inputs are the trace times, the schedule gap
    and the chain's prediction, target clipped to +/-1,800s (log sections 59-60)
12. `lirfbands.garbled_copy` (`garbled_copy_rule`) — last: LIRF no-record rows with a
    doubled-prefix flight string (`ITYTY672`) and a schedule gap in (4h, 8h] are schedule
    copies, predicted at `MVT - SCHED` (20/20 in 2025; log section 57)

`matrix.build()` caches the assembled feature frame, fingerprinted on the feature source
modules, the raw parquet sizes, and the ATFM file. GPU fits (CatBoost) run on a GPU host
(RTX 4070 Ti); everything else runs locally.

Board ladder: v16 301.14 -> v21 280.80 -> v25 269.82 -> v28 268.33 -> v29 267.87 -> v31 260.06
(own-flight expert) -> v32 257.73 (silent-gap rule) -> v33 252.69 (X-Plane stand exits) -> v34 252.14
-> v35 247.54 (ground_v2 extraction) -> v36 246.86 (garbled copy rule) -> v37 246.01 (no-record
expert) -> **v38 244.19** (v7 trace features), rank 21 of 213 on 2026-10-01. `docs/submissions.md` is the per-version record, `docs/state.md` the
current position.

## Rules that constrain the code

`docs/rules-and-compliance.md` is canonical. The ones that bite:

- **Never tune against a submitted score.** Parameters, thresholds and blend weights come
  from training data only. Uploading variants to probe the hidden set is forbidden by the
  organisers and by project policy.
- **Five submissions per UTC day**, and a rejected version number is permanently spent —
  re-uploading the same filename returns the same stale rejection. `prc.submit` checks the
  bucket before uploading; don't bypass it. Two versions were lost to this.
- **External data must be open-licensed and declared** (the wrapper's `external_data`). In use:
  METAR (Iowa State, public domain), OurAirports (public domain), EUROCONTROL daily performance
  files (ruled open 2026-09-17), **ADSB.lol** archives (ODbL; ruled open 2026-09-17) and the
  **X-Plane Scenery Gateway** stand coordinates (GPL v2+). **OpenSky state vectors are
  forbidden** by organiser ruling. Meteostat is partly CC-BY-NC and fails eligibility.
- **There is no prediction-time cutoff** (organiser, 2026-09-19): the model is for post-ops, and
  open trajectory data of the scored flight itself may be used.

## Evaluating a change

- **Validation is unfiltered.** Never drop extreme targets from a holdout; doing so once
  reported 346 against a real 602.
- **Measure against the residual the current model leaves, not against the target.**
  Stand×runway explains 39.5% of taxi variance and adds nothing, because the model already
  extracts it. Features that correlate with the target but not the residual are redundant —
  this has killed four ideas.
- **Confirm on 12 folds before submitting.** A single split once overstated a gain by 19×.
  Screen on few folds, confirm on twelve.
- **Judge a change on its paired twelve-fold evidence, not on board transfer.** Report the
  baseline by name, the months won, and drop-1/10/50 as a concentration diagnostic. Board
  scores are history, not evidence: do not derive thresholds, transfer multipliers, priorities
  or predicted scores from them (audit findings 5 and 11). Small consistent gains can ship,
  preferably bundled.
- The stored "v29 OOF" (`analyse_error_anatomy.v29_oof`) is a reconstruction of the shipped
  chain, not an exact nested OOF; name it as such in every claim (audit finding 1).
- **Hold out whole months, never random rows**, and select hyperparameters on months
  disjoint from January and July — reusing rounds tuned on the ranking months leaked into
  exactly the folds that matter. January/July 2025 are calendar-matched *selection
  diagnostics*, not untouched holdouts.
- Report bulk and tail RMSE separately; the full number swings on tail prevalence, which
  varies 5× month to month.

`scripts/rehearse_submission.py` runs the real scoring path over a stand-in ranking set
built from 2025 Jan+Jul, blanked as the organisers blank theirs. Its **leakage probe**
scores the same rows with `BLOCK_TIME_UTC_mvt` present and absent and requires identical
predictions — run it before any submission.
