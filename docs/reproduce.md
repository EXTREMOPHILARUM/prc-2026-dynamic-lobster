# Reproducing the submission (v48, RMSE 240.51)

Two levels of reproduction, with different completeness:

1. **Prediction from the fitted components: complete.** Given the fitted files in `models/`
   (not published: `models/` is git-ignored and the fitted files are about 640 MB; section 3 refits them) and the feature files in `data/interim/`,
   `prc.submit.predict_frame` regenerates the submitted predictions. Section 4.
2. **Training every component from raw data: documented, with known gaps.** The model grew over 48
   versions. Several components were assembled by hand or fitted with flags no script records.
   Section 5 lists each gap. Section 3 is the full sequence as far as the repo establishes it.

Machine notes: CatBoost fits marked **[GPU]** ran on an RTX 4070 Ti. Steps marked **[RAW]** need the
raw adsb.lol archives (~hundreds of GB) and hard-code `~/prc_data_challenge` and
`~/Data/adsb_raw<year>` as paths. Everything else runs on a laptop. Long runs take 25-60 minutes per
12-fold run; launch them with `python -u` in the background.

## 0. The final-phase submission

The final phase scores Jan, Feb, Jun and Jul 2026 (`final_ranking.parquet`, `final_submitting.parquet`).
`models/mechanism_v48_final.json` is `mechanism_v48.json` with the 2026 own-flight feature file replaced by
`adsb_own_2026_v10final.parquet`, rebuilt over all four months:

    PRC_RANKING_SET=final uv run python -u scripts/adsb/own_flight_features.py 2026 --allow-ranking --stands-v2 --ground-v2 --v7 --v8 --v9 --v10 --tag _v10final
    PRC_RANKING_SET=final uv run python -m prc.submit --final --model models/mechanism_v48_final

The Feb/Jun 2026 adsb.lol days are listed in `scripts/adsb/days_final_2026_feb_jun.txt` (ingest with
`DAYS_FILE=... scripts/adsb/reingest_v2.sh`, extraction with `scripts/adsb/extract_v2.sh`); 2026-06-07, 06-23 and
06-25 could not be fetched, and their departures are served without the trace experts.

## 1. What the model is

A chain over reporting regimes, served by `prc.submit.predict_frame`. Stages, in order: base
decomposition (3 LightGBM seeds, residual to `MVT - AOBT_3`) -> schedule-fallback gate -> LOBT residual
expert -> Rome identity gate -> Rome band table / copy mixture / schedule rules -> CatBoost members ->
LightGBM stacker -> LOBT clip -> own-flight ADS-B expert mix (matched rows) -> LOBT clip ->
no-record ADS-B expert -> Rome garbled-string copy rule. `README.md` ("Architecture") describes each
stage; `docs/experiment-log.md` records the evidence for each.

## 2. The wrapper and its components

`models/mechanism_v48.json` (kind `mechanism_bands`) names every component. v48 is v47 with
`inner_stem` swapped to the v48 base.

| Wrapper key | Files | Produced by |
| --- | --- | --- |
| `inner_stem` `mechanism_mechanism_v48b_inner` (+ `seed_stems` s100-s102) | `.json`, `.gate.txt`, `decomposition_mechanism_v48b_s{100,101,102}.*` | `scripts/refit_v20.py --out mechanism_v48b --atfm --uncapped --rome-rules --rwystate --base-params '{"feature_fraction": 0.5, "num_leaves": 511}' --inner-from mechanism_mechanism_v26_inner` |
| `lobt_expert_stem` | `lobt_expert.txt` | `scripts/refit_v17.py` |
| `lobt_alphas` | inline | `alphas_oof` of `data/interim/lobt_alphas_v21.json` (producer: gap 4) |
| `lobt_alphas_by_season` | inline | `scripts/experiment_seasonal_blend.py` -> `seasonal_blend_metrics.json`, copied by hand |
| `regime_stem` `lirf_regime` | `.json`, `.gate.txt` | `scripts/fit_lirfregime.py --inner mechanism_v17 --out mechanism_v18` |
| `band_table` | inline | `scripts/refit_v19.py` (deterministic; identical since v19) |
| `lirf_moe_stem` `lirf_moe_8h_cat` **[GPU]** | `.json`, `.clf.txt`, `.reg.cbm` | `scripts/fit_lirfmoe.py --max-gap-h 8 --regressor catboost --stem lirf_moe_8h_cat` (gap 3) |
| `cat_members` **[GPU]** | `catmember_long`, `catmember_long_s7`, `catmember_lobt` (`.cbm`, `.json`) | `scripts/fit_catmember.py --iterations 10000 --lr 0.03 --stem catmember_long`; `--anchor lobt --stem catmember_lobt`; seed-7 member: gap 1 |
| `stacker_stem` `stacker_mid`, weight 0.5 | `.txt`, `.json` | `scripts/fit_stacker.py --stem stacker_mid --inner mechanism_v27 --out mechanism_v28` |
| `own_features` | `data/interim/adsb_own_{2025,2026}_v10.parquet` | step 13 below |
| `own_expert_mix` | seven boosters (`ownexpert_v10lag`, `_v10lag_s8`, `_daily_s7`, `_v10f_s7`, `_v10f_s8`, `_cb_v9` **[GPU]**, `_lean_v9`) | `scripts/experiment_own_expert_rich.py <config> --months --src adsb_own_2025_v{9,10}.parquet --save <stem>` (gaps 6-7) |
| `own_norec_stem` `ownexpert_norec_v2` | `.txt` | `scripts/experiment_own_norec.py l31 --v7 --save ownexpert_norec_v2` |
| `garbled_copy_rule`, `rome_rules`, `lobt_clip`, `own_expert_*` flags | inline | set by hand |
| `external_data` | metadata | not read at serving time; see the README's external-data table |

## 3. Full sequence

**A. Setup and data**

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env                                  # S3 keys from s3-console.opensky-network.org
uv run python -m prc.s3 pull                          # training, ranking, submitting parquet
uv run python scripts/fetch_metar.py
uv run python scripts/fetch_ourairports.py
uv run python scripts/fetch_eurocontrol_daily.py --download
uv run python -u scripts/fetch_xplane_stands.py
```

**B. ADS-B ground extraction [RAW]** (the GPU host)

```bash
setsid nohup scripts/adsb/reingest_v2.sh >> data/interim/reingest_v2.log 2>&1 &   # runs ground_points.py --v2 per day
setsid nohup scripts/adsb/extract_v2.sh  >> data/interim/extract_v2.log 2>&1 &
scripts/adsb/fetch_one_replica.sh <day> <variant>   # 2025-01-11 staging-0; 2025-05-28..06-11 staging-0tmp; 2025-10-15 staging-0
```

**C. Own-flight feature files** (read `ground_v2.parquet`)

```bash
uv run python -u scripts/adsb/own_flight_features.py 2025 --stands-v2 --ground-v2 --tag _v6
uv run python -u scripts/adsb/own_flight_features.py 2025 --stands-v2 --ground-v2 --v7 --tag _v7
uv run python -u scripts/adsb/own_flight_features.py 2025 --stands-v2 --ground-v2 --v7 --v8 --v9 --tag _v9
uv run python -u scripts/adsb/own_flight_features.py 2025 --stands-v2 --ground-v2 --v7 --v8 --v9 --v10 --tag _v10
uv run python -u scripts/adsb/own_flight_features.py 2026 --allow-ranking --stands-v2 --ground-v2 --v7 --v8 --v9 --v10 --tag _v10
```

**D. Boosting-round selection**

```bash
uv run python scripts/select_rounds.py
uv run python -u scripts/select_rounds_ctx.py --inner 4 10
uv run python -u scripts/select_rounds_v20.py --lr 0.02 --max-rounds 20000 --inner 4 10
```

**E. Earlier-generation fits that later stages read**

```bash
uv run python -u scripts/cross_validate_v17.py --threads 8
uv run python scripts/refit_v17.py                       # lobt_expert, decomposition_ctx, mechanism_v17
uv run python -u scripts/refit_lobt_alphas.py --threads 8
uv run python scripts/fit_lirfregime.py --inner mechanism_v17 --out mechanism_v18
uv run python -u scripts/experiment_lirf_regime.py --baseline v17_oof_predictions.parquet
uv run python -u scripts/refit_v19.py                    # band table
```

**F. Out-of-fold predictions the stacker and experts are fitted on**

```bash
uv run python -u scripts/cross_validate_v19.py --threads 10 --seeds 3
uv run python -u scripts/cross_validate_v20.py --uncapped --rome-rules --tag v21
uv run python -u scripts/experiment_catboost_gpu.py --iterations 5000 --depth 8      # [GPU] catgpu_d8
uv run python -u scripts/experiment_catboost_gpu.py --target lobt                    # [GPU] catgpu_lobt
uv run python -u scripts/experiment_stacker.py --mid
uv run python -u scripts/experiment_own_expert_bundle.py l255_r1200_lr025 --src adsb_own_2025_v6.parquet --features 6 --tag _v6b
uv run python -u scripts/experiment_own_expert_bundle.py l255_r1200_lr025 --src adsb_own_2025_v7.parquet --features 7 --tag _v7
```

**G. Production fits** (section 2 gives each command)

**H. Assemble, rehearse, submit**

```bash
# models/mechanism_v48.json: v47 wrapper with inner_stem = mechanism_mechanism_v48b_inner (hand edit)
uv run python -u scripts/rehearse_submission.py --model models/mechanism_v48   # leakage probe must PASS
uv run python -m prc.submit --version 48 --model models/mechanism_v48
```

## 4. Reproducing the predictions from fitted components

With `models/` and the `data/interim/adsb_own_2026_v10.parquet` feature file in place:

```bash
uv run python -u scripts/rehearse_submission.py --model models/mechanism_v48
uv run python -m prc.submit --version <N> --model models/mechanism_v48    # writes submissions/, no upload without --upload
```

## 5. Known gaps

Stated plainly so nobody reproducing this has to discover them:

1. `catmember_long_s7` has `random_seed: 7`, but `fit_catmember.py` has no seed option; how the seed
   was set is not recorded.
2. The `--weight`, `--inner` and `--out` values of the three CatBoost member fits are not recorded, and
   the `cat_members` list was written into the wrapper by hand.
3. `lirf_moe_8h_cat`: `--inner` / `--out` not recorded. The target encodings are identical across the
   candidate inners, so this does not change the fitted model.
4. The producer of `data/interim/lobt_alphas_v21.json` is not in the repo.
5. `lobt_alphas_by_season` was copied into the wrapper by hand.
6. The own-expert mix weights came from a non-negative least-squares fit on the ten non-scored months'
   scope rows (log section 63); that fit has no script.
7. The config-to-stem pairing of the seven mix experts is inferred from each booster's features and
   parameters, not recorded verbatim.
8. `v16_oof_predictions.parquet` (column `short_expert`) has no producing script in the repo.
9. The flags of `lirf_unmatched_moe_oof.parquet` and the tags of the `catgpu_*` OOF files are inferred
   from file names.
10. The wrapper files themselves are assembled by hand, version on version.

The stacker and the own-flight experts are fitted on stored out-of-fold files from the v21/v29 era
(`analyse_error_anatomy.v29_oof`, audit finding 1), not on a nested out-of-fold run of v48.
