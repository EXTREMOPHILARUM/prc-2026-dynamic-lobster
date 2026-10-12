# Public competitor repositories reviewed (2026-09-16 to 18)

All GPLv3. Clones live in this session's scratchpad only; nothing was copied. What each does,
what we ported (with the measured effect on our folds/board), and what did not transfer.

**Outcome (2026-09-19).** We are at 267.87, ahead of every public entry reviewed here:
elegant-alligator 268.56, zestful-fountain 273.6, OpenAir 277.0, jubilant-vase, kind-mango.
The exception is merry-quicksand at 257.55, whose one edge over us is an ADS-B stacker at
covered airports — organiser-permitted, ported, and measured at ~0.05 correlation with our
residual, i.e. no value on our pipeline (log section 24). **Ranks 1–16 have no public code**,
so nothing above us is available to learn from.

| team (rank, score at review) | repo | their one >10s step | ported here |
| --- | --- | --- | --- |
| merry-quicksand (7, 257.6) | saurabhdxt-labs/prc-challenge | LIRF no-record rows anchored on MVT-SCHED with a residual specialist; later an ADS-B stacker (ADSB.lol) at covered airports | Rome rules (schedule floor, same-day exact) -0.9; ADS-B tested and NEGATIVE (log s.24) |
| elegant-alligator (18, 268.6) | javidmardanov/PRC-Data-Challenge-2026 | missing-row mixture of experts (identity classifier x copy-free regressor) | LIRF unmatched MoE -4.1 board (v24/v25); LOBT +/-3606 bound; LOBT residual expert -3.0 fold; arrival stand/runway context; seasonal blend weights -0.3 |
| zestful-fountain (23, 273.6) | Phoenix-Ops-LTD/prc2026-taxiout | LIRF unmatched -> MVT-SCHED baseline (461 -> 298) | neighbour NM taxi past/future, arrival taxi-in aggregates (inside v17's -5.4); deep CatBoost member (v22, -4.3 board) |
| likable-eagle / OpenAir (27, 277.0) | ChandanHegde07/OpenAir | LIRF unmatched T = D - G with G learned on all LIRF rows | gate-hold form tested: -8.9 fold on unmatched (beaten by the expert), +7.1 on matched (NEGATIVE) |
| jubilant-vase (66, 299.3) | sergiuv11/prc-data-challenge-2026 | single LightGBM + EDDF/EGLL/EHAM specialists 50/50 | specialists tested: flat |
| kind-mango (67, 299.3; refuses AOBT) | skylinkapi/prc-data-challenge-2026-kind-mango | Step-A band table, ITY340 rule, linear_tree + signed-log | finer band edges -1.9 fold; fallback-rate encoders + isotonic gate (-0.9); linear_tree NEGATIVE; disruption meters 0.0 |

Also read: vibrant-lollipop (P10 reference + ATFM, no scores), knowledgeable-helicopter
(CatBoost, no board score), scaffolds (fenil264, victoralcadi, Enzoferrariz, Necklacesaura).
jolly-lobster's repo (LeeMarshall1113/prc-taxiout-2026, rank 6) went private before review.
No repo exists for ranks 1-5 or 8-17 as of 2026-09-18.

Common to every team above us: read the NM clocks per row, treat Rome no-record rows as a
regime, use the ranking month's own movements as context. Common negatives across repos: weather
beyond METAR basics, Huber/log targets, winsorised training targets (-30s for merry-quicksand),
per-airport post-hoc calibration, more seeds beyond 3-5.
