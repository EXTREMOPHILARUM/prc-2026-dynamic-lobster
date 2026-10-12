"""Score the ranking set and write/upload a submission parquet.

    uv run python -m prc.submit --version 1
    uv run python -m prc.submit --version 1 --upload
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc import (
    arrivals,
    atfm,
    catmember,
    config,
    context,
    data,
    dayoffset,
    decomposition,
    features,
    lirfbands,
    lirfmoe,
    lirfregime,
    lobt,
    mechanism,
    ownexpert,
    pubcal,
    rwystate,
    s3,
    sequencing,
    stacker,
    standrwy,
    tencode,
    twostage,
    weather,
)


def _next_free_version() -> int:
    used = {
        int(p.stem.rsplit("_v", 1)[1])
        for p in config.SUBMISSIONS.glob(f"{config.TEAM}_v*.parquet")
        if p.stem.rsplit("_v", 1)[-1].isdigit()
    }
    return max(used, default=0) + 1


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_model(stem: Path) -> tuple[dict, object]:
    """Load a model of any supported kind. Returns (metadata, handle)."""
    is_composite = not stem.suffix and Path(f"{stem}.json").exists()
    if not is_composite:
        model = lgb.Booster(model_file=str(stem))
        return json.loads(stem.with_suffix(".json").read_text()) | {"kind": "single"}, model

    meta = json.loads(Path(f"{stem}.json").read_text())
    kind = meta.get("kind")
    if kind == "mechanism_bands":
        inner_meta, handle = load_model(stem.parent / meta["inner_stem"])
        meta["_inner_meta"] = inner_meta
        return meta, handle
    if kind in ("mechanism", "mechanism_day"):
        ordinary_stem = meta["ordinary_stem"]
        inner = json.loads((stem.parent / f"{ordinary_stem}.json").read_text()) \
            if kind == "mechanism_day" else meta
        if inner.get("seed_stems"):
            models = [decomposition.load(stem.parent / s)[0] for s in inner["seed_stems"]]
        else:
            models, _ = decomposition.load(stem.parent / inner["ordinary_stem"])
        gate = lgb.Booster(model_file=str(
            stem.parent / f"{ordinary_stem}.gate.txt" if kind == "mechanism_day"
            else f"{stem}.gate.txt"))
        meta["_inner"] = inner
        return meta, (models, gate)
    if kind == "decomposition":
        models, meta = decomposition.load(stem)
        return meta, models
    models, meta = twostage.load(stem)
    return meta, models


def predict_frame(
    meta: dict,
    handle: object,
    rank: pd.DataFrame,
    movements: pd.DataFrame | None = None,
) -> np.ndarray:
    """Score a ranking-shaped frame. The single path used by submissions and rehearsals.

    `rank` holds the departure rows to score. `movements` is the full movement frame
    those departures came from, needed only by the arrival features, which read the
    arrival rows `rank` has already had filtered out. Omitting it when the model wants
    them is an error rather than a silent column of NaNs.
    """
    if meta.get("kind") == "mechanism_bands":
        inner_meta = meta["_inner_meta"]
        X = assemble_features(inner_meta, rank, movements)
        pred, ordinary, gate_p = _predict_parts(inner_meta, handle, rank, X)
        expert_pred = None
        if meta.get("lobt_expert_stem"):
            expert = lgb.Booster(model_file=str(config.MODELS / f"{meta['lobt_expert_stem']}.txt"))
            airport_arr = X["ADEP_mvt"].astype("string").to_numpy()
            expert_pred = lobt.predict(expert, X)
            if meta.get("lobt_alphas_by_season"):
                month_arr = pd.to_datetime(rank["MVT_TIME_UTC_mvt"], utc=True).dt.month.to_numpy()
                season = np.where(np.isin(month_arr, (11, 12, 1, 2, 3)), "winter",
                                  np.where(np.isin(month_arr, (5, 6, 7, 8, 9)), "summer", "global"))
                pred = pred.copy()
                for name in ("winter", "summer", "global"):
                    rows = season == name
                    if rows.any():
                        alphas = meta["lobt_alphas_by_season"].get(name, meta["lobt_alphas"])
                        pred[rows] = lobt.apply(pred[rows], expert_pred[rows], airport_arr[rows], alphas)
            else:
                pred = lobt.apply(pred, expert_pred, airport_arr, meta["lobt_alphas"])
        if meta.get("regime_stem"):
            regime = lirfregime.load(config.MODELS / meta["regime_stem"])
            pred = lirfregime.apply(pred, lirfregime.probability(regime, rank, X), X)
        moe_after_bands = None
        if meta.get("lirf_moe_stem"):
            inner_json = json.loads((config.MODELS / f"{meta['inner_stem']}.json").read_text())
            moe_after_bands = (lirfmoe.load(config.MODELS / meta["lirf_moe_stem"]), inner_json["target_encoding"])
        elif meta.get("short_expert_stem"):
            booster = lgb.Booster(
                model_file=str(config.MODELS / f"{meta['short_expert_stem']}.txt"))
            pred = lirfbands.apply_short_expert(pred, X, booster)
        pred = lirfbands.apply(pred, X, meta["band_table"])
        if moe_after_bands is not None:
            # the mixture serves its scope after the band table so an extended scope overrides the 4-8h band
            pred = lirfmoe.apply(pred, moe_after_bands[0], rank, X, moe_after_bands[1])
        if meta.get("rome_rules"):
            pred = lirfbands.rome_rules(pred, rank, X)
        members = meta.get("cat_members") or (
            [{"stem": meta["cat_member_stem"], "weight": meta["cat_weight"]}] if meta.get("cat_member_stem") else [])
        member_preds = []
        if members:
            from catboost import CatBoostRegressor
            base_pred = pred.copy()
            for spec in members:
                cm = json.loads((config.MODELS / f"{spec['stem']}.json").read_text())
                model = CatBoostRegressor(); model.load_model(str(config.MODELS / f"{spec['stem']}.cbm"))
                Xc = X[cm["features"]] if all(c in X.columns for c in cm["features"]) else X
                member = catmember.predict(model, rank, Xc, cm["categoricals"], cm["target_encoding"], cm.get("anchor", "aobt"))
                # each member shifts the prediction from the same pre-blend base by its weight
                shift = catmember.apply(base_pred, member, spec["weight"]) - base_pred
                pred = pred + shift
                # the stacker's "cat" input is the mean of the members it was fitted on
                if meta.get("stacker_member_mean", "aobt") == "all" or cm.get("anchor", "aobt") == "aobt":
                    member_preds.append(member)
        if meta.get("stacker_stem"):
            if not member_preds or ordinary is None:
                raise SystemExit("the stacker needs CatBoost members and a mechanism inner model")
            model, _ = stacker.load(config.MODELS / meta["stacker_stem"])
            matched = X[decomposition.ANCHOR_COL].notna().to_numpy()
            F = stacker.frame(pred, base_pred, np.mean(member_preds, axis=0), ordinary, gate_p,
                              expert_pred if expert_pred is not None else np.zeros(len(pred)), X)
            pred = stacker.apply(pred, model, F, matched, float(meta.get("stacker_weight", 1.0)))
        if meta.get("lobt_clip"):
            pred = lirfbands.lobt_clip(pred, X)
        if meta.get("own_expert_stem") or meta.get("own_expert_stems") or meta.get("own_expert_mix"):
            # fitted on the clipped chain output, as the OOF it learned from was built. `own_expert_mix` is a weighted
            # set of experts ({"stems": [...], "weight": w}; seeds within one entry are averaged), all reading one frame
            mix = meta.get("own_expert_mix") or [{"stems": meta.get("own_expert_stems") or [meta["own_expert_stem"]],
                                                  "weight": 1.0}]
            members = []
            for entry in mix:
                bs = [ownexpert.CatExpert(str(config.MODELS / f"{stem}.cbm")) if (config.MODELS / f"{stem}.cbm").exists()
                      else lgb.Booster(model_file=str(config.MODELS / f"{stem}.txt")) for stem in entry["stems"]]
                members.append((bs[0] if len(bs) == 1 else ownexpert.Bagged(bs), float(entry["weight"])))
            if abs(sum(w for _, w in members) - 1.0) > 1e-6:
                raise SystemExit("own_expert_mix weights must sum to one")
            own_cols = ownexpert.OWN_BY_VERSION[int(meta.get("own_feature_version", 1))]
            own = ownexpert.load_features(meta["own_features"], rank[config.ID_COL], own_cols)
            anchor = np.where(X[lirfbands.MISSING_COL].to_numpy() == 1, np.nan,
                              X[decomposition.ANCHOR_COL].to_numpy(dtype="float64"))
            f = ownexpert.frame(X["ADEP_mvt"].astype("string").to_numpy(), rank["MVT_TIME_UTC_mvt"],
                                np.clip(pred, 0, None), anchor, own)
            lean = None
            if meta.get("own_expert_blend"):
                # the lean v7-input expert on the same chain output, blended in for diversity (log section 63)
                spec = meta["own_expert_blend"]
                lean = (ownexpert.apply(lgb.Booster(model_file=str(config.MODELS / f"{spec['stem']}.txt")), f),
                        float(spec["weight"]))
            if meta.get("own_expert_rich"):
                # the rich expert also reads the chain's matrix (log section 63)
                f = ownexpert.with_rich(f, X, ownexpert.rich_columns(X, meta["own_expert_rich"]))
            if meta.get("own_expert_daily"):
                # per airport-day reception over the frame's own joined departures (log section 63)
                f = pd.concat([f, ownexpert.daily_reception(f, rank["MVT_TIME_UTC_mvt"])], axis=1)
            if meta.get("own_expert_lag"):
                # per-stand first-sighting lag encodings, tables fitted with the booster (log section 63)
                tables = json.loads((config.MODELS / f"{meta['own_expert_lag']}.lag.json").read_text())
                for name, v in ownexpert.lag_apply(tables, rank).items():
                    f[name] = v
            pred = np.sum([w * ownexpert.apply(b, f) for b, w in members], axis=0)
            if lean is not None:
                # both leave rows outside the expert's scope at the chain output, so the blend touches scope rows only
                pred = (1.0 - lean[1]) * pred + lean[1] * lean[0]
            if meta.get("lobt_clip"):
                pred = lirfbands.lobt_clip(pred, X)
        if meta.get("own_norec_stem"):
            # no-record rows outside Rome whose own trace joined (log section 59); fitted on the chain
            # output at this point, which the matched expert and the LOBT clip leave unchanged there
            booster = lgb.Booster(model_file=str(config.MODELS / f"{meta['own_norec_stem']}.txt"))
            own_cols = ownexpert.OWN_BY_VERSION[int(meta.get("own_norec_feature_version", 6))]
            own = ownexpert.load_features(meta["own_features"], rank[config.ID_COL], own_cols)
            f = ownexpert.frame(X["ADEP_mvt"].astype("string").to_numpy(), rank["MVT_TIME_UTC_mvt"],
                                np.clip(pred, 0, None), np.full(len(pred), np.nan), own)
            f["gap"] = X[lirfbands.SCHED_COL].to_numpy(dtype="float64")
            pred = ownexpert.apply_norec(booster, f, X[lirfbands.MISSING_COL].to_numpy())
        if meta.get("garbled_copy_rule"):
            # last, so no later stage can move a row the rule has pinned to its schedule gap
            pred = lirfbands.garbled_copy(pred, rank, X)
        if meta.get("pubcal_offsets"):
            # airport-month means to the published monthly taxi-out (log section 62); only rows under
            # pubcal.CAP move, which the garbled rule's pinned rows (gap > 4h) never are
            shifted, rows = pubcal.apply(pred, X["ADEP_mvt"].astype("string").to_numpy(), rank["MVT_TIME_UTC_mvt"],
                                         pubcal.load_published(), meta["pubcal_offsets"],
                                         float(meta.get("pubcal_vmin", pubcal.VMIN)))
            if meta.get("lobt_clip"):
                shifted[rows] = lirfbands.lobt_clip(shifted, X)[rows]
            pred = shifted
        return np.clip(pred, 0, None)

    X = assemble_features(meta, rank, movements)
    return _predict_matrix(meta, handle, rank, X)


def assemble_features(
    meta: dict, rank: pd.DataFrame, movements: pd.DataFrame | None
) -> pd.DataFrame:
    """The feature matrix for a ranking-shaped frame, in the model's column order."""
    extra: list[str] = []
    if meta.get("use_arrivals"):
        if movements is None:
            raise SystemExit(
                "this model uses arrival features, which need the full movement frame; "
                "pass movements=<unfiltered frame> to predict_frame"
            )
        rank = arrivals.add(rank, movements)
        extra += meta.get("arrival_features", arrivals.FEATURES)
    if meta.get("use_context"):
        if movements is None:
            raise SystemExit("this model uses context features; pass movements=<unfiltered frame>")
        rank = context.add(rank, movements)
        extra += context.FEATURES
    if meta.get("use_atfm"):
        rank = atfm.attach(rank)
        extra += atfm.FEATURES
    if meta.get("use_weather"):
        rank = weather.attach(rank)
        extra += weather.FEATURES
    if meta.get("use_sequencing"):
        rank = sequencing.add(rank)
        extra += sequencing.FEATURES
    if meta.get("use_rwystate"):
        # runway state and gate retention from the frame's own departures (log section 63)
        rank = rank.join(rwystate.compute(rank))
        extra += rwystate.FEATURES
    if meta.get("use_standrwy"):
        rank = rank.join(standrwy.compute(rank))
        extra += standrwy.FEATURES
    X, _ = features.build(rank, categories=meta["category_levels"])
    if extra:
        X = X.join(rank[extra])
    if meta.get("target_encoding"):
        X = X.join(tencode.transform(rank, meta["target_encoding"]))
    absent = [c for c in meta["features"] if c not in X.columns]
    if absent:
        raise SystemExit(f"features missing at scoring time: {absent}")
    return X[meta["features"]]


def _predict_matrix(meta: dict, handle: object, rank: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    return _predict_parts(meta, handle, rank, X)[0]


def _predict_parts(meta: dict, handle: object, rank: pd.DataFrame, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Prediction plus, for mechanism models, the ordinary base and the schedule-fallback gate probability."""
    kind = meta.get("kind", "single")
    if kind in ("mechanism", "mechanism_day"):
        models, gate = handle
        inner = meta.get("_inner", meta)
        ordinary_meta = json.loads(
            (config.MODELS / f"{inner['ordinary_stem']}.json").read_text())
        members = models if isinstance(models, list) else [models]
        uncapped = bool(inner.get("matched_uncapped", ordinary_meta.get("matched_uncapped", False)))
        preds = []
        for m in members:
            comp = decomposition.predict_components(
                m, X,
                lirf_tail_scale=ordinary_meta["lirf_tail_scale"],
                other_tail_scale=ordinary_meta["other_tail_scale"],
                matched_uncapped=uncapped)
            if uncapped:
                m_rows = X[decomposition.ANCHOR_COL].notna().to_numpy()
                base_only = m_rows.copy()
                if inner.get("unmatched_other_base_only"):
                    base_only |= ~(X["ADEP_mvt"].astype("string") == "LIRF").to_numpy()
                preds.append(np.where(base_only, comp["base"], comp["prediction"]))
            else:
                preds.append(comp["prediction"])
        ordinary = np.mean(preds, axis=0)
        probability = gate.predict(X[meta["gate_features"]])
        pred = mechanism.blend(
            probability, X[meta["proxy_col"]].to_numpy(dtype="float64"), ordinary)
        if kind == "mechanism_day":
            bands = dayoffset.band(rank, X)
            day_p = dayoffset.probability(bands, meta["day_rates"])
            pred = dayoffset.apply(pred, day_p)
        return pred, ordinary, probability
    if kind == "decomposition":
        return decomposition.predict(
            handle, X, lirf_tail_scale=meta["lirf_tail_scale"],
            other_tail_scale=meta["other_tail_scale"]), None, None
    if kind == "single":
        return np.clip(handle.predict(X[handle.feature_name()]), 0, None), None, None
    return twostage.predict(handle, X), None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int)
    ap.add_argument("--final", action="store_true",
                    help="the final phase's single submission <team>_final.parquet (needs PRC_RANKING_SET=final)")
    ap.add_argument("--model", default=str(config.MODELS / "lgbm_baseline.txt"),
                    help="single booster .txt, or the stem models/twostage for the "
                         "two-stage model")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing submission for this version number (don't: the "
             "version number is the only handle the scorer gives us)",
    )
    args = ap.parse_args()
    if args.final != (config.RANKING_SET == "final"):
        raise SystemExit("--final goes with PRC_RANKING_SET=final (and only with it)")
    if not args.final and args.version is None:
        raise SystemExit("--version is required (or --final)")

    stem = Path(args.model)
    meta, handle = load_model(stem)
    movements = data.load_ranking()
    rank = features.departures(movements).copy()
    rank["_pred"] = predict_frame(meta, handle, rank, movements=movements)

    template = data.load_submission_template()
    out = template.drop(columns=[config.TARGET], errors="ignore").merge(
        rank[[config.ID_COL, "_pred"]], on=config.ID_COL, how="left"
    )
    missing = out["_pred"].isna().sum()
    if missing:
        raise ValueError(f"{missing} template rows had no prediction — check {config.ID_COL} join")
    # the ranking script matches the template schema exactly: int32 seconds
    out[config.TARGET] = out.pop("_pred").round().astype("int32")
    out = out[list(template.columns)]
    assert out[config.ID_COL].equals(template[config.ID_COL]), "row order/IDs diverged from template"

    path = config.SUBMISSIONS / (f"{config.TEAM}_final.parquet" if args.final else f"{config.TEAM}_v{args.version}.parquet")
    if path.exists() and not args.force:
        raise SystemExit(
            f"{path.name} already exists. Submissions are write-once: the scorer keys "
            f"results by version number, so overwriting one silently detaches the score "
            f"in the bucket from the file on disk. Use the next free version "
            f"(--version {_next_free_version()}), or --force if you really mean it."
        )
    out.to_parquet(path, index=False)
    print(f"wrote {path}  rows={len(out):,}  mean={out[config.TARGET].mean():.1f}s")

    # provenance: which model produced this file, and what it scored locally
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "version": "final" if args.final else args.version,
                "model": str(stem),
                "kind": meta.get("kind", "single"),
                "holdout": {k: v for k, v in meta.items() if k.startswith("rmse")},
                "git_commit": _git_commit(),
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )

    if args.upload:
        if args.final and any(o["Key"] == path.name for o in s3.list_objects(config.SUBMIT_BUCKET)):
            raise SystemExit(f"{path.name} is already in the bucket: the final submission is single, not replaced")
        used = s3.uploads_today()
        if len(used) >= s3.DAILY_LIMIT:
            raise SystemExit(
                f"{len(used)} of {s3.DAILY_LIMIT} submissions already used today "
                f"(UTC): {', '.join(used)}.\n"
                f"The quota resets at 00:00 UTC. A rejected version number is spent, "
                f"so uploading now would burn v{args.version} for nothing."
            )
        print(f"submissions used today (UTC): {len(used)} of {s3.DAILY_LIMIT}")
        s3.push(path)
        # keep the scorer's own response next to the file's provenance (audit finding 13)
        result = s3.wait_for_result(path.stem)
        if result is None:
            print("no scorer result yet; fetch it later with `uv run python -m prc.s3 results`")
        else:
            print(f"scorer result saved to {result.relative_to(config.ROOT)}:")
            print(result.read_text())


if __name__ == "__main__":
    main()
