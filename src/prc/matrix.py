"""Cached feature matrix.

Every experiment rebuilds the same thing: read 315 MB of parquet, construct 58 features,
join METAR by merge_asof, compute sequencing. That is minutes of single-threaded pandas
before the first tree is grown, repeated on every run, and it dominates wall-clock time
more than the model fits do.

This caches the built matrix to parquet, keyed by a fingerprint of the inputs and the
feature-producing code. Change a feature module and the fingerprint changes, so a stale
cache cannot be used by accident - the failure mode that would make this dangerous.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from prc import arrivals, atfm, config, context, data, features, sequencing, weather

CACHE_DIR = config.DATA_INTERIM / "matrix_cache"

#: modules whose contents change what the matrix contains
FINGERPRINT_SOURCES = ("features.py", "weather.py", "sequencing.py", "arrivals.py", "context.py", "atfm.py")


def _fingerprint(
    kind: str, use_weather: bool, use_sequencing: bool, use_arrivals: bool,
    use_context: bool = False, use_atfm: bool = False, atfm_lag_days: int = 0, jitter_seed: int | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(
        f"{kind}|weather={use_weather}|sequencing={use_sequencing}"
        f"|arrivals={use_arrivals}|context={use_context}|atfm={use_atfm}".encode()
    )
    if jitter_seed is not None:
        h.update(f"jitter={jitter_seed}".encode() + (Path(__file__).parent / "rehearsal.py").read_bytes())
    if kind == "rank" and config.RANKING_SET != "ranking":
        h.update(f"ranking_set={config.RANKING_SET}".encode())
    if use_atfm and atfm.PATH.exists():
        h.update(f"atfm:{atfm.PATH.stat().st_size}".encode())
    if use_atfm and atfm_lag_days:
        h.update(f"atfm_lag={atfm_lag_days}".encode())
    src = Path(__file__).parent
    for name in FINGERPRINT_SOURCES:
        h.update((src / name).read_bytes())
    for parquet in sorted((config.DATA_RAW).glob("*.parquet")):
        stat = parquet.stat()
        h.update(f"{parquet.name}:{stat.st_size}".encode())
    return h.hexdigest()[:16]


def build(
    kind: str = "train",
    use_weather: bool = True,
    use_sequencing: bool = True,
    use_arrivals: bool = False,
    use_context: bool = False,
    use_atfm: bool = False,
    atfm_lag_days: int = 0,
    jitter_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (raw frame, feature matrix, categorical columns), cached on disk.

    `kind` is "train" (labelled 2025 departures) or "rank" (the 2026 ranking departures). `jitter_seed` (train only)
    perturbs movement and block times as the final phase does (`rehearsal.jitter_times`) before any feature is built.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if jitter_seed is not None and kind != "train":
        raise ValueError("jitter_seed applies to the training matrix only")
    tag = _fingerprint(kind, use_weather, use_sequencing, use_arrivals, use_context, use_atfm, atfm_lag_days, jitter_seed)
    raw_path = CACHE_DIR / f"{kind}_{tag}_raw.parquet"
    x_path = CACHE_DIR / f"{kind}_{tag}_X.parquet"
    meta_path = CACHE_DIR / f"{kind}_{tag}_meta.json"

    if raw_path.exists() and x_path.exists() and meta_path.exists():
        cats = json.loads(meta_path.read_text())["categoricals"]
        raw = pd.read_parquet(raw_path)
        X = pd.read_parquet(x_path)
        for c in cats:
            X[c] = X[c].astype("category")
        return raw, X, cats

    # arrival features need the arrival rows, so the *full* movement frame is loaded and
    # the departures selected from it rather than filtered at load time
    movements = data.load_training() if kind == "train" else data.load_ranking()
    if jitter_seed is not None:
        from prc import rehearsal
        movements = rehearsal.jitter_times(movements, jitter_seed)
    if kind == "train":
        raw = features.departures_with_target(movements).reset_index(drop=True)
    else:
        raw = features.departures(movements).reset_index(drop=True)

    extra: list[str] = []
    if use_arrivals:
        raw = arrivals.add(raw, movements)
        extra += arrivals.FEATURES
    if use_context:
        raw = context.add(raw, movements)
        extra += context.FEATURES
    if use_atfm:
        raw = atfm.attach(raw, lag_days=atfm_lag_days)
        extra += atfm.FEATURES
    if use_weather:
        raw = weather.attach(raw)
        extra += weather.FEATURES
    if use_sequencing:
        raw = sequencing.add(raw)
        extra += sequencing.FEATURES

    X, cats = features.build(raw)
    X = X.drop(columns=["month"], errors="ignore").join(raw[extra])

    raw.to_parquet(raw_path, index=False)
    X.to_parquet(x_path, index=False)
    meta_path.write_text(json.dumps({"categoricals": cats, "features": list(X.columns)}))
    return raw, X, cats


def clear() -> int:
    """Delete every cached matrix. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*"))
    for f in files:
        f.unlink()
    return len(files)
