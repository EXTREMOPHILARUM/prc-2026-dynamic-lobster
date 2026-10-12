"""Thin boto3 wrapper for the OpenSky MinIO buckets.

Usage:
    uv run python -m prc.s3 ls                    # list competition data bucket
    uv run python -m prc.s3 ls --bucket prc-2026-dynamic-lobster
    uv run python -m prc.s3 pull                  # download every object not yet local
    uv run python -m prc.s3 pull --prefix training_
    uv run python -m prc.s3 push submissions/dynamic-lobster_v1.parquet
    uv run python -m prc.s3 results               # save every scorer result JSON (docs/reference/scorer-results)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from botocore.config import Config

from prc import config


def client():
    if not config.S3_ACCESS_KEY or not config.S3_SECRET_KEY:
        raise RuntimeError(
            "PRC_S3_ACCESS_KEY / PRC_S3_SECRET_KEY are unset. Copy .env.example to .env and "
            "fill in an access key generated at https://s3-console.opensky-network.org"
        )
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def list_objects(bucket: str, prefix: str = "") -> list[dict]:
    paginator = client().get_paginator("list_objects_v2")
    out: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def pull(bucket: str, prefix: str = "", dest: Path = config.DATA_RAW, force: bool = False) -> None:
    c = client()
    dest.mkdir(parents=True, exist_ok=True)
    for obj in list_objects(bucket, prefix):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        target = dest / Path(key).name
        if target.exists() and not force and target.stat().st_size == obj["Size"]:
            print(f"skip  {key}")
            continue
        print(f"get   {key} -> {target} ({obj['Size'] / 1e6:.1f} MB)")
        c.download_file(bucket, key, str(target))


#: the organisers cap submissions per UTC day; a rejected version number cannot be reused
DAILY_LIMIT = 5


def uploads_today(bucket: str = config.SUBMIT_BUCKET) -> list[str]:
    """Submission files already uploaded in the current UTC day.

    The quota is keyed to UTC, not local time - reasoning from a local date has burned
    two version numbers (v6 and v11). Always ask the bucket.
    """
    from datetime import UTC, datetime

    today = datetime.now(UTC).date()
    return sorted(
        Path(obj["Key"]).stem
        for obj in list_objects(bucket)
        if obj["Key"].endswith(".parquet") and obj["LastModified"].date() == today
    )


def push(path: Path, bucket: str = config.SUBMIT_BUCKET) -> None:
    client().upload_file(str(path), bucket, path.name)
    print(f"put   {path.name} -> s3://{bucket}/{path.name}")


RESULTS_DIR = config.ROOT / "docs" / "reference" / "scorer-results"
RESULT_SUFFIX = ".parquet_result.json"


def save_results(bucket: str = config.SUBMIT_BUCKET, dest: Path = RESULTS_DIR) -> list[Path]:
    """Download every scorer result JSON in the submission bucket, verbatim, into `dest`.

    The scorer's own response is the record of what a version scored (audit finding 13); these
    files are small, carry no credentials, and are tracked in git."""
    dest.mkdir(parents=True, exist_ok=True)
    c = client(); out = []
    for obj in list_objects(bucket):
        if obj["Key"].endswith(RESULT_SUFFIX):
            path = dest / obj["Key"].replace(RESULT_SUFFIX, ".result.json")
            path.write_bytes(c.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            out.append(path)
    return out


def wait_for_result(name: str, bucket: str = config.SUBMIT_BUCKET, timeout_s: float = 900.0,
                    dest: Path = RESULTS_DIR) -> Path | None:
    """Poll for one submission's scorer result, save it verbatim, return its path (None on timeout)."""
    import time
    key = f"{name}{RESULT_SUFFIX}"; c = client(); start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if any(o["Key"] == key for o in list_objects(bucket)):
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{name}.result.json"
            path.write_bytes(c.get_object(Bucket=bucket, Key=key)["Body"].read())
            return path
        time.sleep(20)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls")
    p_ls.add_argument("--bucket", default=config.DATA_BUCKET)
    p_ls.add_argument("--prefix", default="")

    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--bucket", default=config.DATA_BUCKET)
    p_pull.add_argument("--prefix", default="")
    p_pull.add_argument("--force", action="store_true")

    p_push = sub.add_parser("push")
    p_push.add_argument("path", type=Path)
    p_push.add_argument("--bucket", default=config.SUBMIT_BUCKET)

    p_res = sub.add_parser("results")
    p_res.add_argument("--bucket", default=config.SUBMIT_BUCKET)

    args = ap.parse_args()
    if args.cmd == "ls":
        for obj in list_objects(args.bucket, args.prefix):
            print(f"{obj['Size']:>12}  {obj['LastModified']:%Y-%m-%d}  {obj['Key']}")
    elif args.cmd == "pull":
        pull(args.bucket, args.prefix, force=args.force)
    elif args.cmd == "results":
        for path in save_results(args.bucket):
            print(f"saved {path.relative_to(config.ROOT)}")
    else:
        push(args.path, args.bucket)


if __name__ == "__main__":
    main()
