"""DIMER transport helpers for validator execution.

The normal contract is a dataset already mounted at DIMER_DATASET_DIR.  The
on-prem Workbench currently gives validator jobs an empty /data volume plus an
S3/MinIO dataset prefix instead.  This module bridges that transport without
changing validator semantics: local files always win, and S3 is used only when
the dataset directory has no files.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MAX_STAGE_BYTES = 1 << 30


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    value = default if not raw else int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _has_local_files(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and any(path.is_file() for path in dataset_dir.rglob("*"))


def _safe_relative_key(prefix: str, key: str) -> Path | None:
    normalized_prefix = prefix.rstrip("/") + "/"
    if not key.startswith(normalized_prefix):
        raise ValueError(f"S3 object {key!r} is outside configured prefix {normalized_prefix!r}")
    relative = key[len(normalized_prefix):]
    if not relative or relative.endswith("/"):
        return None
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe S3 dataset object key: {key!r}")
    return Path(*posix.parts)


def _s3_client():
    import boto3

    kwargs: dict[str, Any] = {}
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def stage_dataset_if_needed(dataset_dir: Path | None = None) -> dict[str, Any]:
    """Populate an empty local dataset directory from DIMER's S3 prefix.

    Returns a small status record for logging/tests.  A configured S3 prefix that
    contains no files is an error: silently validating an empty directory would
    hide a transport failure as a dataset-format failure.
    """
    target_dir = dataset_dir or Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
    if _has_local_files(target_dir):
        return {"attempted": False, "source": "local"}

    bucket = os.getenv("S3_BUCKET", "").strip()
    prefix = os.getenv("S3_DATASET_PREFIX", "").strip()
    if not bucket or not prefix:
        return {"attempted": False, "source": "local-empty"}

    max_stage_bytes = _positive_int_env("DIMER_MAX_S3_STAGE_BYTES", DEFAULT_MAX_STAGE_BYTES)
    client = _s3_client()
    objects: list[tuple[str, Path, int]] = []
    total_bytes = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            relative = _safe_relative_key(prefix, key)
            if relative is None:
                continue
            size = int(item.get("Size") or 0)
            if size < 0:
                raise ValueError(f"S3 object {key!r} reported an invalid size {size}")
            total_bytes += size
            if total_bytes > max_stage_bytes:
                raise ValueError(
                    f"S3 dataset staging would exceed {max_stage_bytes:,} bytes "
                    f"(configured by DIMER_MAX_S3_STAGE_BYTES)"
                )
            objects.append((key, relative, size))

    if not objects:
        raise FileNotFoundError(f"S3 dataset prefix s3://{bucket}/{prefix} contains no files")

    target_dir.mkdir(parents=True, exist_ok=True)
    for key, relative, _size in objects:
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(destination))

    return {
        "attempted": True,
        "source": "s3",
        "fileCount": len(objects),
        "totalBytes": total_bytes,
    }
