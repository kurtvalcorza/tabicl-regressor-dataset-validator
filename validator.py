"""DIMER dataset validator for TabICLv2 regression."""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

TEMPLATE_NAME = "tabicl-regressor-dataset-validator"
DATASET_DIR = Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset"))
RESULT_PATH = Path(os.getenv("DIMER_RESULT_PATH", "/data/dataset-validations/result.json"))
DONE_CALLBACK = os.getenv("DIMER_DONE_CALLBACK", "").strip()
MIN_TRAIN_ROWS = 50
MIN_EVAL_ROWS = 10
MAX_FEATURES = 2_000

# Numeric/limit configuration. Module-level DEFAULTS (plain literals so import
# never fails); the values used are (re)loaded from the environment inside run()
# via _load_limits(), so a malformed platform value produces a structured
# failure result.json instead of an import-time crash.
CALLBACK_TIMEOUT_SECONDS = 10.0
MAX_SAMPLE_FILES = 25
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1 << 30
MAX_SINGLE_CSV_BYTES = 512 << 20
MAX_COMPRESSION_RATIO = 100.0
MAX_TOTAL_ROWS = 5_000_000


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _load_limits() -> None:
    """Re-read numeric limits from the environment inside the protected path.

    Malformed values raise here (caught by main() -> failure result.json)
    instead of crashing at import before result/callback handling exists.
    """
    # Parse and validate into locals FIRST; publish globals only after every
    # check passes, so a malformed value never leaves an invalid global behind
    # (the crash-path failure callback must still use a valid timeout).
    callback_timeout = _float_env("DIMER_CALLBACK_TIMEOUT_SECONDS", 10.0)
    max_sample = _int_env("DIMER_MAX_SAMPLE_FILES", 25)
    max_archive = _int_env("DIMER_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1 << 30)
    max_single = _int_env("DIMER_MAX_SINGLE_CSV_BYTES", 512 << 20)
    max_ratio = _float_env("DIMER_MAX_COMPRESSION_RATIO", 100.0)
    max_rows = _int_env("DIMER_MAX_TOTAL_ROWS", 5_000_000)
    for _name, _val in (
        ("DIMER_CALLBACK_TIMEOUT_SECONDS", callback_timeout),
        ("DIMER_MAX_COMPRESSION_RATIO", max_ratio),
    ):
        if not math.isfinite(_val) or _val <= 0:
            raise ValueError(f"{_name} must be a positive finite number, got {_val!r}")
    for _name, _val in (
        ("DIMER_MAX_SAMPLE_FILES", max_sample),
        ("DIMER_MAX_ARCHIVE_UNCOMPRESSED_BYTES", max_archive),
        ("DIMER_MAX_SINGLE_CSV_BYTES", max_single),
        ("DIMER_MAX_TOTAL_ROWS", max_rows),
    ):
        if _val <= 0:
            raise ValueError(f"{_name} must be a positive integer, got {_val!r}")
    global CALLBACK_TIMEOUT_SECONDS, MAX_SAMPLE_FILES, MAX_ARCHIVE_UNCOMPRESSED_BYTES
    global MAX_SINGLE_CSV_BYTES, MAX_COMPRESSION_RATIO, MAX_TOTAL_ROWS
    CALLBACK_TIMEOUT_SECONDS = callback_timeout
    MAX_SAMPLE_FILES = max_sample
    MAX_ARCHIVE_UNCOMPRESSED_BYTES = max_archive
    MAX_SINGLE_CSV_BYTES = max_single
    MAX_COMPRESSION_RATIO = max_ratio
    MAX_TOTAL_ROWS = max_rows


def _json_env(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def _upload_result_to_s3(content: str) -> None:
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    bucket = os.getenv("S3_BUCKET", "").strip()
    key = os.getenv("S3_RESULT_KEY", "").strip()
    if not all((endpoint, bucket, key, os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))):
        return
    try:
        import boto3
        boto3.client("s3", endpoint_url=endpoint).put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"Best-effort S3 upload failed: {exc}")


def write_result(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    RESULT_PATH.write_text(content, encoding="utf-8")
    _upload_result_to_s3(content)


def notify_done_callback() -> dict[str, Any]:
    if not DONE_CALLBACK:
        return {"attempted": False}
    parsed = urlparse(DONE_CALLBACK)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "error": f"unsupported callback scheme {parsed.scheme!r}"}
    try:
        response = requests.post(DONE_CALLBACK, timeout=CALLBACK_TIMEOUT_SECONDS)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _normalize_member(name: str) -> str | None:
    if not name or name.endswith("/"):
        return None
    normalized = name.replace("\\", "/")
    parts = Path(normalized).parts
    # Reject absolute paths and parent-directory traversal BEFORE stripping
    # anything (Path() has already collapsed any leading "./").
    if normalized.startswith("/") or ".." in parts:
        raise ValueError(f"unsafe archive member: {name}")
    if len(parts) > 1 and parts[0].lower() in {"dataset", "datasets"}:
        parts = parts[1:]
    return "/".join(parts) if parts else None


@dataclass(frozen=True)
class Entry:
    logical_path: str
    source: Any
    size: int


class DatasetSource:
    def __init__(self) -> None:
        self.archive_name: str | None = None
        self.source_type = "directory"
        self._archive: zipfile.ZipFile | None = None
        self._entries: list[Entry] = []
        zips = sorted(DATASET_DIR.glob("*.zip"))
        if len(zips) > 1:
            raise ValueError(f"multiple dataset zip files found: {[p.name for p in zips]}")
        if zips:
            archive_path = zips[0]
            self.archive_name = archive_path.name
            self.source_type = "zip"
            self._archive = zipfile.ZipFile(archive_path)
            total = 0
            for info in self._archive.infolist():
                logical = _normalize_member(info.filename)
                if logical is None:
                    continue
                # Zip-bomb guard: reject pathological compression ratios.
                compressed = int(getattr(info, "compress_size", 0) or 0)
                uncompressed = int(info.file_size)
                if compressed > 0 and (uncompressed / compressed) > MAX_COMPRESSION_RATIO:
                    self._archive.close()
                    raise ValueError(
                        f"archive member {info.filename!r} has compression ratio "
                        f"{uncompressed / compressed:.0f}:1; limit is {MAX_COMPRESSION_RATIO:.0f}:1"
                    )
                total += uncompressed
                self._entries.append(Entry(logical, info, uncompressed))
            if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError(f"archive expands to {total:,} bytes; limit is {MAX_ARCHIVE_UNCOMPRESSED_BYTES:,}")
        else:
            for path in sorted(DATASET_DIR.rglob("*")):
                if path.is_file():
                    self._entries.append(Entry(str(path.relative_to(DATASET_DIR)), path, int(path.stat().st_size)))

    @property
    def files(self) -> list[str]:
        return sorted(entry.logical_path for entry in self._entries)

    def candidates(self, stem: str) -> list[Entry]:
        return sorted([
            entry for entry in self._entries
            if Path(entry.logical_path).suffix.lower() == ".csv"
            and Path(entry.logical_path).stem.lower() == stem.lower()
        ], key=lambda e: e.logical_path)

    def unique_csv(self, stem: str, required: bool = False) -> Entry | None:
        matches = self.candidates(stem)
        if len(matches) > 1:
            raise ValueError(f"multiple {stem}.csv candidates found: {[m.logical_path for m in matches]}")
        if not matches:
            if required:
                raise FileNotFoundError(f"no {stem}.csv found")
            return None
        if matches[0].size > MAX_SINGLE_CSV_BYTES:
            raise ValueError(f"{matches[0].logical_path} is {matches[0].size:,} bytes; single-CSV limit is {MAX_SINGLE_CSV_BYTES:,}")
        return matches[0]

    def read_csv(self, entry: Entry, nrows: int | None = None) -> pd.DataFrame:
        if self._archive is not None:
            with self._archive.open(entry.source) as handle:
                return pd.read_csv(handle, nrows=nrows)
        return pd.read_csv(entry.source, nrows=nrows)

    def has_nested_zip(self) -> bool:
        return any(Path(entry.logical_path).suffix.lower() == ".zip" for entry in self._entries)

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()


def _check(name: str, successful: bool, message: str) -> dict[str, Any]:
    return {"name": name, "successful": bool(successful), "message": message}


def build_checks(source: DatasetSource, preprocessing: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_column = str(preprocessing.get("target_column") or "target").strip()
    drop_columns = [c.strip() for c in str(preprocessing.get("drop_columns") or "").split(",") if c.strip()]
    checks: list[dict[str, Any]] = []
    # classNames is a mandatory DIMER metadata field. Regression has no classes,
    # so it is always the empty array — per the docs' "even if empty" contract.
    meta: dict[str, Any] = {"targetColumn": target_column, "dropColumns": drop_columns, "classNames": []}

    checks.append(_check(
        "target_not_dropped",
        target_column not in drop_columns,
        f"target_column {target_column!r} must not appear in drop_columns."
        if target_column in drop_columns
        else f"target_column {target_column!r} is not listed in drop_columns.",
    ))
    checks.append(_check("no_nested_zip", not source.has_nested_zip(), "No nested zip found." if not source.has_nested_zip() else "A nested zip was found; upload CSVs directly."))
    try:
        train_entry = source.unique_csv("train", required=True)
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("train_csv_unique", False, str(exc)))
        return checks, meta
    checks.append(_check("train_csv_unique", True, f"Using {train_entry.logical_path}."))

    try:
        # Bounded read: cap memory at MAX_TOTAL_ROWS+1 rows so a pathological
        # file is rejected during load, not after fully materializing it.
        train = source.read_csv(train_entry, nrows=MAX_TOTAL_ROWS + 1)
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("train_csv_parses", False, f"train.csv could not be parsed: {exc}"))
        return checks, meta
    over_cap = len(train) > MAX_TOTAL_ROWS
    checks.append(_check(
        "train_csv_parses",
        True,
        f"Parsed {'>' if over_cap else ''}{len(train)} rows x {train.shape[1]} columns.",
    ))

    columns = list(train.columns)
    meta.update({"columns": columns, "rowCount": (f">{MAX_TOTAL_ROWS}" if over_cap else int(len(train)))})
    checks.append(_check(
        "row_count_within_limit",
        not over_cap,
        f"train.csv exceeds the {MAX_TOTAL_ROWS}-row operational limit."
        if over_cap
        else f"train.csv has {len(train)} rows; operational limit is {MAX_TOTAL_ROWS}.",
    ))
    if over_cap:
        return checks, meta  # don't run downstream checks on a truncated frame

    has_target = target_column in columns
    checks.append(_check("target_column_present", has_target, f"Target column {target_column!r} found." if has_target else f"Target column {target_column!r} not found."))
    if not has_target:
        return checks, meta

    numeric_target = pd.to_numeric(train[target_column], errors="coerce")
    finite_mask = numeric_target.notna() & np.isfinite(numeric_target.to_numpy(dtype=float, na_value=np.nan))
    usable_rows = int(finite_mask.sum())
    distinct_values = int(numeric_target[finite_mask].nunique())
    meta.update({"usableTrainRows": usable_rows, "distinctTargetValues": distinct_values})
    checks.append(_check("target_is_numeric", bool(numeric_target.notna().sum() == train[target_column].notna().sum()), "All non-null target values are numeric."))
    checks.append(_check("target_is_finite", usable_rows == int(train[target_column].notna().sum()), f"{usable_rows} finite numeric target rows found."))
    checks.append(_check("minimum_usable_rows", usable_rows >= MIN_TRAIN_ROWS, f"{usable_rows} usable rows; need at least {MIN_TRAIN_ROWS}."))
    checks.append(_check("target_has_variation", distinct_values >= 2, f"Target has {distinct_values} distinct finite value(s); regression needs variation."))

    feature_columns = [c for c in columns if c != target_column and c not in drop_columns]
    meta["featureColumnCount"] = len(feature_columns)
    checks.append(_check("feature_columns_present", len(feature_columns) >= 1, f"{len(feature_columns)} feature column(s) remain after exclusions."))
    checks.append(_check("feature_count_supported", len(feature_columns) <= MAX_FEATURES, f"{len(feature_columns)} features; configured operational limit is {MAX_FEATURES}."))

    train_set = set(columns)
    for stem in ("val", "test"):
        try:
            entry = source.unique_csv(stem, required=False)
        except Exception as exc:  # noqa: BLE001
            checks.append(_check(f"{stem}_csv_unique", False, str(exc)))
            continue
        if entry is None:
            continue
        checks.append(_check(f"{stem}_csv_unique", True, f"Using {entry.logical_path}."))
        try:
            frame = source.read_csv(entry)  # full split, not just a 5-row schema sample
        except Exception as exc:  # noqa: BLE001
            checks.append(_check(f"{stem}_csv_parses", False, f"{stem}.csv could not be parsed: {exc}"))
            continue
        same = set(frame.columns) == train_set
        checks.append(_check(f"{stem}_schema_matches_train", same, f"{stem}.csv schema matches train.csv." if same else f"{stem}.csv columns differ from train.csv."))
        if not same or target_column not in frame.columns:
            continue
        # _clean_frame can empty a split by removing every non-finite target;
        # verify the full split still has enough usable numeric targets to score.
        numeric = pd.to_numeric(frame[target_column], errors="coerce")
        finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        n_usable = int(finite.sum())
        checks.append(_check(
            f"{stem}_has_usable_targets",
            n_usable >= MIN_EVAL_ROWS,
            f"{n_usable} finite numeric {stem} targets after cleaning; need at least {MIN_EVAL_ROWS}.",
        ))

    return checks, meta


def run() -> int:
    _load_limits()
    preprocessing = _json_env("DIMER_PREPROCESSING_ARGS_JSON")
    pipeline_metadata = _json_env("DIMER_PIPELINE_METADATA_JSON")
    source = DatasetSource()
    try:
        checks, check_meta = build_checks(source, preprocessing)
        successful = all(check["successful"] for check in checks)
        payload = {
            "successful": successful,
            "message": "TabICLv2 regression dataset validation succeeded." if successful else "TabICLv2 regression dataset validation failed — see checks.",
            "datasetSummary": {
                "source": source.source_type,
                "archive": source.archive_name,
                "fileCount": len(source.files),
                "extensions": dict(Counter(Path(p).suffix.lower() or "<none>" for p in source.files)),
                "sampleFiles": source.files[:MAX_SAMPLE_FILES],
            },
            "checks": checks,
            # taskType: DIMER metadata -> baked DIMER_TASK_TYPE env (Custom/Other
            # pipelines) -> model-family literal.
            "metadata": {"template": TEMPLATE_NAME, "taskType": pipeline_metadata.get("taskType") or os.getenv("DIMER_TASK_TYPE") or "tabular_regression", **check_meta},
        }
        write_result(payload)
        log(f"Callback: {json.dumps(notify_done_callback(), sort_keys=True)}")
        return 0 if successful else 1
    finally:
        source.close()


def main() -> int:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001
        payload = {"successful": False, "message": "TabICLv2 dataset validator crashed.", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "metadata": {"template": TEMPLATE_NAME, "classNames": []}}
        try:
            write_result(payload)
            notify_done_callback()
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
