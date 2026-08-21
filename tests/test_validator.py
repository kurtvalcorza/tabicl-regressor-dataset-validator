import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("validator", Path(__file__).parents[1] / "validator.py")
validator = importlib.util.module_from_spec(SPEC)
# Register before exec so @dataclass field processing can resolve the module
# under `from __future__ import annotations` (dataclasses looks it up in
# sys.modules by __module__; unregistered -> AttributeError on 3.11).
sys.modules["validator"] = validator
SPEC.loader.exec_module(validator)


@pytest.fixture(autouse=True)
def _reset_limits():
    validator._load_limits()
    yield


def test_numeric_target_and_usable_rows(tmp_path, monkeypatch):
    frame = pd.DataFrame({"x": range(60), "target": [float(i) for i in range(60)]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, meta = validator.build_checks(source, {})
    finally:
        source.close()
    assert all(c["successful"] for c in checks)
    assert meta["usableTrainRows"] == 60


def test_duplicate_train_rejected(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pd.DataFrame({"x": [1], "target": [1.0]}).to_csv(tmp_path / "a" / "train.csv", index=False)
    pd.DataFrame({"x": [2], "target": [2.0]}).to_csv(tmp_path / "b" / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {})
    finally:
        source.close()
    assert not next(c for c in checks if c["name"] == "train_csv_unique")["successful"]


def test_non_numeric_target_rejected(tmp_path, monkeypatch):
    pd.DataFrame({"x": range(60), "target": ["bad"] * 60}).to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {})
    finally:
        source.close()
    assert not next(c for c in checks if c["name"] == "target_is_numeric")["successful"]


def test_malformed_numeric_env_yields_structured_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_SINGLE_CSV_BYTES", "not-an-int")
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(validator, "RESULT_PATH", tmp_path / "result.json")
    assert validator.main() == 1
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["successful"] is False


def test_compression_ratio_rejected(tmp_path, monkeypatch):
    zpath = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("train.csv", "x,target\n" + ("0,0\n" * 200_000))  # highly compressible
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    with pytest.raises(ValueError, match="compression ratio"):
        validator.DatasetSource()


def test_row_count_limit_enforced_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_TOTAL_ROWS", "10")
    validator._load_limits()
    pd.DataFrame({"x": range(60), "target": [float(i) for i in range(60)]}).to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, meta = validator.build_checks(source, {})
    finally:
        source.close()
    names = {c["name"]: c["successful"] for c in checks}
    assert names["row_count_within_limit"] is False
    assert "target_is_numeric" not in names  # early return, no downstream checks
    assert meta["rowCount"] == ">10"


def test_non_finite_limit_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", "inf")
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(validator, "RESULT_PATH", tmp_path / "result.json")
    assert validator.main() == 1
    assert json.loads((tmp_path / "result.json").read_text())["successful"] is False


def test_invalid_timeout_does_not_corrupt_global(monkeypatch):
    monkeypatch.setenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "-5")
    prior = validator.CALLBACK_TIMEOUT_SECONDS
    with pytest.raises(ValueError):
        validator._load_limits()
    assert validator.CALLBACK_TIMEOUT_SECONDS == prior
    assert validator.CALLBACK_TIMEOUT_SECONDS > 0


def test_normalize_member_rejects_traversal_and_absolute():
    assert validator._normalize_member("train.csv") == "train.csv"
    assert validator._normalize_member("./train.csv") == "train.csv"
    assert validator._normalize_member("dataset/train.csv") == "train.csv"
    for hostile in ("../train.csv", "../../etc/passwd", "/train.csv", "dataset/../secret.csv"):
        with pytest.raises(ValueError, match="unsafe archive member"):
            validator._normalize_member(hostile)


def test_target_in_drop_columns_is_rejected(tmp_path, monkeypatch):
    pd.DataFrame({"x": range(60), "target": [float(i) for i in range(60)]}).to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {"target_column": "target", "drop_columns": "target,x"})
    finally:
        source.close()
    assert not next(c for c in checks if c["name"] == "target_not_dropped")["successful"]


def test_val_split_target_usability_is_validated(tmp_path, monkeypatch):
    pd.DataFrame({"x": range(60), "target": [float(i) for i in range(60)]}).to_csv(tmp_path / "train.csv", index=False)
    # val schema matches but every target is non-numeric -> 0 usable after cleaning
    pd.DataFrame({"x": range(20), "target": ["bad"] * 20}).to_csv(tmp_path / "val.csv", index=False)
    monkeypatch.setattr(validator, "DATASET_DIR", tmp_path)
    source = validator.DatasetSource()
    try:
        checks, _ = validator.build_checks(source, {})
    finally:
        source.close()
    assert not next(c for c in checks if c["name"] == "val_has_usable_targets")["successful"]


def test_validate_entrypoint_delegates_to_validator():
    # The DIMER-facing `validate.py` entrypoint must expose the same `main` as
    # the `validator.py` implementation module (no behavioral fork).
    spec = importlib.util.spec_from_file_location("validate", Path(__file__).parents[1] / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate"] = mod
    spec.loader.exec_module(mod)
    assert mod.main is validator.main
