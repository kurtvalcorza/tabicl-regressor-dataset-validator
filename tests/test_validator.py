import importlib.util
import sys
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
