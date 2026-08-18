import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location("validator", Path(__file__).parents[1] / "validator.py")
validator = importlib.util.module_from_spec(SPEC)
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
