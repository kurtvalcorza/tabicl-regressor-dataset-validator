from __future__ import annotations

import json
from pathlib import Path

import pytest

import dimer_entrypoint
import dimer_transport
import validator


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **_kwargs):
        return self.pages


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.downloads = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator([
            {"Contents": [{"Key": key, "Size": len(body)} for key, body in self.objects.items()]}
        ])

    def download_file(self, bucket, key, destination):
        self.downloads.append((bucket, key, destination))
        Path(destination).write_bytes(self.objects[key])


def _configure_s3(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "datasets")
    monkeypatch.setenv("S3_DATASET_PREFIX", "workbench/sessions/s1/dataset/")


def test_local_dataset_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / "train.csv").write_text("x,target\n1,1.0\n", encoding="utf-8")
    _configure_s3(monkeypatch)
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: pytest.fail("S3 must not be used"))

    status = dimer_transport.stage_dataset_if_needed(tmp_path)

    assert status == {"attempted": False, "source": "local"}


def test_empty_directory_is_staged_from_s3(tmp_path, monkeypatch):
    _configure_s3(monkeypatch)
    prefix = "workbench/sessions/s1/dataset/"
    fake = FakeS3({
        prefix + "train.csv": b"x,target\n1,1.0\n2,2.0\n",
        prefix + "nested/val.csv": b"x,target\n3,3.0\n4,4.0\n",
    })
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    status = dimer_transport.stage_dataset_if_needed(tmp_path)

    assert status["attempted"] is True
    assert status["fileCount"] == 2
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "nested" / "val.csv").exists()


def test_unsafe_s3_key_is_rejected(tmp_path, monkeypatch):
    _configure_s3(monkeypatch)
    prefix = "workbench/sessions/s1/dataset/"
    fake = FakeS3({prefix + "../escape.csv": b"x,target\n1,1.0\n"})
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    with pytest.raises(ValueError, match="unsafe S3 dataset object key"):
        dimer_transport.stage_dataset_if_needed(tmp_path)
    assert not (tmp_path.parent / "escape.csv").exists()


def test_empty_s3_prefix_is_a_transport_failure(tmp_path, monkeypatch):
    _configure_s3(monkeypatch)
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: FakeS3({}))

    with pytest.raises(FileNotFoundError, match="contains no files"):
        dimer_transport.stage_dataset_if_needed(tmp_path)


def test_stage_size_limit_is_enforced_before_download(tmp_path, monkeypatch):
    _configure_s3(monkeypatch)
    monkeypatch.setenv("DIMER_MAX_S3_STAGE_BYTES", "4")
    prefix = "workbench/sessions/s1/dataset/"
    fake = FakeS3({prefix + "train.csv": b"12345"})
    monkeypatch.setattr(dimer_transport, "_s3_client", lambda: fake)

    with pytest.raises(ValueError, match="would exceed"):
        dimer_transport.stage_dataset_if_needed(tmp_path)
    assert fake.downloads == []


def test_entrypoint_staging_failure_persists_result_and_callbacks(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    calls = []
    monkeypatch.setattr(dimer_entrypoint, "stage_dataset_if_needed", lambda: (_ for _ in ()).throw(RuntimeError("stage failed")))
    monkeypatch.setattr(validator, "RESULT_PATH", result_path)
    monkeypatch.setattr(validator, "notify_done_callback", lambda: (calls.append(1), {"attempted": True})[1])

    assert dimer_entrypoint.main() == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["successful"] is False
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["metadata"]["classNames"] == []
    assert calls == [1]
