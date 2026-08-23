# DIMER validator contract

This repository owns the TabICLv2 **regression** dataset-validator worker. It does not own DIMER Workbench orchestration.

## Dataset transport

The preferred worker contract is a dataset already materialized under `DIMER_DATASET_DIR` (default `/data/dataset`). Existing local/PVC execution therefore takes precedence.

For the current DIMER on-prem validator job, `/data` may be an empty `emptyDir` while the dataset remains in MinIO/S3. When all of the following are present and the local dataset directory contains no files, the container entrypoint stages the prefix before calling the existing validator:

- `S3_BUCKET`
- `S3_DATASET_PREFIX`
- optional `S3_ENDPOINT_URL`
- standard AWS credential environment variables

`DIMER_MAX_S3_STAGE_BYTES` bounds the total object bytes staged (default 1 GiB). Object paths containing parent traversal or absolute paths are rejected.

The staging layer is transport-only. It does not change validation semantics in `validator.py`.

## Result and callback

The validator writes `DIMER_RESULT_PATH` and, when `S3_RESULT_KEY` is configured, best-effort uploads that JSON to `S3_BUCKET`. `DIMER_DONE_CALLBACK` is attempted exactly once on normal validation/crash paths. A staging failure also emits a structured failure result and attempts the callback.

## Preprocessing values: DIMER-side dependency

The validator already consumes `DIMER_PREPROCESSING_ARGS_JSON` for `target_column`, `drop_columns`, `max_train_rows`, and `validation_split`.

**Current DIMER on-prem dataset-validation jobs must be updated by the platform owner to inject the resolved preprocessing values. This repository does not modify DIMER.** Until that platform change lands, validation uses this worker's defaults when the variable is absent, while the later fine-tuning job may receive user-selected values.

That mismatch is documented as a platform contract dependency and must be closed before production sign-off for non-default preprocessing.

## Platform follow-ups (documentation only)

DIMER should eventually:

1. inject the resolved `DIMER_PREPROCESSING_ARGS_JSON` into validator jobs;
2. optionally own S3-to-local staging centrally (for example via an init container), in which case this worker detects the already-populated local dataset and skips its compatibility staging;
3. validate/version worker result envelopes rather than relying on ad hoc dictionary access.

No `dimer-backend` changes are part of this repository change.
