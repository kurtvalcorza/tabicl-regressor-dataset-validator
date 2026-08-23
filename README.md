# tabicl-regressor-dataset-validator

DIMER dataset validator for the TabICLv2 regressor pipeline. It checks that an uploaded
tabular-regression dataset zip meets the CSV contract before fine-tuning runs.

- Runs as a CPU Kubernetes Job.
- DIMER builds the root `Dockerfile` into an image and runs `validate.py` (a thin entrypoint that
  delegates to the tested `validator.py`).
- Pairs with `tabicl-regressor-finetuner`.

The complete pipeline documentation, dataset specification, and the fine-tuner are in the
[tabicl-regressor-pipeline](https://github.com/kurtvalcorza/tabicl-regressor-pipeline) project.

## Contract summary

The dataset zip must contain a `train.csv` with a **finite numeric** `target` column; every other
non-dropped column is a feature (numeric or categorical). Missing, non-numeric, `NaN`, and
infinite target values are excluded from the usable-row count and removed before training. At least
**50 usable rows** and **2 distinct finite target values** are required. Optional `val.csv` /
`test.csv` must share the schema and still carry enough usable numeric targets to score after
cleaning.

The validator reports pass/fail per check in `result.json` and rejects duplicate split candidates,
nested zips, path-traversal members, and oversized / zip-bomb archives (≤1 GiB uncompressed,
≤512 MiB per CSV, compression-ratio guard). A predominantly non-numeric target triggers
wrong-pipeline guidance toward the classifier. All limits are overridable by platform environment
variables. See the project's dataset specification for the full rules.
