# tabicl-regressor-dataset-validator

DIMER dataset validator for TabICLv2 tabular regression.

Validates `train.csv` plus optional `val.csv` / `test.csv`, enforces a numeric regression target, deterministic split-file resolution, usable-row and feature-count limits, and archive size/path safety before fine-tuning.

Pairs with `tabicl-regressor-finetuner`. Full documentation lives in `tabicl-regressor-pipeline`.
