"""Container entrypoint that adapts DIMER transport before validation."""
from __future__ import annotations

import json
import traceback

from dimer_transport import stage_dataset_if_needed


def _staging_failure(exc: Exception) -> int:
    import validator

    payload = {
        "successful": False,
        "message": "TabICLv2 regression dataset staging failed.",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "metadata": {
            "template": validator.TEMPLATE_NAME,
            "taskType": validator._resolve_task_type("tabular_regression"),
            "classNames": [],
        },
    }
    try:
        validator.write_result(payload)
    except Exception as write_exc:  # noqa: BLE001
        validator.log(f"Failed to persist staging failure: {write_exc}")
    try:
        validator.log(
            f"Callback: {json.dumps(validator.notify_done_callback(), sort_keys=True)}"
        )
    except Exception as cb_exc:  # noqa: BLE001
        validator.log(f"Callback delivery (best-effort) failed: {cb_exc}")
    return 1


def main() -> int:
    try:
        status = stage_dataset_if_needed()
        if status.get("attempted"):
            print(f"[dimer-transport] staged dataset from S3: {json.dumps(status, sort_keys=True)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        return _staging_failure(exc)

    from validator import main as validator_main

    return validator_main()


if __name__ == "__main__":
    raise SystemExit(main())
