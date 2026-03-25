from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.kernelbench_v3_reference_runner import _load_case_config, _run_case

import torch


def _emit_failure(code: str, message: str, **details: object) -> None:
    payload = {
        "code": code,
        "fix_family": "visible_eval_failure",
        "likely_next_actions": ["inspect_quality", "patch_candidate", "eval"],
        **details,
    }
    print(f"GPC_FAILURE_JSON:{json.dumps(payload, sort_keys=True)}", file=sys.stderr)
    raise SystemExit(message)


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def main() -> None:
    payload = _load_payload()
    case_config_path = Path(str(payload["case_config_path"]))
    case = _load_case_config(case_config_path)
    problem_path = Path(str(payload["problem_path"]))
    expected = _run_case(problem_path, case, "visible", torch.device("cpu"))
    actual = payload.get("visible_result")
    common = {
        "section": "visible_tests",
        "benchmark_source": payload.get("benchmark_source"),
        "benchmark_case_id": payload.get("benchmark_case_id"),
        "benchmark_case_version": payload.get("benchmark_case_version"),
        "provenance_kind": payload.get("provenance_kind"),
        "case_config_path": str(case_config_path),
        "problem_path": str(problem_path),
    }
    if not isinstance(actual, dict):
        _emit_failure("kernelbench_v3_visible_result_not_mapping", f"Visible KB-v3 result is not a mapping: {actual}", **common)
    if actual.get("shape") != expected.get("shape"):
        _emit_failure("kernelbench_v3_visible_shape_mismatch", "Unexpected visible KB-v3 shape", observed=actual.get("shape"), expected=expected.get("shape"), **common)
    if not math.isclose(float(actual.get("sum", 0.0)), float(expected.get("sum", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure("kernelbench_v3_visible_sum_mismatch", "Unexpected visible KB-v3 sum", observed=actual.get("sum"), expected=expected.get("sum"), **common)
    if not math.isclose(float(actual.get("mean", 0.0)), float(expected.get("mean", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure("kernelbench_v3_visible_mean_mismatch", "Unexpected visible KB-v3 mean", observed=actual.get("mean"), expected=expected.get("mean"), **common)
    if "sha256" not in actual:
        _emit_failure("kernelbench_v3_visible_missing_sha256", "Visible KB-v3 result is missing sha256", **common)


if __name__ == "__main__":
    main()
