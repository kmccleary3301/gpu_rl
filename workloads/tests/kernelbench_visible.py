from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.kernelbench_reference_runner import _load_case_config, _run_case

import torch


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
    expected = _run_case(Path(str(payload["problem_path"])), case, "visible", torch.device("cpu"))
    actual = payload.get("visible_result")
    if not isinstance(actual, dict):
        raise SystemExit(f"Visible KernelBench result is not a mapping: {actual}")
    if actual.get("shape") != expected.get("shape"):
        raise SystemExit(f"Unexpected visible KernelBench shape: {actual.get('shape')} != {expected.get('shape')}")
    if not math.isclose(float(actual.get("sum", 0.0)), float(expected.get("sum", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        raise SystemExit(f"Unexpected visible KernelBench sum: {actual.get('sum')} != {expected.get('sum')}")
    if not math.isclose(float(actual.get("mean", 0.0)), float(expected.get("mean", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        raise SystemExit(f"Unexpected visible KernelBench mean: {actual.get('mean')} != {expected.get('mean')}")
    if "sha256" not in actual:
        raise SystemExit("Visible KernelBench result is missing sha256")


if __name__ == "__main__":
    main()
