from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.computeeval_reference_runner import _load_problem, visible_summary


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def main() -> None:
    payload = _load_payload()
    problem_path = Path(str(payload["problem_path"]))
    expected = visible_summary(_load_problem(problem_path))
    actual = payload.get("visible_result")
    if actual != expected:
        raise SystemExit(f"Unexpected ComputeEval visible result: {actual} != {expected}")


if __name__ == "__main__":
    main()
