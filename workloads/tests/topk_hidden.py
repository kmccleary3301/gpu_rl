from __future__ import annotations

import json
import os
from pathlib import Path


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def _assert_case(case: dict[str, object], expected_indices: list[int], expected_scores: list[float]) -> None:
    indices = case.get("indices")
    scores = case.get("scores")
    if indices != expected_indices:
        raise SystemExit(f"Unexpected indices: {indices} != {expected_indices}")
    if scores != expected_scores:
        raise SystemExit(f"Unexpected scores: {scores} != {expected_scores}")


def main() -> None:
    payload = _load_payload()
    _assert_case(payload["hidden_negative"], [1, 3, 2], [2.2, 2.2, 0.0])
    _assert_case(payload["hidden_ties"], [0, 1, 3], [0.5, 0.5, 0.5])


if __name__ == "__main__":
    main()
