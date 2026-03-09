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
    _assert_case(payload["visible_small"], [1, 3], [0.91, 0.72])
    visible_batch = payload["visible_batch"]
    if not isinstance(visible_batch, list) or len(visible_batch) != 2:
        raise SystemExit("visible_batch should contain two rows")
    _assert_case(visible_batch[0], [2, 4], [0.9, 0.7])
    _assert_case(visible_batch[1], [1, 3], [0.95, 0.85])


if __name__ == "__main__":
    main()
