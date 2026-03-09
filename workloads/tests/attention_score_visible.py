from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.triton_attention_score_shared import VISIBLE_K, VISIBLE_Q, reference_attention_scores


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def _assert_scores_close(actual: object, expected: list[list[float]]) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise SystemExit(f"Unexpected visible attention scores: {actual}")
    actual_tensor = torch.tensor(actual, dtype=torch.float32)
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    if actual_tensor.shape != expected_tensor.shape or not torch.allclose(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4):
        raise SystemExit(f"Unexpected visible attention scores: {actual}")


def main() -> None:
    payload = _load_payload()
    expected = reference_attention_scores(VISIBLE_Q, VISIBLE_K, causal=True)
    _assert_scores_close(payload.get("visible_attention_scores"), expected)


if __name__ == "__main__":
    main()
