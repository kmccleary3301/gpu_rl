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


def _assert_close_list(actual: list[float], expected: list[float], atol: float = 1e-5) -> None:
    if len(actual) != len(expected):
        raise SystemExit(f"Length mismatch: {actual} != {expected}")
    for lhs, rhs in zip(actual, expected, strict=True):
        if abs(float(lhs) - float(rhs)) > atol:
            raise SystemExit(f"Value mismatch: {actual} != {expected}")


def main() -> None:
    payload = _load_payload()
    visible = payload.get("visible_routing")
    if not isinstance(visible, dict):
        raise SystemExit(f"Unexpected visible routing payload: {visible}")
    if visible.get("indices") != [1, 0]:
        raise SystemExit(f"Unexpected visible routing indices: {visible.get('indices')}")
    _assert_close_list(list(visible.get("values", [])), [0.7, 1.0])


if __name__ == "__main__":
    main()
