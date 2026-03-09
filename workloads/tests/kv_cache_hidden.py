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


def _assert_rows_close(actual: object, expected: list[list[float]]) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise SystemExit(f"Unexpected hidden kv rows: {actual}")
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected, strict=True)):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            raise SystemExit(f"Unexpected hidden row shape at row {row_index}: {actual_row}")
        for col_index, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row, strict=True)):
            if abs(float(actual_value) - expected_value) > 1e-6:
                raise SystemExit(
                    f"Unexpected hidden kv row value at row {row_index}, col {col_index}: {actual_value} != {expected_value}"
                )


def main() -> None:
    payload = _load_payload()
    expected = [
        [2.0, 2.5, 3.0, 3.5],
        [8.0, 8.5, 9.0, 9.5],
        [2.0, 2.5, 3.0, 3.5],
        [4.0, 4.5, 5.0, 5.5],
    ]
    _assert_rows_close(payload.get("hidden_kv_rows"), expected)


if __name__ == "__main__":
    main()
