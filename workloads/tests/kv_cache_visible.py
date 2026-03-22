from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _emit_failure(code: str, message: str, **details: object) -> None:
    payload = {"code": code, **details}
    print(f"GPC_FAILURE_JSON:{json.dumps(payload, sort_keys=True)}", file=sys.stderr)
    raise SystemExit(message)


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def _assert_rows_close(actual: object, expected: list[list[float]]) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        _emit_failure(
            "visible_kv_shape_mismatch",
            f"Unexpected visible kv rows: {actual}",
            section="visible_tests",
            expected_rows=len(expected),
            actual_rows=len(actual) if isinstance(actual, list) else None,
        )
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected, strict=True)):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            _emit_failure(
                "visible_kv_row_shape_mismatch",
                f"Unexpected visible row shape at row {row_index}: {actual_row}",
                section="visible_tests",
                row_index=row_index,
                expected_cols=len(expected_row),
                actual_cols=len(actual_row) if isinstance(actual_row, list) else None,
            )
        for col_index, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row, strict=True)):
            if abs(float(actual_value) - expected_value) > 1e-6:
                _emit_failure(
                    "visible_kv_value_mismatch",
                    f"Unexpected visible kv row value at row {row_index}, col {col_index}: {actual_value} != {expected_value}",
                    section="visible_tests",
                    row_index=row_index,
                    col_index=col_index,
                    actual=float(actual_value),
                    expected=expected_value,
                )


def main() -> None:
    payload = _load_payload()
    expected = [
        [9.0, 10.0, 11.0, 12.0],
        [1.0, 2.0, 3.0, 4.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    _assert_rows_close(payload.get("visible_kv_rows"), expected)


if __name__ == "__main__":
    main()
