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
            "hidden_kv_shape_mismatch",
            f"Unexpected hidden kv rows: {actual}",
            section="hidden_tests",
            expected_rows=len(expected),
            actual_rows=len(actual) if isinstance(actual, list) else None,
        )
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected, strict=True)):
        if not isinstance(actual_row, list) or len(actual_row) != len(expected_row):
            _emit_failure(
                "hidden_kv_row_shape_mismatch",
                f"Unexpected hidden row shape at row {row_index}: {actual_row}",
                section="hidden_tests",
                row_index=row_index,
                expected_cols=len(expected_row),
                actual_cols=len(actual_row) if isinstance(actual_row, list) else None,
            )
        for col_index, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row, strict=True)):
            if abs(float(actual_value) - expected_value) > 1e-6:
                _emit_failure(
                    "hidden_kv_value_mismatch",
                    f"Unexpected hidden kv row value at row {row_index}, col {col_index}: {actual_value} != {expected_value}",
                    section="hidden_tests",
                    row_index=row_index,
                    col_index=col_index,
                    actual=float(actual_value),
                    expected=expected_value,
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
    summary = payload.get("optimization_summary")
    if not isinstance(summary, dict):
        _emit_failure(
            "missing_optimization_summary",
            "Missing optimization_summary",
            section="hidden_tests",
            expected_any_of=["optimization_summary"],
        )
    strategy_change = summary.get("strategy_change")
    candidate_ref = summary.get("candidate_ref")
    if strategy_change != "replace_cpu_indexing_path_with_triton_paged_gather_kernel":
        _emit_failure(
            "unexpected_optimization_strategy_change",
            f"Unexpected optimization strategy change: {strategy_change}",
            section="hidden_tests",
            observed=strategy_change,
        )
    if candidate_ref != "workloads/reference/triton_kv_cache_gather_optimize_candidate.py":
        _emit_failure(
            "unexpected_optimize_candidate_ref",
            f"Unexpected optimize candidate ref: {candidate_ref}",
            section="hidden_tests",
            observed=candidate_ref,
        )


if __name__ == "__main__":
    main()
