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
                    f"Unexpected hidden kv value at row {row_index}, col {col_index}: {actual_value} != {expected_value}",
                    section="hidden_tests",
                    row_index=row_index,
                    col_index=col_index,
                    actual=float(actual_value),
                    expected=expected_value,
                )


def main() -> None:
    payload = _load_payload()
    expected = [
        [12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75],
        [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75],
        [12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75],
        [14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75],
        [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75],
        [10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75],
    ]
    _assert_rows_close(payload.get("hidden_kv_rows"), expected)
    summary = payload.get("optimization_summary")
    if not isinstance(summary, dict):
        _emit_failure(
            "missing_optimization_summary",
            "Missing optimization_summary",
            section="hidden_tests",
            expected_any_of=["optimization_summary"],
            suspected_region="optimization_summary",
            likely_next_actions=["patch_candidate", "eval"],
            fix_family="missing_summary",
            confidence=0.95,
        )
    accepted = {
        (
            "replace_cpu_indexing_path_with_triton_paged_gather_harder_candidate",
            "workloads/reference/triton_kv_cache_gather_harder_optimize_candidate.py",
        ),
        (
            "supersede_triton_paged_gather_harder_candidate_with_ranked_variant",
            "workloads/reference/triton_kv_cache_gather_harder_optimize_candidate_v2.py",
        ),
    }
    observed = (summary.get("strategy_change"), summary.get("candidate_ref"))
    if observed not in accepted:
        _emit_failure(
            "unexpected_optimization_strategy_change",
            f"Unexpected optimization summary: {observed}",
            section="hidden_tests",
            observed_strategy_change=summary.get("strategy_change"),
            observed_candidate_ref=summary.get("candidate_ref"),
            likely_next_actions=["inspect_quality", "patch_candidate", "eval"],
            fix_family="unexpected_summary",
            confidence=0.9,
        )


if __name__ == "__main__":
    main()
