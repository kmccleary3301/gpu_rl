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


def main() -> None:
    payload = _load_payload()
    expected = [8.0, 3.0, 6.0]
    if payload.get("hidden_row_sum") != expected:
        _emit_failure(
            "hidden_row_sum_mismatch",
            f"Unexpected hidden row sums: {payload.get('hidden_row_sum')} != {expected}",
            section="hidden_tests",
            expected=expected,
            observed=payload.get("hidden_row_sum"),
        )
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
    accepted_pairs = {
        (
            "replace_cpu_reference_path_with_triton_row_sum_kernel",
            "workloads/reference/triton_row_sum_optimize_candidate.py",
        ),
        (
            "supersede_first_row_sum_candidate_with_second_triton_wrapper_variant",
            "workloads/reference/triton_row_sum_optimize_candidate_v2.py",
        ),
    }
    if (strategy_change, candidate_ref) not in accepted_pairs:
        _emit_failure(
            "unexpected_branching_optimization_summary",
            f"Unexpected optimization summary pair: {(strategy_change, candidate_ref)}",
            section="hidden_tests",
            observed={
                "strategy_change": strategy_change,
                "candidate_ref": candidate_ref,
            },
            expected_any_of=[
                {
                    "strategy_change": expected_strategy,
                    "candidate_ref": expected_candidate,
                }
                for expected_strategy, expected_candidate in sorted(accepted_pairs)
            ],
        )
    if candidate_ref == "workloads/reference/triton_row_sum_optimize_candidate_v2.py":
        supersedes = summary.get("supersedes_candidate_ref")
        if supersedes != "workloads/reference/triton_row_sum_optimize_candidate.py":
            _emit_failure(
                "missing_superseded_candidate_ref",
                f"Unexpected superseded candidate ref: {supersedes}",
                section="hidden_tests",
                observed=supersedes,
            )


if __name__ == "__main__":
    main()
