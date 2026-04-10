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


def _assert_close_list(actual: list[float], expected: list[float], *, atol: float = 1e-5) -> None:
    if len(actual) != len(expected):
        _emit_failure(
            "hidden_routing_shape_mismatch",
            f"Length mismatch: {actual} != {expected}",
            section="hidden_tests",
            expected_len=len(expected),
            actual_len=len(actual),
        )
    for index, (lhs, rhs) in enumerate(zip(actual, expected, strict=True)):
        if abs(float(lhs) - float(rhs)) > atol:
            _emit_failure(
                "hidden_routing_value_mismatch",
                f"Value mismatch at position {index}: {lhs} != {rhs}",
                section="hidden_tests",
                index=index,
                actual=float(lhs),
                expected=float(rhs),
            )


def main() -> None:
    payload = _load_payload()
    hidden = payload.get("hidden_routing")
    if not isinstance(hidden, dict):
        _emit_failure(
            "hidden_routing_payload_missing",
            f"Unexpected hidden routing payload: {hidden}",
            section="hidden_tests",
        )
    if hidden.get("indices") != [1, 0, 6, 0, 0]:
        _emit_failure(
            "hidden_routing_indices_mismatch",
            f"Unexpected hidden routing indices: {hidden.get('indices')}",
            section="hidden_tests",
            observed=hidden.get("indices"),
            expected=[1, 0, 6, 0, 0],
        )
    _assert_close_list(list(hidden.get("values", [])), [3.0, 4.2, -2.5, 0.501, 7.0])
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
            "replace_cpu_stable_argmax_path_with_triton_kernel_widest_candidate",
            "workloads/reference/triton_routing_argmax_hardest_optimize_candidate.py",
        ),
        (
            "supersede_triton_kernel_widest_candidate_with_ranked_variant",
            "workloads/reference/triton_routing_argmax_hardest_optimize_candidate_v2.py",
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
