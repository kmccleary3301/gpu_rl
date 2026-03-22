from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.triton_attention_score_shared import HIDDEN_K, HIDDEN_Q, reference_attention_scores


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


def _assert_scores_close(actual: object, expected: list[list[float]]) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        _emit_failure(
            "hidden_attention_score_shape_mismatch",
            f"Unexpected hidden attention scores: {actual}",
            section="hidden_tests",
            expected_rows=len(expected),
            actual_rows=len(actual) if isinstance(actual, list) else None,
        )
    actual_tensor = torch.tensor(actual, dtype=torch.float32)
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    if actual_tensor.shape != expected_tensor.shape or not torch.allclose(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4):
        diff = (actual_tensor - expected_tensor).abs()
        flat_index = int(diff.argmax().item())
        row_index, col_index = divmod(flat_index, int(diff.shape[1]))
        _emit_failure(
            "hidden_attention_score_mismatch",
            f"Unexpected hidden attention scores: {actual}",
            section="hidden_tests",
            row_index=row_index,
            col_index=col_index,
            actual=float(actual_tensor[row_index, col_index].item()),
            expected=float(expected_tensor[row_index, col_index].item()),
            max_abs_diff=float(diff.max().item()),
        )


def main() -> None:
    payload = _load_payload()
    expected = reference_attention_scores(HIDDEN_Q, HIDDEN_K, causal=True)
    _assert_scores_close(payload.get("hidden_attention_scores"), expected)
    optimization_summary = payload.get("optimization_summary")
    if isinstance(optimization_summary, dict):
        strategy_change = optimization_summary.get("strategy_change")
        candidate_ref = optimization_summary.get("candidate_ref")
        accepted = {
            (
                "replace_cpu_reference_path_with_tiled_triton_kernel_candidate",
                "workloads/reference/triton_attention_score_optimize_candidate.py",
            ),
            (
                "supersede_tiled_triton_kernel_candidate_with_ranked_variant",
                "workloads/reference/triton_attention_score_optimize_candidate_v2.py",
            ),
        }
        if (strategy_change, candidate_ref) not in accepted:
            _emit_failure(
                "unexpected_optimization_strategy_change",
                f"Unexpected optimization strategy change: {strategy_change}",
                section="hidden_tests",
                observed=strategy_change,
                observed_candidate_ref=candidate_ref,
            )
        return
    reformulation_summary = payload.get("reformulation_summary")
    if not isinstance(reformulation_summary, dict):
        _emit_failure(
            "missing_optimization_summary",
            "Missing reformulation_summary or optimization_summary",
            section="hidden_tests",
            expected_any_of=["optimization_summary", "reformulation_summary"],
        )
    strategy_change = reformulation_summary.get("strategy_change")
    optimized_ref = reformulation_summary.get("optimized_ref")
    if strategy_change != "replace_naive_python_loops_with_tiled_triton_kernel":
        _emit_failure(
            "unexpected_reformulation_strategy_change",
            f"Unexpected strategy change: {strategy_change}",
            section="hidden_tests",
            observed=strategy_change,
        )
    if optimized_ref != "workloads/reference/triton_attention_score_reformulate_candidate.py":
        _emit_failure(
            "unexpected_reformulate_candidate_ref",
            f"Unexpected optimized ref: {optimized_ref}",
            section="hidden_tests",
            observed=optimized_ref,
        )


if __name__ == "__main__":
    main()
