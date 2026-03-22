from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.kernelbench_reference_runner import _load_case_config, _run_case

import torch


def _failure_defaults(code: str) -> dict[str, object]:
    if code == "missing_optimization_summary":
        return {
            "fix_family": "missing_optimization_summary",
            "likely_next_actions": ["patch_candidate", "bench", "compare", "eval"],
        }
    if code in {"unexpected_optimization_strategy_change", "unexpected_optimize_candidate_ref"}:
        return {
            "fix_family": "optimization_summary_mismatch",
            "likely_next_actions": ["inspect_quality", "patch_candidate", "compare", "eval"],
        }
    if "shape_mismatch" in code:
        return {
            "fix_family": "shape_mismatch",
            "likely_next_actions": ["inspect_quality", "patch_candidate", "eval"],
        }
    if "sum_mismatch" in code or "mean_mismatch" in code:
        return {
            "fix_family": "numerical_mismatch",
            "likely_next_actions": ["inspect_quality", "patch_candidate", "eval"],
        }
    return {
        "fix_family": "public_optimize_failure",
        "likely_next_actions": ["inspect_quality", "patch_candidate", "eval"],
    }


def _emit_failure(code: str, message: str, **details: object) -> None:
    payload = {"code": code, **_failure_defaults(code), **details}
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
    case_config_path = Path(str(payload["case_config_path"]))
    case = _load_case_config(case_config_path)
    expected = _run_case(Path(str(payload["problem_path"])), case, "hidden", torch.device("cpu"))
    actual = payload.get("hidden_result")
    if not isinstance(actual, dict):
        _emit_failure("kernelbench_hidden_result_not_mapping", f"Hidden KernelBench result is not a mapping: {actual}", section="hidden_tests")
    if actual.get("shape") != expected.get("shape"):
        _emit_failure(
            "kernelbench_hidden_shape_mismatch",
            f"Unexpected hidden KernelBench shape: {actual.get('shape')} != {expected.get('shape')}",
            section="hidden_tests",
            observed=actual.get("shape"),
            expected=expected.get("shape"),
        )
    if not math.isclose(float(actual.get("sum", 0.0)), float(expected.get("sum", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure(
            "kernelbench_hidden_sum_mismatch",
            f"Unexpected hidden KernelBench sum: {actual.get('sum')} != {expected.get('sum')}",
            section="hidden_tests",
            observed=actual.get("sum"),
            expected=expected.get("sum"),
        )
    if not math.isclose(float(actual.get("mean", 0.0)), float(expected.get("mean", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure(
            "kernelbench_hidden_mean_mismatch",
            f"Unexpected hidden KernelBench mean: {actual.get('mean')} != {expected.get('mean')}",
            section="hidden_tests",
            observed=actual.get("mean"),
            expected=expected.get("mean"),
        )
    if "sha256" not in actual:
        _emit_failure("kernelbench_hidden_missing_sha256", "Hidden KernelBench result is missing sha256", section="hidden_tests")
    summary = payload.get("optimization_summary")
    if not isinstance(summary, dict):
        _emit_failure("missing_optimization_summary", "Missing optimization_summary", section="hidden_tests", expected_any_of=["optimization_summary"])
    case_name = case_config_path.name
    accepted_pairs = {
        "level1_023_softmax.json": {
            (
                "promote_curated_kernelbench_softmax_candidate_wrapper",
                "workloads/reference/kernelbench_softmax_optimize_candidate.py",
            ),
        },
        "level1_023_softmax_wide.json": {
            (
                "promote_curated_kernelbench_softmax_wide_candidate_wrapper",
                "workloads/reference/kernelbench_softmax_wide_optimize_candidate.py",
            ),
            (
                "supersede_curated_kernelbench_softmax_wide_candidate_wrapper",
                "workloads/reference/kernelbench_softmax_wide_optimize_candidate_v2.py",
            ),
        },
    }
    observed_pair = (summary.get("strategy_change"), summary.get("candidate_ref"))
    if observed_pair not in accepted_pairs.get(case_name, set()):
        _emit_failure(
            "unexpected_optimization_strategy_change",
            f"Unexpected optimization strategy change: {summary.get('strategy_change')}",
            section="hidden_tests",
            observed=summary.get("strategy_change"),
            observed_candidate_ref=summary.get("candidate_ref"),
            case_config_path=str(case_config_path),
        )
    if case_name == "level1_023_softmax_wide.json" and observed_pair[0] == "supersede_curated_kernelbench_softmax_wide_candidate_wrapper":
        if summary.get("supersedes_candidate_ref") != "workloads/reference/kernelbench_softmax_wide_optimize_candidate.py":
            _emit_failure(
                "unexpected_supersedes_candidate_ref",
                f"Unexpected supersedes candidate ref: {summary.get('supersedes_candidate_ref')}",
                section="hidden_tests",
                observed=summary.get("supersedes_candidate_ref"),
                case_config_path=str(case_config_path),
            )
    elif case_name == "level1_023_softmax.json" and summary.get("candidate_ref") != "workloads/reference/kernelbench_softmax_optimize_candidate.py":
        _emit_failure(
            "unexpected_optimize_candidate_ref",
            f"Unexpected optimize candidate ref: {summary.get('candidate_ref')}",
            section="hidden_tests",
            observed=summary.get("candidate_ref"),
        )


if __name__ == "__main__":
    main()
