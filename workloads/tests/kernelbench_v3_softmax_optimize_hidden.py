from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workloads.reference.kernelbench_v3_reference_runner import _load_case_config, _run_case

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
    common = {
        "section": "hidden_tests",
        "benchmark_source": payload.get("benchmark_source"),
        "benchmark_case_id": payload.get("benchmark_case_id"),
        "benchmark_case_version": payload.get("benchmark_case_version"),
        "provenance_kind": payload.get("provenance_kind"),
        "case_config_path": str(case_config_path),
        "problem_path": str(payload.get("problem_path")),
    }
    if not isinstance(actual, dict):
        _emit_failure("kernelbench_v3_hidden_result_not_mapping", "Hidden KB-v3 optimize result is not a mapping", **common)
    if actual.get("shape") != expected.get("shape"):
        _emit_failure("kernelbench_v3_hidden_shape_mismatch", "Unexpected hidden KB-v3 optimize shape", observed=actual.get("shape"), expected=expected.get("shape"), **common)
    if not math.isclose(float(actual.get("sum", 0.0)), float(expected.get("sum", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure("kernelbench_v3_hidden_sum_mismatch", "Unexpected hidden KB-v3 optimize sum", observed=actual.get("sum"), expected=expected.get("sum"), **common)
    if not math.isclose(float(actual.get("mean", 0.0)), float(expected.get("mean", 0.0)), rel_tol=1e-4, abs_tol=1e-5):
        _emit_failure("kernelbench_v3_hidden_mean_mismatch", "Unexpected hidden KB-v3 optimize mean", observed=actual.get("mean"), expected=expected.get("mean"), **common)
    if "sha256" not in actual:
        _emit_failure("kernelbench_v3_hidden_missing_sha256", "Hidden KB-v3 optimize result is missing sha256", **common)
    if payload.get("provenance_kind") != case.get("provenance_kind"):
        _emit_failure("kernelbench_v3_provenance_kind_mismatch", "Unexpected KB-v3 optimize provenance kind", observed=payload.get("provenance_kind"), expected=case.get("provenance_kind"), **common)
    summary = payload.get("optimization_summary")
    if not isinstance(summary, dict):
        _emit_failure("missing_optimization_summary", "Missing optimization_summary", expected_any_of=["optimization_summary"], **common)
    accepted_pairs = {
        "level1_023_online_softmax_streaming_harder.json": {
            (
                "promote_curated_kernelbench_v3_online_softmax_streaming_harder_candidate_wrapper",
                "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_candidate.py",
            ),
            (
                "supersede_curated_kernelbench_v3_online_softmax_streaming_harder_candidate_wrapper",
                "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_candidate_v2.py",
            ),
        },
    }
    observed_pair = (summary.get("strategy_change"), summary.get("candidate_ref"))
    if observed_pair not in accepted_pairs.get(case_config_path.name, set()):
        _emit_failure(
            "unexpected_optimization_strategy_change",
            "Unexpected KB-v3 optimize strategy change",
            observed=summary.get("strategy_change"),
            observed_candidate_ref=summary.get("candidate_ref"),
            **common,
        )
    if observed_pair[0] == "supersede_curated_kernelbench_v3_online_softmax_streaming_harder_candidate_wrapper":
        if summary.get("supersedes_candidate_ref") != "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_candidate.py":
            _emit_failure(
                "unexpected_supersedes_candidate_ref",
                "Unexpected KB-v3 optimize supersedes candidate ref",
                observed=summary.get("supersedes_candidate_ref"),
                expected="workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_candidate.py",
                **common,
            )


if __name__ == "__main__":
    main()
