from __future__ import annotations

import json
from pathlib import Path

from gpu_cockpit.engine.optimize_patch_registry import get_optimize_patch_spec


ROOT = Path(__file__).resolve().parents[1]


def test_phase6a_online_softmax_streaming_harder_patch_spec() -> None:
    spec = get_optimize_patch_spec("task/online_softmax_streaming_harder/eval/v1")
    assert spec is not None
    assert spec["patch_target_file"] == "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_patchable_candidate.py"
    assert spec["positive_patch_source_file"] == "workloads/reference/kernelbench_v3_online_softmax_streaming_harder_optimize_candidate.py"
    assert spec["public_benchmark"]["benchmark_case_id"] == "kernelbench-v3/level1/23_online_softmax_streaming_harder"
    assert len(spec["positive_attempt_plans"]) == 3


def test_phase6a_online_softmax_streaming_harder_case_is_harder_than_softmax_wide() -> None:
    bridge_case = json.loads(
        (
            ROOT
            / "workloads"
            / "public_benchmarks"
            / "kernelbench_v3"
            / "v3_1"
            / "cases"
            / "level1_023_online_softmax_streaming_harder.json"
        ).read_text(encoding="utf-8")
    )
    prior_case = json.loads(
        (
            ROOT
            / "workloads"
            / "public_benchmarks"
            / "kernelbench_v3"
            / "v3_1"
            / "cases"
            / "level1_023_softmax_wide.json"
        ).read_text(encoding="utf-8")
    )
    assert bridge_case["benchmark"]["module_overrides"]["dim"] > prior_case["benchmark"]["module_overrides"]["dim"]
    assert bridge_case["hidden"]["module_overrides"]["dim"] > prior_case["hidden"]["module_overrides"]["dim"]
