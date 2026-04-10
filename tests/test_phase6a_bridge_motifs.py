from __future__ import annotations

from gpu_cockpit.engine.optimize_patch_registry import get_optimize_patch_spec


def test_phase6a_locality_kv_patch_spec_present() -> None:
    spec = get_optimize_patch_spec("task/kv_cache_gather_locality_harder/eval/v1")
    assert spec is not None
    assert spec["patch_target_file"] == "workloads/reference/triton_kv_cache_gather_locality_harder_optimize_patchable_candidate.py"
    assert len(spec["positive_attempt_plans"]) == 3


def test_phase6a_tileaware_routing_patch_spec_present() -> None:
    spec = get_optimize_patch_spec("task/routing_dispatch_tileaware_harder/eval/v1")
    assert spec is not None
    assert spec["patch_target_file"] == "workloads/reference/triton_routing_dispatch_tileaware_harder_optimize_patchable_candidate.py"
    assert len(spec["positive_attempt_plans"]) == 3
