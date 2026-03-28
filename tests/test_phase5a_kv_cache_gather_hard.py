from __future__ import annotations

import unittest
from pathlib import Path

from gpu_cockpit.engine.optimize_patch_registry import get_optimize_patch_spec
from gpu_cockpit.engine.task_registry import TaskRegistry


ROOT = Path(__file__).resolve().parents[1]


class Phase5AKVCacheGatherHardTests(unittest.TestCase):
    def test_task_registry_loads_hard_kv_task(self) -> None:
        registry = TaskRegistry(ROOT)
        task = registry.get("task/kv_cache_gather_hard/eval/v1")

        self.assertEqual(task.operator_family, "kv_cache_gather")
        self.assertEqual(task.difficulty, "hard")
        self.assertEqual(
            task.reference_impl_ref,
            "workloads/reference/triton_kv_cache_gather_hard_optimize_patchable_candidate.py",
        )
        self.assertEqual(task.hidden_tests_ref, "workloads/tests/kv_cache_gather_hard_hidden.py")

    def test_optimize_patch_registry_exposes_three_attempt_hard_kv_plan(self) -> None:
        spec = get_optimize_patch_spec("task/kv_cache_gather_hard/eval/v1")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(
            spec["patch_target_file"],
            "workloads/reference/triton_kv_cache_gather_hard_optimize_patchable_candidate.py",
        )
        self.assertEqual(len(spec["positive_attempt_plans"]), 3)
        self.assertEqual(
            spec["positive_attempt_plans"][1]["patch_source_file"],
            "workloads/reference/triton_kv_cache_gather_hard_optimize_candidate.py",
        )
        self.assertEqual(
            spec["positive_attempt_plans"][2]["patch_source_file"],
            "workloads/reference/triton_kv_cache_gather_hard_optimize_candidate_v2.py",
        )


if __name__ == "__main__":
    unittest.main()
