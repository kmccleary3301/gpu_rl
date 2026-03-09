from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import list_action_space, run_scripted_reference_episode
from gpu_cockpit.engine.knowledge import build_knowledge_index


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_environment"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")
        shutil.copytree(ROOT / "knowledge", self.tmp_root / "knowledge")
        build_knowledge_index(self.tmp_root)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_action_space_contains_expected_v0_actions(self) -> None:
        action_names = {row.action_name for row in list_action_space()}
        self.assertTrue({"run", "bench", "eval", "inspect", "compare", "replay", "build", "adapter_show", "knowledge_query"} <= action_names)

    def test_scripted_reference_episode_smoke_eval(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/smoke/eval/v1",
            ["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            section="eval",
        )
        self.assertEqual(episode.task_id, "task/smoke/eval/v1")
        self.assertGreaterEqual(len(episode.steps), 3)
        self.assertEqual(episode.episode_kind, "scripted_reference")
        self.assertEqual(episode.terminal_state, "success")
        self.assertIn("knowledge_query", [step.action.action_type for step in episode.steps])
        self.assertIn("eval", [step.action.action_type for step in episode.steps])
        self.assertTrue(episode.steps[-1].terminal)

    def test_scripted_reference_episode_triton_task(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_row_sum/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_candidate.py", "--benchmark-repeats", "3"],
            section="eval",
        )
        self.assertEqual(episode.terminal_state, "success")
        self.assertIn("eval", [step.action.action_type for step in episode.steps])

    def test_scripted_reference_episode_public_benchmark_task(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/kernelbench/level1/32_hardtanh/eval/v1",
            [
                "python3",
                "workloads/reference/kernelbench_reference_runner.py",
                "--case-config",
                "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_032_hardtanh.json",
                "--benchmark-repeats",
                "3",
            ],
            section="eval",
        )
        self.assertEqual(episode.terminal_state, "success")
        self.assertIn("knowledge_query", [step.action.action_type for step in episode.steps])

    def test_scripted_reference_episode_debug_workflow(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
            step_budget=6,
        )
        actions = [step.action.action_type for step in episode.steps]
        self.assertEqual(episode.task_verb, "debug")
        self.assertIn("build", actions)
        self.assertIn("eval", actions)
        self.assertIn("replay", actions)
        self.assertEqual(episode.metadata["workflow"], "debug")

    def test_scripted_reference_episode_reformulate_workflow_uses_baseline_compare(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/attention_reformulate/eval/v1",
            ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            step_budget=6,
        )
        actions = [step.action.action_type for step in episode.steps]
        self.assertEqual(episode.task_verb, "reformulate")
        self.assertIn("bench", actions)
        self.assertIn("compare", actions)
        self.assertEqual(episode.metadata["workflow"], "reformulate")
