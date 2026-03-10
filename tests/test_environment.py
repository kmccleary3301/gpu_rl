from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import TrajectoryEpisode
from gpu_cockpit.engine.environment import initialize_environment_state, list_action_space, run_scripted_reference_episode, step_environment
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
        self.assertTrue({"run", "bench", "eval", "inspect", "inspect_quality", "compare", "replay", "build", "patch_candidate", "adapter_show", "knowledge_query"} <= action_names)

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
        self.assertIn("patch_candidate", actions)
        self.assertEqual(episode.metadata["workflow"], "debug")
        self.assertEqual(episode.metadata["training_example_kind"], "positive_sft_example")

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
        self.assertIn("patch_candidate", actions)
        self.assertEqual(episode.metadata["workflow"], "reformulate")

    def test_scripted_reference_episode_debug_negative_workflow_retains_failed_repair_trace(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
            workflow="debug_negative",
            step_budget=12,
        )
        self.assertEqual(episode.task_verb, "debug")
        self.assertEqual(episode.terminal_state, "failure")
        self.assertEqual(episode.metadata["training_example_kind"], "negative_debug_example")
        self.assertEqual(episode.governance.episode_governance_kind, "usable_negative_debug")
        self.assertTrue(episode.metadata["negative_transition_trace"])
        self.assertGreaterEqual(episode.metadata["patch_attempt_count"], 1)

    def test_scripted_reference_episode_reformulate_negative_workflow_retains_failed_transform_trace(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/attention_reformulate/eval/v1",
            ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            workflow="reformulate_negative",
            step_budget=12,
        )
        self.assertEqual(episode.task_verb, "reformulate")
        self.assertEqual(episode.terminal_state, "success")
        self.assertEqual(episode.metadata["training_example_kind"], "negative_reformulate_example")
        self.assertEqual(episode.governance.episode_governance_kind, "usable_negative_transition")
        self.assertTrue(episode.metadata["negative_transition_trace"])
        self.assertGreaterEqual(episode.metadata["patch_attempt_count"], 1)

    def test_patch_candidate_action_applies_workspace_change_and_tracks_candidate_state(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/reduction_debug/eval/v1", step_budget=4)
        original_path = self.tmp_root / "workloads" / "reference" / "triton_row_sum_broken_kernel.py"
        repaired_path = self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py"
        original_text = original_path.read_text(encoding="utf-8")
        repaired_text = repaired_path.read_text(encoding="utf-8")

        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="patch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            patch_text=repaired_text,
            patch_intent="restore the row mask so the last column participates in the reduction",
            patch_expected_effect="make the broken kernel behavior match the repaired kernel",
            patch_transition_kind="repaired",
        )

        self.assertNotEqual(original_text, repaired_text)
        self.assertEqual(original_path.read_text(encoding="utf-8"), repaired_text)
        self.assertEqual(step.action.action_type, "patch_candidate")
        self.assertEqual(step.step_label, "patch_action")
        self.assertEqual(step.transition_kind, "repaired")
        self.assertIsNotNone(step.patch_hash)
        self.assertEqual(step.input_candidate_id, None)
        self.assertEqual(step.output_candidate_id, next_state.current_candidate_id)
        self.assertEqual(step.observation.observation_type, "candidate_patch")
        self.assertIn("eval", step.recommended_next_actions)
        self.assertIsNotNone(next_state.current_candidate_id)
        self.assertIsNotNone(next_state.current_candidate_run_ref)
