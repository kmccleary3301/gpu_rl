from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import TrajectoryAction, TrajectoryEpisode, TrajectoryObservation, TrajectoryStep
from gpu_cockpit.engine.environment import _derive_reward_ledger, _scripted_patch_plan, initialize_environment_state, list_action_space, run_scripted_reference_episode, step_environment
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
        self.assertTrue(
            {
                "run",
                "bench",
                "eval",
                "inspect",
                "inspect_quality",
                "compare",
                "replay",
                "build",
                "patch_candidate",
                "branch_candidate",
                "revert_candidate",
                "promote_candidate",
                "adapter_show",
                "knowledge_query",
            }
            <= action_names
        )

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
            ["python3", "workloads/reference/triton_row_sum_optimize_candidate.py", "--benchmark-repeats", "3"],
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
        self.assertEqual(next_state.current_candidate_parent_id, None)
        self.assertIsNotNone(next_state.current_candidate_run_ref)
        self.assertEqual(next_state.current_candidate_status, "patched")
        self.assertEqual(next_state.candidate_history, [next_state.current_candidate_id])

    def test_branch_candidate_action_tracks_lineage_without_workspace_change(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/reduction_debug/eval/v1", step_budget=5)
        target_path = self.tmp_root / "workloads" / "reference" / "triton_row_sum_broken_kernel.py"
        original_text = target_path.read_text(encoding="utf-8")
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")

        state, _ = step_environment(
            self.tmp_root,
            state,
            action_name="patch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            patch_text=repaired_text,
            patch_intent="repair the row sum kernel",
            patch_transition_kind="repaired",
        )
        patched_candidate_id = state.current_candidate_id

        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="branch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_intent="branch for an alternate reduction-mask experiment",
            branch_label="alt_mask_branch",
        )

        self.assertEqual(target_path.read_text(encoding="utf-8"), repaired_text)
        self.assertEqual(step.action.action_type, "branch_candidate")
        self.assertEqual(step.transition_kind, "branched")
        self.assertEqual(step.input_candidate_id, patched_candidate_id)
        self.assertEqual(next_state.current_candidate_parent_id, patched_candidate_id)
        self.assertEqual(next_state.current_candidate_status, "draft")
        self.assertEqual(next_state.metadata["last_candidate_operation"], "branch")
        self.assertEqual(next_state.metadata["last_branch_label"], "alt_mask_branch")
        self.assertEqual(next_state.candidate_history, [patched_candidate_id, next_state.current_candidate_id])
        self.assertEqual(next_state.candidate_run_history, [state.current_candidate_run_ref, next_state.current_candidate_run_ref])
        self.assertEqual(next_state.candidate_lineage_events[-1]["action_name"], "branch_candidate")
        self.assertNotEqual(original_text, repaired_text)

    def test_revert_candidate_action_restores_prior_text_snapshot(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/reduction_debug/eval/v1", step_budget=5)
        target_path = self.tmp_root / "workloads" / "reference" / "triton_row_sum_broken_kernel.py"
        broken_text = target_path.read_text(encoding="utf-8")
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")

        state, _ = step_environment(
            self.tmp_root,
            state,
            action_name="patch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            patch_text=repaired_text,
            patch_intent="repair the row sum kernel",
            patch_transition_kind="repaired",
        )
        repaired_candidate_id = state.current_candidate_id

        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="revert_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_intent="revert to the broken baseline for comparison",
            revert_target_candidate_id=repaired_candidate_id,
        )

        self.assertEqual(target_path.read_text(encoding="utf-8"), broken_text)
        self.assertEqual(step.action.action_type, "revert_candidate")
        self.assertEqual(step.transition_kind, "reverted")
        self.assertEqual(step.input_candidate_id, repaired_candidate_id)
        self.assertEqual(next_state.current_candidate_parent_id, repaired_candidate_id)
        self.assertEqual(next_state.current_candidate_status, "reverted")
        self.assertEqual(next_state.metadata["last_candidate_operation"], "revert")
        self.assertEqual(next_state.metadata["last_revert_target_candidate_id"], repaired_candidate_id)
        self.assertEqual(next_state.candidate_lineage_events[-1]["action_name"], "revert_candidate")

    def test_reward_ledger_tracks_recovery_supersede_and_near_miss_shaping(self) -> None:
        patch_step = TrajectoryStep(
            step_index=0,
            action=TrajectoryAction(action_type="patch_candidate"),
            observation=TrajectoryObservation(observation_type="candidate_patch"),
            reward_components={},
            reward_total=0.0,
        )
        compare_step = TrajectoryStep(
            step_index=1,
            action=TrajectoryAction(action_type="compare"),
            observation=TrajectoryObservation(
                observation_type="comparison",
                projection={
                    "optimize_delta_summary": {
                        "correctness_change": "recovered",
                        "perf_change": "regressed",
                    }
                },
            ),
            reward_components={"tool_cost": -0.02},
            reward_total=-0.02,
        )
        revert_step = TrajectoryStep(
            step_index=2,
            action=TrajectoryAction(action_type="revert_candidate"),
            observation=TrajectoryObservation(observation_type="candidate_patch"),
            reward_components={},
            reward_total=0.0,
        )
        terminal_eval_step = TrajectoryStep(
            step_index=3,
            action=TrajectoryAction(action_type="eval"),
            observation=TrajectoryObservation(observation_type="quality_gate"),
            reward_components={},
            reward_total=0.0,
            terminal=True,
            terminal_state="three_attempt_positive_complete",
        )
        ledger = _derive_reward_ledger(
            task_ref="task/reduction_row_sum_branching/eval/v1",
            task_verb="optimize",
            terminal_state="three_attempt_positive_complete",
            steps=[patch_step, compare_step, revert_step, terminal_eval_step],
            final_eval_envelope={
                "correctness_gate": "fail",
                "determinism_gate": "pass",
                "perf_gate": "not_run",
            },
        )
        self.assertLess(ledger.total_shaping_components["candidate_regression_penalty"], 0.0)
        self.assertGreater(ledger.total_shaping_components["best_known_supersede_bonus"], 0.0)
        self.assertGreater(ledger.total_shaping_components["revert_recovery_bonus"], 0.0)
        self.assertGreater(ledger.total_shaping_components["near_miss_progress_bonus"], 0.0)
        self.assertIn("near_miss_progress_bonus_awarded", ledger.entries[-1].notes)

    def test_promote_candidate_action_marks_candidate_promoted(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/reduction_debug/eval/v1", step_budget=5)
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")

        state, _ = step_environment(
            self.tmp_root,
            state,
            action_name="patch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            patch_text=repaired_text,
            patch_intent="repair the row sum kernel",
            patch_transition_kind="repaired",
        )
        patched_candidate_id = state.current_candidate_id

        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="promote_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_intent="promote the repaired kernel as the preferred candidate",
            promote_label="preferred_fix_v1",
        )

        self.assertEqual(step.action.action_type, "promote_candidate")
        self.assertEqual(step.transition_kind, "promoted")
        self.assertEqual(step.input_candidate_id, patched_candidate_id)
        self.assertEqual(next_state.current_candidate_parent_id, patched_candidate_id)
        self.assertEqual(next_state.current_candidate_status, "promoted")
        self.assertEqual(next_state.metadata["last_candidate_operation"], "promote")
        self.assertEqual(next_state.metadata["last_promote_label"], "preferred_fix_v1")
        self.assertEqual(next_state.candidate_lineage_events[-1]["action_name"], "promote_candidate")

    def test_attention_score_optimize_patch_plan_targets_patchable_candidate(self) -> None:
        plan = _scripted_patch_plan(self.tmp_root, "task/attention_score/eval/v1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["target_file"], "workloads/reference/triton_attention_score_optimize_patchable_candidate.py")
        self.assertEqual(plan["patch_kind"], "perf_transform")
        self.assertEqual(
            plan["eval_command"],
            ["python3", "workloads/reference/triton_attention_score_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_reduction_row_sum_optimize_patch_plan_targets_patchable_candidate(self) -> None:
        plan = _scripted_patch_plan(self.tmp_root, "task/reduction_row_sum/eval/v1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["target_file"], "workloads/reference/triton_row_sum_optimize_patchable_candidate.py")
        self.assertEqual(plan["patch_kind"], "perf_transform")
        self.assertEqual(
            plan["eval_command"],
            ["python3", "workloads/reference/triton_row_sum_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kv_cache_gather_optimize_patch_plan_targets_patchable_candidate(self) -> None:
        plan = _scripted_patch_plan(self.tmp_root, "task/kv_cache_gather/eval/v1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["target_file"], "workloads/reference/triton_kv_cache_gather_optimize_patchable_candidate.py")
        self.assertEqual(plan["patch_kind"], "perf_transform")
        self.assertEqual(
            plan["eval_command"],
            ["python3", "workloads/reference/triton_kv_cache_gather_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kernelbench_sum_reduction_optimize_patch_plan_targets_patchable_candidate(self) -> None:
        plan = _scripted_patch_plan(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["target_file"], "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py")
        self.assertEqual(plan["patch_kind"], "perf_transform")
        self.assertIsNone(plan["post_patch_build_spec"])
        self.assertEqual(
            plan["eval_command"],
            ["python3", "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kernelbench_softmax_optimize_patch_plan_targets_patchable_candidate(self) -> None:
        plan = _scripted_patch_plan(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1")

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["target_file"], "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py")
        self.assertEqual(plan["patch_kind"], "perf_transform")
        self.assertIsNone(plan["post_patch_build_spec"])
        self.assertEqual(
            plan["eval_command"],
            ["python3", "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_initial_bench_sets_baseline_compare_anchor(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/smoke/eval/v1", step_budget=4)

        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="bench",
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            section="summary",
        )

        self.assertEqual(step.action.action_type, "bench")
        self.assertIsNotNone(next_state.comparison_anchor_run_ref)
        self.assertEqual(next_state.comparison_anchor_label, "baseline_bench")
