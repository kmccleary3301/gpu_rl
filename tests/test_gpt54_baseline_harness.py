from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_gpt54_first_wave_baseline as harness

from gpu_cockpit.engine.environment import initialize_environment_state, list_action_space


class GPT54BaselineHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_gpt54_harness"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_attention_score_task_context_enables_bounded_patch_iteration(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")

        self.assertEqual(
            ctx["default_command"],
            ["python3", "workloads/reference/triton_attention_score_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )
        self.assertIsInstance(ctx["patch"], dict)
        self.assertEqual(ctx["patch"]["patch_target_file"], "workloads/reference/triton_attention_score_optimize_patchable_candidate.py")
        self.assertEqual(ctx["patch"]["patch_kind"], "perf_transform")
        self.assertEqual(ctx["patch"]["post_patch_build_spec"], "workloads/reference/triton_attention_score_kernel.py:get_build_spec")

    def test_attention_score_allowed_actions_include_patch_and_build(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        action_specs = [spec.model_dump(mode="json") for spec in list_action_space()]

        allowed = harness._allowed_actions(action_specs, ctx)

        self.assertIn("patch_candidate", allowed)
        self.assertIn("branch_candidate", allowed)
        self.assertIn("revert_candidate", allowed)
        self.assertIn("promote_candidate", allowed)
        self.assertIn("build", allowed)
        self.assertIn("bench", allowed)
        self.assertIn("compare", allowed)

    def test_attention_score_bench_switches_from_baseline_to_candidate_after_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6)

        before_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            before_patch["command"],
            ["python3", "workloads/reference/triton_attention_score_baseline.py", "--benchmark-repeats", "20"],
        )

        state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        after_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            after_patch["command"],
            ["python3", "workloads/reference/triton_attention_score_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_state_snapshot_surfaces_candidate_lineage_summary(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6).model_copy(
            update={
                "current_candidate_id": "cand_child",
                "current_candidate_parent_id": "cand_parent",
                "current_candidate_run_ref": "runs/branch_child",
                "current_candidate_status": "draft",
                "best_known_candidate_id": "cand_parent",
                "best_known_candidate_parent_id": None,
                "best_known_candidate_run_ref": "runs/patch_parent",
                "candidate_history": ["cand_parent", "cand_child"],
                "candidate_run_history": ["runs/patch_parent", "runs/branch_child"],
                "candidate_lineage_events": [
                    {"action_name": "patch_candidate", "candidate_id": "cand_parent", "candidate_role": "patched_candidate", "summary": "repair the parent candidate"},
                    {
                        "action_name": "branch_candidate",
                        "candidate_id": "cand_sibling",
                        "parent_candidate_id": "cand_parent",
                        "candidate_role": "branched_candidate",
                        "summary": "branch a sibling candidate",
                    },
                    {
                        "action_name": "branch_candidate",
                        "candidate_id": "cand_child",
                        "parent_candidate_id": "cand_parent",
                        "candidate_role": "branched_candidate",
                        "summary": "branch for a focused alternate optimization",
                    },
                ],
            }
        )

        snapshot = harness._state_snapshot(state)

        self.assertEqual(snapshot["current_candidate_parent_id"], "cand_parent")
        self.assertEqual(snapshot["current_candidate_status"], "draft")
        self.assertEqual(snapshot["candidate_history"], ["cand_parent", "cand_child"])
        self.assertEqual(snapshot["candidate_lineage"]["history_length"], 2)
        self.assertEqual(snapshot["candidate_lineage"]["current_parent_candidate_id"], "cand_parent")
        self.assertEqual(snapshot["candidate_lineage"]["candidate_role"], "branched_candidate")
        self.assertEqual(snapshot["candidate_lineage"]["parent_candidate_ref"], "cand_parent")
        self.assertEqual(snapshot["candidate_lineage"]["sibling_candidate_refs"], ["cand_sibling"])
        self.assertEqual(snapshot["candidate_lineage"]["why_this_candidate_exists"], "branch for a focused alternate optimization")
        self.assertEqual(snapshot["candidate_lineage"]["best_known_candidate_id"], "cand_parent")

    def test_branch_revert_promote_kwargs_require_existing_candidate(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6).model_copy(
            update={"current_candidate_id": "cand_current"}
        )

        branch_kwargs = harness._resolve_action_kwargs("branch_candidate", ctx, state, None)
        self.assertEqual(branch_kwargs["task_ref"], "task/attention_score/eval/v1")
        self.assertIn("branch_label", branch_kwargs)

        revert_kwargs = harness._resolve_action_kwargs("revert_candidate", ctx, state, None)
        self.assertEqual(revert_kwargs["revert_target_candidate_id"], "cand_current")

        promote_kwargs = harness._resolve_action_kwargs("promote_candidate", ctx, state, None)
        self.assertEqual(promote_kwargs["promote_label"], "preferred_candidate")

    def test_two_attempt_positive_patch_resolution_advances_attempt_index(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "two_attempt_positive_v1"
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=10)

        first_patch = harness._resolve_action_kwargs("patch_candidate", ctx, state, None)
        self.assertEqual(first_patch["candidate_attempt_index"], 1)
        self.assertEqual(first_patch["patch_kind"], "no_op")

        state = state.model_copy(
            update={
                "current_candidate_id": "cand_a",
                "current_candidate_attempt_index": 1,
            }
        )
        second_patch = harness._resolve_action_kwargs("patch_candidate", ctx, state, None)
        self.assertEqual(second_patch["candidate_attempt_index"], 2)
        self.assertEqual(second_patch["patch_kind"], "perf_transform")

    def test_two_attempt_positive_requires_branch_then_second_patch_then_second_compare(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "two_attempt_positive_v1"
        budgets = {
            "step_budget": 12,
            "max_retries": 1,
            "max_patches": 2,
            "max_branches": 1,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 2,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 1,
            "branches": 0,
            "reverts": 0,
            "promotes": 0,
            "compares": 1,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 2,
        }
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=12).model_copy(
            update={
                "last_run_ref": "runs/bench_candidate_a",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate_a", "runs/bench_candidate_a"],
                "current_candidate_id": "candidate_a",
                "current_candidate_attempt_index": 1,
                "comparison_anchor_run_ref": "runs/bench_baseline",
                "metadata": {"last_eval_solved": False},
            }
        )

        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["branch_candidate", "patch_candidate"])
        self.assertFalse(harness._action_allowed_in_state("patch_candidate", state, ctx, counters, budgets))

        counters["branches"] = 1
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["patch_candidate", "bench"])
        self.assertTrue(harness._action_allowed_in_state("patch_candidate", state, ctx, counters, budgets))

        counters["patches"] = 2
        counters["bench_actions"] = 3
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["compare", "promote_candidate"])
        self.assertFalse(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))

        counters["compares"] = 2
        self.assertTrue(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))

    def test_three_attempt_positive_requires_second_branch_and_third_compare(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        budgets = {
            "step_budget": 14,
            "max_retries": 1,
            "max_patches": 3,
            "max_branches": 2,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 3,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 2,
            "branches": 1,
            "reverts": 0,
            "promotes": 0,
            "compares": 2,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 3,
        }
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=14).model_copy(
            update={
                "last_run_ref": "runs/bench_candidate_b",
                "run_history": [
                    "runs/bench_baseline",
                    "runs/patch_candidate_a",
                    "runs/bench_candidate_a",
                    "runs/patch_candidate_b",
                    "runs/bench_candidate_b",
                ],
                "current_candidate_id": "candidate_b",
                "current_candidate_attempt_index": 2,
                "comparison_anchor_run_ref": "runs/bench_candidate_a",
                "metadata": {"last_eval_solved": False},
            }
        )

        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["branch_candidate", "patch_candidate"])
        self.assertFalse(harness._action_allowed_in_state("patch_candidate", state, ctx, counters, budgets))

        counters["branches"] = 2
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["patch_candidate", "bench"])
        self.assertTrue(harness._action_allowed_in_state("patch_candidate", state, ctx, counters, budgets))

        counters["patches"] = 3
        counters["bench_actions"] = 4
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["compare", "promote_candidate"])
        self.assertFalse(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))

        counters["compares"] = 3
        self.assertTrue(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))

    def test_three_attempt_positive_draft_branch_requires_patch_before_bench_or_compare(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        budgets = {
            "step_budget": 15,
            "max_retries": 1,
            "max_patches": 3,
            "max_branches": 2,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 3,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 2,
            "branches": 2,
            "reverts": 0,
            "promotes": 0,
            "compares": 2,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 4,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=15).model_copy(
            update={
                "last_run_ref": "runs/branch_candidate_c",
                "current_candidate_id": "candidate_c_draft",
                "current_candidate_parent_id": "candidate_b",
                "current_candidate_status": "draft",
                "current_candidate_attempt_index": 3,
                "comparison_anchor_run_ref": "runs/bench_baseline",
            }
        )

        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["patch_candidate", "inspect_quality"])
        self.assertTrue(harness._action_allowed_in_state("patch_candidate", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("bench", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("compare", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))

    def test_three_attempt_positive_late_stage_blocks_knowledge_query(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        budgets = {
            "step_budget": 15,
            "max_retries": 1,
            "max_patches": 3,
            "max_branches": 2,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 3,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 2,
            "branches": 1,
            "reverts": 0,
            "promotes": 0,
            "compares": 2,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 3,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=15).model_copy(
            update={
                "last_run_ref": "runs/compare_candidate_b",
                "current_candidate_id": "candidate_b",
                "current_candidate_status": "patched",
                "current_candidate_attempt_index": 2,
                "comparison_anchor_run_ref": "runs/bench_candidate_a",
            }
        )

        self.assertFalse(harness._action_allowed_in_state("knowledge_query", state, ctx, counters, budgets))
        self.assertTrue(harness._action_allowed_in_state("branch_candidate", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("inspect_quality", state, ctx, counters, budgets))

    def test_multi_candidate_negative_optimize_hints_require_branch_then_revert_before_eval(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "negative")
        ctx["multi_candidate_mode"] = "branch_revert_negative_v1"
        budgets = {
            "step_budget": 10,
            "max_retries": 1,
            "max_patches": 1,
            "max_branches": 1,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 1,
            "branches": 0,
            "reverts": 0,
            "promotes": 0,
            "compares": 1,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 2,
        }
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=10).model_copy(
            update={
                "last_run_ref": "runs/bench_candidate",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate", "runs/bench_candidate"],
                "current_candidate_id": "candidate_optimize_01",
                "comparison_anchor_run_ref": "runs/bench_baseline",
                "metadata": {"last_eval_solved": False},
            }
        )

        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["branch_candidate", "revert_candidate"])
        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

        counters["branches"] = 1
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["revert_candidate", "eval"])
        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

        counters["reverts"] = 1
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["eval"])
        self.assertTrue(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

    def test_multi_candidate_negative_stop_after_revert_eval(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "negative")
        ctx["multi_candidate_mode"] = "branch_revert_negative_v1"
        counters = {
            "patches": 1,
            "branches": 1,
            "reverts": 1,
            "promotes": 0,
            "compares": 1,
            "eval_actions": 1,
            "replays": 0,
        }
        should_stop, terminal_reason = harness._should_stop(
            ctx,
            {
                "action_name": "eval",
                "observation": {"status": "failed"},
            },
            counters,
            {
                "max_patches": 1,
                "max_compares": 1,
                "max_replays": 1,
            },
        )
        self.assertTrue(should_stop)
        self.assertEqual(terminal_reason, "multi_candidate_negative_complete")

    def test_multi_candidate_positive_requires_branch_before_promote_and_eval(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "branch_promote_positive_v1"
        budgets = {
            "step_budget": 10,
            "max_retries": 1,
            "max_patches": 1,
            "max_branches": 1,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 1,
            "branches": 0,
            "reverts": 0,
            "promotes": 0,
            "compares": 1,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 2,
        }
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=10).model_copy(
            update={
                "last_run_ref": "runs/bench_candidate",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate", "runs/bench_candidate"],
                "current_candidate_id": "candidate_optimize_01",
                "comparison_anchor_run_ref": "runs/bench_baseline",
                "metadata": {"last_eval_solved": False},
            }
        )

        self.assertFalse(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

        counters["branches"] = 1
        self.assertTrue(harness._action_allowed_in_state("promote_candidate", state, ctx, counters, budgets))
        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

        counters["promotes"] = 1
        self.assertFalse(harness._action_allowed_in_state("branch_candidate", state, ctx, counters, budgets))
        self.assertTrue(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

    def test_attention_score_optimize_fallback_prefers_patch_then_compare_then_eval(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        budgets = {
            "step_budget": 8,
            "max_retries": 1,
            "max_patches": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 1,
            "patches": 0,
            "compares": 0,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 1,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6).model_copy(
            update={"last_run_ref": "runs/baseline_bench", "run_history": ["runs/baseline_bench"]}
        )

        self.assertEqual(harness._fallback_action(ctx, state, dict(counters), budgets), "patch_candidate")

        patched_state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        patched_counters = dict(counters)
        patched_counters["patches"] = 1
        self.assertEqual(harness._fallback_action(ctx, patched_state, dict(patched_counters), budgets), "bench")

        compared_counters = dict(patched_counters)
        compared_counters["bench_actions"] = 2
        self.assertEqual(harness._fallback_action(ctx, patched_state, dict(compared_counters), budgets), "compare")

        post_compare_counters = dict(compared_counters)
        post_compare_counters["compares"] = 1
        self.assertEqual(harness._fallback_action(ctx, patched_state, dict(post_compare_counters), budgets), "eval")

    def test_three_attempt_positive_fallback_requires_bench_then_compare_before_first_branch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        budgets = {
            "step_budget": 15,
            "max_retries": 1,
            "max_patches": 3,
            "max_branches": 2,
            "max_reverts": 1,
            "max_promotes": 1,
            "max_compares": 3,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=15).model_copy(
            update={
                "last_run_ref": "runs/patch_candidate_a",
                "current_candidate_id": "candidate_a",
                "current_candidate_status": "patched",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate_a"],
            }
        )
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 1,
            "patches": 1,
            "branches": 0,
            "reverts": 0,
            "promotes": 0,
            "compares": 0,
            "replays": 0,
            "eval_actions": 0,
            "bench_actions": 1,
        }
        self.assertEqual(harness._fallback_action(ctx, state, dict(counters), budgets), "bench")

        counters["bench_actions"] = 2
        state = state.model_copy(update={"last_run_ref": "runs/bench_candidate_a", "run_history": ["runs/bench_baseline", "runs/patch_candidate_a", "runs/bench_candidate_a"]})
        self.assertEqual(harness._fallback_action(ctx, state, dict(counters), budgets), "compare")

        counters["compares"] = 1
        self.assertEqual(harness._fallback_action(ctx, state, dict(counters), budgets), "branch_candidate")

    def test_attention_score_eval_is_rejected_until_compare_after_candidate_bench(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        budgets = {
            "step_budget": 8,
            "max_retries": 1,
            "max_patches": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 1,
            "compares": 0,
            "replays": 0,
            "eval_actions": 1,
            "bench_actions": 1,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6).model_copy(
            update={
                "last_run_ref": "runs/build_candidate",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate", "runs/build_candidate"],
                "current_candidate_id": "candidate_optimize_01",
                "comparison_anchor_run_ref": "runs/bench_baseline",
            }
        )

        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))
        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["bench", "eval"])

        counters["bench_actions"] = 2
        state = state.model_copy(
            update={
                "last_run_ref": "runs/bench_candidate",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate", "runs/build_candidate", "runs/bench_candidate"],
            }
        )

        self.assertFalse(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))
        self.assertTrue(harness._action_allowed_in_state("compare", state, ctx, counters, budgets))

        hints = harness._controller_hints(ctx, harness._state_snapshot(state), counters, budgets)
        self.assertEqual(hints["priority_actions"], ["compare", "eval"])

    def test_attention_score_after_compare_allows_only_eval_closeout(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        budgets = {
            "step_budget": 8,
            "max_retries": 1,
            "max_patches": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 0,
            "patches": 1,
            "compares": 1,
            "replays": 0,
            "eval_actions": 1,
            "bench_actions": 2,
        }
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=6).model_copy(
            update={
                "last_run_ref": "runs/bench_candidate",
                "run_history": ["runs/bench_baseline", "runs/patch_candidate", "runs/bench_candidate"],
                "current_candidate_id": "candidate_optimize_01",
                "comparison_anchor_run_ref": "runs/bench_baseline",
                "metadata": {"last_eval_solved": False},
            }
        )

        self.assertTrue(harness._action_allowed_in_state("eval", state, ctx, counters, budgets))

    def test_observation_packet_v3_current_strips_compare_digest_and_candidate_lineage(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        ctx["interface_profile"] = "v3_current"
        state = initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=8).model_copy(
            update={
                "current_candidate_id": "cand_b",
                "current_candidate_parent_id": "cand_a",
                "current_candidate_status": "patched",
                "candidate_lineage_events": [
                    {"action_name": "patch_candidate", "candidate_id": "cand_a", "candidate_role": "patched_candidate", "summary": "candidate a"},
                    {"action_name": "branch_candidate", "candidate_id": "cand_b", "parent_candidate_id": "cand_a", "candidate_role": "branched_candidate", "summary": "candidate b"},
                ],
            }
        )
        packet = harness._observation_packet(
            task_ctx=ctx,
            allowed_actions=["bench", "compare", "eval"],
            state_snapshot=harness._state_snapshot(state),
            budgets={"step_budget": 8, "max_retries": 1, "max_patches": 1, "max_compares": 1, "max_replays": 1, "max_knowledge_queries": 1},
            counters={"model_calls": 0, "provider_failures": 0, "failed_tool_calls": 0, "controller_rejections": 0, "knowledge_queries": 0, "patches": 1, "branches": 0, "reverts": 0, "promotes": 0, "compares": 1, "replays": 0, "eval_actions": 0, "bench_actions": 2},
            step_records=[
                {
                    "step_index": 3,
                    "step_label": "compare_action",
                    "action_name": "compare",
                    "reward_total": -0.01,
                    "reward_components": {"tool_cost": -0.01},
                    "recommended_next_actions": ["inspect_quality", "patch_candidate", "bench"],
                    "transition_kind": None,
                    "observation": {
                        "type": "comparison",
                        "status": None,
                        "run_id": "runs/bench_candidate",
                        "task_id": "task/attention_score/eval/v1",
                        "summary_ref": None,
                        "salient_artifact_refs": [],
                        "projection_excerpt": {
                            "lhs_run_id": "lhs",
                            "rhs_run_id": "rhs",
                            "lhs_status": "ok",
                            "rhs_status": "ok",
                            "optimize_delta_summary": {"correctness_change": "preserved_fail"},
                            "candidate_delta_brief": {"rhs_candidate_id": "cand_b"},
                        },
                    },
                }
            ],
        )

        self.assertNotIn("candidate_lineage", packet["state"])
        self.assertEqual(packet["last_step"]["recommended_next_actions"], [])
        self.assertNotIn("optimize_delta_summary", packet["last_step"]["observation"]["projection_excerpt"])

    def test_observation_packet_compare_localization_profile_preserves_failure_signal_without_lineage(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/attention_score/eval/v1", "positive")
        ctx["interface_profile"] = "compare_plus_localization_v1"
        packet = harness._observation_packet(
            task_ctx=ctx,
            allowed_actions=["inspect_quality", "patch_candidate", "eval"],
            state_snapshot=harness._state_snapshot(initialize_environment_state(self.tmp_root, "task/attention_score/eval/v1", step_budget=8)),
            budgets={"step_budget": 8, "max_retries": 1, "max_patches": 1, "max_compares": 1, "max_replays": 1, "max_knowledge_queries": 1},
            counters={"model_calls": 0, "provider_failures": 0, "failed_tool_calls": 0, "controller_rejections": 0, "knowledge_queries": 0, "patches": 0, "branches": 0, "reverts": 0, "promotes": 0, "compares": 0, "replays": 0, "eval_actions": 1, "bench_actions": 1},
            step_records=[
                {
                    "step_index": 1,
                    "step_label": "verification_action",
                    "action_name": "eval",
                    "reward_total": -0.05,
                    "reward_components": {"tool_cost": -0.05},
                    "recommended_next_actions": ["inspect_quality", "patch_candidate", "eval"],
                    "transition_kind": None,
                    "observation": {
                        "type": "eval_projection",
                        "status": "failed",
                        "run_id": "runs/eval_candidate",
                        "task_id": "task/attention_score/eval/v1",
                        "summary_ref": "summary.json",
                        "salient_artifact_refs": [],
                        "projection_excerpt": {
                            "failure_localization": {"hidden_tests": {"code": "hidden_attention_score_mismatch"}},
                            "gate_summary": {"correctness_gate": "fail"},
                        },
                    },
                }
            ],
        )

        self.assertNotIn("candidate_lineage", packet["state"])
        self.assertIn("failure_localization", packet["last_step"]["observation"]["projection_excerpt"])
        self.assertEqual(packet["last_step"]["recommended_next_actions"], ["inspect_quality", "patch_candidate", "eval"])

    def test_reduction_row_sum_task_context_enables_bounded_patch_iteration(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")

        self.assertEqual(
            ctx["default_command"],
            ["python3", "workloads/reference/triton_row_sum_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )
        self.assertIsInstance(ctx["patch"], dict)
        self.assertEqual(ctx["patch"]["patch_target_file"], "workloads/reference/triton_row_sum_optimize_patchable_candidate.py")
        self.assertEqual(ctx["patch"]["patch_kind"], "perf_transform")
        self.assertEqual(ctx["patch"]["post_patch_build_spec"], "workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec")

    def test_reduction_row_sum_allowed_actions_include_patch_and_build(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        action_specs = [spec.model_dump(mode="json") for spec in list_action_space()]

        allowed = harness._allowed_actions(action_specs, ctx)

        self.assertIn("patch_candidate", allowed)
        self.assertIn("build", allowed)
        self.assertIn("bench", allowed)
        self.assertIn("compare", allowed)

    def test_reduction_row_sum_bench_switches_from_baseline_to_candidate_after_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/reduction_row_sum/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/reduction_row_sum/eval/v1", step_budget=6)

        before_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            before_patch["command"],
            ["python3", "workloads/reference/triton_row_sum_baseline.py", "--benchmark-repeats", "50"],
        )

        state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        after_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            after_patch["command"],
            ["python3", "workloads/reference/triton_row_sum_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kv_cache_gather_task_context_enables_bounded_patch_iteration(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kv_cache_gather/eval/v1", "positive")

        self.assertEqual(
            ctx["default_command"],
            ["python3", "workloads/reference/triton_kv_cache_gather_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )
        self.assertIsInstance(ctx["patch"], dict)
        self.assertEqual(ctx["patch"]["patch_target_file"], "workloads/reference/triton_kv_cache_gather_optimize_patchable_candidate.py")
        self.assertEqual(ctx["patch"]["patch_kind"], "perf_transform")
        self.assertEqual(ctx["patch"]["post_patch_build_spec"], "workloads/reference/triton_kv_cache_gather_kernel.py:get_build_spec")

    def test_kv_cache_gather_bench_switches_from_baseline_to_candidate_after_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kv_cache_gather/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/kv_cache_gather/eval/v1", step_budget=6)

        before_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            before_patch["command"],
            ["python3", "workloads/reference/triton_kv_cache_gather_baseline.py", "--benchmark-repeats", "50"],
        )

        state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        after_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            after_patch["command"],
            ["python3", "workloads/reference/triton_kv_cache_gather_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kernelbench_sum_reduction_task_context_enables_bounded_patch_iteration(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1", "positive")

        self.assertEqual(
            ctx["default_command"],
            ["python3", "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )
        self.assertIsInstance(ctx["patch"], dict)
        self.assertEqual(ctx["patch"]["patch_target_file"], "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py")
        self.assertEqual(ctx["patch"]["patch_kind"], "perf_transform")
        self.assertIsNone(ctx["patch"]["post_patch_build_spec"])

    def test_kernelbench_sum_reduction_allowed_actions_exclude_build_without_build_spec(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1", "positive")
        action_specs = [spec.model_dump(mode="json") for spec in list_action_space()]

        allowed = harness._allowed_actions(action_specs, ctx)

        self.assertIn("patch_candidate", allowed)
        self.assertIn("bench", allowed)
        self.assertIn("compare", allowed)
        self.assertNotIn("build", allowed)

    def test_kernelbench_sum_reduction_bench_switches_from_baseline_to_candidate_after_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1", step_budget=6)

        before_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            before_patch["command"],
            [
                "python3",
                "workloads/reference/kernelbench_reference_runner.py",
                "--case-config",
                "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_047_sum_reduction.json",
                "--benchmark-repeats",
                "50",
            ],
        )

        state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        after_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            after_patch["command"],
            ["python3", "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kernelbench_sum_reduction_negative_context_uses_no_op_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/47_sum_reduction/eval/v1", "negative")

        self.assertEqual(ctx["patch"]["patch_kind"], "no_op")
        self.assertEqual(
            ctx["patch"]["patch_text"],
            (self.tmp_root / "workloads/reference/kernelbench_sum_reduction_optimize_patchable_candidate.py").read_text(encoding="utf-8"),
        )

    def test_kernelbench_softmax_task_context_enables_bounded_patch_iteration(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", "positive")

        self.assertEqual(
            ctx["default_command"],
            ["python3", "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )
        self.assertIsInstance(ctx["patch"], dict)
        self.assertEqual(ctx["patch"]["patch_target_file"], "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py")
        self.assertEqual(ctx["patch"]["patch_kind"], "perf_transform")
        self.assertIsNone(ctx["patch"]["post_patch_build_spec"])

    def test_kernelbench_softmax_allowed_actions_exclude_build_without_build_spec(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", "positive")
        action_specs = [spec.model_dump(mode="json") for spec in list_action_space()]

        allowed = harness._allowed_actions(action_specs, ctx)

        self.assertIn("patch_candidate", allowed)
        self.assertIn("bench", allowed)
        self.assertIn("compare", allowed)
        self.assertNotIn("build", allowed)

    def test_kernelbench_softmax_bench_switches_from_baseline_to_candidate_after_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", "positive")
        state = initialize_environment_state(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", step_budget=6)

        before_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            before_patch["command"],
            [
                "python3",
                "workloads/reference/kernelbench_reference_runner.py",
                "--case-config",
                "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_023_softmax.json",
                "--benchmark-repeats",
                "50",
            ],
        )

        state = state.model_copy(update={"current_candidate_id": "candidate_optimize_01"})
        after_patch = harness._resolve_action_kwargs("bench", ctx, state, None)
        self.assertEqual(
            after_patch["command"],
            ["python3", "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py", "--benchmark-repeats", "2"],
        )

    def test_kernelbench_softmax_negative_context_uses_no_op_patch(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", "negative")

        self.assertEqual(ctx["patch"]["patch_kind"], "no_op")
        self.assertEqual(
            ctx["patch"]["patch_text"],
            (self.tmp_root / "workloads/reference/kernelbench_softmax_optimize_patchable_candidate.py").read_text(encoding="utf-8"),
        )

    def test_negative_optimize_trace_stops_after_replay_once_failure_is_established(self) -> None:
        ctx = harness._task_context(self.tmp_root, "task/kernelbench/level1/23_softmax/eval/v1", "negative")
        budgets = {
            "step_budget": 8,
            "max_retries": 1,
            "max_patches": 1,
            "max_compares": 1,
            "max_replays": 1,
            "max_knowledge_queries": 1,
        }
        counters = {
            "model_calls": 0,
            "provider_failures": 0,
            "failed_tool_calls": 0,
            "controller_rejections": 0,
            "knowledge_queries": 1,
            "patches": 1,
            "compares": 1,
            "replays": 1,
            "eval_actions": 1,
            "bench_actions": 2,
        }
        should_stop, terminal_reason = harness._should_stop(
            ctx,
            {
                "action_name": "replay",
                "observation": {"status": "failed"},
            },
            counters,
            budgets,
        )

        self.assertTrue(should_stop)
        self.assertEqual(terminal_reason, "negative_trace_complete")
