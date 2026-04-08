from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from gpu_cockpit.engine.environment import _refresh_candidate_tree_state, initialize_environment_state

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_local_optimize_checkpoint_eval as local_eval


class LocalOptimizeCheckpointEvalTests(unittest.TestCase):
    def test_partial_payload_contains_expected_fields(self) -> None:
        root = Path("/home/kmccleary/projects/gpu_code_agents/gpu_rl")
        state = initialize_environment_state(root, "task/attention_score/eval/v1", policy_id="policy_test", step_budget=17)
        payload = local_eval._partial_payload(
            phase="initialized",
            task_ctx={"task_ref": "task/attention_score/eval/v1", "variant": "positive", "verb": "optimize"},
            state=state,
            model_label="model_test",
            terminal_reason="budget_exhausted",
            counters={"model_calls": 0},
            step_records=[],
            model_turns=[],
            extra={"turn_index": 0},
        )
        self.assertEqual(payload["phase"], "initialized")
        self.assertEqual(payload["task_ref"], "task/attention_score/eval/v1")
        self.assertEqual(payload["variant"], "positive")
        self.assertEqual(payload["model"], "model_test")
        self.assertEqual(payload["step_count"], 0)
        self.assertIn("state", payload)
        self.assertEqual(payload["turn_index"], 0)

    def test_write_json_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "episode.partial.json"
            local_eval._write_json(out_path, {"phase": "completed", "success": True})
            self.assertTrue(out_path.exists())
            self.assertIn('"phase": "completed"', out_path.read_text(encoding="utf-8"))

    def test_current_allowed_actions_filters_to_state_legal_actions(self) -> None:
        root = Path("/home/kmccleary/projects/gpu_code_agents/gpu_rl")
        state = initialize_environment_state(root, "task/attention_score/eval/v1", policy_id="policy_test", step_budget=17)
        task_ctx = local_eval.harness._task_context(root, "task/attention_score/eval/v1", "positive")
        task_ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        allowed = local_eval._current_allowed_actions(
            base_allowed_actions=["bench", "patch_candidate", "compare", "eval", "knowledge_query"],
            state=state,
            task_ctx=task_ctx,
            counters={
                "model_calls": 0,
                "provider_failures": 0,
                "failed_tool_calls": 0,
                "controller_rejections": 0,
                "knowledge_queries": 0,
                "patches": 0,
                "branches": 0,
                "reverts": 0,
                "promotes": 0,
                "compares": 0,
                "replays": 0,
                "eval_actions": 0,
                "bench_actions": 0,
            },
            budgets={
                "step_budget": 17,
                "max_retries": 1,
                "max_patches": 3,
                "max_branches": 2,
                "max_reverts": 1,
                "max_promotes": 1,
                "max_compares": 3,
                "max_replays": 1,
                "max_knowledge_queries": 1,
            },
        )
        self.assertEqual(allowed, ["bench", "patch_candidate", "knowledge_query"])

    def test_current_allowed_actions_drops_root_bench_after_baseline(self) -> None:
        root = Path("/home/kmccleary/projects/gpu_code_agents/gpu_rl")
        state = initialize_environment_state(root, "task/routing_argmax_hard/eval/v1", policy_id="policy_test", step_budget=17)
        task_ctx = local_eval.harness._task_context(root, "task/routing_argmax_hard/eval/v1", "positive")
        task_ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        state = state.model_copy(
            update={
                "last_run_ref": "/tmp/baseline_bench",
                "comparison_anchor_run_ref": "/tmp/baseline_bench",
                "comparison_anchor_label": "baseline_bench",
            }
        )
        state = _refresh_candidate_tree_state(state)
        allowed = local_eval._current_allowed_actions(
            base_allowed_actions=["bench", "patch_candidate", "compare", "eval", "knowledge_query"],
            state=state,
            task_ctx=task_ctx,
            counters={
                "model_calls": 1,
                "provider_failures": 0,
                "failed_tool_calls": 0,
                "controller_rejections": 0,
                "knowledge_queries": 0,
                "patches": 0,
                "branches": 0,
                "reverts": 0,
                "promotes": 0,
                "compares": 0,
                "replays": 0,
                "eval_actions": 0,
                "bench_actions": 1,
            },
            budgets={
                "step_budget": 17,
                "max_retries": 1,
                "max_patches": 3,
                "max_branches": 2,
                "max_reverts": 1,
                "max_promotes": 1,
                "max_compares": 3,
                "max_replays": 1,
                "max_knowledge_queries": 1,
            },
        )
        self.assertEqual(allowed, ["patch_candidate", "knowledge_query"])

    def test_three_attempt_hints_prefer_branch_after_first_compare(self) -> None:
        root = Path("/home/kmccleary/projects/gpu_code_agents/gpu_rl")
        task_ctx = local_eval.harness._task_context(root, "task/attention_score/eval/v1", "positive")
        task_ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        state_snapshot = {
            "last_run_ref": "/tmp/bench_run",
            "current_candidate_id": "cand_alpha",
            "current_candidate_status": "benchmarked",
            "current_endgame_recommendation": "compare",
            "current_legal_next_actions": ["bench", "branch_candidate", "revert_candidate", "compare"],
            "candidate_lineage": {
                "endgame_recommendation": "compare",
                "legal_next_actions": ["bench", "branch_candidate", "revert_candidate", "compare"],
            },
            "metadata": {},
        }
        hints = local_eval.harness._controller_hints(
            task_ctx,
            state_snapshot,
            {
                "model_calls": 4,
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
            },
            {
                "step_budget": 17,
                "max_retries": 1,
                "max_patches": 3,
                "max_branches": 2,
                "max_reverts": 1,
                "max_promotes": 1,
                "max_compares": 3,
                "max_replays": 1,
                "max_knowledge_queries": 1,
            },
        )
        self.assertEqual(hints["priority_actions"][:2], ["branch_candidate", "patch_candidate"])

    def test_three_attempt_disallows_bench_after_first_compare_before_branch(self) -> None:
        root = Path("/home/kmccleary/projects/gpu_code_agents/gpu_rl")
        task_ctx = local_eval.harness._task_context(root, "task/attention_score/eval/v1", "positive")
        task_ctx["multi_candidate_mode"] = "three_attempt_positive_v1"
        state = initialize_environment_state(root, "task/attention_score/eval/v1", policy_id="policy_test", step_budget=17)
        state = state.model_copy(
            update={
                "current_candidate_id": "cand_alpha",
                "current_candidate_status": "benchmarked",
                "current_candidate_run_ref": "/tmp/candidate_run",
                "candidate_lineage_events": [
                    {
                        "candidate_id": "cand_alpha",
                        "candidate_role": "patched_candidate",
                        "parent_candidate_id": None,
                        "run_ref": "/tmp/candidate_run",
                        "status": "benchmarked",
                    }
                ],
            }
        )
        state = _refresh_candidate_tree_state(state)
        allowed = local_eval._current_allowed_actions(
            base_allowed_actions=["bench", "branch_candidate", "patch_candidate", "compare", "revert_candidate"],
            state=state,
            task_ctx=task_ctx,
            counters={
                "model_calls": 4,
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
            },
            budgets={
                "step_budget": 17,
                "max_retries": 1,
                "max_patches": 3,
                "max_branches": 2,
                "max_reverts": 1,
                "max_promotes": 1,
                "max_compares": 3,
                "max_replays": 1,
                "max_knowledge_queries": 1,
            },
        )
        self.assertNotIn("bench", allowed)
        self.assertIn("branch_candidate", allowed)


if __name__ == "__main__":
    unittest.main()
