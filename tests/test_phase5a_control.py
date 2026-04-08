from __future__ import annotations

import unittest

from gpu_cockpit.contracts.environment import AgentEnvironmentState
from gpu_cockpit.engine.environment import _refresh_candidate_tree_state


class Phase5AControlTests(unittest.TestCase):
    def test_root_after_baseline_bench_requires_patch_or_query(self) -> None:
        state = AgentEnvironmentState(
            episode_id="env_episode_test",
            policy_id="policy_test",
            task_id="task/routing_argmax_hard/eval/v1",
            step_budget_total=17,
            step_budget_remaining=16,
            steps_taken=1,
            last_run_ref="/tmp/baseline_bench",
            comparison_anchor_run_ref="/tmp/baseline_bench",
            comparison_anchor_label="baseline_bench",
        )

        refreshed = _refresh_candidate_tree_state(state)

        self.assertEqual(refreshed.current_legal_next_actions, ["patch_candidate", "knowledge_query"])
        self.assertNotIn("bench", refreshed.current_legal_next_actions)

    def test_patched_candidate_prefers_bench_over_compare(self) -> None:
        state = AgentEnvironmentState(
            episode_id="env_episode_test",
            policy_id="policy_test",
            task_id="task/routing_argmax_hard/eval/v1",
            step_budget_total=15,
            step_budget_remaining=13,
            steps_taken=2,
            current_candidate_id="cand_alpha",
            current_candidate_run_ref="/tmp/patch_run",
            current_candidate_status="patched",
            current_candidate_attempt_index=1,
            comparison_anchor_run_ref="/tmp/baseline_bench",
            comparison_anchor_label="baseline_bench",
            candidate_history=["cand_alpha"],
            candidate_run_history=["/tmp/patch_run"],
            candidate_lineage_events=[
                {
                    "candidate_id": "cand_alpha",
                    "candidate_role": "patched_candidate",
                    "parent_candidate_id": None,
                    "run_ref": "/tmp/patch_run",
                    "status": "patched",
                }
            ],
        )

        refreshed = _refresh_candidate_tree_state(state)

        self.assertEqual(refreshed.current_endgame_recommendation, "bench")
        self.assertIn("bench", refreshed.current_legal_next_actions)
        self.assertIn("build", refreshed.current_legal_next_actions)
        self.assertNotIn("compare", refreshed.current_legal_next_actions)
        self.assertNotIn("eval", refreshed.current_legal_next_actions)


if __name__ == "__main__":
    unittest.main()
