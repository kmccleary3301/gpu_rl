from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from gpu_cockpit.engine.environment import initialize_environment_state

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


if __name__ == "__main__":
    unittest.main()
