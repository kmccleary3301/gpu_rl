from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.inspector import inspect_run, load_run_summary
from gpu_cockpit.engine.replay import validate_run_bundle


class GoldenRunTests(unittest.TestCase):
    def test_checked_in_golden_run_bundle_is_replayable(self) -> None:
        fixture = ROOT / "tests" / "golden_runs" / "smoke_bundle_v1"
        summary = inspect_run(ROOT, str(fixture))
        self.assertEqual(summary["run_id"], "golden_smoke_v1")

        run_summary = load_run_summary(ROOT, str(fixture))
        self.assertEqual(run_summary.status, "ok")

        replay_status = validate_run_bundle(ROOT, str(fixture))
        self.assertEqual(replay_status["status"], "ok")
        self.assertEqual(replay_status["missing_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
