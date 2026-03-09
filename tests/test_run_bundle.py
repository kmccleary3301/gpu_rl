from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RunSpec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.engine import RunBundleWriter


class RunBundleWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_run_bundle"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_initialize_and_write_artifact(self) -> None:
        writer = RunBundleWriter(root=self.tmp_root)
        run_spec = RunSpec(
            run_id="run_test_bundle",
            created_at=datetime.now(tz=UTC),
            task_ref="task/test/diagnose/v1",
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_docker",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=1,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        run_dir = writer.initialize(run_spec)
        artifact = writer.write_artifact(
            relative_path="notes/hello.txt",
            kind="text_note",
            content="hello\n",
            mime="text/plain",
        )

        self.assertEqual(run_dir.name, "run_test_bundle")
        self.assertTrue((run_dir / "manifest.json").exists())
        self.assertTrue((run_dir / "events.jsonl").exists())
        self.assertTrue((run_dir / artifact.path).exists())
        self.assertEqual(len(writer.list_artifacts()), 1)

    def test_artifact_budget_is_enforced(self) -> None:
        writer = RunBundleWriter(root=self.tmp_root)
        run_spec = RunSpec(
            run_id="run_budget_test",
            created_at=datetime.now(tz=UTC),
            task_ref="task/test/diagnose/v1",
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_docker",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=1,
                artifact_mb=1,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        with self.assertRaises(RuntimeError):
            writer.write_artifact(
                relative_path="notes/too_big.txt",
                kind="text_note",
                content="x" * (2 * 1024 * 1024),
                mime="text/plain",
            )


if __name__ == "__main__":
    unittest.main()
