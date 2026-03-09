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
from gpu_cockpit.engine.benchmark import run_subprocess_benchmark, run_task_benchmark
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.task_registry import TaskRegistry


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_benchmark"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_run_subprocess_benchmark(self) -> None:
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="bench_test_001",
            created_at=datetime.now(tz=UTC),
            task_ref="task/smoke/diagnose/v1",
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=3,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        perf = run_subprocess_benchmark(
            writer,
            command=["python3", "-c", "print('bench')"],
            warmups=1,
            repeats=3,
        )
        self.assertGreaterEqual(perf.steady_state_ms_p50, 0.0)
        self.assertTrue((self.tmp_root / "runs" / "bench_test_001" / "perf" / "benchmark.json").exists())

    def test_run_task_benchmark_with_baseline(self) -> None:
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="bench_test_baseline",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=3,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        perf = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
        )
        self.assertIsNotNone(perf.baseline_steady_state_ms_p50)
        self.assertGreaterEqual(perf.speedup_vs_baseline, 1.0)


if __name__ == "__main__":
    unittest.main()
