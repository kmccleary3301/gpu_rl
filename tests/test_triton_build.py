from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.backends.triton.build import compile_triton_build_spec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.contracts.run import RunSpec
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _cuda_ready() -> bool:
    try:
        import torch
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


@unittest.skipUnless(_cuda_ready(), "CUDA Triton toolchain is not available")
class TritonBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_triton_build"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.writer = RunBundleWriter(self.tmp_root)
        self.writer.initialize(
            RunSpec(
                run_id="triton_build_test",
                created_at=datetime(2026, 3, 9, tzinfo=UTC),
                task_ref="task/reduction_row_sum/eval/v1",
                mode="human",
                target_backend="triton",
                target_vendor="nvidia",
                executor="local_host",
                policy_pack="balanced",
                budgets=BudgetSpec(wall_seconds=60, compile_attempts=1, bench_runs=0, profile_runs=0, artifact_mb=128),
                seed_pack=SeedPack(global_seed=1, input_seed=2),
            )
        )

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_compile_triton_build_spec_emits_ir_and_triview(self) -> None:
        record = compile_triton_build_spec(
            self.writer,
            ROOT,
            "workloads/reference/triton_row_sum_kernel.py:get_build_spec",
        )
        self.assertEqual(record.compiler, "triton")
        self.assertTrue((self.writer.run_dir / "build" / "tri_view.json").exists())
        self.assertTrue((self.writer.run_dir / "build" / "ptx.txt").exists())
        self.assertTrue((self.writer.run_dir / "build" / "ttir.mlir").exists())
        self.assertTrue((self.writer.run_dir / "build" / "ttgir.mlir").exists())
        self.assertTrue((self.writer.run_dir / "build" / "llir.ll").exists())
        self.assertTrue((self.writer.run_dir / "build" / "source_map_summary.json").exists())
        tri_view = (self.writer.run_dir / "build" / "tri_view.json").read_text(encoding="utf-8")
        self.assertIn('"correlation_method": "ptx_loc_source_map_v1"', tri_view)


if __name__ == "__main__":
    unittest.main()
