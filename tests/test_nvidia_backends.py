from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.backends.nvidia.disassembly import emit_disassembly_nvidia
from gpu_cockpit.backends.nvidia.ncu import profile_kernel_nvidia
from gpu_cockpit.backends.nvidia.sanitizer import sanitize_nvidia
from gpu_cockpit.contracts import RunSpec, TaskSpec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.contracts.task import NumericalPolicy, PerfProtocol
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.executors.base import CommandExecutor, CommandResult


class FakeNvidiaToolExecutor(CommandExecutor):
    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del env, cwd, timeout
        if command[:1] == ["ncu"] and "--export" in command:
            prefix = Path(command[command.index("--export") + 1])
            prefix.parent.mkdir(parents=True, exist_ok=True)
            prefix.with_suffix(".ncu-rep").write_text("fake report\n", encoding="utf-8")
            return CommandResult(command=command, exit_code=0, stdout="ncu profile ok\n", stderr="", duration_ms=30)
        if command[:1] == ["ncu"] and "--import" in command:
            csv_stdout = "\n".join(
                [
                    "ID,Kernel Name,Metric Name,Metric Unit,Metric Value",
                    "1,my_kernel,sm__warps_active.avg.pct_of_peak_sustained_active,%,52.0",
                    "1,my_kernel,launch__registers_per_thread,register/thread,48",
                    "1,my_kernel,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,71.5",
                    "1,my_kernel,sm__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
                    "1,my_kernel,lts__t_sector_hit_rate.pct,%,88.0",
                    "1,my_kernel,gpu__time_duration.sum,ns,3000000",
                ]
            )
            return CommandResult(command=command, exit_code=0, stdout=csv_stdout + "\n", stderr="", duration_ms=20)
        if command[:1] == ["compute-sanitizer"]:
            log_path = Path(command[command.index("--log-file") + 1])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_text = "\n".join(
                [
                    "========= Invalid __global__ read of size 4 at my_kernel() in /tmp/demo.cu:42",
                    "========= Error summary: 1 error",
                ]
            )
            log_path.write_text(log_text + "\n", encoding="utf-8")
            return CommandResult(command=command, exit_code=1, stdout="", stderr=log_text + "\n", duration_ms=25)
        if command[:2] == ["cuobjdump", "--dump-ptx"]:
            return CommandResult(command=command, exit_code=0, stdout=".visible .entry demo_kernel() {\n  ret;\n}\n", stderr="", duration_ms=10)
        if command[:1] == ["nvdisasm"]:
            return CommandResult(command=command, exit_code=0, stdout="/*0000*/ MOV R1, c[0x0][0x20] ;\n/*0010*/ EXIT ;\n", stderr="", duration_ms=10)
        raise AssertionError(f"Unexpected command: {command}")


def _build_writer(root: Path) -> RunBundleWriter:
    task = TaskSpec(
        task_id="task/test/nvidia/v1",
        prompt="profile me",
        verb="optimize",
        operator_family="test",
        difficulty="unit",
        allowed_backends=["triton"],
        numerical_policy=NumericalPolicy(dtype_matrix=["fp32"], rtol=1e-5, atol=1e-6),
        perf_protocol=PerfProtocol(warmups=1, repeats=1, timer="wall_clock"),
        required_artifacts=[],
    )
    run_spec = RunSpec(
        run_id="run_test_nvidia_backends",
        created_at=datetime(2026, 3, 9, tzinfo=UTC),
        task_ref=task.task_id,
        mode="human",
        target_backend="triton",
        target_vendor="nvidia",
        executor="local_host",
        policy_pack="balanced",
        budgets=BudgetSpec(wall_seconds=10, compile_attempts=1, bench_runs=1, profile_runs=1, artifact_mb=16),
        seed_pack=SeedPack(global_seed=1, input_seed=2),
        tags=["test"],
        tool_versions={},
    )
    writer = RunBundleWriter(root)
    writer.initialize(run_spec)
    return writer


class NvidiaBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_nvidia_backends"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.writer = _build_writer(self.tmp_root)
        self.executor = FakeNvidiaToolExecutor()

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_profile_kernel_nvidia_normalizes_metrics(self) -> None:
        with patch("gpu_cockpit.backends.nvidia.ncu.shutil.which", return_value="/usr/bin/ncu"):
            report = profile_kernel_nvidia(
                self.writer,
                ["python3", "-c", "print('hello')"],
                profile_pack="memory",
                executor=self.executor,
            )
        self.assertEqual(report.backend, "nvidia_ncu")
        self.assertEqual(report.kernel_name, "my_kernel")
        self.assertEqual(report.classification, "memory_bound")
        self.assertAlmostEqual(report.occupancy or 0.0, 52.0)
        self.assertAlmostEqual(report.dram_throughput_pct_peak or 0.0, 71.5)
        self.assertEqual(report.profiled_kernel_count, 1)
        self.assertTrue((self.writer.run_dir / "profiles" / "kernel" / "summary.json").exists())
        payload = json.loads((self.writer.run_dir / "profiles" / "kernel" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["kernel_name"], "my_kernel")

    def test_sanitize_nvidia_normalizes_findings(self) -> None:
        with patch("gpu_cockpit.backends.nvidia.sanitizer.shutil.which", return_value="/usr/bin/compute-sanitizer"):
            report = sanitize_nvidia(
                self.writer,
                ["python3", "-c", "print('hello')"],
                tool="memcheck",
                executor=self.executor,
            )
        self.assertFalse(report.passed)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.severity_counts["error"], 1)
        self.assertEqual(report.category_counts["memory_access_error"], 1)
        self.assertEqual(report.findings[0].category, "memory_access_error")
        self.assertEqual(report.findings[0].line, 42)
        self.assertTrue((self.writer.run_dir / "sanitize" / "memcheck_summary.json").exists())

    def test_emit_disassembly_nvidia_builds_triview(self) -> None:
        source_path = self.tmp_root / "demo_kernel.py"
        source_path.write_text("def demo_kernel(x):\n    return x + 1\n", encoding="utf-8")
        binary_path = self.tmp_root / "demo.fatbin"
        binary_path.write_bytes(b"fakefatbin")
        with (
            patch("gpu_cockpit.backends.nvidia.disassembly.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name in {"cuobjdump", "nvdisasm"} else None),
        ):
            record = emit_disassembly_nvidia(
                self.writer,
                ["python3", str(source_path)],
                source_file=str(source_path),
                binary_file=str(binary_path),
                executor=self.executor,
            )
        self.assertEqual(record.status, "ok")
        self.assertTrue((self.writer.run_dir / "build" / "tri_view.json").exists())
        tri_view = json.loads((self.writer.run_dir / "build" / "tri_view.json").read_text(encoding="utf-8"))
        self.assertEqual(tri_view["backend"], "nvidia_disassembly")
        self.assertGreaterEqual(len(tri_view["lines"]), 1)


if __name__ == "__main__":
    unittest.main()
