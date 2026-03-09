from __future__ import annotations

import json
from datetime import UTC, datetime
import platform
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.backends.amd.rocprof import profile_kernel_amd, trace_system_amd
from gpu_cockpit.contracts import RunSpec, TaskSpec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.contracts.task import NumericalPolicy, PerfProtocol
from gpu_cockpit.engine.doctor import _collect_amd_fingerprints, _parse_rocm_smi_power_limits, _parse_rocminfo_devices
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.executors.base import CommandExecutor, CommandResult
from gpu_cockpit.contracts.doctor import ToolStatus


ROCINFO_SAMPLE = """
=====================
HSA Agents
=====================
******* Agent 1 *******
  Name:                    AMD EPYC
  Marketing Name:          AMD EPYC
  Device Type:             CPU
******* Agent 2 *******
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI250X
  Device Type:             GPU
  Global Memory Size:      68719476736
"""

ROCM_SMI_SAMPLE = """
GPU[0]          : Card series:          AMD Instinct MI250X
GPU[0]          : Power Cap (W):        500W
"""


class FakeAmdToolExecutor(CommandExecutor):
    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del env, cwd, timeout
        if command[:1] == ["/opt/rocm/bin/rocprof"] and "--hip-trace" in command:
            trace_path = Path(command[command.index("-o") + 1])
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text('{"trace":"ok"}\n', encoding="utf-8")
            return CommandResult(command=command, exit_code=0, stdout="rocprof trace ok\n", stderr="", duration_ms=20)
        if command[:1] == ["/opt/rocm/bin/rocprof"] and "--stats" in command:
            csv_path = Path(command[command.index("-o") + 1])
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "\n".join(
                    [
                        "KernelName,DurationNs,OccupancyPct,VGPR,DRAMPct,ComputePct,L2HitPct",
                        "amd_kernel,4000000,48,40,72,35,81",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            return CommandResult(command=command, exit_code=0, stdout="rocprof profile ok\n", stderr="", duration_ms=30)
        raise AssertionError(f"Unexpected command: {command}")


def _build_writer(root: Path) -> RunBundleWriter:
    task = TaskSpec(
        task_id="task/test/amd/v1",
        prompt="profile me",
        verb="optimize",
        operator_family="test",
        difficulty="unit",
        allowed_backends=["hip"],
        numerical_policy=NumericalPolicy(dtype_matrix=["fp32"], rtol=1e-5, atol=1e-6),
        perf_protocol=PerfProtocol(warmups=1, repeats=1, timer="wall_clock"),
        required_artifacts=[],
    )
    run_spec = RunSpec(
        run_id="run_test_amd_backends",
        created_at=datetime(2026, 3, 9, tzinfo=UTC),
        task_ref=task.task_id,
        mode="human",
        target_backend="hip",
        target_vendor="amd",
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


class AmdBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_amd_backends"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.writer = _build_writer(self.tmp_root)
        self.executor = FakeAmdToolExecutor()

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_trace_system_amd_emits_summary(self) -> None:
        with patch("gpu_cockpit.backends.amd.rocprof.shutil.which", side_effect=lambda name: "/opt/rocm/bin/rocprof" if name == "rocprof" else None):
            summary = trace_system_amd(self.writer, ["python3", "-c", "print('amd trace')"], executor=self.executor)
        self.assertEqual(summary.backend, "amd_rocprof_trace")
        self.assertTrue((self.writer.run_dir / "traces" / "system" / "summary.json").exists())
        self.assertTrue((self.writer.run_dir / "traces" / "system" / "summary.md").exists())

    def test_profile_kernel_amd_normalizes_metrics(self) -> None:
        with patch("gpu_cockpit.backends.amd.rocprof.shutil.which", side_effect=lambda name: "/opt/rocm/bin/rocprof" if name == "rocprof" else None):
            report = profile_kernel_amd(self.writer, ["python3", "-c", "print('amd profile')"], executor=self.executor)
        self.assertEqual(report.backend, "amd_rocprof")
        self.assertEqual(report.kernel_name, "amd_kernel")
        self.assertEqual(report.classification, "memory_bound")
        self.assertAlmostEqual(report.occupancy or 0.0, 48.0)
        self.assertTrue((self.writer.run_dir / "profiles" / "kernel" / "summary.json").exists())
        self.assertTrue((self.writer.run_dir / "profiles" / "kernel" / "summary.md").exists())
        payload = json.loads((self.writer.run_dir / "profiles" / "kernel" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["kernel_name"], "amd_kernel")

    def test_parse_rocminfo_devices_extracts_gpu_rows(self) -> None:
        devices = _parse_rocminfo_devices(ROCINFO_SAMPLE)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["name"], "gfx90a")
        self.assertEqual(devices[0]["marketing_name"], "AMD Instinct MI250X")

    def test_parse_rocm_smi_power_limits_extracts_watts(self) -> None:
        self.assertEqual(_parse_rocm_smi_power_limits(ROCM_SMI_SAMPLE), {"card0": 500})

    def test_collect_amd_fingerprints_parses_realistic_outputs(self) -> None:
        tool_statuses = [
            ToolStatus(name="rocminfo", path="/opt/rocm/bin/rocminfo", version="ROCk module version 6.2.0", available=True),
            ToolStatus(name="rocm-smi", path="/opt/rocm/bin/rocm-smi", version="ROCm System Management Interface v6.2.0", available=True),
            ToolStatus(name="hipcc", path="/opt/rocm/bin/hipcc", version="HIP version: 6.2.0", available=True),
            ToolStatus(name="rocprof", path="/opt/rocm/bin/rocprof", version="rocprof version 6.2.0", available=True),
        ]
        with (
            patch("gpu_cockpit.engine.doctor.shutil.which", side_effect=lambda name: f"/opt/rocm/bin/{name}" if name in {"rocminfo", "rocm-smi"} else None),
            patch(
                "gpu_cockpit.engine.doctor._run_command",
                side_effect=lambda args: ROCINFO_SAMPLE
                if args == ["rocminfo"]
                else ROCM_SMI_SAMPLE
                if args == ["rocm-smi", "--showproductname", "--showpower"]
                else None,
            ),
            patch("gpu_cockpit.engine.doctor.platform.release", return_value=platform.release()),
        ):
            fingerprints = _collect_amd_fingerprints(tool_statuses)
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(fingerprints[0].vendor, "amd")
        self.assertEqual(fingerprints[0].gpu_name, "AMD Instinct MI250X")
        self.assertEqual(fingerprints[0].arch, "gfx90a")
        self.assertEqual(fingerprints[0].power_limit_w, 500)
        self.assertEqual(fingerprints[0].runtime_version, "6.2.0")
