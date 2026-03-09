from __future__ import annotations

from pathlib import Path
import json
import shutil as shell_shutil
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.runner import run_task


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_runner"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        workloads_src = ROOT / "workloads"
        workloads_dst = self.tmp_root / "workloads"
        shutil.copytree(workloads_src, workloads_dst)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_run_task_without_trace(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('runner smoke')"],
            trace_system=False,
        )
        self.assertTrue((run_dir / "manifest.json").exists())
        self.assertTrue((run_dir / "prompt" / "task_spec.json").exists())
        self.assertTrue((run_dir / "meta" / "doctor_report.json").exists())
        self.assertTrue((run_dir / "command" / "summary.json").exists())
        self.assertTrue((run_dir / "summary.json").exists())
        self.assertTrue((run_dir / "summary.md").exists())

    def test_hidden_test_refs_stay_out_of_prompt_bundle(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        public_task = json.loads((run_dir / "prompt" / "task_spec.json").read_text(encoding="utf-8"))
        full_task = json.loads((run_dir / "meta" / "task_spec_full.json").read_text(encoding="utf-8"))
        self.assertIsNone(public_task["hidden_tests_ref"])
        self.assertEqual(full_task["hidden_tests_ref"], "workloads/tests/smoke_hidden.py")

    @unittest.skipUnless(shell_shutil.which("docker"), "docker is not available")
    def test_run_task_with_local_docker_executor(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('docker runner smoke')"],
            trace_system=False,
            executor="local_docker",
        )
        stdout = (run_dir / "command" / "stdout.txt").read_text(encoding="utf-8")
        self.assertIn("docker runner smoke", stdout)

    def test_run_task_emits_profile_and_sanitizer_artifacts(self) -> None:
        def fake_profile(writer, command, profile_pack="quick", executor=None):
            del command, executor
            writer.write_artifact(
                relative_path="profiles/kernel/summary.json",
                kind="kernel_profile_summary",
                content='{"backend":"nvidia_ncu","profiler":"ncu","command":[],"profile_pack":"quick","exit_code":0,"duration_ms":1,"kernel_name":"demo_kernel","classification":"memory_bound","profiled_kernel_count":1,"top_kernels":[],"warnings":[]}\n',
                mime="application/json",
                semantic_tags=["profile"],
            )
            writer.write_artifact(
                relative_path="profiles/kernel/summary.md",
                kind="kernel_profile_markdown",
                content="# profile\n",
                mime="text/markdown",
                semantic_tags=["profile"],
            )
            return __import__("gpu_cockpit.contracts", fromlist=["ProfileReport"]).ProfileReport(
                profiler="ncu",
                backend="nvidia_ncu",
                command=[],
                profile_pack=profile_pack,
                exit_code=0,
                duration_ms=1,
                kernel_name="demo_kernel",
                classification="memory_bound",
                profiled_kernel_count=1,
                top_kernels=[],
            )

        def fake_sanitize(writer, command, tool="memcheck", executor=None):
            del command, executor
            writer.write_artifact(
                relative_path=f"sanitize/{tool}_summary.json",
                kind="sanitizer_summary",
                content='{"backend":"nvidia_compute_sanitizer","tool":"memcheck","command":[],"exit_code":0,"duration_ms":1,"passed":true,"error_count":0,"warning_count":0,"findings":[],"warnings":[]}\n',
                mime="application/json",
                semantic_tags=["sanitize"],
            )
            writer.write_artifact(
                relative_path=f"sanitize/{tool}_summary.md",
                kind="sanitizer_markdown",
                content="# sanitize\n",
                mime="text/markdown",
                semantic_tags=["sanitize"],
            )
            return __import__("gpu_cockpit.contracts", fromlist=["SanitizerReport"]).SanitizerReport(
                backend="nvidia_compute_sanitizer",
                tool=tool,
                command=[],
                exit_code=0,
                duration_ms=1,
                passed=True,
            )

        def fake_bottleneck(writer, profile_report, sanitizer_report=None):
            del profile_report, sanitizer_report
            writer.write_artifact(
                relative_path="bottlenecks/primary.json",
                kind="bottleneck_card",
                content='{"card_id":"bn_test","run_id":"test","subject":"demo_kernel","primary_bottleneck":"memory_bound","confidence":0.8,"evidence":[],"why":"test","next_actions":[]}\n',
                mime="application/json",
                semantic_tags=["bottleneck"],
            )

        with (
            patch("gpu_cockpit.engine.runner.profile_kernel_nvidia", side_effect=fake_profile),
            patch("gpu_cockpit.engine.runner.sanitize_nvidia", side_effect=fake_sanitize),
            patch("gpu_cockpit.engine.runner.build_bottleneck_card", side_effect=fake_bottleneck),
        ):
            run_dir = run_task(
                root=self.tmp_root,
                task_ref="task/smoke/diagnose/v1",
                command=["python3", "-c", "print('instrumented run')"],
                trace_system=False,
                profile_kernel=True,
                sanitize=True,
            )
        self.assertTrue((run_dir / "profiles" / "kernel" / "summary.json").exists())
        self.assertTrue((run_dir / "sanitize" / "memcheck_summary.json").exists())
        self.assertTrue((run_dir / "bottlenecks" / "primary.json").exists())

    def test_run_task_degrades_when_nvidia_tools_are_missing(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('graceful degrade')"],
            trace_system=False,
            profile_kernel=True,
            sanitize=True,
        )
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("ncu is not installed.", summary["warnings"])
        self.assertIn("compute-sanitizer is not installed.", summary["warnings"])

    def test_run_task_emits_triview_artifacts(self) -> None:
        source_path = self.tmp_root / "demo_kernel.py"
        source_path.write_text("def demo_kernel(x):\n    return x * 2\n", encoding="utf-8")

        def fake_disassembly(writer, command, source_file=None, binary_file=None, ptx_file=None, sass_file=None, executor=None):
            del command, binary_file, ptx_file, sass_file, executor
            if source_file:
                writer.write_artifact(
                    relative_path="build/source.txt",
                    kind="source_text",
                    content=Path(source_file).read_text(encoding="utf-8"),
                    mime="text/plain",
                    semantic_tags=["build", "source"],
                )
            writer.write_artifact(
                relative_path="build/tri_view.json",
                kind="tri_view",
                content='{"backend":"nvidia_disassembly","source_ref":"build/source.txt","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[],"warnings":[]}\n',
                mime="application/json",
                semantic_tags=["build", "triview"],
            )
            writer.write_artifact(
                relative_path="build/tri_view.md",
                kind="tri_view_markdown",
                content="# tri view\n",
                mime="text/markdown",
                semantic_tags=["build", "triview"],
            )
            writer.write_artifact(
                relative_path="build/source_ptx_sass_map.json",
                kind="source_ptx_sass_map",
                content='{"backend":"nvidia_disassembly","lines":[],"warnings":[]}\n',
                mime="application/json",
                semantic_tags=["build", "triview", "map"],
            )
            writer.write_artifact(
                relative_path="build/source_map_summary.json",
                kind="source_map_summary",
                content='{"backend":"nvidia_disassembly","line_count":0,"unique_source_lines":0,"mapped_ptx_lines":0,"mapped_sass_lines":0,"source_spans":[],"warnings":[]}\n',
                mime="application/json",
                semantic_tags=["build", "triview", "summary"],
            )
            writer.write_artifact(
                relative_path="build/ptx.txt",
                kind="ptx_text",
                content=".visible .entry demo() {}\n",
                mime="text/plain",
                semantic_tags=["build", "ptx"],
            )
            writer.write_artifact(
                relative_path="build/sass.txt",
                kind="sass_text",
                content="MOV R1, R2;\n",
                mime="text/plain",
                semantic_tags=["build", "sass"],
            )
            writer.write_artifact(
                relative_path="build/build_record.json",
                kind="build_record",
                content='{"compiler":"manual","compiler_version":"unknown","flags":[],"status":"ok","duration_ms":1,"stdout_ref":null,"stderr_ref":null,"ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","ptxas_stats_ref":null,"binary_hash":"abc"}\n',
                mime="application/json",
                semantic_tags=["build", "record"],
            )
            return __import__("gpu_cockpit.contracts", fromlist=["BuildRecord"]).BuildRecord(
                compiler="manual",
                compiler_version="unknown",
                status="ok",
                duration_ms=1,
                ptx_ref="build/ptx.txt",
                sass_ref="build/sass.txt",
                binary_hash="abc",
            )

        with patch("gpu_cockpit.engine.runner.emit_disassembly_nvidia", side_effect=fake_disassembly):
            run_dir = run_task(
                root=self.tmp_root,
                task_ref="task/smoke/diagnose/v1",
                command=["python3", str(source_path)],
                emit_disassembly=True,
                source_file=str(source_path),
            )
        self.assertTrue((run_dir / "build" / "tri_view.json").exists())
        self.assertTrue((run_dir / "build" / "build_record.json").exists())
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("build/source_map_summary.json", summary["key_artifacts"])

    @unittest.skipUnless(shell_shutil.which("python3"), "python3 is not available")
    def test_run_task_can_use_triton_build_spec_flag(self) -> None:
        try:
            import torch
            import triton  # noqa: F401
        except Exception:
            self.skipTest("torch/triton unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/reduction_row_sum/eval/v1",
            command=["python3", "workloads/reference/triton_row_sum_candidate.py", "--benchmark-repeats", "5"],
            emit_disassembly=True,
            triton_build_spec="workloads/reference/triton_row_sum_kernel.py:get_build_spec",
        )
        self.assertTrue((run_dir / "build" / "tri_view.json").exists())
        self.assertTrue((run_dir / "build" / "ttir.mlir").exists())


if __name__ == "__main__":
    unittest.main()
