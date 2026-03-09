from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.indexer import list_runs
from gpu_cockpit.engine.inspector import compare_runs, inspect_run
from gpu_cockpit.engine.runner import run_task


class InspectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_inspector"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_compare_two_runs(self) -> None:
        lhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('lhs')"],
            trace_system=False,
        )
        rhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('rhs')"],
            trace_system=False,
        )
        comparison = compare_runs(self.tmp_root, str(lhs), str(rhs))
        self.assertEqual(comparison.lhs_status, "ok")
        self.assertEqual(comparison.rhs_status, "ok")
        self.assertEqual(comparison.lhs_missing_required_artifacts, [])
        self.assertEqual(comparison.rhs_missing_required_artifacts, [])

    def test_inspect_run_projects_required_artifacts(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('inspect projection')"],
            trace_system=False,
        )
        inspected = inspect_run(self.tmp_root, str(run_dir))
        self.assertIn("projection", inspected)
        self.assertEqual(inspected["projection"]["missing_required_artifacts"], [])
        self.assertIn("evidence_quality", inspected["projection"])
        self.assertGreater(inspected["projection"]["evidence_quality"]["overall_score"], 0.0)

    def test_inspect_run_can_select_build_section(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('inspect build section')"],
            trace_system=False,
        )
        (run_dir / "build").mkdir(parents=True, exist_ok=True)
        (run_dir / "build" / "build_record.json").write_text(
            '{"compiler":"triton","compiler_version":"3.3.1","status":"ok","binary_hash":"abc123","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt"}\n',
            encoding="utf-8",
        )
        (run_dir / "build" / "tri_view.json").write_text(
            '{"backend":"triton_nvidia","correlation_method":"ptx_loc_source_map_v1","source_path":"kernel.py","source_ref":"build/source.txt","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[],"warnings":[]}\n',
            encoding="utf-8",
        )
        section = inspect_run(self.tmp_root, str(run_dir), section="build")
        self.assertIn("build_record", section)
        self.assertIn("build_projection", section)

    def test_list_runs_filters_by_task_and_status(self) -> None:
        run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('indexed run')"],
            trace_system=False,
        )
        rows = list_runs(self.tmp_root, task_id="task/smoke/diagnose/v1", status="ok", limit=5)
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "task/smoke/diagnose/v1")

    def test_compare_surfaces_profile_and_sanitizer_fields(self) -> None:
        lhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('lhs profile')"],
            trace_system=False,
        )
        rhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('rhs profile')"],
            trace_system=False,
        )
        (lhs / "profiles" / "kernel").mkdir(parents=True, exist_ok=True)
        (rhs / "profiles" / "kernel").mkdir(parents=True, exist_ok=True)
        (lhs / "sanitize").mkdir(parents=True, exist_ok=True)
        (rhs / "sanitize").mkdir(parents=True, exist_ok=True)
        (lhs / "profiles" / "kernel" / "summary.json").write_text(
            '{"classification":"memory_bound","kernel_name":"lhs_kernel"}\n',
            encoding="utf-8",
        )
        (rhs / "profiles" / "kernel" / "summary.json").write_text(
            '{"classification":"compute_bound","kernel_name":"rhs_kernel"}\n',
            encoding="utf-8",
        )
        (lhs / "sanitize" / "memcheck_summary.json").write_text(
            '{"passed":true,"error_count":0}\n',
            encoding="utf-8",
        )
        (rhs / "sanitize" / "memcheck_summary.json").write_text(
            '{"passed":false,"error_count":2}\n',
            encoding="utf-8",
        )
        (lhs / "build").mkdir(parents=True, exist_ok=True)
        (rhs / "build").mkdir(parents=True, exist_ok=True)
        (lhs / "build" / "ptx.txt").write_text("visible ptx\n", encoding="utf-8")
        (lhs / "build" / "sass.txt").write_text("visible sass\n", encoding="utf-8")
        (rhs / "build" / "ptx.txt").write_text("changed ptx\n", encoding="utf-8")
        (rhs / "build" / "sass.txt").write_text("changed sass\n", encoding="utf-8")
        (lhs / "build" / "build_record.json").write_text(
            '{"compiler":"triton","status":"ok","binary_hash":"lhs-hash","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt"}\n',
            encoding="utf-8",
        )
        (rhs / "build" / "build_record.json").write_text(
            '{"compiler":"triton","status":"ok","binary_hash":"rhs-hash","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt"}\n',
            encoding="utf-8",
        )
        (lhs / "build" / "tri_view.json").write_text(
            '{"backend":"triton_nvidia","correlation_method":"ptx_loc_source_map_v1","source_path":"lhs.py","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[{"source_line":10,"ptx_line":20,"sass_line":30},{"source_line":11,"ptx_line":21,"sass_line":31}],"warnings":[]}\n',
            encoding="utf-8",
        )
        (rhs / "build" / "tri_view.json").write_text(
            '{"backend":"triton_nvidia","correlation_method":"heuristic_anchor_alignment_v2","source_path":"rhs.py","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[{"source_line":7,"ptx_line":17,"sass_line":27}],"warnings":[]}\n',
            encoding="utf-8",
        )
        comparison = compare_runs(self.tmp_root, str(lhs), str(rhs))
        self.assertEqual(comparison.lhs_profile_classification, "memory_bound")
        self.assertEqual(comparison.rhs_profile_classification, "compute_bound")
        self.assertTrue(comparison.lhs_sanitizer_passed)
        self.assertEqual(comparison.rhs_sanitizer_error_count, 2)
        self.assertTrue(comparison.lhs_triview_present)
        self.assertTrue(comparison.rhs_triview_present)
        self.assertEqual(comparison.lhs_triview_correlation_method, "ptx_loc_source_map_v1")
        self.assertEqual(comparison.rhs_triview_source_path, "rhs.py")
        self.assertEqual(comparison.lhs_triview_line_count, 2)
        self.assertEqual(comparison.rhs_triview_line_count, 1)
        self.assertTrue(comparison.build_binary_hash_changed)
        self.assertIsNotNone(comparison.lhs_evidence_score)
        self.assertIsNotNone(comparison.rhs_evidence_score)
        self.assertTrue(comparison.lhs_sft_ready)

    def test_inspect_run_projects_build_projection(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('build projection')"],
            trace_system=False,
        )
        (run_dir / "build").mkdir(parents=True, exist_ok=True)
        (run_dir / "build" / "source.txt").write_text("def kernel():\n    pass\n", encoding="utf-8")
        (run_dir / "build" / "ptx.txt").write_text("ld.param.u64 %rd1, [kernel];\n", encoding="utf-8")
        (run_dir / "build" / "sass.txt").write_text("S2R R2, SR_TID.X;\n", encoding="utf-8")
        (run_dir / "build" / "build_record.json").write_text(
            '{"compiler":"triton","compiler_version":"3.3.1","status":"ok","binary_hash":"abc123","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt"}\n',
            encoding="utf-8",
        )
        (run_dir / "build" / "tri_view.json").write_text(
            '{"backend":"triton_nvidia","correlation_method":"ptx_loc_source_map_v1","source_path":"kernel.py","source_ref":"build/source.txt","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[{"source_line":9,"ptx_line":27,"sass_line":2},{"source_line":10,"ptx_line":31,"sass_line":3}],"warnings":[]}\n',
            encoding="utf-8",
        )
        inspected = inspect_run(self.tmp_root, str(run_dir))
        build_projection = inspected["projection"]["build_projection"]
        self.assertEqual(build_projection["record"]["compiler"], "triton")
        self.assertEqual(build_projection["tri_view"]["line_count"], 2)
        self.assertEqual(build_projection["tri_view"]["unique_source_lines"], 2)
        self.assertIn("ptx", build_projection["artifact_hashes"])

    def test_inspect_and_compare_amd_golden_bundle(self) -> None:
        amd_bundle = ROOT / "tests" / "golden_runs" / "amd_bundle_v1"
        inspected = inspect_run(ROOT, str(amd_bundle), section="profile")
        self.assertEqual(inspected["profile_summary"]["backend"], "amd_rocprof")
        self.assertEqual(inspected["bottleneck_card"]["primary_bottleneck"], "memory_bound")

        rhs = self.tmp_root / "amd_bundle_variant"
        shutil.copytree(amd_bundle, rhs)
        (rhs / "summary.json").write_text(
            (amd_bundle / "summary.json").read_text(encoding="utf-8").replace('"duration_ms": 42', '"duration_ms": 55'),
            encoding="utf-8",
        )
        (rhs / "profiles" / "kernel" / "summary.json").write_text(
            (amd_bundle / "profiles" / "kernel" / "summary.json").read_text(encoding="utf-8").replace('"classification": "memory_bound"', '"classification": "compute_bound"'),
            encoding="utf-8",
        )
        comparison = compare_runs(ROOT, str(amd_bundle), str(rhs))
        self.assertEqual(comparison.lhs_profile_classification, "memory_bound")
        self.assertEqual(comparison.rhs_profile_classification, "compute_bound")
        self.assertEqual(comparison.lhs_status, "ok")

    def test_inspect_run_can_select_quality_section(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        section = inspect_run(self.tmp_root, str(run_dir), section="quality")
        self.assertIn("evidence_quality", section)
        self.assertIn("benchmark_reporting", section["evidence_quality"])
        self.assertIn("training_readiness", section)
        self.assertEqual(section["training_readiness"]["training_example_kind"], "unusable")

    def test_compare_golden_reformulate_pair_surfaces_training_transition(self) -> None:
        comparison = compare_runs(
            ROOT,
            str(ROOT / "tests" / "golden_runs" / "attention_reformulate_baseline_bench_v1"),
            str(ROOT / "tests" / "golden_runs" / "attention_reformulate_eval_bundle_v1"),
        )
        self.assertEqual(comparison.trainworthiness_change, "unusable->positive_rl_trace")
        self.assertEqual(comparison.rhs_training_example_kind, "positive_rl_trace")

    def test_inspect_golden_debug_bundle_surfaces_failure_triage(self) -> None:
        section = inspect_run(ROOT, str(ROOT / "tests" / "golden_runs" / "reduction_debug_bundle_v1"), section="quality")
        self.assertEqual(section["failure_triage"]["task_verb"], "debug")
        self.assertIn("correctness/correctness.json", section["failure_triage"]["likely_artifacts"])

    def test_compare_golden_public_benchmark_bundles(self) -> None:
        comparison = compare_runs(
            ROOT,
            str(ROOT / "tests" / "golden_runs" / "kernelbench_public_eval_v1"),
            str(ROOT / "tests" / "golden_runs" / "computeeval_public_eval_v1"),
        )
        self.assertEqual(comparison.lhs_status, "ok")
        self.assertEqual(comparison.rhs_status, "ok")
        self.assertTrue(comparison.lhs_rl_trace_ready)
        self.assertTrue(comparison.rhs_rl_trace_ready)


if __name__ == "__main__":
    unittest.main()
