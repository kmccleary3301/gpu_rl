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
from gpu_cockpit.engine.patching import apply_patch_candidate, branch_candidate, promote_candidate
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
        (lhs / "build" / "source.txt").write_text("def lhs_kernel():\n    return 1\n", encoding="utf-8")
        (rhs / "build" / "source.txt").write_text("def rhs_kernel():\n    return 2\n", encoding="utf-8")
        (lhs / "build" / "ttir.mlir").write_text("tt.func @lhs()\n", encoding="utf-8")
        (rhs / "build" / "ttir.mlir").write_text("tt.func @rhs()\n", encoding="utf-8")
        (lhs / "build" / "ttgir.mlir").write_text("ttg.func @lhs()\n", encoding="utf-8")
        (rhs / "build" / "ttgir.mlir").write_text("ttg.func @rhs()\n", encoding="utf-8")
        (lhs / "build" / "llir.ll").write_text("define void @lhs() {}\n", encoding="utf-8")
        (rhs / "build" / "llir.ll").write_text("define void @rhs() {}\n", encoding="utf-8")
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
            '{"backend":"triton_nvidia","correlation_method":"ptx_loc_source_map_v1","source_path":"lhs.py","source_ref":"build/source.txt","ttir_ref":"build/ttir.mlir","ttgir_ref":"build/ttgir.mlir","llir_ref":"build/llir.ll","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[{"source_line":10,"ptx_line":20,"sass_line":30},{"source_line":11,"ptx_line":21,"sass_line":31}],"warnings":[]}\n',
            encoding="utf-8",
        )
        (rhs / "build" / "tri_view.json").write_text(
            '{"backend":"triton_nvidia","correlation_method":"heuristic_anchor_alignment_v2","source_path":"rhs.py","source_ref":"build/source.txt","ttir_ref":"build/ttir.mlir","ttgir_ref":"build/ttgir.mlir","llir_ref":"build/llir.ll","ptx_ref":"build/ptx.txt","sass_ref":"build/sass.txt","lines":[{"source_line":7,"ptx_line":17,"sass_line":27}],"warnings":[]}\n',
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
        self.assertTrue(comparison.build_source_hash_changed)
        self.assertTrue(comparison.build_ttir_hash_changed)
        self.assertTrue(comparison.build_ttgir_hash_changed)
        self.assertTrue(comparison.build_llir_hash_changed)
        self.assertTrue(comparison.build_ptx_hash_changed)
        self.assertTrue(comparison.build_sass_hash_changed)
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
        self.assertIn("training_trace_triage", section)
        self.assertIn("profile_triage", section)
        self.assertEqual(section["training_readiness"]["training_example_kind"], "unusable")

    def test_compare_surfaces_public_benchmark_optimize_digest(self) -> None:
        lhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('lhs public benchmark')"],
            trace_system=False,
        )
        rhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('rhs public benchmark')"],
            trace_system=False,
        )
        (lhs / "command").mkdir(parents=True, exist_ok=True)
        (rhs / "command").mkdir(parents=True, exist_ok=True)
        (lhs / "command" / "stdout.txt").write_text(
            '{"benchmark_source":"kernelbench","benchmark_case_id":"kernelbench/level1/23_softmax","benchmark_case_version":"v0.1","case_config_path":"/tmp/softmax_case.json","problem_path":"/tmp/softmax.py","optimization_summary":{"strategy_change":"baseline_kernelbench_reference","candidate_ref":"workloads/reference/kernelbench_reference_runner.py","baseline_ref":"workloads/reference/kernelbench_reference_runner.py","case_config_ref":"workloads/public_benchmarks/kernelbench/v0_1/cases/level1_023_softmax.json"}}\n',
            encoding="utf-8",
        )
        (rhs / "command" / "stdout.txt").write_text(
            '{"benchmark_source":"kernelbench","benchmark_case_id":"kernelbench/level1/23_softmax","benchmark_case_version":"v0.1","case_config_path":"/tmp/softmax_case.json","problem_path":"/tmp/softmax.py","optimization_summary":{"strategy_change":"promote_curated_kernelbench_softmax_candidate_wrapper","candidate_ref":"workloads/reference/kernelbench_softmax_optimize_candidate.py","baseline_ref":"workloads/reference/kernelbench_reference_runner.py","case_config_ref":"workloads/public_benchmarks/kernelbench/v0_1/cases/level1_023_softmax.json"}}\n',
            encoding="utf-8",
        )
        (lhs / "eval").mkdir(parents=True, exist_ok=True)
        (rhs / "eval").mkdir(parents=True, exist_ok=True)
        (lhs / "perf").mkdir(parents=True, exist_ok=True)
        (rhs / "perf").mkdir(parents=True, exist_ok=True)
        (lhs / "eval" / "eval_envelope.json").write_text('{"correctness_gate":"pass","final_score":0.8}\n', encoding="utf-8")
        (rhs / "eval" / "eval_envelope.json").write_text('{"correctness_gate":"pass","final_score":1.0}\n', encoding="utf-8")
        (lhs / "perf" / "benchmark.json").write_text(
            '{"steady_state_ms_p50":10.0,"timing_method":"wall_clock","split_compile_from_run":true,"benchmark_scope":"tool.run_benchmark","candidate_command_sha256":"lhs_sha"}\n',
            encoding="utf-8",
        )
        (rhs / "perf" / "benchmark.json").write_text(
            '{"steady_state_ms_p50":8.0,"timing_method":"wall_clock","split_compile_from_run":true,"benchmark_scope":"tool.run_benchmark","candidate_command_sha256":"rhs_sha"}\n',
            encoding="utf-8",
        )

        comparison = compare_runs(self.tmp_root, str(lhs), str(rhs))

        self.assertEqual(comparison.optimize_delta_summary["benchmark_case_id"], "kernelbench/level1/23_softmax")
        self.assertEqual(comparison.optimize_delta_summary["benchmark_source"], "kernelbench")
        self.assertEqual(
            comparison.optimize_delta_summary["optimization_strategy_change"],
            "promote_curated_kernelbench_softmax_candidate_wrapper",
        )
        self.assertEqual(
            comparison.optimize_delta_summary["candidate_ref"],
            "workloads/reference/kernelbench_softmax_optimize_candidate.py",
        )
        self.assertEqual(comparison.compare_type, "baseline_to_candidate")
        self.assertEqual(comparison.benchmark_provenance["benchmark_case_id"], "kernelbench/level1/23_softmax")
        self.assertEqual(comparison.benchmark_provenance["benchmark_source"], "kernelbench")
        self.assertEqual(comparison.perf_localization["rhs"]["timing_method"], "wall_clock")
        self.assertEqual(comparison.perf_localization["rhs"]["benchmark_scope"], "tool.run_benchmark")
        self.assertEqual(comparison.perf_localization["rhs"]["candidate_command_sha256"], "rhs_sha")
        self.assertIn("Compare is anchored to public benchmark case `kernelbench/level1/23_softmax`.", comparison.summary_lines)
        self.assertIn(
            "Optimization strategy changed from `baseline_kernelbench_reference` to `promote_curated_kernelbench_softmax_candidate_wrapper`.",
            comparison.summary_lines,
        )

    def test_compare_inherits_candidate_lineage_and_optimize_summary_from_parent_patch_run(self) -> None:
        lhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('{}')"],
            trace_system=False,
        )
        replacement_text = (self.tmp_root / "workloads" / "reference" / "triton_attention_score_optimize_candidate_v2.py").read_text(
            encoding="utf-8"
        )
        patch_run, _, candidate_state, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/smoke/eval/v1",
            target_file="workloads/reference/triton_attention_score_optimize_patchable_candidate.py",
            replacement_text=replacement_text,
            intent="promote the attention candidate into the ranked optimize variant",
            expected_effect="surface the ranked optimization summary in downstream bench runs",
            patch_kind="perf_transform",
            transition_kind="reformulated",
        )
        rhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=[
                "python3",
                "-c",
                (
                    "import json; "
                    "print(json.dumps({'optimization_summary': {"
                    "'strategy_change': 'supersede_tiled_triton_kernel_candidate_with_ranked_variant', "
                    "'candidate_ref': 'workloads/reference/triton_attention_score_optimize_candidate_v2.py', "
                    "'baseline_ref': 'workloads/reference/triton_attention_score_baseline.py', "
                    "'supersedes_candidate_ref': 'workloads/reference/triton_attention_score_optimize_candidate.py'"
                    "}}, sort_keys=True))"
                ),
            ],
            trace_system=False,
            lineage={"parent_run_id": patch_run.name},
        )

        comparison = compare_runs(self.tmp_root, str(lhs), str(rhs))

        self.assertEqual(comparison.rhs_candidate_id, candidate_state.candidate_id)
        self.assertEqual(comparison.rhs_parent_candidate_id, candidate_state.parent_candidate_id)
        self.assertEqual(
            comparison.optimize_delta_summary["optimization_strategy_change"],
            "supersede_tiled_triton_kernel_candidate_with_ranked_variant",
        )
        self.assertEqual(
            comparison.optimize_delta_summary["candidate_ref"],
            "workloads/reference/triton_attention_score_optimize_candidate_v2.py",
        )
        self.assertEqual(comparison.candidate_delta_brief["rhs_candidate_id"], candidate_state.candidate_id)

    def test_compare_golden_reformulate_pair_surfaces_training_transition(self) -> None:
        comparison = compare_runs(
            ROOT,
            str(ROOT / "tests" / "golden_runs" / "attention_reformulate_baseline_bench_v1"),
            str(ROOT / "tests" / "golden_runs" / "attention_reformulate_eval_bundle_v1"),
        )
        self.assertEqual(comparison.trainworthiness_change, "unusable->benchmark_only")
        self.assertEqual(comparison.rhs_training_example_kind, "benchmark_only")
        self.assertTrue(comparison.rhs_benchmark_ready)
        self.assertFalse(comparison.rhs_rl_trace_ready)
        self.assertTrue(any("Trainworthiness changed" in line for line in comparison.summary_lines))
        self.assertEqual(comparison.optimize_delta_summary["correctness_change"], "unknown")
        self.assertIn("inspect_quality", comparison.recommended_next_actions)

    def test_inspect_golden_debug_bundle_surfaces_failure_triage(self) -> None:
        section = inspect_run(ROOT, str(ROOT / "tests" / "golden_runs" / "reduction_debug_bundle_v1"), section="quality")
        self.assertEqual(section["failure_triage"]["task_verb"], "debug")
        self.assertIn("correctness/correctness.json", section["failure_triage"]["likely_artifacts"])
        self.assertIn("summary_lines", section["profile_triage"])
        self.assertIn("perf_localization", section)

    def test_compare_golden_public_benchmark_bundles(self) -> None:
        comparison = compare_runs(
            ROOT,
            str(ROOT / "tests" / "golden_runs" / "kernelbench_public_eval_v1"),
            str(ROOT / "tests" / "golden_runs" / "computeeval_public_eval_v1"),
        )
        self.assertEqual(comparison.lhs_status, "ok")
        self.assertEqual(comparison.rhs_status, "ok")
        self.assertTrue(comparison.lhs_benchmark_ready)
        self.assertTrue(comparison.rhs_benchmark_ready)
        self.assertEqual(comparison.lhs_training_example_kind, "benchmark_only")
        self.assertEqual(comparison.rhs_training_example_kind, "benchmark_only")
        self.assertFalse(comparison.lhs_rl_trace_ready)
        self.assertFalse(comparison.rhs_rl_trace_ready)

    def test_inspect_and_compare_patch_bundles_surface_candidate_fields(self) -> None:
        broken_path = self.tmp_root / "workloads" / "reference" / "triton_row_sum_broken_kernel.py"
        original_text = broken_path.read_text(encoding="utf-8")
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")

        lhs_run_dir, _, lhs_candidate, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=repaired_text,
            intent="repair the broken row-sum kernel mask",
            expected_effect="restore full-column coverage",
            patch_kind="bug_fix",
            transition_kind="repaired",
        )
        rhs_run_dir, _, rhs_candidate, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=original_text,
            intent="revert the patch to recover the original broken kernel",
            expected_effect="return to the previous state",
            patch_kind="refactor",
            transition_kind="reverted",
            parent_run_ref=str(lhs_run_dir),
            parent_run_id=lhs_run_dir.name,
            parent_candidate_id=lhs_candidate.candidate_id,
        )

        section = inspect_run(self.tmp_root, str(lhs_run_dir), section="transition")
        self.assertTrue(section["candidate_projection"]["patch_present"])
        self.assertEqual(section["candidate_projection"]["candidate_state"]["candidate_id"], lhs_candidate.candidate_id)

        comparison = compare_runs(self.tmp_root, str(lhs_run_dir), str(rhs_run_dir))
        self.assertTrue(comparison.lhs_patch_present)
        self.assertTrue(comparison.rhs_patch_present)
        self.assertEqual(comparison.lhs_candidate_id, lhs_candidate.candidate_id)
        self.assertEqual(comparison.rhs_candidate_id, rhs_candidate.candidate_id)
        self.assertEqual(comparison.lhs_transition_kind, "repaired")
        self.assertEqual(comparison.rhs_transition_kind, "reverted")
        self.assertTrue(comparison.patch_hash_changed)
        self.assertIn("lhs", comparison.perf_localization)
        self.assertIn("rhs", comparison.perf_localization)
        self.assertEqual(comparison.lineage_relationship, "lhs_parent_of_rhs")
        self.assertTrue(comparison.parent_child_related)
        self.assertEqual(comparison.rhs_parent_candidate_id, lhs_candidate.candidate_id)

    def test_compare_surfaces_optimize_delta_summary_and_actions(self) -> None:
        lhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        rhs = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        (lhs / "eval").mkdir(parents=True, exist_ok=True)
        (rhs / "eval").mkdir(parents=True, exist_ok=True)
        (lhs / "eval" / "eval_envelope.json").write_text(
            '{"correctness_gate":"fail","final_score":0.0,"determinism_gate":"pass","perf_gate":"blocked"}\n',
            encoding="utf-8",
        )
        (rhs / "eval" / "eval_envelope.json").write_text(
            '{"correctness_gate":"pass","final_score":0.8,"determinism_gate":"pass","perf_gate":"fail"}\n',
            encoding="utf-8",
        )
        (lhs / "perf").mkdir(parents=True, exist_ok=True)
        (rhs / "perf").mkdir(parents=True, exist_ok=True)
        (lhs / "perf" / "benchmark.json").write_text('{"steady_state_ms_p50":10.0}\n', encoding="utf-8")
        (rhs / "perf" / "benchmark.json").write_text('{"steady_state_ms_p50":12.5}\n', encoding="utf-8")
        (rhs / "candidate").mkdir(parents=True, exist_ok=True)
        (rhs / "candidate" / "state.json").write_text(
            '{"candidate_id":"cand_rhs","parent_candidate_id":"cand_lhs","status":"patched","origin_kind":"patch"}\n',
            encoding="utf-8",
        )
        (rhs / "candidate" / "transition.json").write_text(
            '{"transition_kind":"reformulated"}\n',
            encoding="utf-8",
        )
        (rhs / "patches").mkdir(parents=True, exist_ok=True)
        (rhs / "patches" / "applied_patch.json").write_text(
            '{"patch_kind":"perf_transform","patch_hash":"abc123"}\n',
            encoding="utf-8",
        )

        comparison = compare_runs(self.tmp_root, str(lhs), str(rhs))

        self.assertEqual(comparison.optimize_delta_summary["correctness_change"], "recovered")
        self.assertEqual(comparison.optimize_delta_summary["perf_change"], "regressed")
        self.assertEqual(comparison.optimize_delta_summary["patch_change"], "none->perf_transform")
        self.assertEqual(comparison.recommended_next_actions, ["inspect_quality", "patch_candidate", "bench"])
        self.assertTrue(any("Correctness recovered" in line for line in comparison.summary_lines))

    def test_compare_candidate_siblings_surfaces_same_parent_lineage(self) -> None:
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")
        patch_run_dir, _, patched_candidate, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=repaired_text,
            intent="repair the row-sum kernel",
            expected_effect="restore full-column coverage",
            patch_kind="bug_fix",
            transition_kind="repaired",
        )
        branch_run_dir, branch_candidate_state, _ = branch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            intent="branch for alternate reduction tuning",
            branch_label="sibling_a",
            parent_run_ref=str(patch_run_dir),
            parent_run_id=patch_run_dir.name,
            parent_candidate_id=patched_candidate.candidate_id,
        )
        promote_run_dir, promoted_candidate_state, _ = promote_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            intent="promote a second sibling candidate",
            promotion_label="sibling_b",
            parent_run_ref=str(patch_run_dir),
            parent_run_id=patch_run_dir.name,
            parent_candidate_id=patched_candidate.candidate_id,
        )

        comparison = compare_runs(self.tmp_root, str(branch_run_dir), str(promote_run_dir))

        self.assertEqual(comparison.lineage_relationship, "same_parent")
        self.assertFalse(comparison.parent_child_related)
        self.assertEqual(comparison.lhs_parent_candidate_id, patched_candidate.candidate_id)
        self.assertEqual(comparison.rhs_parent_candidate_id, patched_candidate.candidate_id)
        self.assertEqual(comparison.lhs_candidate_id, branch_candidate_state.candidate_id)
        self.assertEqual(comparison.rhs_candidate_id, promoted_candidate_state.candidate_id)
        self.assertEqual(comparison.candidate_delta_brief["lineage_relationship"], "same_parent")
        self.assertEqual(comparison.candidate_delta_brief["lhs_transition_kind"], "branched")
        self.assertEqual(comparison.candidate_delta_brief["rhs_transition_kind"], "promoted")
        self.assertEqual(comparison.candidate_delta_brief["lhs_candidate_role"], "branched_candidate")
        self.assertEqual(comparison.candidate_delta_brief["rhs_candidate_role"], "promoted_candidate")
        self.assertEqual(comparison.candidate_delta_brief["lhs_candidate_role_group"], "branch")
        self.assertEqual(comparison.candidate_delta_brief["rhs_candidate_role_group"], "promoted")
        self.assertEqual(
            comparison.candidate_delta_brief["sibling_candidate_refs"],
            [branch_candidate_state.candidate_id, promoted_candidate_state.candidate_id],
        )
        self.assertEqual(comparison.lhs_candidate_role_group, "branch")
        self.assertEqual(comparison.rhs_candidate_role_group, "promoted")
        self.assertTrue(comparison.optimize_delta_summary["lineage_scopes"]["sibling_delta"]["available"])

    def test_inspect_run_prefers_localized_next_actions(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        (run_dir / "correctness").mkdir(parents=True, exist_ok=True)
        (run_dir / "correctness" / "failure_localization.json").write_text(
            '{"hidden_tests":{"code":"kernelbench_hidden_sum_mismatch","fix_family":"numerical_mismatch","likely_next_actions":["inspect_quality","patch_candidate","eval"]}}\n',
            encoding="utf-8",
        )

        inspected = inspect_run(self.tmp_root, str(run_dir), section="quality")

        self.assertEqual(inspected["recommended_next_actions"], ["inspect_quality", "patch_candidate", "eval"])


if __name__ == "__main__":
    unittest.main()
