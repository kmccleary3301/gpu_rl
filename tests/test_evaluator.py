from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.contracts.run import RunSpec
from gpu_cockpit.contracts.trace import SystemTraceSummary
from gpu_cockpit.engine.benchmark import run_task_benchmark
from gpu_cockpit.engine.command_runner import run_command
from gpu_cockpit.engine.evaluator import run_evaluation_hooks
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.task_registry import TaskRegistry


def _cuda_torch_triton_ready() -> bool:
    try:
        import torch
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_eval"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_eval_hooks_pass_for_smoke_task(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_bundle",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command_summary = run_command(
            writer,
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
            command_summary=command_summary,
            perf_report=run_task_benchmark(
                writer=writer,
                root=self.tmp_root,
                task=task,
                command=command_summary.command,
            ),
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.determinism_gate, "pass")
        self.assertEqual(envelope.perf_gate, "pass")
        self.assertEqual(envelope.final_score, 1.0)
        self.assertTrue((self.tmp_root / "runs" / "eval_test_bundle" / "correctness" / "determinism.json").exists())

    def test_antihack_fails_on_forbidden_pattern(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_antihack",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command_summary = run_command(
            writer,
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK'); print('CPU_FALLBACK')"],
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
            command_summary=command_summary,
            perf_report=run_task_benchmark(
                writer=writer,
                root=self.tmp_root,
                task=task,
                command=command_summary.command,
            ),
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertFalse(correctness.hidden_tests_ok)
        self.assertFalse(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.final_score, 0.0)

    def test_determinism_failure_drops_eval_score(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/diagnose/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_determinism",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command_summary = run_command(
            writer,
            command=["python3", "-c", "import random; print(random.random())"],
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
            command_summary=command_summary,
            determinism_runs=3,
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(anti_hack.passed)
        self.assertFalse(determinism.passed)
        self.assertIn("determinism_failed", correctness.failures)
        self.assertEqual(envelope.determinism_gate, "fail")
        self.assertEqual(envelope.final_score, 0.0)

    def test_perf_gate_fails_for_slower_than_baseline_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_perf_fail",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command_summary = run_command(
            writer,
            command=["python3", "-c", "import time; time.sleep(0.05); print('GPU_COCKPIT_SMOKE_OK')"],
        )
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
            command_summary=command_summary,
            perf_report=perf_report,
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.perf_gate, "fail")
        self.assertEqual(envelope.final_score, 0.8)

    def test_antihack_scans_directory_trees_for_forbidden_patterns(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/smoke/eval/v1")
        source_dir = self.tmp_root / "generated"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "kernel.py").write_text("import torch\nx = torch.topk(y, 4)\n", encoding="utf-8")

        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_tree_scan",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command_summary = run_command(
            writer,
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command_summary.command,
            command_summary=command_summary,
            scan_paths=[source_dir],
        )
        self.assertFalse(anti_hack.passed)

    def test_topk_router_task_passes_with_reference_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/topk_router/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_topk_router",
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
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/topk_router_candidate.py", "--benchmark-repeats", "20"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "topk_router_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertFalse(anti_hack.library_shortcut_detected)
        self.assertIn(str(self.tmp_root / "workloads" / "reference" / "topk_router_candidate.py"), anti_hack.scanned_locations)
        self.assertGreater(envelope.final_score, 0.0)

    def test_kernelbench_hardtanh_task_passes_with_reference_runner(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/kernelbench/level1/32_hardtanh/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_kernelbench_hardtanh",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="cuda",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = [
            "python3",
            "workloads/reference/kernelbench_reference_runner.py",
            "--case-config",
            "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_032_hardtanh.json",
            "--benchmark-repeats",
            "5",
        ]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)

    def test_kernelbench_layernorm_task_passes_with_reference_runner(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/kernelbench/level1/40_layernorm/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_kernelbench_layernorm",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="cuda",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = [
            "python3",
            "workloads/reference/kernelbench_reference_runner.py",
            "--case-config",
            "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_040_layernorm.json",
            "--benchmark-repeats",
            "3",
        ]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_reduction_row_sum_task_passes_with_triton_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/reduction_row_sum/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_reduction_row_sum",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_row_sum_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_row_sum_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertGreaterEqual(envelope.final_score, 0.8)

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_routing_argmax_task_passes_with_triton_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/routing_argmax/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_routing_argmax",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_routing_argmax_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_routing_argmax_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_kv_cache_gather_task_passes_with_triton_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/kv_cache_gather/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_kv_cache_gather",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_kv_cache_gather_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_kv_cache_gather_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_attention_score_task_passes_with_triton_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/attention_score/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_attention_score",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_attention_score_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_attention_score_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)

    def test_profile_diagnose_task_passes_with_reference_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/profile_diagnose/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_profile_diagnose",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="cuda",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/profile_diagnose_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "profile_diagnose_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertGreaterEqual(envelope.final_score, 0.8)

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_reduction_debug_task_passes_with_reference_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/reduction_debug/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_reduction_debug",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_row_sum_debug_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertEqual(envelope.perf_gate, "pass")

    @unittest.skipUnless(_cuda_torch_triton_ready(), "CUDA Triton stack unavailable")
    def test_attention_reformulate_task_passes_with_reference_candidate(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/attention_reformulate/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_attention_reformulate",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=120,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=128,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "5"]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "triton_attention_score_reformulate_candidate.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertEqual(envelope.perf_gate, "pass")

    def test_computeeval_task_passes_with_reference_runner(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/computeeval/cuda_16/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_computeeval",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="cuda",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = [
            "python3",
            "workloads/reference/computeeval_reference_runner.py",
            "--problem",
            "workloads/public_benchmarks/computeeval/2025_1/problems/cuda_16.json",
            "--benchmark-repeats",
            "20",
        ]
        command_summary = run_command(writer, command=command)
        perf_report = run_task_benchmark(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
        )
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=perf_report,
            scan_paths=[self.tmp_root / "workloads" / "reference" / "computeeval_reference_runner.py"],
        )
        self.assertTrue(correctness.compile_ok)
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertEqual(envelope.correctness_gate, "pass")
        self.assertEqual(envelope.perf_gate, "pass")

    def test_computeeval_library_case_preserves_provenance(self) -> None:
        registry = TaskRegistry(self.tmp_root)
        task = registry.get("task/computeeval/cuda_31/eval/v1")
        writer = RunBundleWriter(self.tmp_root)
        run_spec = RunSpec(
            run_id="eval_test_computeeval_provenance",
            created_at=datetime.now(tz=UTC),
            task_ref=task.task_id,
            mode="human",
            target_backend="cuda",
            target_vendor="nvidia",
            executor="local_host",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=0,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        writer.initialize(run_spec)
        command = [
            "python3",
            "workloads/reference/computeeval_reference_runner.py",
            "--problem",
            "workloads/public_benchmarks/computeeval/2025_1/problems/cuda_31.json",
            "--benchmark-repeats",
            "10",
        ]
        command_summary = run_command(writer, command=command)
        correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
            writer=writer,
            root=self.tmp_root,
            task=task,
            command=command,
            command_summary=command_summary,
            perf_report=run_task_benchmark(
                writer=writer,
                root=self.tmp_root,
                task=task,
                command=command,
            ),
        )
        self.assertTrue(correctness.visible_tests_ok)
        self.assertTrue(correctness.hidden_tests_ok)
        self.assertTrue(anti_hack.passed)
        self.assertTrue(determinism.passed)
        self.assertGreaterEqual(envelope.final_score, 0.8)


if __name__ == "__main__":
    unittest.main()
