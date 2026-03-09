from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.adapter_registry import describe_adapter, get_adapter, list_adapter_cases, list_adapters


class AdapterRegistryTests(unittest.TestCase):
    def test_list_adapters_reports_file_backed_cases(self) -> None:
        adapters = list_adapters(ROOT)
        smoke = next(row for row in adapters if row["name"] == "smoke")
        topk = next(row for row in adapters if row["name"] == "topk_router")
        reduction = next(row for row in adapters if row["name"] == "reduction_sum")
        reduction_debug = next(row for row in adapters if row["name"] == "reduction_debug")
        routing = next(row for row in adapters if row["name"] == "routing_argmax")
        attention = next(row for row in adapters if row["name"] == "attention_score")
        attention_reformulate = next(row for row in adapters if row["name"] == "attention_reformulate")
        kv_cache = next(row for row in adapters if row["name"] == "kv_cache_gather")
        diagnose = next(row for row in adapters if row["name"] == "profile_diagnose")
        kernelbench = next(row for row in adapters if row["name"] == "kernelbench")
        computeeval = next(row for row in adapters if row["name"] == "computeeval")
        self.assertEqual(smoke["case_count"], 2)
        self.assertIn("case/smoke/eval/v1", smoke["cases"])
        self.assertEqual(topk["case_count"], 1)
        self.assertIn("case/topk_router/eval/v1", topk["cases"])
        self.assertEqual(reduction["case_count"], 1)
        self.assertIn("case/reduction_row_sum/eval/v1", reduction["cases"])
        self.assertEqual(reduction_debug["case_count"], 1)
        self.assertIn("case/reduction_debug/eval/v1", reduction_debug["cases"])
        self.assertEqual(routing["case_count"], 1)
        self.assertIn("case/routing_argmax/eval/v1", routing["cases"])
        self.assertEqual(attention["case_count"], 1)
        self.assertIn("case/attention_score/eval/v1", attention["cases"])
        self.assertEqual(attention_reformulate["case_count"], 1)
        self.assertIn("case/attention_reformulate/eval/v1", attention_reformulate["cases"])
        self.assertEqual(kv_cache["case_count"], 1)
        self.assertIn("case/kv_cache_gather/eval/v1", kv_cache["cases"])
        self.assertEqual(diagnose["case_count"], 1)
        self.assertIn("case/profile_diagnose/eval/v1", diagnose["cases"])
        self.assertEqual(kernelbench["case_count"], 11)
        self.assertIn("case/kernelbench/level1/32_hardtanh/v0_1", kernelbench["cases"])
        self.assertIn("case/kernelbench/level1/23_softmax_wide/v0_1", kernelbench["cases"])
        self.assertEqual(computeeval["case_count"], 8)
        self.assertIn("case/computeeval/2025_1/cuda_10/v1", computeeval["cases"])
        self.assertIn("case/computeeval/2025_1/cuda_16_streams_audit/v1", computeeval["cases"])

    def test_load_smoke_case_and_task(self) -> None:
        adapter = get_adapter("smoke")
        case = adapter.load_case(ROOT, "case/smoke/eval/v1")
        task = adapter.load_task(ROOT, "case/smoke/eval/v1")
        self.assertEqual(case.task_ref, "task/smoke/eval/v1")
        self.assertEqual(task.task_id, "task/smoke/eval/v1")
        self.assertEqual(case.adapter, "smoke")

    def test_list_adapter_cases(self) -> None:
        cases = list_adapter_cases(ROOT, "smoke")
        self.assertEqual({case.case_id for case in cases}, {"case/smoke/diagnose/v1", "case/smoke/eval/v1"})

    def test_load_topk_case_and_task(self) -> None:
        adapter = get_adapter("topk_router")
        case = adapter.load_case(ROOT, "case/topk_router/eval/v1")
        task = adapter.load_task(ROOT, "case/topk_router/eval/v1")
        self.assertEqual(case.task_ref, "task/topk_router/eval/v1")
        self.assertEqual(case.operator_family, "routing_topk")
        self.assertEqual(task.task_id, "task/topk_router/eval/v1")

    def test_load_reduction_case_and_task(self) -> None:
        adapter = get_adapter("reduction_sum")
        case = adapter.load_case(ROOT, "case/reduction_row_sum/eval/v1")
        task = adapter.load_task(ROOT, "case/reduction_row_sum/eval/v1")
        self.assertEqual(case.task_ref, "task/reduction_row_sum/eval/v1")
        self.assertEqual(case.operator_family, "reduction_sum")
        self.assertEqual(task.task_id, "task/reduction_row_sum/eval/v1")

    def test_load_reduction_debug_case_and_task(self) -> None:
        adapter = get_adapter("reduction_debug")
        case = adapter.load_case(ROOT, "case/reduction_debug/eval/v1")
        task = adapter.load_task(ROOT, "case/reduction_debug/eval/v1")
        self.assertEqual(case.task_ref, "task/reduction_debug/eval/v1")
        self.assertEqual(task.verb, "debug")
        self.assertEqual(task.task_id, "task/reduction_debug/eval/v1")

    def test_load_routing_argmax_case_and_task(self) -> None:
        adapter = get_adapter("routing_argmax")
        case = adapter.load_case(ROOT, "case/routing_argmax/eval/v1")
        task = adapter.load_task(ROOT, "case/routing_argmax/eval/v1")
        self.assertEqual(case.task_ref, "task/routing_argmax/eval/v1")
        self.assertEqual(case.operator_family, "routing_argmax")
        self.assertEqual(task.task_id, "task/routing_argmax/eval/v1")

    def test_load_kv_cache_gather_case_and_task(self) -> None:
        adapter = get_adapter("kv_cache_gather")
        case = adapter.load_case(ROOT, "case/kv_cache_gather/eval/v1")
        task = adapter.load_task(ROOT, "case/kv_cache_gather/eval/v1")
        self.assertEqual(case.task_ref, "task/kv_cache_gather/eval/v1")
        self.assertEqual(case.operator_family, "kv_cache_gather")
        self.assertEqual(task.task_id, "task/kv_cache_gather/eval/v1")

    def test_load_attention_score_case_and_task(self) -> None:
        adapter = get_adapter("attention_score")
        case = adapter.load_case(ROOT, "case/attention_score/eval/v1")
        task = adapter.load_task(ROOT, "case/attention_score/eval/v1")
        self.assertEqual(case.task_ref, "task/attention_score/eval/v1")
        self.assertEqual(case.operator_family, "attention_score_tile")
        self.assertEqual(task.task_id, "task/attention_score/eval/v1")

    def test_load_attention_reformulate_case_and_task(self) -> None:
        adapter = get_adapter("attention_reformulate")
        case = adapter.load_case(ROOT, "case/attention_reformulate/eval/v1")
        task = adapter.load_task(ROOT, "case/attention_reformulate/eval/v1")
        self.assertEqual(case.task_ref, "task/attention_reformulate/eval/v1")
        self.assertEqual(task.verb, "reformulate")
        self.assertEqual(task.task_id, "task/attention_reformulate/eval/v1")

    def test_load_profile_diagnose_case_and_task(self) -> None:
        adapter = get_adapter("profile_diagnose")
        case = adapter.load_case(ROOT, "case/profile_diagnose/eval/v1")
        task = adapter.load_task(ROOT, "case/profile_diagnose/eval/v1")
        self.assertEqual(case.task_ref, "task/profile_diagnose/eval/v1")
        self.assertEqual(case.operator_family, "profile_diagnose")
        self.assertEqual(task.task_id, "task/profile_diagnose/eval/v1")

    def test_load_kernelbench_case_and_task(self) -> None:
        adapter = get_adapter("kernelbench")
        case = adapter.load_case(ROOT, "case/kernelbench/level1/32_hardtanh/v0_1")
        task = adapter.load_task(ROOT, "case/kernelbench/level1/32_hardtanh/v0_1")
        self.assertEqual(case.source_benchmark, "KernelBench")
        self.assertEqual(case.source_case_version, "v0.1")
        self.assertEqual(task.task_id, "task/kernelbench/level1/32_hardtanh/eval/v1")

    def test_describe_kernelbench_reports_grouped_coverage(self) -> None:
        summary = describe_adapter(ROOT, "kernelbench")
        self.assertEqual(summary["case_count"], 11)
        self.assertEqual(summary["by_benchmark_level"]["level1"], 11)
        self.assertEqual(summary["by_operator_family"]["normalization_layernorm"], 2)
        self.assertIn("import_manifest", summary)
        self.assertGreaterEqual(summary["by_eval_type"]["correctness-heavy"], 9)
        self.assertGreaterEqual(summary["by_tag"]["attention-adjacent"], 1)
        self.assertEqual(summary["by_tag"]["curated-variant"], 3)

    def test_load_computeeval_case_and_task(self) -> None:
        adapter = get_adapter("computeeval")
        case = adapter.load_case(ROOT, "case/computeeval/2025_1/cuda_31/v1")
        task = adapter.load_task(ROOT, "case/computeeval/2025_1/cuda_31/v1")
        self.assertEqual(case.source_benchmark, "ComputeEval")
        self.assertEqual(case.source_case_version, "2025-1")
        self.assertEqual(case.metadata["official_task_id"], "CUDA/31")
        self.assertEqual(task.task_id, "task/computeeval/cuda_31/eval/v1")

    def test_describe_computeeval_reports_grouped_coverage(self) -> None:
        summary = describe_adapter(ROOT, "computeeval")
        self.assertEqual(summary["case_count"], 8)
        self.assertEqual(summary["by_release"]["2025-1"], 8)
        self.assertGreaterEqual(summary["by_operator_family"]["cuda_kernel_launch"], 1)
        self.assertIn("import_manifest", summary)
        self.assertGreaterEqual(summary["by_eval_type"]["correctness-heavy"], 8)
        self.assertGreaterEqual(summary["by_tag"]["metadata-heavy"], 4)
        self.assertEqual(summary["by_tag"]["curated-variant"], 3)


if __name__ == "__main__":
    unittest.main()
