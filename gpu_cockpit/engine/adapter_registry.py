from __future__ import annotations

from pathlib import Path

from gpu_cockpit.contracts import BenchmarkCaseSpec
from gpu_cockpit.workloads.adapters.base import BenchmarkAdapter
from gpu_cockpit.workloads.adapters.attention_score import AttentionScoreBenchmarkAdapter
from gpu_cockpit.workloads.adapters.attention_reformulate import AttentionReformulateBenchmarkAdapter
from gpu_cockpit.workloads.adapters.computeeval import ComputeEvalBenchmarkAdapter
from gpu_cockpit.workloads.adapters.kv_cache_gather import KVCacheGatherBenchmarkAdapter
from gpu_cockpit.workloads.adapters.kernelbench import KernelBenchBenchmarkAdapter
from gpu_cockpit.workloads.adapters.profile_diagnose import ProfileDiagnoseBenchmarkAdapter
from gpu_cockpit.workloads.adapters.reduction_debug import ReductionDebugBenchmarkAdapter
from gpu_cockpit.workloads.adapters.reduction_sum import ReductionSumBenchmarkAdapter
from gpu_cockpit.workloads.adapters.routing_argmax import RoutingArgmaxBenchmarkAdapter
from gpu_cockpit.workloads.adapters.smoke import SmokeBenchmarkAdapter
from gpu_cockpit.workloads.adapters.topk_router import TopKRouterBenchmarkAdapter


def _all_adapters() -> list[BenchmarkAdapter]:
    return [
        SmokeBenchmarkAdapter(),
        TopKRouterBenchmarkAdapter(),
        ReductionSumBenchmarkAdapter(),
        ReductionDebugBenchmarkAdapter(),
        RoutingArgmaxBenchmarkAdapter(),
        AttentionScoreBenchmarkAdapter(),
        AttentionReformulateBenchmarkAdapter(),
        KVCacheGatherBenchmarkAdapter(),
        ProfileDiagnoseBenchmarkAdapter(),
        KernelBenchBenchmarkAdapter(),
        ComputeEvalBenchmarkAdapter(),
    ]


def get_adapter(name: str) -> BenchmarkAdapter:
    for adapter in _all_adapters():
        if adapter.name == name:
            return adapter
    raise KeyError(f"Unknown adapter: {name}")


def list_adapters(root: Path) -> list[dict[str, object]]:
    return [adapter.describe(root) for adapter in _all_adapters()]


def list_adapter_cases(root: Path, adapter_name: str) -> list[BenchmarkCaseSpec]:
    return get_adapter(adapter_name).list_cases(root)


def describe_adapter(root: Path, adapter_name: str) -> dict[str, object]:
    return get_adapter(adapter_name).describe(root)
