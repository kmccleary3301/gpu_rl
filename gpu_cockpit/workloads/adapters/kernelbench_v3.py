from __future__ import annotations

import json
from pathlib import Path

from gpu_cockpit.contracts import BenchmarkCaseSpec, TaskSpec
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.workloads.adapters.base import BenchmarkAdapter


class KernelBenchV3BenchmarkAdapter(BenchmarkAdapter):
    name = "kernelbench_v3"
    case_dir = Path("workloads/benchmarks/kernelbench_v3_cases")
    manifest_path = Path("workloads/public_benchmarks/kernelbench_v3/v3_1/manifest.json")

    def _case_paths(self, root: Path) -> list[Path]:
        directory = root / self.case_dir
        if not directory.exists():
            return []
        return sorted(directory.glob("*.json"))

    def list_cases(self, root: Path) -> list[BenchmarkCaseSpec]:
        return [BenchmarkCaseSpec.model_validate_json(path.read_text(encoding="utf-8")) for path in self._case_paths(root)]

    def load_case(self, root: Path, case_id: str) -> BenchmarkCaseSpec:
        for case in self.list_cases(root):
            if case.case_id == case_id:
                return case
        raise KeyError(f"Unknown KernelBench-v3 benchmark case: {case_id}")

    def load_task(self, root: Path, case_id: str) -> TaskSpec:
        case = self.load_case(root, case_id)
        registry = TaskRegistry(root)
        return registry.get(case.task_ref)

    def describe(self, root: Path) -> dict[str, object]:
        payload = super().describe(root)
        manifest_file = root / self.manifest_path
        manifest = None
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        by_level: dict[str, int] = {}
        by_provenance_kind: dict[str, int] = {}
        by_track: dict[str, int] = {}
        for case in self.list_cases(root):
            benchmark_level = str(case.metadata.get("benchmark_level", case.difficulty or "unknown"))
            provenance_kind = str(case.metadata.get("provenance_kind", "unknown"))
            track = str(case.metadata.get("official_track", "unknown"))
            by_level[benchmark_level] = by_level.get(benchmark_level, 0) + 1
            by_provenance_kind[provenance_kind] = by_provenance_kind.get(provenance_kind, 0) + 1
            by_track[track] = by_track.get(track, 0) + 1
        payload.update(
            {
                "import_manifest": manifest,
                "by_benchmark_level": by_level,
                "by_provenance_kind": by_provenance_kind,
                "by_track": by_track,
            }
        )
        return payload
