from __future__ import annotations

from pathlib import Path

from gpu_cockpit.contracts import BenchmarkCaseSpec, TaskSpec
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.workloads.adapters.base import BenchmarkAdapter


class AttentionReformulateBenchmarkAdapter(BenchmarkAdapter):
    name = "attention_reformulate"
    case_dir = Path("workloads/benchmarks/attention_reformulate_cases")

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
        raise KeyError(f"Unknown attention-reformulate benchmark case: {case_id}")

    def load_task(self, root: Path, case_id: str) -> TaskSpec:
        case = self.load_case(root, case_id)
        registry = TaskRegistry(root)
        return registry.get(case.task_ref)
