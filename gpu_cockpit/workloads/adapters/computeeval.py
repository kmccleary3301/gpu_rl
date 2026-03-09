from __future__ import annotations

import json
from pathlib import Path

from gpu_cockpit.contracts import BenchmarkCaseSpec, TaskSpec
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.workloads.adapters.base import BenchmarkAdapter


class ComputeEvalBenchmarkAdapter(BenchmarkAdapter):
    name = "computeeval"
    case_dir = Path("workloads/benchmarks/computeeval_cases")
    manifest_path = Path("workloads/public_benchmarks/computeeval/2025_1/manifest.json")

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
        raise KeyError(f"Unknown ComputeEval benchmark case: {case_id}")

    def load_task(self, root: Path, case_id: str) -> TaskSpec:
        case = self.load_case(root, case_id)
        return TaskRegistry(root).get(case.task_ref)

    def describe(self, root: Path) -> dict[str, object]:
        payload = super().describe(root)
        manifest_file = root / self.manifest_path
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else None
        release_counts: dict[str, int] = {}
        eval_type_counts: dict[str, int] = {}
        operator_family_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}
        for case in self.list_cases(root):
            release = str(case.metadata.get("release", case.source_case_version or "unknown"))
            release_counts[release] = release_counts.get(release, 0) + 1
            operator_family_counts[case.operator_family] = operator_family_counts.get(case.operator_family, 0) + 1
            for eval_type in case.metadata.get("eval_types", []):
                key = str(eval_type)
                eval_type_counts[key] = eval_type_counts.get(key, 0) + 1
            for tag in case.tags:
                if tag in {"public-benchmark", "computeeval"}:
                    continue
                tag_text = str(tag)
                tag_counts[tag_text] = tag_counts.get(tag_text, 0) + 1
        payload.update(
            {
                "import_manifest": manifest,
                "by_release": release_counts,
                "by_operator_family": operator_family_counts,
                "by_eval_type": eval_type_counts,
                "by_tag": tag_counts,
            }
        )
        return payload
