from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gpu_cockpit.contracts import BenchmarkCaseSpec, TaskSpec


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def list_cases(self, root: Path) -> list[BenchmarkCaseSpec]:
        raise NotImplementedError

    @abstractmethod
    def load_case(self, root: Path, case_id: str) -> BenchmarkCaseSpec:
        raise NotImplementedError

    @abstractmethod
    def load_task(self, root: Path, case_id: str) -> TaskSpec:
        raise NotImplementedError

    def describe(self, root: Path) -> dict[str, object]:
        cases = self.list_cases(root)
        by_difficulty: dict[str, int] = {}
        by_operator_family: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        source_benchmarks: dict[str, int] = {}
        for case in cases:
            if case.difficulty:
                by_difficulty[case.difficulty] = by_difficulty.get(case.difficulty, 0) + 1
            if case.operator_family:
                by_operator_family[case.operator_family] = by_operator_family.get(case.operator_family, 0) + 1
            if case.source_benchmark:
                source_benchmarks[case.source_benchmark] = source_benchmarks.get(case.source_benchmark, 0) + 1
            for tag in case.tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
        return {
            "name": self.name,
            "case_count": len(cases),
            "cases": [case.case_id for case in cases],
            "by_difficulty": by_difficulty,
            "by_operator_family": by_operator_family,
            "by_tag": by_tag,
            "source_benchmarks": source_benchmarks,
        }
