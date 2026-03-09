from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_problem(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def visible_summary(problem: dict[str, Any]) -> dict[str, Any]:
    context_files = problem.get("context_files", [])
    return {
        "prompt_sha256": _sha256_text(str(problem.get("prompt", ""))),
        "build_command": problem.get("build_command"),
        "test_command": problem.get("test_command"),
        "min_cuda_toolkit": problem.get("min_cuda_toolkit"),
        "arch_list": problem.get("arch_list", []),
        "context_file_count": len(context_files),
        "context_paths": [item["path"] for item in context_files],
        "context_digest": _sha256_payload(context_files),
    }


def hidden_summary(problem: dict[str, Any]) -> dict[str, Any]:
    test_files = problem.get("test_files", [])
    metadata = problem.get("metadata", {})
    return {
        "test_file_count": len(test_files),
        "test_paths": [item["path"] for item in test_files],
        "test_digest": _sha256_payload(test_files),
        "source_references": problem.get("source_references"),
        "timeout_seconds": problem.get("timeout_seconds"),
        "difficulty": metadata.get("difficulty"),
        "tags": metadata.get("tags", []),
        "release_tags": metadata.get("releases", []),
    }


def _benchmark(problem: dict[str, Any], repeats: int) -> None:
    for _ in range(repeats):
        _ = visible_summary(problem)
        _ = hidden_summary(problem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    problem_path = Path(args.problem).resolve()
    problem = _load_problem(problem_path)
    _benchmark(problem, args.benchmark_repeats)
    print(
        json.dumps(
            {
                "benchmark_source": "computeeval",
                "benchmark_release": "2025-1",
                "benchmark_task_id": problem["task_id"],
                "problem_path": str(problem_path),
                "visible_result": visible_summary(problem),
                "hidden_result": hidden_summary(problem),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
