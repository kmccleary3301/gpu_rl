from __future__ import annotations

import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "workloads" / "public_benchmarks" / "computeeval" / "2025_1"
RELEASE = "2025-1"
RELEASE_URL = "https://raw.githubusercontent.com/NVIDIA/compute-eval/main/data/releases/2025-1-problems.tar.gz"
CURATED_TASK_IDS = ["CUDA/0", "CUDA/10", "CUDA/16", "CUDA/31", "CUDA/35"]


def _safe_problem_name(task_id: str) -> str:
    return task_id.lower().replace("/", "_")


def _load_release_rows() -> list[dict[str, object]]:
    fd, archive_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    urllib.request.urlretrieve(RELEASE_URL, archive_path)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.extractfile("problems.jsonl")
            if member is None:
                raise RuntimeError("ComputeEval release archive is missing problems.jsonl")
            return [json.loads(line) for line in member.read().decode("utf-8").splitlines() if line.strip()]
    finally:
        Path(archive_path).unlink(missing_ok=True)


def main() -> None:
    rows = _load_release_rows()
    selected = [row for row in rows if row.get("task_id") in CURATED_TASK_IDS]
    if len(selected) != len(CURATED_TASK_IDS):
        found = {str(row.get("task_id")) for row in selected}
        missing = [task_id for task_id in CURATED_TASK_IDS if task_id not in found]
        raise RuntimeError(f"Missing curated ComputeEval task ids from release {RELEASE}: {missing}")

    problems_dir = DESTINATION / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    manifest_cases: list[dict[str, object]] = []

    for row in sorted(selected, key=lambda item: str(item["task_id"])):
        task_id = str(row["task_id"])
        problem_name = _safe_problem_name(task_id)
        out_path = problems_dir / f"{problem_name}.json"
        out_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        written_paths.append(str(out_path.relative_to(ROOT)))
        metadata = row.get("metadata", {})
        manifest_cases.append(
            {
                "task_id": task_id,
                "path": str(out_path.relative_to(ROOT)),
                "difficulty": metadata.get("difficulty"),
                "tags": metadata.get("tags", []),
                "min_cuda_toolkit": row.get("min_cuda_toolkit"),
                "type": row.get("type"),
            }
        )

    manifest = {
        "benchmark_name": "ComputeEval",
        "benchmark_version": RELEASE,
        "source_repo": "https://github.com/NVIDIA/compute-eval",
        "release_url": RELEASE_URL,
        "release_format": "datapack",
        "curation_batch": "computeeval_2025_1_batch1",
        "case_count": len(manifest_cases),
        "included_cases": manifest_cases,
    }
    manifest_path = DESTINATION / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "written_paths": written_paths}, indent=2))


if __name__ == "__main__":
    main()
