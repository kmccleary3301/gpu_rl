from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import run_scripted_reference_episode
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.trajectory import export_episode_dataset


CURATED_REFERENCE_EPISODES = [
    {
        "task": "task/reduction_row_sum/eval/v1",
        "command": ["python3", "workloads/reference/triton_row_sum_candidate.py", "--benchmark-repeats", "5"],
        "section": "eval",
        "include_build": True,
        "triton_build_spec": "workloads/reference/triton_row_sum_kernel.py:get_build_spec",
    },
    {
        "task": "task/profile_diagnose/eval/v1",
        "command": ["python3", "workloads/reference/profile_diagnose_candidate.py"],
        "section": "quality",
    },
    {
        "task": "task/reduction_debug/eval/v1",
        "command": ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
        "section": "quality",
        "include_build": True,
        "triton_build_spec": "workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
    },
    {
        "task": "task/attention_reformulate/eval/v1",
        "command": ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
        "section": "quality",
        "step_budget": 6,
    },
    {
        "task": "task/kernelbench/level1/32_hardtanh/eval/v1",
        "command": [
            "python3",
            "workloads/reference/kernelbench_reference_runner.py",
            "--case-config",
            "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_032_hardtanh.json",
            "--benchmark-repeats",
            "2",
        ],
        "section": "eval",
    },
    {
        "task": "task/computeeval/cuda_0/eval/v1",
        "command": [
            "python3",
            "workloads/reference/computeeval_reference_runner.py",
            "--problem",
            "workloads/public_benchmarks/computeeval/2025_1/problems/cuda_0.json",
            "--benchmark-repeats",
            "2",
        ],
        "section": "eval",
    },
]


def main() -> None:
    build_knowledge_index(ROOT)
    episodes = []
    for spec in CURATED_REFERENCE_EPISODES:
        episodes.append(
            run_scripted_reference_episode(
                ROOT,
                spec["task"],
                list(spec["command"]),
                policy_id="scripted_reference_v2",
                step_budget=int(spec.get("step_budget", 6)),
                section=str(spec.get("section", "quality")),
                include_build=bool(spec.get("include_build", False)),
                triton_build_spec=spec.get("triton_build_spec"),
            )
        )
    out_dir = ROOT / "datasets" / "seed_reference_v2"
    manifest_path = export_episode_dataset(
        episodes,
        out_dir,
        policy_id="scripted_reference_v2",
        split="seed",
        metadata={
            "collection_kind": "scripted_multi_verb_reference",
            "tasks": [spec["task"] for spec in CURATED_REFERENCE_EPISODES],
        },
    )
    print(
        json.dumps(
            {
                "episode_count": len(episodes),
                "tasks": [spec["task"] for spec in CURATED_REFERENCE_EPISODES],
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
