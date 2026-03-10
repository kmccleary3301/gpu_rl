from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import run_scripted_reference_episode
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft
from gpu_cockpit.engine.trajectory import export_episode_dataset, write_episode


def _copy_run_dir(run_ref: str, destination: Path) -> None:
    src = Path(run_ref)
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(src, destination)


def _rewrite_episode_source_run(episode, source_run_ref: str):
    return episode.model_copy(update={"source_run_ref": source_run_ref})


def _rewrite_episode_run_ref_map(episode):
    run_ref_map = episode.metadata.get("run_ref_map", {})
    rewritten: dict[str, str] = {}
    if isinstance(run_ref_map, dict):
        for run_id, run_ref in run_ref_map.items():
            if not isinstance(run_id, str) or not isinstance(run_ref, str):
                continue
            destination = ROOT / "tests" / "golden_runs" / run_id
            _copy_run_dir(run_ref, destination)
            rewritten[run_id] = f"../../golden_runs/{run_id}"
    metadata = dict(episode.metadata)
    metadata["run_ref_map"] = rewritten
    return episode.model_copy(update={"metadata": metadata})


def main() -> int:
    tmp_root = ROOT / "tests" / "tmp_transition_goldens"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "workloads", tmp_root / "workloads")
    shutil.copytree(ROOT / "knowledge", tmp_root / "knowledge")
    build_knowledge_index(tmp_root)

    debug_episode = run_scripted_reference_episode(
        tmp_root,
        "task/reduction_debug/eval/v1",
        ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
        section="quality",
        include_build=True,
        triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
    )
    reformulate_episode = run_scripted_reference_episode(
        tmp_root,
        "task/attention_reformulate/eval/v1",
        ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
        section="quality",
        step_budget=12,
    )
    debug_negative_episode = run_scripted_reference_episode(
        tmp_root,
        "task/reduction_debug/eval/v1",
        ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
        section="quality",
        include_build=True,
        triton_build_spec="workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
        workflow="debug_negative",
        step_budget=12,
    )
    reformulate_negative_episode = run_scripted_reference_episode(
        tmp_root,
        "task/attention_reformulate/eval/v1",
        ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
        section="quality",
        workflow="reformulate_negative",
        step_budget=12,
    )

    debug_eval_destination = ROOT / "tests" / "golden_runs" / "reduction_debug_patch_eval_v1"
    reformulate_eval_destination = ROOT / "tests" / "golden_runs" / "attention_reformulate_patch_eval_v1"
    debug_negative_destination = ROOT / "tests" / "golden_runs" / "reduction_debug_negative_eval_v1"
    reformulate_negative_destination = ROOT / "tests" / "golden_runs" / "attention_reformulate_negative_eval_v1"
    _copy_run_dir(debug_episode.source_run_ref, debug_eval_destination)
    _copy_run_dir(reformulate_episode.source_run_ref, reformulate_eval_destination)
    _copy_run_dir(debug_negative_episode.source_run_ref, debug_negative_destination)
    _copy_run_dir(reformulate_negative_episode.source_run_ref, reformulate_negative_destination)
    debug_episode = _rewrite_episode_source_run(debug_episode, "../../golden_runs/reduction_debug_patch_eval_v1")
    reformulate_episode = _rewrite_episode_source_run(reformulate_episode, "../../golden_runs/attention_reformulate_patch_eval_v1")
    debug_negative_episode = _rewrite_episode_source_run(debug_negative_episode, "../../golden_runs/reduction_debug_negative_eval_v1")
    reformulate_negative_episode = _rewrite_episode_source_run(reformulate_negative_episode, "../../golden_runs/attention_reformulate_negative_eval_v1")
    debug_episode = _rewrite_episode_run_ref_map(debug_episode)
    reformulate_episode = _rewrite_episode_run_ref_map(reformulate_episode)
    debug_negative_episode = _rewrite_episode_run_ref_map(debug_negative_episode)
    reformulate_negative_episode = _rewrite_episode_run_ref_map(reformulate_negative_episode)

    write_episode(debug_episode, ROOT / "tests" / "golden_episodes" / "reduction_debug_patch_episode_v1.json")
    write_episode(reformulate_episode, ROOT / "tests" / "golden_episodes" / "attention_reformulate_patch_episode_v1.json")
    write_episode(debug_negative_episode, ROOT / "tests" / "golden_episodes" / "reduction_debug_negative_episode_v1.json")
    write_episode(reformulate_negative_episode, ROOT / "tests" / "golden_episodes" / "attention_reformulate_negative_episode_v1.json")

    transition_dataset_dir = ROOT / "tests" / "golden_datasets" / "transition_collection_v1"
    if transition_dataset_dir.exists():
        shutil.rmtree(transition_dataset_dir, ignore_errors=True)
    export_episode_dataset(
        [debug_episode, reformulate_episode],
        transition_dataset_dir,
        policy_id="scripted_reference_v1",
        split="seed",
        metadata={"fixture": "transition_collection_v1"},
    )

    negative_transition_dataset_dir = ROOT / "tests" / "golden_datasets" / "transition_negative_collection_v1"
    if negative_transition_dataset_dir.exists():
        shutil.rmtree(negative_transition_dataset_dir, ignore_errors=True)
    export_episode_dataset(
        [debug_negative_episode, reformulate_negative_episode],
        negative_transition_dataset_dir,
        policy_id="scripted_reference_v1",
        split="seed",
        metadata={"fixture": "transition_negative_collection_v1"},
    )

    transition_sft_dir = ROOT / "tests" / "golden_datasets" / "transition_sft_v1"
    if transition_sft_dir.exists():
        shutil.rmtree(transition_sft_dir, ignore_errors=True)
    package_trajectory_dataset_as_sft(
        ROOT,
        transition_dataset_dir,
        transition_sft_dir,
        split="train",
        patch_bearing_only=True,
        governance_allowlist=["usable_positive_sft"],
        transition_kind_allowlist=["repaired", "reformulated"],
    )

    negative_transition_sft_dir = ROOT / "tests" / "golden_datasets" / "transition_negative_sft_v1"
    if negative_transition_sft_dir.exists():
        shutil.rmtree(negative_transition_sft_dir, ignore_errors=True)
    package_trajectory_dataset_as_sft(
        ROOT,
        negative_transition_dataset_dir,
        negative_transition_sft_dir,
        split="train",
        patch_bearing_only=True,
        governance_allowlist=["usable_negative_debug", "usable_negative_transition"],
        transition_kind_allowlist=["patch_applied"],
    )

    current_candidate_run_ref = str(debug_episode.metadata.get("current_candidate_run_ref", ""))
    if current_candidate_run_ref:
        _copy_run_dir(current_candidate_run_ref, ROOT / "tests" / "golden_runs" / "reduction_debug_patch_transition_v1")

    shutil.rmtree(tmp_root, ignore_errors=True)
    print("tests/golden_episodes/reduction_debug_patch_episode_v1.json")
    print("tests/golden_episodes/attention_reformulate_patch_episode_v1.json")
    print("tests/golden_episodes/reduction_debug_negative_episode_v1.json")
    print("tests/golden_episodes/attention_reformulate_negative_episode_v1.json")
    print("tests/golden_datasets/transition_collection_v1")
    print("tests/golden_datasets/transition_negative_collection_v1")
    print("tests/golden_datasets/transition_sft_v1")
    print("tests/golden_datasets/transition_negative_sft_v1")
    print("tests/golden_runs/reduction_debug_patch_transition_v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
