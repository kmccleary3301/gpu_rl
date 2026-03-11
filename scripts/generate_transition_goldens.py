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


def _rewrite_episode_identity(episode, episode_id: str):
    return episode.model_copy(update={"episode_id": episode_id})


def _rewrite_refs_in_value(payload, ref_map: dict[str, str]):
    if isinstance(payload, str):
        return ref_map.get(payload, payload)
    if isinstance(payload, list):
        return [_rewrite_refs_in_value(item, ref_map) for item in payload]
    if isinstance(payload, dict):
        return {key: _rewrite_refs_in_value(value, ref_map) for key, value in payload.items()}
    return payload


def _rewrite_episode_refs(episode, ref_map: dict[str, str]):
    payload = episode.model_dump(mode="json")
    rewritten = _rewrite_refs_in_value(payload, ref_map)
    return type(episode).model_validate(rewritten)


def _rewrite_run_dir_refs(run_dir: Path, ref_map: dict[str, str]) -> None:
    for path in run_dir.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewritten = _rewrite_refs_in_value(payload, ref_map)
        path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")


def _remove_stale_timestamped_goldens() -> None:
    golden_runs_dir = ROOT / "tests" / "golden_runs"
    for path in golden_runs_dir.iterdir():
        if path.name.startswith(("run_20", "bench_20", "eval_20", "patch_20")):
            shutil.rmtree(path, ignore_errors=True)

    golden_dataset_dirs = [
        ROOT / "tests" / "golden_datasets" / "transition_collection_v1" / "episodes",
        ROOT / "tests" / "golden_datasets" / "transition_negative_collection_v1" / "episodes",
        ROOT / "tests" / "golden_datasets" / "transition_sft_v1" / "examples",
        ROOT / "tests" / "golden_datasets" / "transition_negative_sft_v1" / "examples",
    ]
    for directory in golden_dataset_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.name.startswith(("env_episode_20", "sft_env_episode_20")):
                path.unlink(missing_ok=True)


def _canonicalize_golden_run_refs(tmp_root: Path) -> None:
    golden_runs_dir = ROOT / "tests" / "golden_runs"
    ref_map: dict[str, str] = {}
    for path in golden_runs_dir.iterdir():
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        ref_map[str(tmp_root / "runs" / run_id)] = f"tests/golden_runs/{path.name}"
    for path in golden_runs_dir.iterdir():
        if path.is_dir():
            _rewrite_run_dir_refs(path, ref_map)


def _rewrite_episode_run_ref_map(episode, fixture_prefix: str, current_candidate_ref: str | None = None):
    run_ref_map = episode.metadata.get("run_ref_map", {})
    rewritten: dict[str, str] = {}
    original_ref_to_rewritten: dict[str, str] = {}
    original_ref_to_destination: dict[str, Path] = {}
    prefix_counts: dict[str, int] = {}
    if isinstance(run_ref_map, dict):
        for run_id, run_ref in run_ref_map.items():
            if not isinstance(run_id, str) or not isinstance(run_ref, str):
                continue
            prefix = run_id.split("_", 1)[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            alias = f"{fixture_prefix}_{prefix}_{prefix_counts[prefix]:02d}"
            destination = ROOT / "tests" / "golden_runs" / alias
            _copy_run_dir(run_ref, destination)
            rewritten[alias] = f"../../golden_runs/{alias}"
            original_ref_to_rewritten[run_ref] = f"../../golden_runs/{alias}"
            original_ref_to_destination[run_ref] = destination
        for destination in original_ref_to_destination.values():
            _rewrite_run_dir_refs(destination, original_ref_to_rewritten)
    metadata = dict(episode.metadata)
    metadata["run_ref_map"] = rewritten
    if current_candidate_ref and current_candidate_ref in original_ref_to_rewritten:
        metadata["current_candidate_run_ref"] = original_ref_to_rewritten[current_candidate_ref]
    rewritten_episode = episode.model_copy(update={"metadata": metadata})
    return _rewrite_episode_refs(rewritten_episode, original_ref_to_rewritten), original_ref_to_rewritten


def main() -> int:
    _remove_stale_timestamped_goldens()

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
    stable_ref_map = {
        str(debug_episode.source_run_ref): "../../golden_runs/reduction_debug_patch_eval_v1",
        str(reformulate_episode.source_run_ref): "../../golden_runs/attention_reformulate_patch_eval_v1",
        str(debug_negative_episode.source_run_ref): "../../golden_runs/reduction_debug_negative_eval_v1",
        str(reformulate_negative_episode.source_run_ref): "../../golden_runs/attention_reformulate_negative_eval_v1",
    }
    debug_episode = _rewrite_episode_source_run(debug_episode, "../../golden_runs/reduction_debug_patch_eval_v1")
    reformulate_episode = _rewrite_episode_source_run(reformulate_episode, "../../golden_runs/attention_reformulate_patch_eval_v1")
    debug_negative_episode = _rewrite_episode_source_run(debug_negative_episode, "../../golden_runs/reduction_debug_negative_eval_v1")
    reformulate_negative_episode = _rewrite_episode_source_run(reformulate_negative_episode, "../../golden_runs/attention_reformulate_negative_eval_v1")
    current_candidate_run_ref = str(debug_episode.metadata.get("current_candidate_run_ref", ""))
    if current_candidate_run_ref:
        _copy_run_dir(current_candidate_run_ref, ROOT / "tests" / "golden_runs" / "reduction_debug_patch_transition_v1")
        stable_ref_map[current_candidate_run_ref] = "../../golden_runs/reduction_debug_patch_transition_v1"

    debug_episode, debug_run_ref_map = _rewrite_episode_run_ref_map(
        debug_episode,
        "reduction_debug_patch",
        current_candidate_ref=current_candidate_run_ref or None,
    )
    if current_candidate_run_ref:
        debug_metadata = dict(debug_episode.metadata)
        debug_metadata["current_candidate_run_ref"] = "../../golden_runs/reduction_debug_patch_transition_v1"
        debug_episode = debug_episode.model_copy(update={"metadata": debug_metadata})
    reformulate_episode, reformulate_run_ref_map = _rewrite_episode_run_ref_map(
        reformulate_episode,
        "attention_reformulate_patch",
        current_candidate_ref=str(reformulate_episode.metadata.get("current_candidate_run_ref", "")) or None,
    )
    debug_negative_episode, debug_negative_run_ref_map = _rewrite_episode_run_ref_map(
        debug_negative_episode,
        "reduction_debug_negative",
        current_candidate_ref=str(debug_negative_episode.metadata.get("current_candidate_run_ref", "")) or None,
    )
    reformulate_negative_episode, reformulate_negative_run_ref_map = _rewrite_episode_run_ref_map(
        reformulate_negative_episode,
        "attention_reformulate_negative",
        current_candidate_ref=str(reformulate_negative_episode.metadata.get("current_candidate_run_ref", "")) or None,
    )
    all_ref_maps = {}
    for mapping in (
        debug_run_ref_map,
        reformulate_run_ref_map,
        debug_negative_run_ref_map,
        reformulate_negative_run_ref_map,
        stable_ref_map,
    ):
        all_ref_maps.update(mapping)
    _rewrite_run_dir_refs(debug_eval_destination, all_ref_maps)
    _rewrite_run_dir_refs(reformulate_eval_destination, all_ref_maps)
    _rewrite_run_dir_refs(debug_negative_destination, all_ref_maps)
    _rewrite_run_dir_refs(reformulate_negative_destination, all_ref_maps)
    if current_candidate_run_ref:
        _rewrite_run_dir_refs(ROOT / "tests" / "golden_runs" / "reduction_debug_patch_transition_v1", all_ref_maps)

    debug_episode = _rewrite_episode_identity(debug_episode, "fixture_reduction_debug_patch_episode_v1")
    reformulate_episode = _rewrite_episode_identity(reformulate_episode, "fixture_attention_reformulate_patch_episode_v1")
    debug_negative_episode = _rewrite_episode_identity(debug_negative_episode, "fixture_reduction_debug_negative_episode_v1")
    reformulate_negative_episode = _rewrite_episode_identity(reformulate_negative_episode, "fixture_attention_reformulate_negative_episode_v1")

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

    _canonicalize_golden_run_refs(tmp_root)

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
