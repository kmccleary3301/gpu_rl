from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gpu_cockpit.contracts import SFTDatasetManifest, SFTExample, TrajectoryEpisode
from gpu_cockpit.engine.task_registry import TaskRegistry


PROMPT_FAMILY_MAP = {
    "synthesize": "synthesize",
    "optimize": "optimize",
    "diagnose": "diagnose",
    "debug": "debug",
    "reformulate": "reformulate",
    "fuse": "reformulate",
    "port": "reformulate",
}

DEFAULT_ALLOWED_TRAINING_KINDS = {
    "positive_sft_example",
    "positive_rl_trace",
    "negative_debug_example",
    "negative_reformulate_example",
}


def _load_manifest(dataset_dir: Path) -> dict[str, object]:
    manifest_path = dataset_dir / "trajectory_dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing trajectory dataset manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_episodes(dataset_dir: Path) -> list[tuple[str, TrajectoryEpisode]]:
    manifest = _load_manifest(dataset_dir)
    episodes: list[tuple[str, TrajectoryEpisode]] = []
    for relative_ref in manifest.get("episode_refs", []):
        relative_path = str(relative_ref)
        episode_path = dataset_dir / relative_path
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        episodes.append((relative_path, TrajectoryEpisode.model_validate(payload)))
    return episodes


def _render_prompt(task_prompt: str, prompt_family: str, episode: TrajectoryEpisode) -> str:
    action_names = ", ".join(step.action.action_type for step in episode.steps)
    return "\n".join(
        [
            f"Task family: {prompt_family}",
            f"Task id: {episode.task_id}",
            "",
            task_prompt,
            "",
            "Produce a compact tool-using plan for the task and keep the actions consistent with the cockpit tool API.",
            f"Observed reference episode actions: {action_names}",
        ]
    )


def _render_response(episode: TrajectoryEpisode) -> str:
    lines = []
    for step in episode.steps:
        reward_suffix = f" reward={step.reward_total:.3f}"
        transition_suffix = f" transition={step.transition_kind}" if step.transition_kind else ""
        if step.observation.run_id:
            lines.append(f"{step.step_index + 1}. {step.action.action_type} -> run={step.observation.run_id}{transition_suffix}{reward_suffix}")
        else:
            lines.append(f"{step.step_index + 1}. {step.action.action_type}{transition_suffix}{reward_suffix}")
    lines.append(f"terminal_state={episode.terminal_state}")
    return "\n".join(lines)


def _is_public_benchmark(task_id: str) -> bool:
    return task_id.startswith("task/kernelbench/") or task_id.startswith("task/computeeval/")


def _benchmark_name(task_id: str) -> str | None:
    if task_id.startswith("task/kernelbench/"):
        return "KernelBench"
    if task_id.startswith("task/computeeval/"):
        return "ComputeEval"
    return None


def package_trajectory_dataset_as_sft(
    root: Path,
    dataset_dir: Path,
    out_dir: Path,
    *,
    split: str = "train",
    include_failures: bool = True,
    only_public_benchmarks: bool = False,
    include_benchmark_only: bool = False,
    patch_bearing_only: bool = False,
    verb_allowlist: list[str] | None = None,
    governance_allowlist: list[str] | None = None,
    transition_kind_allowlist: list[str] | None = None,
    allowed_training_example_kinds: list[str] | None = None,
) -> Path:
    registry = TaskRegistry(root)
    episodes = _load_episodes(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_refs: list[str] = []
    task_ids: list[str] = []
    prompt_family_counts: dict[str, int] = {}
    verb_counts: dict[str, int] = {}
    operator_family_counts: dict[str, int] = {}
    training_example_kind_counts: dict[str, int] = {}
    episode_governance_counts: dict[str, int] = {}
    patch_kind_counts: dict[str, int] = {}
    transition_kind_counts: dict[str, int] = {}
    patch_bearing_example_count = 0
    allowed_kinds = set(allowed_training_example_kinds or DEFAULT_ALLOWED_TRAINING_KINDS)
    if not include_benchmark_only:
        allowed_kinds.discard("benchmark_only")
    allowed_governance = set(governance_allowlist or [])
    allowed_transitions = set(transition_kind_allowlist or [])
    for relative_ref, episode in episodes:
        if not include_failures and episode.terminal_state != "success":
            continue
        task = registry.get(episode.task_id)
        if only_public_benchmarks and not _is_public_benchmark(task.task_id):
            continue
        if verb_allowlist and task.verb not in set(verb_allowlist):
            continue
        prompt_family = PROMPT_FAMILY_MAP.get(task.verb, "general")
        readiness = str(episode.governance.training_example_kind if episode.governance else episode.metadata.get("training_example_kind", "unusable"))
        if readiness not in allowed_kinds:
            continue
        patch_kinds = sorted(
            {
                str(step.action.metadata["patch_kind"])
                for step in episode.steps
                if isinstance(step.action.metadata.get("patch_kind"), str)
            }
        )
        transition_kinds = [step.transition_kind for step in episode.steps if isinstance(step.transition_kind, str)]
        governance_kind = str(episode.governance.episode_governance_kind if episode.governance else episode.metadata.get("episode_governance_kind", "unusable"))
        governance_reasons = list(episode.governance.reasons) if episode.governance else list(episode.metadata.get("episode_governance_reasons", []))
        if patch_bearing_only and not patch_kinds:
            continue
        if allowed_governance and governance_kind not in allowed_governance:
            continue
        if allowed_transitions and not (allowed_transitions & set(transition_kinds)):
            continue
        example = SFTExample(
            example_id=f"sft_{episode.episode_id}",
            created_at=datetime.now(tz=UTC),
            split=split,
            task_id=episode.task_id,
            prompt_family=prompt_family,
            prompt=_render_prompt(task.prompt, prompt_family, episode),
            response=_render_response(episode),
            source_episode_ref=relative_ref,
            metadata={
                "task_verb": task.verb,
                "operator_family": task.operator_family,
                "backend": task.allowed_backends[0] if task.allowed_backends else None,
                "feature_requirements": list(task.feature_requirements),
                "terminal_state": episode.terminal_state,
                "episode_kind": episode.episode_kind,
                "benchmark_name": _benchmark_name(task.task_id),
                "evidence_score": episode.metadata.get("evidence_score"),
                "training_example_kind": readiness,
                "episode_governance_kind": governance_kind,
                "episode_governance_reasons": governance_reasons,
                "patch_present": bool(patch_kinds),
                "patch_kinds": patch_kinds,
                "transition_kinds": transition_kinds,
                "candidate_lineage_depth": episode.metadata.get("candidate_lineage_depth", 0),
            },
        )
        example_path = examples_dir / f"{example.example_id}.json"
        example_path.write_text(json.dumps(example.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        example_refs.append(str(example_path.relative_to(out_dir)))
        task_ids.append(example.task_id)
        prompt_family_counts[prompt_family] = prompt_family_counts.get(prompt_family, 0) + 1
        verb_counts[task.verb] = verb_counts.get(task.verb, 0) + 1
        operator_family = task.operator_family or "unknown"
        operator_family_counts[operator_family] = operator_family_counts.get(operator_family, 0) + 1
        training_example_kind_counts[readiness] = training_example_kind_counts.get(readiness, 0) + 1
        episode_governance_counts[governance_kind] = episode_governance_counts.get(governance_kind, 0) + 1
        if patch_kinds:
            patch_bearing_example_count += 1
            for patch_kind in patch_kinds:
                patch_kind_counts[patch_kind] = patch_kind_counts.get(patch_kind, 0) + 1
        for transition_kind in transition_kinds:
            transition_kind_counts[transition_kind] = transition_kind_counts.get(transition_kind, 0) + 1
    manifest = SFTDatasetManifest(
        dataset_id=f"sft_dataset_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now(tz=UTC),
        split=split,
        example_count=len(example_refs),
        example_refs=example_refs,
        task_ids=sorted(set(task_ids)),
        metadata={
            "source_dataset": str(dataset_dir),
            "include_failures": include_failures,
            "only_public_benchmarks": only_public_benchmarks,
            "include_benchmark_only": include_benchmark_only,
            "patch_bearing_only": patch_bearing_only,
            "verb_allowlist": list(verb_allowlist or []),
            "governance_allowlist": sorted(allowed_governance),
            "transition_kind_allowlist": sorted(allowed_transitions),
            "allowed_training_example_kinds": sorted(allowed_kinds),
            "prompt_family_counts": prompt_family_counts,
            "verb_counts": verb_counts,
            "operator_family_counts": operator_family_counts,
            "training_example_kind_counts": training_example_kind_counts,
            "episode_governance_counts": episode_governance_counts,
            "patch_kind_counts": patch_kind_counts,
            "transition_kind_counts": transition_kind_counts,
            "patch_bearing_example_count": patch_bearing_example_count,
        },
    )
    manifest_path = out_dir / "sft_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return manifest_path


def validate_sft_dataset(out_dir: Path) -> dict[str, object]:
    manifest_path = out_dir / "sft_dataset_manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "missing": ["sft_dataset_manifest.json"]}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_examples = [
        str(relative_ref)
        for relative_ref in payload.get("example_refs", [])
        if not (out_dir / str(relative_ref)).exists()
    ]
    return {
        "status": "ok" if not missing_examples else "failed",
        "example_count": int(payload.get("example_count", 0)),
        "missing_examples": missing_examples,
        "manifest": payload,
    }
