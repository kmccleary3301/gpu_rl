from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import (
    EpisodeReadinessReport,
    ReadinessDecision,
    TrajectoryAction,
    TrajectoryDatasetManifest,
    TrajectoryEpisode,
    TrajectoryObservation,
    TrajectoryStep,
)
from gpu_cockpit.engine.inspector import inspect_run, resolve_run_dir
from gpu_cockpit.engine.task_registry import TaskRegistry


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_episode_run_ref(dataset_dir: Path, run_ref: str) -> Path:
    path = Path(run_ref)
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _environment_hash(run_dir: Path) -> str | None:
    environment_path = run_dir / "replay" / "environment.json"
    if not environment_path.exists():
        return None
    return hashlib.sha256(environment_path.read_bytes()).hexdigest()


def _infer_action_type(run_id: str) -> str:
    if run_id.startswith("eval_"):
        return "eval"
    if run_id.startswith("bench_"):
        return "bench"
    return "run"


def _derive_reward(summary_payload: dict[str, Any], run_dir: Path) -> tuple[dict[str, float], float]:
    envelope = _load_optional_json(run_dir / "eval" / "eval_envelope.json") or {}
    reward_components = envelope.get("reward_components")
    if isinstance(reward_components, dict):
        normalized = {
            str(key): float(value)
            for key, value in reward_components.items()
            if isinstance(value, (int, float))
        }
    else:
        normalized = {}
    final_score = envelope.get("final_score")
    if isinstance(final_score, (int, float)):
        return normalized, float(final_score)
    return normalized or {"completion": 1.0 if summary_payload.get("status") == "ok" else 0.0}, 1.0 if summary_payload.get("status") == "ok" else 0.0


def _select_projection_section(summary_payload: dict[str, Any], section: str) -> dict[str, Any]:
    projection = summary_payload.get("projection", {})
    if not isinstance(projection, dict):
        projection = {}
    if section == "full":
        return projection
    if section == "summary":
        return {
            "run": {
                key: summary_payload.get(key)
                for key in [
                    "run_id",
                    "task_id",
                    "status",
                    "trace_enabled",
                    "backend",
                    "vendor",
                    "parent_run_id",
                    "candidate_id",
                    "parent_candidate_id",
                    "patch_present",
                    "patch_kind",
                    "transition_kind",
                    "candidate_role",
                    "exit_code",
                    "duration_ms",
                    "key_artifacts",
                ]
            }
        }
    section_map = {
        "build": ["build_record", "tri_view", "build_projection"],
        "eval": ["correctness_summary", "determinism_summary", "anti_hack_summary", "eval_envelope", "gate_summary"],
        "profile": ["profile_summary", "sanitizer_summary", "bottleneck_card"],
        "replay": ["replay_validation", "replay_pack"],
        "quality": ["evidence_quality", "training_readiness"],
    }
    keys = section_map.get(section)
    if keys is None:
        raise ValueError(f"Unknown inspect section: {section}")
    return {key: projection.get(key) for key in keys}


def write_episode(episode: TrajectoryEpisode, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(episode.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return out_path


def _episode_terminal_state(episode: TrajectoryEpisode) -> str:
    if episode.terminal_state:
        return episode.terminal_state
    if episode.steps and episode.steps[-1].terminal_state:
        return str(episode.steps[-1].terminal_state)
    return "unknown"


def _count_by(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def _episode_has_patch(episode: TrajectoryEpisode) -> bool:
    return any(step.action.action_type == "patch_candidate" for step in episode.steps)


def _episode_lineage_issues(episode: TrajectoryEpisode) -> list[str]:
    issues: list[str] = []
    current_candidate_id: str | None = None
    for step in episode.steps:
        if current_candidate_id is not None and step.input_candidate_id not in {None, current_candidate_id}:
            issues.append(f"input_candidate_mismatch:{step.step_index}")
        if step.action.action_type == "patch_candidate" and not step.output_candidate_id:
            issues.append(f"patch_missing_output_candidate:{step.step_index}")
        if step.output_candidate_id:
            current_candidate_id = step.output_candidate_id
    return issues


def _governance_from_run_evidence(evidence_quality: dict[str, Any]) -> EpisodeReadinessReport:
    training_example_kind = str(evidence_quality.get("training_example_kind", "unusable"))
    if training_example_kind in {"positive_sft_example", "positive_rl_trace"}:
        governance_kind = "usable_positive_sft"
    elif training_example_kind == "negative_debug_example":
        governance_kind = "usable_negative_debug"
    elif training_example_kind == "benchmark_only":
        governance_kind = "benchmark_only"
    else:
        governance_kind = "unusable"
    benchmark_payload = evidence_quality.get("benchmark_reporting", {})
    sft_payload = evidence_quality.get("sft_collection", {})
    rl_payload = evidence_quality.get("rl_reward_trace", {})
    return EpisodeReadinessReport(
        episode_governance_kind=governance_kind,
        training_example_kind=training_example_kind,
        benchmark_collection=ReadinessDecision(
            eligible=bool(benchmark_payload.get("eligible", False)),
            reasons=[str(reason) for reason in benchmark_payload.get("reasons", [])],
        ),
        sft_collection=ReadinessDecision(
            eligible=bool(sft_payload.get("eligible", False)),
            reasons=[str(reason) for reason in sft_payload.get("reasons", [])],
        ),
        rl_reward_trace=ReadinessDecision(
            eligible=bool(rl_payload.get("eligible", False)),
            reasons=[str(reason) for reason in rl_payload.get("reasons", [])],
        ),
        has_build_evidence=bool(evidence_quality.get("build_completeness", 0.0)),
        has_profile_evidence=bool(evidence_quality.get("profile_completeness", 0.0)),
        patch_bearing=False,
        reasons=[str(reason) for reason in evidence_quality.get("notes", [])],
        notes=[f"evidence_score:{evidence_quality.get('overall_score', 0.0)}"],
    )


def export_episode_dataset(
    episodes: list[TrajectoryEpisode],
    out_dir: Path,
    *,
    policy_id: str,
    split: str = "seed",
    metadata: dict[str, object] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_refs: list[str] = []
    task_ids: list[str] = []
    verbs: list[str] = []
    operator_families: list[str] = []
    terminal_states: list[str] = []
    readiness_labels: list[str] = []
    governance_labels: list[str] = []
    transition_kinds: list[str] = []
    patch_kinds: list[str] = []
    patch_bearing_episode_count = 0
    lineage_safe_episode_count = 0
    successful_episode_count = 0
    failed_episode_count = 0
    patch_bearing_negative_episode_count = 0
    usable_positive_episode_count = 0
    usable_negative_episode_count = 0
    for episode in episodes:
        out_path = episodes_dir / f"{episode.episode_id}.json"
        write_episode(episode, out_path)
        episode_refs.append(str(out_path.relative_to(out_dir)))
        task_ids.append(episode.task_id)
        if episode.task_verb:
            verbs.append(episode.task_verb)
        if episode.operator_family:
            operator_families.append(episode.operator_family)
        terminal_state = _episode_terminal_state(episode)
        terminal_states.append(terminal_state)
        readiness_labels.append(str(episode.governance.training_example_kind if episode.governance else episode.metadata.get("training_example_kind", "unusable")))
        governance_labels.append(str(episode.governance.episode_governance_kind if episode.governance else episode.metadata.get("episode_governance_kind", "unusable")))
        patch_bearing = _episode_has_patch(episode)
        if patch_bearing:
            patch_bearing_episode_count += 1
        if not _episode_lineage_issues(episode):
            lineage_safe_episode_count += 1
        governance_kind = str(episode.governance.episode_governance_kind if episode.governance else episode.metadata.get("episode_governance_kind", "unusable"))
        if governance_kind == "usable_positive_sft":
            usable_positive_episode_count += 1
        if governance_kind in {"usable_negative_debug", "usable_negative_transition"}:
            usable_negative_episode_count += 1
            if patch_bearing:
                patch_bearing_negative_episode_count += 1
        for step in episode.steps:
            if step.transition_kind:
                transition_kinds.append(step.transition_kind)
            patch_kind = step.action.metadata.get("patch_kind")
            if isinstance(patch_kind, str):
                patch_kinds.append(patch_kind)
        if terminal_state == "success":
            successful_episode_count += 1
        else:
            failed_episode_count += 1
    manifest = TrajectoryDatasetManifest(
        dataset_id=f"trajectory_dataset_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now(tz=UTC),
        policy_id=policy_id,
        split=split,
        episode_count=len(episode_refs),
        successful_episode_count=successful_episode_count,
        failed_episode_count=failed_episode_count,
        patch_bearing_episode_count=patch_bearing_episode_count,
        patch_bearing_negative_episode_count=patch_bearing_negative_episode_count,
        lineage_safe_episode_count=lineage_safe_episode_count,
        usable_positive_episode_count=usable_positive_episode_count,
        usable_negative_episode_count=usable_negative_episode_count,
        episode_refs=episode_refs,
        task_ids=sorted(set(task_ids)),
        verb_counts=_count_by(verbs),
        operator_family_counts=_count_by(operator_families),
        terminal_state_counts=_count_by(terminal_states),
        readiness_counts=_count_by(readiness_labels),
        episode_governance_counts=_count_by(governance_labels),
        transition_kind_counts=_count_by(transition_kinds),
        patch_kind_counts=_count_by(patch_kinds),
        metadata=metadata or {},
    )
    manifest_path = out_dir / "trajectory_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return manifest_path


def capture_run_episode(root: Path, run_ref: str, *, policy_id: str = "reference_policy_v1", section: str = "full") -> TrajectoryEpisode:
    run_dir = resolve_run_dir(root, run_ref)
    summary_payload = inspect_run(root, str(run_dir), section="full")
    replay_command = _load_optional_json(run_dir / "replay" / "command.json") or {}
    command = replay_command.get("command", [])
    if not isinstance(command, list):
        command = []
    reward_components, final_reward = _derive_reward(summary_payload, run_dir)
    evidence_quality = summary_payload.get("projection", {}).get("evidence_quality", {})
    governance = _governance_from_run_evidence(evidence_quality)
    key_artifacts = summary_payload.get("key_artifacts", [])
    if not isinstance(key_artifacts, list):
        key_artifacts = []
    artifact_refs = _dedupe(
        [str(path) for path in key_artifacts]
        + [
            "summary.json",
            "summary.md",
            "replay/replay_pack.json" if (run_dir / "replay" / "replay_pack.json").exists() else "",
            "replay/command.json" if (run_dir / "replay" / "command.json").exists() else "",
            "replay/environment.json" if (run_dir / "replay" / "environment.json").exists() else "",
        ]
    )
    artifact_refs = [path for path in artifact_refs if path]
    observation_projection = _select_projection_section(summary_payload, section)
    observation = TrajectoryObservation(
        observation_type="run_bundle_projection",
        run_id=str(summary_payload.get("run_id")),
        task_id=str(summary_payload.get("task_id")),
        status=str(summary_payload.get("status")),
        backend=str(summary_payload.get("backend")),
        vendor=str(summary_payload.get("vendor")),
        summary_ref="summary.json",
        artifact_refs=artifact_refs,
        salient_artifact_refs=artifact_refs[:3],
        projection=observation_projection,
    )
    action = TrajectoryAction(
        action_type=_infer_action_type(str(summary_payload.get("run_id"))),
        step_kind="tool_call",
        command=[str(item) for item in command],
        target_run_id=str(summary_payload.get("run_id")),
        target_task_id=str(summary_payload.get("task_id")),
        artifact_refs=[
            "replay/command.json",
            "replay/environment.json",
        ]
        if (run_dir / "replay" / "command.json").exists()
        else [],
        metadata={
            "section": section,
            "trace_enabled": bool(summary_payload.get("trace_enabled")),
        },
    )
    terminal_state = "success" if summary_payload.get("status") == "ok" else "failure"
    task = TaskRegistry(root).get(str(summary_payload.get("task_id")))
    final_reward = sum(float(value) for value in reward_components.values())
    return TrajectoryEpisode(
        episode_id=f"episode_{summary_payload.get('run_id')}",
        created_at=datetime.now(tz=UTC),
        policy_id=policy_id,
        task_id=str(summary_payload.get("task_id")),
        task_verb=task.verb,
        operator_family=task.operator_family,
        source_run_id=str(summary_payload.get("run_id")),
        source_run_ref=str(run_dir),
        episode_kind="single_run_projection",
        steps=[
            TrajectoryStep(
                step_index=0,
                step_label="verification_action",
                action=action,
                observation=observation,
                reward_components=reward_components,
                reward_total=final_reward,
                transition_kind="evaluated" if action.action_type == "eval" else "benchmarked" if action.action_type == "bench" else None,
                recommended_next_actions=[],
                terminal=True,
                terminal_state=terminal_state,
            )
        ],
        final_reward=final_reward,
        terminal_state=terminal_state,
        environment_hash=_environment_hash(run_dir),
        artifact_refs=artifact_refs,
        governance=governance,
        metadata={
            "section": section,
            "run_status": summary_payload.get("status"),
            "has_eval_envelope": (run_dir / "eval" / "eval_envelope.json").exists(),
            "run_ref_map": {str(summary_payload.get("run_id")): str(run_dir)},
            "training_example_kind": evidence_quality.get("training_example_kind", "unusable"),
            "episode_governance_kind": governance.episode_governance_kind,
            "evidence_score": evidence_quality.get("overall_score", 0.0),
        },
    )


def write_trajectory_episode(root: Path, run_ref: str, out_path: Path, *, policy_id: str = "reference_policy_v1", section: str = "full") -> Path:
    episode = capture_run_episode(root, run_ref, policy_id=policy_id, section=section)
    return write_episode(episode, out_path)


def export_trajectory_dataset(
    root: Path,
    run_refs: list[str],
    out_dir: Path,
    *,
    policy_id: str = "reference_policy_v1",
    split: str = "seed",
    section: str = "full",
) -> Path:
    episodes = [capture_run_episode(root, run_ref, policy_id=policy_id, section=section) for run_ref in run_refs]
    return export_episode_dataset(
        episodes,
        out_dir,
        policy_id=policy_id,
        split=split,
        metadata={"section": section},
    )


def validate_trajectory_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "trajectory_dataset_manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "missing": ["trajectory_dataset_manifest.json"], "missing_episode_refs": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_refs = manifest.get("episode_refs", [])
    missing_episode_refs = [
        str(relative_path)
        for relative_path in episode_refs
        if not (dataset_dir / str(relative_path)).exists()
    ]
    invalid_episode_refs: list[str] = []
    invalid_step_orders: list[str] = []
    missing_terminal_states: list[str] = []
    broken_source_run_refs: list[str] = []
    broken_artifact_refs: list[str] = []
    missing_governance_refs: list[str] = []
    broken_candidate_lineage_refs: list[str] = []
    for relative_path in episode_refs:
        episode_path = dataset_dir / str(relative_path)
        if not episode_path.exists():
            continue
        try:
            episode = TrajectoryEpisode.model_validate(json.loads(episode_path.read_text(encoding="utf-8")))
        except Exception:
            invalid_episode_refs.append(str(relative_path))
            continue
        source_run_dir = Path(episode.source_run_ref)
        if episode.source_run_ref and not source_run_dir.is_absolute():
            source_run_dir = (dataset_dir / source_run_dir).resolve()
        run_ref_map_payload = episode.metadata.get("run_ref_map", {})
        run_ref_dirs: list[Path] = []
        if isinstance(run_ref_map_payload, dict):
            for run_ref in run_ref_map_payload.values():
                if isinstance(run_ref, str) and run_ref:
                    resolved = _resolve_episode_run_ref(dataset_dir, run_ref)
                    if resolved.exists():
                        run_ref_dirs.append(resolved)
        if episode.source_run_ref and not source_run_dir.exists():
            broken_source_run_refs.append(str(relative_path))
        step_indices = [step.step_index for step in episode.steps]
        if step_indices != list(range(len(step_indices))):
            invalid_step_orders.append(str(relative_path))
        if not episode.steps or not episode.steps[-1].terminal or not _episode_terminal_state(episode):
            missing_terminal_states.append(str(relative_path))
        if episode.governance is None:
            missing_governance_refs.append(str(relative_path))
        lineage_issues = _episode_lineage_issues(episode)
        if lineage_issues:
            broken_candidate_lineage_refs.append(f"{relative_path}:{','.join(lineage_issues)}")
        candidate_dirs = [source_run_dir] if source_run_dir.exists() else []
        candidate_dirs.extend(run_ref_dirs)
        if candidate_dirs:
            for artifact_ref in episode.artifact_refs:
                if artifact_ref and not any((run_dir / artifact_ref).exists() for run_dir in candidate_dirs):
                    broken_artifact_refs.append(f"{relative_path}:{artifact_ref}")
    return {
        "status": "ok"
        if not missing_episode_refs
        and not invalid_episode_refs
        and not invalid_step_orders
        and not missing_terminal_states
        and not broken_source_run_refs
        and not broken_artifact_refs
        and not broken_candidate_lineage_refs
        else "failed",
        "episode_count": int(manifest.get("episode_count", 0)),
        "missing_episode_refs": missing_episode_refs,
        "invalid_episode_refs": invalid_episode_refs,
        "invalid_step_orders": invalid_step_orders,
        "missing_terminal_states": missing_terminal_states,
        "missing_governance_refs": missing_governance_refs,
        "broken_candidate_lineage_refs": broken_candidate_lineage_refs,
        "broken_source_run_refs": broken_source_run_refs,
        "broken_artifact_refs": broken_artifact_refs,
        "manifest": manifest,
    }
