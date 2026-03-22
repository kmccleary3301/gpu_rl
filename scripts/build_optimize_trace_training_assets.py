from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import EpisodeReadinessReport, LearningRewardTrace, OptimizeTraceSnapshot, OptimizeTraceSnapshots, ReadinessDecision, RewardLedger, RewardLedgerEntry, TrajectoryAction, TrajectoryEpisode, TrajectoryObservation, TrajectoryStep
from gpu_cockpit.engine.optimize_patch_registry import get_optimize_patch_spec
from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.engine.trajectory import export_episode_dataset, validate_trajectory_dataset


TRAIN_REFS = [
    "artifacts/baselines/gpt54_attention_score_bounded_patch_probe_v1/batch_v6_forced_eval_closeout_retry1/task__attention_score__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_bounded_patch_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kv_cache_gather_bounded_patch_probe_v1/batch_v2_retry1/task__kv_cache_gather__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_sum_reduction_bounded_patch_probe_v1/batch_v1_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_multi_candidate_positive_probe_v1/batch_v2_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_positive_probe_v1/batch_v2_retry1/task__kernelbench__level1__23_softmax__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_public_negative_bounded_patch_probe_v1/batch_v2_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__negative.json",
    "artifacts/baselines/gpt54_reduction_row_sum_multi_candidate_negative_probe_v1/batch_v2_retry1/task__reduction_row_sum__eval__v1__negative.json",
    "artifacts/baselines/gpt54_kernelbench_public_negative_bounded_patch_probe_v1/batch_v2_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_reduction_row_sum_two_attempt_positive_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
]

DEV_REFS = [
    "artifacts/baselines/gpt54_kernelbench_softmax_bounded_patch_probe_v1/batch_v3_public_signal_retry2/task__kernelbench__level1__23_softmax__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_second_wave_kernel_push_v1/batch_v1/task__attention_score__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
]


def _load_payload(relative_ref: str) -> dict[str, object]:
    return json.loads((ROOT / relative_ref).read_text(encoding="utf-8"))


def _quality_bucket(payload: dict[str, object]) -> str:
    if bool(payload.get("success")):
        return "usable_positive"
    terminal_reason = str(payload.get("terminal_reason", ""))
    if terminal_reason in {"negative_trace_complete", "multi_candidate_negative_complete"}:
        return "usable_negative"
    return "excluded"


def _governance(payload: dict[str, object]) -> EpisodeReadinessReport:
    quality_bucket = _quality_bucket(payload)
    if quality_bucket == "usable_positive":
        training_example_kind = "positive_sft_example"
        episode_governance_kind = "usable_positive_sft"
    elif quality_bucket == "usable_negative":
        training_example_kind = "negative_reformulate_example"
        episode_governance_kind = "usable_negative_transition"
    else:
        training_example_kind = "unusable"
        episode_governance_kind = "unusable"
    patch_bearing = int(payload.get("counters", {}).get("patches", 0)) > 0 if isinstance(payload.get("counters"), dict) else False
    reasons = [f"quality_bucket:{quality_bucket}"]
    if patch_bearing:
        reasons.append("patch_bearing")
    return EpisodeReadinessReport(
        episode_governance_kind=episode_governance_kind,
        training_example_kind=training_example_kind,
        benchmark_collection=ReadinessDecision(eligible=bool(payload.get("success")), reasons=[]),
        sft_collection=ReadinessDecision(eligible=quality_bucket in {"usable_positive", "usable_negative"}, reasons=[]),
        rl_reward_trace=ReadinessDecision(eligible=False, reasons=["gpt54_baseline_trace"]),
        has_build_evidence=False,
        has_profile_evidence=False,
        patch_bearing=patch_bearing,
        reasons=reasons,
        notes=["converted_from_gpt54_baseline_episode"],
    )


def _patch_metadata(task_ref: str, variant: str, action_name: str) -> dict[str, object]:
    if action_name != "patch_candidate":
        return {}
    spec = get_optimize_patch_spec(task_ref) or {}
    is_negative = variant == "negative"
    patch_kind_key = "negative_patch_kind" if is_negative else "positive_patch_kind"
    transition_kind_key = "negative_transition_kind" if is_negative else "positive_transition_kind"
    return {
        "patch_kind": spec.get(patch_kind_key),
        "transition_kind": spec.get(transition_kind_key),
    }


def _step_from_payload_step(task_ref: str, variant: str, step: dict[str, object]) -> TrajectoryStep:
    observation = step.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}
    projection = observation.get("projection_excerpt", {})
    if not isinstance(projection, dict):
        projection = {}
    action_name = str(step.get("action_name"))
    return TrajectoryStep(
        step_index=int(step.get("step_index", 0)),
        step_label=str(step.get("step_label")) if step.get("step_label") is not None else None,
        action=TrajectoryAction(
            action_type=action_name,
            step_kind=str(step.get("step_label")) if step.get("step_label") is not None else None,
            target_run_id=str(observation.get("run_id")) if observation.get("run_id") is not None else None,
            target_task_id=str(observation.get("task_id")) if observation.get("task_id") is not None else None,
            artifact_refs=[str(item) for item in observation.get("salient_artifact_refs", []) if isinstance(item, str)],
            metadata=_patch_metadata(task_ref, variant, action_name),
        ),
        observation=TrajectoryObservation(
            observation_type=str(observation.get("type", "projection")),
            run_id=str(observation.get("run_id")) if observation.get("run_id") is not None else None,
            task_id=str(observation.get("task_id")) if observation.get("task_id") is not None else None,
            status=str(observation.get("status")) if observation.get("status") is not None else None,
            summary_ref=str(observation.get("summary_ref")) if observation.get("summary_ref") is not None else None,
            salient_artifact_refs=[str(item) for item in observation.get("salient_artifact_refs", []) if isinstance(item, str)],
            projection=projection,
        ),
        reward_components={
            str(key): float(value)
            for key, value in (step.get("reward_components") or {}).items()
            if isinstance(value, (int, float))
        },
        reward_total=float(step.get("reward_total", 0.0)),
        transition_kind=str(step.get("transition_kind")) if step.get("transition_kind") is not None else None,
        recommended_next_actions=[str(item) for item in step.get("recommended_next_actions", []) if isinstance(item, str)],
        terminal=False,
        terminal_state=None,
    )


def _annotate_candidate_lineage(steps: list[TrajectoryStep], candidate_history: list[str]) -> list[TrajectoryStep]:
    next_candidate_index = 0
    current_candidate_id: str | None = None
    annotated: list[TrajectoryStep] = []
    for step in steps:
        input_candidate_id = current_candidate_id
        output_candidate_id = step.output_candidate_id
        if step.action.action_type in {"patch_candidate", "branch_candidate", "revert_candidate", "promote_candidate"}:
            if next_candidate_index < len(candidate_history):
                output_candidate_id = candidate_history[next_candidate_index]
            else:
                output_candidate_id = f"converted_{step.action.action_type}_{step.step_index}"
            next_candidate_index += 1
            current_candidate_id = output_candidate_id
        else:
            output_candidate_id = current_candidate_id
        annotated.append(
            step.model_copy(
                update={
                    "input_candidate_id": input_candidate_id,
                    "output_candidate_id": output_candidate_id,
                }
            )
        )
    return annotated


def _optimize_snapshots(steps: list[TrajectoryStep]) -> OptimizeTraceSnapshots | None:
    candidate_snapshots: list[OptimizeTraceSnapshot] = []
    compare_snapshots: list[OptimizeTraceSnapshot] = []
    failure_localization_snapshots: list[OptimizeTraceSnapshot] = []
    for step in steps:
        projection = step.observation.projection if isinstance(step.observation.projection, dict) else {}
        candidate_projection = projection.get("candidate_projection")
        if isinstance(candidate_projection, dict) and candidate_projection:
            candidate_snapshots.append(
                OptimizeTraceSnapshot(
                    step_index=step.step_index,
                    action_type=step.action.action_type,
                    snapshot_kind="candidate_state",
                    run_id=step.observation.run_id,
                    payload=candidate_projection,
                )
            )
        if step.action.action_type == "compare":
            compare_snapshots.append(
                OptimizeTraceSnapshot(
                    step_index=step.step_index,
                    action_type=step.action.action_type,
                    snapshot_kind="compare",
                    run_id=step.observation.run_id,
                    payload={
                        key: projection.get(key)
                        for key in ["optimize_delta_summary", "candidate_delta_brief", "recommended_next_actions", "summary_lines"]
                        if projection.get(key) is not None
                    },
                )
            )
        failure_localization = projection.get("failure_localization")
        if isinstance(failure_localization, dict) and failure_localization:
            failure_localization_snapshots.append(
                OptimizeTraceSnapshot(
                    step_index=step.step_index,
                    action_type=step.action.action_type,
                    snapshot_kind="failure_localization",
                    run_id=step.observation.run_id,
                    payload=failure_localization,
                )
            )
    if not candidate_snapshots and not compare_snapshots and not failure_localization_snapshots:
        return None
    return OptimizeTraceSnapshots(
        candidate_snapshots=candidate_snapshots,
        compare_snapshots=compare_snapshots,
        failure_localization_snapshots=failure_localization_snapshots,
    )


def _learning_reward_trace(payload: dict[str, object]) -> LearningRewardTrace:
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    compare_used = int(counters.get("compares", 0)) > 0
    task_success = bool(payload.get("success"))
    task_outcome = "success" if task_success else str(payload.get("terminal_reason", "failure"))
    trace_usability = "trainable_positive" if task_success else "trainable_negative" if task_outcome in {"negative_trace_complete", "multi_candidate_negative_complete"} else "analysis_only"
    tool_cost = round(
        sum(
            float(value)
            for step in payload.get("steps", [])
            if isinstance(step, dict)
            for key, value in (step.get("reward_components") or {}).items()
            if key == "tool_cost" and isinstance(value, (int, float))
        ),
        4,
    )
    reward_components = {
        "task_success": 0.6 if task_success else 0.0,
        "correctness": 0.25 if task_success else 0.0,
        "determinism": 0.1 if task_success else 0.0,
        "perf_improvement": 0.0,
    }
    shaping_components = {
        "compare_use_bonus": 0.02 if compare_used else 0.0,
        "tool_cost": tool_cost,
        "candidate_regression_penalty": 0.0,
    }
    reward_ledger = RewardLedger(
        task_id=str(payload.get("task_ref")),
        task_verb=str(payload.get("verb")) if payload.get("verb") is not None else None,
        task_outcome=task_outcome,
        trace_usability=trace_usability,
        entries=[
            RewardLedgerEntry(
                step_index=int(step.get("step_index", 0)),
                action_type=str(step.get("action_name")),
                reward_components=reward_components if bool(step.get("terminal_state")) else {},
                shaping_components={
                    "tool_cost": round(
                        sum(
                            float(value)
                            for key, value in (step.get("reward_components") or {}).items()
                            if key == "tool_cost" and isinstance(value, (int, float))
                        ),
                        4,
                    )
                },
                total_delta=round(
                    (
                        sum(reward_components.values()) if bool(step.get("terminal_state")) else 0.0
                    )
                    + sum(
                        float(value)
                        for key, value in (step.get("reward_components") or {}).items()
                        if key == "tool_cost" and isinstance(value, (int, float))
                    ),
                    4,
                ),
                notes=[],
            )
            for step in payload.get("steps", [])
            if isinstance(step, dict)
        ],
        total_reward_components=reward_components,
        total_shaping_components=shaping_components,
        total_reward=round(sum(reward_components.values()) + sum(shaping_components.values()), 4),
    )
    return LearningRewardTrace(
        task_id=str(payload.get("task_ref")),
        task_verb=str(payload.get("verb")) if payload.get("verb") is not None else None,
        terminal_state=task_outcome,
        task_outcome=task_outcome,
        trace_usability=trace_usability,
        task_success=task_success,
        correctness_passed=task_success,
        determinism_passed=True,
        anti_hack_passed=True,
        perf_gate="unknown",
        compare_used=compare_used,
        patch_bearing=int(counters.get("patches", 0)) > 0,
        branch_count=int(counters.get("branches", 0)),
        revert_count=int(counters.get("reverts", 0)),
        promote_count=int(counters.get("promotes", 0)),
        reward_components=reward_components,
        shaping_components=shaping_components,
        excluded_governance_signals=["benchmark_reporting", "sft_collection", "rl_reward_trace_readiness", "evidence_score"],
        total_reward=reward_ledger.total_reward,
        notes=["converted_from_gpt54_baseline_episode"],
        reward_ledger=reward_ledger,
    )


def _episode_from_report(relative_ref: str) -> TrajectoryEpisode:
    payload = _load_payload(relative_ref)
    task_ref = str(payload.get("task_ref"))
    variant = str(payload.get("variant"))
    registry = TaskRegistry(ROOT)
    task = registry.get(task_ref)
    steps = [_step_from_payload_step(task_ref, variant, step) for step in payload.get("steps", []) if isinstance(step, dict)]
    state = payload.get("state", {})
    if not isinstance(state, dict):
        state = {}
    candidate_history = [str(item) for item in state.get("candidate_history", []) if isinstance(item, str)]
    steps = _annotate_candidate_lineage(steps, candidate_history)
    terminal_state = "success" if bool(payload.get("success")) else str(payload.get("terminal_reason", "failure"))
    if steps:
        steps[-1] = steps[-1].model_copy(update={"terminal": True, "terminal_state": terminal_state})
    final_reward = round(sum(step.reward_total for step in steps), 4)
    governance = _governance(payload)
    learning_reward_trace = _learning_reward_trace(payload)
    return TrajectoryEpisode(
        episode_id=f"converted_{Path(relative_ref).stem}",
        created_at=datetime.now(tz=UTC),
        policy_id="gpt54_optimize_trace_converter_v1",
        task_id=task_ref,
        task_verb=task.verb,
        operator_family=task.operator_family,
        source_run_id=str(state.get("last_run_id", "none")),
        source_run_ref=str(state.get("last_run_ref", relative_ref)),
        episode_kind="gpt54_optimize_trace_conversion",
        steps=steps,
        final_reward=final_reward,
        terminal_state=terminal_state,
        artifact_refs=[],
        governance=governance,
        learning_reward_trace=learning_reward_trace,
        reward_ledger=learning_reward_trace.reward_ledger,
        optimize_trace_snapshots=_optimize_snapshots(steps),
        metadata={
            "variant": variant,
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "terminal_reason": payload.get("terminal_reason"),
            "source_episode_ref": relative_ref,
            "training_example_kind": governance.training_example_kind,
            "episode_governance_kind": governance.episode_governance_kind,
            "episode_governance_reasons": governance.reasons,
            "counters": payload.get("counters", {}),
        },
    )


def main() -> int:
    outputs = {
        "train_trajectory": ROOT / "datasets" / "optimize_trace_transition_train_v4",
        "dev_trajectory": ROOT / "datasets" / "optimize_trace_transition_dev_v4",
        "train_sft": ROOT / "datasets" / "optimize_trace_sft_train_v4",
        "dev_sft": ROOT / "datasets" / "optimize_trace_sft_dev_v4",
    }
    train_episodes = [_episode_from_report(ref) for ref in TRAIN_REFS]
    dev_episodes = [_episode_from_report(ref) for ref in DEV_REFS]

    train_manifest = export_episode_dataset(train_episodes, outputs["train_trajectory"], policy_id="gpt54_optimize_trace_converter_v1", split="train")
    dev_manifest = export_episode_dataset(dev_episodes, outputs["dev_trajectory"], policy_id="gpt54_optimize_trace_converter_v1", split="dev")
    train_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["train_trajectory"],
        outputs["train_sft"],
        split="train",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=True,
    )
    dev_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["dev_trajectory"],
        outputs["dev_sft"],
        split="dev",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=True,
    )

    report = {
        "report_id": "optimize_trace_training_assets_v4",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
        "train_trajectory_manifest": str(train_manifest.relative_to(ROOT)),
        "dev_trajectory_manifest": str(dev_manifest.relative_to(ROOT)),
        "train_sft_manifest": str(train_sft_manifest.relative_to(ROOT)),
        "dev_sft_manifest": str(dev_sft_manifest.relative_to(ROOT)),
        "train_trajectory_validation": validate_trajectory_dataset(outputs["train_trajectory"]),
        "dev_trajectory_validation": validate_trajectory_dataset(outputs["dev_trajectory"]),
        "train_sft_validation": validate_sft_dataset(outputs["train_sft"]),
        "dev_sft_validation": validate_sft_dataset(outputs["dev_sft"]),
    }
    out_path = ROOT / "artifacts" / "training" / "optimize_trace_training_assets_report_v4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
