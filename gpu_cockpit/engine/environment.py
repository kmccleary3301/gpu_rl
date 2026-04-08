from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from gpu_cockpit.contracts import (
    AgentActionSpec,
    AgentEnvironmentState,
    BaselineSpec,
    EpisodeReadinessReport,
    EvidenceQualityReport,
    LearningRewardTrace,
    OptimizeTraceSnapshot,
    OptimizeTraceSnapshots,
    ReadinessDecision,
    RewardLedger,
    RewardLedgerEntry,
    RunSummary,
    TrajectoryAction,
    TrajectoryEpisode,
    TrajectoryObservation,
    TrajectoryStep,
)
from gpu_cockpit.engine.adapter_registry import get_adapter
from gpu_cockpit.engine.benchmark import run_task_benchmark
from gpu_cockpit.engine.doctor import collect_doctor_report
from gpu_cockpit.engine.evaluator import run_evaluation_hooks
from gpu_cockpit.engine.indexer import list_runs
from gpu_cockpit.engine.inspector import compare_runs, inspect_run, resolve_run_dir
from gpu_cockpit.engine.knowledge import query_knowledge
from gpu_cockpit.engine.optimize_patch_registry import resolve_optimize_patch_plan
from gpu_cockpit.engine.patching import apply_patch_candidate, branch_candidate, promote_candidate, revert_candidate
from gpu_cockpit.engine.replay import validate_run_bundle, write_replay_pack
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.runner import build_run_id, build_run_spec, run_task, write_run_summary, write_task_artifacts
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.executors import make_executor


ACTION_SPACE: tuple[AgentActionSpec, ...] = (
    AgentActionSpec(
        action_name="run",
        description="Execute a task command and create a run bundle.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
        cost_units=0.05,
        observation_focus="run_summary",
        recommended_verbs=["optimize", "debug", "diagnose", "reformulate"],
    ),
    AgentActionSpec(
        action_name="build",
        description="Compile or emit build/disassembly artifacts for a task.",
        requires_task=True,
        produces_run_bundle=True,
        cost_units=0.03,
        observation_focus="build_projection",
        recommended_verbs=["debug", "reformulate", "optimize"],
    ),
    AgentActionSpec(
        action_name="bench",
        description="Benchmark a task command and emit a perf bundle.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
        cost_units=0.03,
        observation_focus="perf_summary",
        recommended_verbs=["optimize", "reformulate"],
    ),
    AgentActionSpec(
        action_name="eval",
        description="Run task evaluation hooks and emit correctness and reward artifacts.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
        cost_units=0.05,
        observation_focus="eval_summary",
        recommended_verbs=["optimize", "debug", "diagnose", "reformulate"],
    ),
    AgentActionSpec(
        action_name="inspect",
        description="Project a compact section from a run bundle.",
        cost_units=0.01,
        observation_focus="inspection",
        recommended_verbs=["diagnose", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="inspect_build",
        description="Project the build/disassembly section from a run bundle.",
        cost_units=0.01,
        observation_focus="build_projection",
        recommended_verbs=["debug", "reformulate", "optimize"],
    ),
    AgentActionSpec(
        action_name="inspect_profile",
        description="Project the profile/bottleneck section from a run bundle.",
        cost_units=0.01,
        observation_focus="profile_projection",
        recommended_verbs=["diagnose", "optimize", "reformulate"],
    ),
    AgentActionSpec(
        action_name="inspect_quality",
        description="Project quality, failure triage, and training-readiness details from a run bundle.",
        cost_units=0.01,
        observation_focus="quality_projection",
        recommended_verbs=["diagnose", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="patch_candidate",
        description="Apply a scripted patch to a candidate file and emit patch/candidate artifacts.",
        requires_task=True,
        produces_run_bundle=True,
        cost_units=0.04,
        observation_focus="candidate_transition",
        recommended_verbs=["debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="branch_candidate",
        description="Create a new candidate branch from the current candidate lineage point.",
        requires_task=True,
        produces_run_bundle=True,
        cost_units=0.02,
        observation_focus="candidate_transition",
        recommended_verbs=["optimize", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="revert_candidate",
        description="Revert candidate workspace state to an earlier text snapshot and emit candidate lineage artifacts.",
        requires_task=True,
        produces_run_bundle=True,
        cost_units=0.03,
        observation_focus="candidate_transition",
        recommended_verbs=["optimize", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="promote_candidate",
        description="Promote the current candidate into a named ready state without applying a new patch.",
        requires_task=True,
        produces_run_bundle=True,
        cost_units=0.02,
        observation_focus="candidate_transition",
        recommended_verbs=["optimize", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="compare",
        description="Compare two run bundles.",
        cost_units=0.01,
        observation_focus="comparison",
        recommended_verbs=["diagnose", "debug", "reformulate"],
    ),
    AgentActionSpec(
        action_name="replay",
        description="Validate replay completeness for a run bundle.",
        cost_units=0.01,
        observation_focus="replay_validation",
        recommended_verbs=["debug", "diagnose", "reformulate"],
    ),
    AgentActionSpec(
        action_name="adapter_show",
        description="Load a benchmark case and derived task from a registered adapter.",
        cost_units=0.005,
        observation_focus="adapter_case",
        recommended_verbs=["optimize"],
    ),
    AgentActionSpec(
        action_name="knowledge_query",
        description="Query the local knowledge index for related docs, tasks, and runs.",
        cost_units=0.02,
        observation_focus="knowledge_results",
        recommended_verbs=["diagnose", "debug", "reformulate"],
    ),
)


def list_action_space() -> list[AgentActionSpec]:
    return [spec.model_copy(deep=True) for spec in ACTION_SPACE]


def initialize_environment_state(
    root: Path,
    task_ref: str,
    *,
    policy_id: str = "scripted_reference_v1",
    step_budget: int = 5,
) -> AgentEnvironmentState:
    task = TaskRegistry(root).get(task_ref)
    return _refresh_candidate_tree_state(
        AgentEnvironmentState(
        episode_id=f"env_episode_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        policy_id=policy_id,
        task_id=task.task_id,
        step_budget_total=step_budget,
        step_budget_remaining=step_budget,
        metadata={
            "operator_family": task.operator_family,
            "verb": task.verb,
            "difficulty": task.difficulty,
            "allowed_backends": list(task.allowed_backends),
        },
        )
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _derive_eval_rewards(run_dir: Path) -> tuple[dict[str, float], float]:
    eval_envelope_path = run_dir / "eval" / "eval_envelope.json"
    if not eval_envelope_path.exists():
        return {"completion": 1.0 if (run_dir / "summary.json").exists() else 0.0}, 1.0 if (run_dir / "summary.json").exists() else 0.0
    payload = json.loads(eval_envelope_path.read_text(encoding="utf-8"))
    reward_components = payload.get("reward_components", {})
    normalized = {
        str(key): float(value)
        for key, value in reward_components.items()
        if isinstance(value, (int, float))
    }
    final_score = payload.get("final_score")
    return normalized, float(final_score) if isinstance(final_score, (int, float)) else 0.0


def _tool_cost(action_name: str) -> float:
    for spec in ACTION_SPACE:
        if spec.action_name == action_name:
            return float(spec.cost_units)
    return 0.0


def _step_label_for_action(action_name: str) -> str:
    if action_name in {"knowledge_query", "inspect", "inspect_build", "inspect_profile", "inspect_quality"}:
        return "diagnostic_action"
    if action_name in {"patch_candidate", "branch_candidate", "revert_candidate", "promote_candidate"}:
        return "patch_action"
    if action_name in {"eval", "bench", "replay"}:
        return "verification_action"
    if action_name == "compare":
        return "compare_action"
    if action_name == "build":
        return "build_action"
    return "tool_action"


def _salient_artifacts(observation_type: str, projection: dict[str, object], artifact_refs: list[str]) -> list[str]:
    if observation_type == "inspection":
        failure_triage = projection.get("failure_triage")
        if isinstance(failure_triage, dict):
            likely_artifacts = failure_triage.get("likely_artifacts")
            if isinstance(likely_artifacts, list):
                return [str(path) for path in likely_artifacts]
    if observation_type == "candidate_patch":
        preferred = [path for path in artifact_refs if path.startswith("patches/") or path.startswith("candidate/")]
        if preferred:
            return preferred
    if observation_type == "comparison":
        preferred = []
        if projection.get("lhs_patch_present"):
            preferred.append("lhs:patch")
        if projection.get("rhs_patch_present"):
            preferred.append("rhs:patch")
        return preferred
    return artifact_refs[:3]


def _recommended_actions_from_projection(observation_type: str, projection: dict[str, object]) -> list[str]:
    if observation_type in {"inspection", "eval_projection", "run_bundle_projection", "build_projection"}:
        raw = projection.get("recommended_next_actions")
        if isinstance(raw, list):
            return [str(item) for item in raw]
    if observation_type == "candidate_patch":
        return ["build", "eval", "compare", "replay"]
    if observation_type == "comparison":
        raw = projection.get("recommended_next_actions")
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return ["inspect_quality", "patch_candidate", "eval"]
    return []


def _update_state_after_run(state: AgentEnvironmentState, run_ref: str, run_id: str) -> AgentEnvironmentState:
    run_history = list(state.run_history)
    run_history.append(run_ref)
    return state.model_copy(
        update={
            "steps_taken": state.steps_taken + 1,
            "step_budget_remaining": max(0, state.step_budget_remaining - 1),
            "last_run_id": run_id,
            "last_run_ref": run_ref,
            "run_history": run_history,
        }
    )


def _update_state_after_info(state: AgentEnvironmentState) -> AgentEnvironmentState:
    return state.model_copy(
        update={
            "steps_taken": state.steps_taken + 1,
            "step_budget_remaining": max(0, state.step_budget_remaining - 1),
        }
    )


def _candidate_role_group(candidate_role: object | None) -> str | None:
    role = str(candidate_role) if candidate_role is not None else None
    if role is None:
        return None
    if role == "baseline_candidate":
        return "baseline"
    if role in {"working_candidate", "patched_candidate", "synthesized_candidate"}:
        return "trial"
    if role == "branched_candidate":
        return "branch"
    if role == "reverted_candidate":
        return "reverted"
    if role == "promoted_candidate":
        return "promoted"
    if role == "comparison_anchor":
        return "anchor"
    return role


def _candidate_tree_depth(
    lineage_events: list[dict[str, object]],
    candidate_id: str | None,
) -> int | None:
    if candidate_id is None:
        return None
    parent_index: dict[str, str | None] = {}
    for event in lineage_events:
        if not isinstance(event, dict):
            continue
        event_candidate_id = event.get("candidate_id")
        if event_candidate_id is None:
            continue
        parent_index[str(event_candidate_id)] = (
            str(event.get("parent_candidate_id")) if event.get("parent_candidate_id") is not None else None
        )
    if candidate_id not in parent_index:
        return None
    depth = 0
    cursor = parent_index.get(candidate_id)
    seen: set[str] = {candidate_id}
    while cursor is not None and cursor not in seen:
        depth += 1
        seen.add(cursor)
        cursor = parent_index.get(cursor)
    return depth


def _candidate_bucket_lists(lineage_events: list[dict[str, object]]) -> tuple[list[str], list[str], list[str]]:
    latest_event_by_candidate: dict[str, dict[str, object]] = {}
    for event in lineage_events:
        if not isinstance(event, dict):
            continue
        candidate_id = event.get("candidate_id")
        if candidate_id is None:
            continue
        latest_event_by_candidate[str(candidate_id)] = event
    dominated: list[str] = []
    active: list[str] = []
    archived: list[str] = []
    for candidate_id, event in latest_event_by_candidate.items():
        status = str(event.get("status")) if event.get("status") is not None else None
        metadata = event.get("metadata")
        if isinstance(metadata, dict) and metadata.get("dominated") is True:
            dominated.append(candidate_id)
            continue
        if status in {"dominated"}:
            dominated.append(candidate_id)
            continue
        if status in {"promoted", "archived"}:
            archived.append(candidate_id)
            continue
        active.append(candidate_id)
    return dominated, active, archived


def _candidate_legal_next_actions(state: AgentEnvironmentState, candidate_role_group: str | None) -> list[str]:
    if state.current_candidate_id is None:
        if state.comparison_anchor_run_ref is not None:
            return ["patch_candidate", "knowledge_query"]
        return ["bench", "patch_candidate", "knowledge_query"]
    status = state.current_candidate_status
    if status in {"patched", "build_passed"}:
        return ["bench", "build", "patch_candidate", "branch_candidate", "revert_candidate"]
    if status == "dominated":
        return ["revert_candidate", "patch_candidate", "compare", "eval"]
    if status == "promoted":
        return ["eval", "compare", "inspect_quality", "inspect_build"]
    if status == "reverted":
        return ["bench", "compare", "patch_candidate", "branch_candidate", "eval"]
    if candidate_role_group == "branch":
        return ["bench", "compare", "patch_candidate", "revert_candidate", "promote_candidate", "eval"]
    return ["bench", "compare", "patch_candidate", "branch_candidate", "revert_candidate", "promote_candidate", "eval"]


def _refresh_candidate_tree_state(state: AgentEnvironmentState) -> AgentEnvironmentState:
    lineage_events = [event for event in state.candidate_lineage_events if isinstance(event, dict)]
    current_event = None
    if state.current_candidate_id is not None:
        for raw_event in reversed(lineage_events):
            if raw_event.get("candidate_id") == state.current_candidate_id:
                current_event = raw_event
                break
    candidate_role = str(current_event.get("candidate_role")) if isinstance(current_event, dict) and current_event.get("candidate_role") is not None else None
    candidate_role_group = _candidate_role_group(candidate_role)
    tree_depth = _candidate_tree_depth(lineage_events, state.current_candidate_id)
    dominated_candidate_ids, active_candidate_ids, archived_candidate_ids = _candidate_bucket_lists(lineage_events)
    current_status = state.current_candidate_status
    current_branch_state = None
    if candidate_role_group == "branch":
        current_branch_state = "branched"
    elif state.current_candidate_parent_id is not None and state.current_candidate_id is not None:
        current_branch_state = "child_candidate"
    if current_status == "dominated":
        current_branch_state = "dominated"
    current_promote_state = "promoted" if current_status == "promoted" else "not_promoted"
    current_revert_state = "reverted" if current_status == "reverted" else "not_reverted"
    current_supersede_reason = state.current_supersede_reason
    if current_supersede_reason is None and state.current_candidate_id == state.best_known_candidate_id:
        current_supersede_reason = state.best_known_candidate_reason
    if current_status == "promoted":
        current_endgame_recommendation = "eval"
    elif current_status == "reverted":
        current_endgame_recommendation = "bench"
    elif current_status in {"patched", "build_passed"}:
        current_endgame_recommendation = "bench"
    elif current_status == "dominated":
        current_endgame_recommendation = "revert"
    elif current_status in {"eval_passed", "eval_failed"}:
        current_endgame_recommendation = "stop"
    else:
        current_endgame_recommendation = state.current_endgame_recommendation or "compare"
    return state.model_copy(
        update={
            "current_candidate_role": candidate_role,
            "current_candidate_role_group": candidate_role_group,
            "current_candidate_tree_depth": tree_depth,
            "current_branch_state": current_branch_state,
            "current_promote_state": current_promote_state,
            "current_revert_state": current_revert_state,
            "current_supersede_reason": current_supersede_reason,
            "current_endgame_recommendation": current_endgame_recommendation,
            "current_legal_next_actions": _candidate_legal_next_actions(state, candidate_role_group),
            "dominated_candidate_ids": dominated_candidate_ids,
            "active_candidate_ids": active_candidate_ids,
            "archived_candidate_ids": archived_candidate_ids,
        }
    )


def _sibling_candidate_refs(state: AgentEnvironmentState, candidate_id: str | None, parent_candidate_id: str | None) -> list[str]:
    if candidate_id is None or parent_candidate_id is None:
        return []
    sibling_ids: list[str] = []
    for event in state.candidate_lineage_events:
        if not isinstance(event, dict):
            continue
        if str(event.get("parent_candidate_id")) != parent_candidate_id:
            continue
        sibling_candidate_id = event.get("candidate_id")
        if sibling_candidate_id is None:
            continue
        sibling_id = str(sibling_candidate_id)
        if sibling_id != candidate_id and sibling_id not in sibling_ids:
            sibling_ids.append(sibling_id)
    return sibling_ids


def _candidate_tree_brief(state: AgentEnvironmentState) -> dict[str, object]:
    current_candidate_id = state.current_candidate_id
    current_parent_candidate_id = state.current_candidate_parent_id
    current_event = None
    for raw_event in reversed(state.candidate_lineage_events):
        if not isinstance(raw_event, dict):
            continue
        if raw_event.get("candidate_id") == current_candidate_id:
            current_event = raw_event
            break
    candidate_role = current_event.get("candidate_role") if isinstance(current_event, dict) else None
    sibling_refs = _sibling_candidate_refs(state, current_candidate_id, current_parent_candidate_id)
    return {
        "history_length": len(list(getattr(state, "candidate_history", []) or [])),
        "current_candidate_id": current_candidate_id,
        "current_parent_candidate_id": current_parent_candidate_id,
        "current_status": state.current_candidate_status,
        "current_candidate_attempt_index": state.current_candidate_attempt_index,
        "current_candidate_ref": state.current_candidate_run_ref,
        "candidate_role": state.current_candidate_role or candidate_role,
        "candidate_role_group": state.current_candidate_role_group or _candidate_role_group(candidate_role),
        "tree_depth": state.current_candidate_tree_depth,
        "branch_state": state.current_branch_state,
        "promote_state": state.current_promote_state,
        "revert_state": state.current_revert_state,
        "parent_candidate_ref": current_parent_candidate_id,
        "sibling_candidate_refs": sibling_refs,
        "why_this_candidate_exists": current_event.get("summary") if isinstance(current_event, dict) else None,
        "supersede_reason": state.current_supersede_reason,
        "endgame_recommendation": state.current_endgame_recommendation,
        "legal_next_actions": list(state.current_legal_next_actions),
        "recent_events": list(state.candidate_lineage_events[-3:]),
        "best_known_candidate_id": state.best_known_candidate_id,
        "best_known_candidate_parent_id": state.best_known_candidate_parent_id,
        "best_known_candidate_run_ref": state.best_known_candidate_run_ref,
        "best_known_candidate_reason": state.best_known_candidate_reason,
        "dominated_candidate_ids": list(state.dominated_candidate_ids),
        "active_candidate_ids": list(state.active_candidate_ids),
        "archived_candidate_ids": list(state.archived_candidate_ids),
    }


def _update_best_known_candidate_from_compare(
    state: AgentEnvironmentState,
    payload: dict[str, object],
) -> dict[str, object]:
    updates: dict[str, object] = {}
    if state.current_candidate_id is None or state.current_candidate_run_ref is None:
        return updates
    correctness_recovered = payload.get("correctness_recovered")
    perf_improved = payload.get("perf_improved")
    rhs_candidate_id = payload.get("rhs_candidate_id")
    rhs_parent_candidate_id = payload.get("rhs_parent_candidate_id")
    if rhs_candidate_id is None or str(rhs_candidate_id) != state.current_candidate_id:
        return updates
    if correctness_recovered is True or perf_improved is True or state.best_known_candidate_id is None:
        updates["best_known_candidate_id"] = state.current_candidate_id
        updates["best_known_candidate_parent_id"] = rhs_parent_candidate_id
        updates["best_known_candidate_run_ref"] = state.current_candidate_run_ref
        if correctness_recovered is True:
            updates["best_known_candidate_reason"] = "correctness_recovered"
        elif perf_improved is True:
            updates["best_known_candidate_reason"] = "perf_improved"
        else:
            updates["best_known_candidate_reason"] = "first_candidate_anchor"
    return updates


def _compare_tree_updates(
    state: AgentEnvironmentState,
    payload: dict[str, object],
    best_known_updates: dict[str, object],
) -> dict[str, object]:
    if state.current_candidate_id is None:
        return {}
    optimize_delta = payload.get("optimize_delta_summary")
    correctness_change = None
    perf_change = None
    if isinstance(optimize_delta, dict):
        correctness_change = str(optimize_delta.get("correctness_change")) if optimize_delta.get("correctness_change") is not None else None
        perf_change = str(optimize_delta.get("perf_change")) if optimize_delta.get("perf_change") is not None else None
    old_best_candidate_id = state.best_known_candidate_id
    new_best_candidate_id = best_known_updates.get("best_known_candidate_id", state.best_known_candidate_id)
    new_best_reason = best_known_updates.get("best_known_candidate_reason", state.best_known_candidate_reason)
    lineage_events = [*state.candidate_lineage_events]
    updates: dict[str, object] = {}
    current_candidate_id = state.current_candidate_id
    current_candidate_role = state.current_candidate_role
    current_candidate_status = state.current_candidate_status
    is_regression_against_best_known = (
        old_best_candidate_id is not None
        and old_best_candidate_id != current_candidate_id
        and (correctness_change == "regressed" or perf_change == "regressed")
    )
    if is_regression_against_best_known:
        lineage_events.append(
            _candidate_lineage_event(
                action_name="compare",
                candidate_id=current_candidate_id,
                parent_candidate_id=state.current_candidate_parent_id,
                source_candidate_id=current_candidate_id,
                run_ref=state.current_candidate_run_ref or state.last_run_ref or "compare",
                status="dominated",
                transition_kind=None,
                summary="candidate regressed against the best-known branch and was pruned",
                candidate_role=current_candidate_role,
                candidate_attempt_index=state.current_candidate_attempt_index,
                metadata={
                    "dominated": True,
                    "compare_decision": "prune",
                    "best_known_candidate_id": old_best_candidate_id,
                    "regression_against_best_known": True,
                },
            )
        )
        updates.update(
            {
                "current_candidate_status": "dominated",
                "current_supersede_reason": "regressed_against_best_known",
                "current_endgame_recommendation": "revert",
            }
        )
    elif new_best_candidate_id == current_candidate_id and old_best_candidate_id not in {None, current_candidate_id}:
        lineage_events.append(
            _candidate_lineage_event(
                action_name="compare",
                candidate_id=str(old_best_candidate_id),
                parent_candidate_id=state.best_known_candidate_parent_id,
                source_candidate_id=str(old_best_candidate_id),
                run_ref=state.best_known_candidate_run_ref or state.last_run_ref or "compare",
                status="dominated",
                transition_kind=None,
                summary="previous best-known branch was superseded by the current candidate",
                candidate_role=None,
                candidate_attempt_index=state.current_candidate_attempt_index,
                metadata={
                    "dominated": True,
                    "compare_decision": "superseded",
                    "superseded_by_candidate_id": current_candidate_id,
                },
            )
        )
        updates.update(
            {
                "current_candidate_status": current_candidate_status,
                "current_supersede_reason": str(new_best_reason) if new_best_reason is not None else "superseded_previous_best_known",
                "current_endgame_recommendation": "promote",
            }
        )
    if lineage_events != state.candidate_lineage_events:
        updates["candidate_lineage_events"] = lineage_events
    return updates


def _lineage_from_candidate_state(root: Path, state: AgentEnvironmentState) -> dict[str, object]:
    if not state.current_candidate_run_ref:
        return {}
    run_dir = resolve_run_dir(root, state.current_candidate_run_ref)
    transition_payload = inspect_run(root, str(run_dir), section="transition")
    candidate_projection = transition_payload.get("candidate_projection", {}) if isinstance(transition_payload, dict) else {}
    candidate_state = candidate_projection.get("candidate_state", {}) if isinstance(candidate_projection, dict) else {}
    applied_patch = candidate_projection.get("applied_patch") if isinstance(candidate_projection, dict) else None
    transition = candidate_projection.get("transition", {}) if isinstance(candidate_projection, dict) else {}
    operation = candidate_projection.get("operation", {}) if isinstance(candidate_projection, dict) else {}
    candidate_diff_ref = None
    if isinstance(applied_patch, dict):
        candidate_diff_ref = applied_patch.get("diff_ref")
    if candidate_diff_ref is None and isinstance(candidate_state, dict):
        diff_summary = candidate_state.get("diff_summary")
        if isinstance(diff_summary, dict):
            candidate_diff_ref = diff_summary.get("diff_ref")
    sibling_candidate_refs = _sibling_candidate_refs(
        state,
        str(candidate_state.get("candidate_id")) if candidate_state.get("candidate_id") is not None else None,
        str(candidate_state.get("parent_candidate_id")) if candidate_state.get("parent_candidate_id") is not None else None,
    )
    return {
        "parent_run_id": state.last_run_id,
        "candidate_id": candidate_state.get("candidate_id"),
        "parent_candidate_id": candidate_state.get("parent_candidate_id"),
        "source_candidate_id": candidate_state.get("source_candidate_id"),
        "source_run_ref": state.current_candidate_run_ref,
        "patch_ref": "patches/applied_patch.json" if isinstance(applied_patch, dict) else None,
        "diff_ref": candidate_diff_ref,
        "transition_ref": "candidate/transition.json" if transition else None,
        "candidate_role": candidate_state.get("candidate_role"),
        "candidate_role_group": _candidate_role_group(candidate_state.get("candidate_role")),
        "candidate_status": candidate_state.get("status"),
        "candidate_tree_depth": state.current_candidate_tree_depth,
        "candidate_origin_kind": candidate_state.get("origin_kind"),
        "candidate_operation_kind": operation.get("operation_kind") or candidate_state.get("last_operation_kind"),
        "operation_ref": "candidate/operation.json" if operation else None,
        "transition_kind": transition.get("transition_kind"),
        "best_known_candidate_id": state.best_known_candidate_id,
        "best_known_candidate_reason": state.best_known_candidate_reason,
        "supersede_reason": state.current_supersede_reason,
        "branch_state": state.current_branch_state,
        "endgame_recommendation": state.current_endgame_recommendation,
        "legal_next_actions": list(state.current_legal_next_actions),
        "dominated_candidate_ids": list(state.dominated_candidate_ids),
        "active_candidate_ids": list(state.active_candidate_ids),
        "archived_candidate_ids": list(state.archived_candidate_ids),
        "patch_present": isinstance(applied_patch, dict),
        "patch_kind": applied_patch.get("patch_kind") if isinstance(applied_patch, dict) else None,
        "sibling_candidate_refs": sibling_candidate_refs,
    }


def _candidate_lineage_event(
    *,
    action_name: str,
    candidate_id: str,
    parent_candidate_id: str | None,
    run_ref: str,
    status: str | None,
    transition_kind: str | None,
    summary: str | None,
    candidate_role: str | None = None,
    source_candidate_id: str | None = None,
    candidate_attempt_index: int | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "action_name": action_name,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "source_candidate_id": source_candidate_id,
        "run_ref": run_ref,
        "status": status,
        "transition_kind": transition_kind,
        "summary": summary,
        "candidate_role": candidate_role,
        "candidate_attempt_index": candidate_attempt_index,
        "metadata": metadata or {},
    }


def _load_text(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def _scripted_patch_plan(root: Path, task_id: str, *, variant: str = "positive") -> dict[str, object] | None:
    if task_id == "task/reduction_debug/eval/v1":
        if variant == "negative":
            return {
                "target_file": "workloads/reference/triton_row_sum_patchable_candidate.py",
                "patch_text": _load_text(root, "workloads/reference/triton_row_sum_broken_candidate.py"),
                "patch_intent": "attempt a reduction repair but leave the broken last-column mask behavior in place",
                "patch_expected_effect": "keep the candidate reproducibly broken so the episode becomes a usable failed repair trace",
                "patch_kind": "no_op",
                "transition_kind": "patch_applied",
                "eval_command": [
                    "python3",
                    "workloads/reference/triton_row_sum_patchable_candidate.py",
                    "--benchmark-repeats",
                    "2",
                ],
                "pre_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
                "post_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
            }
        return {
            "target_file": "workloads/reference/triton_row_sum_patchable_candidate.py",
            "patch_text": _load_text(root, "workloads/reference/triton_row_sum_debug_candidate.py"),
            "patch_intent": "repair the reduction candidate so it uses the corrected Triton row-sum implementation",
            "patch_expected_effect": "turn the broken reduction candidate into a correct implementation without changing the task contract",
            "patch_kind": "bug_fix",
            "transition_kind": "repaired",
            "eval_command": [
                "python3",
                "workloads/reference/triton_row_sum_patchable_candidate.py",
                "--benchmark-repeats",
                "2",
            ],
            "pre_patch_build_spec": "workloads/reference/triton_row_sum_broken_kernel.py:get_build_spec",
            "post_patch_build_spec": "workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
        }
    if task_id == "task/attention_reformulate/eval/v1":
        if variant == "negative":
            return {
                "target_file": "workloads/reference/triton_attention_score_weak_baseline.py",
                "patch_text": _load_text(root, "workloads/reference/triton_attention_score_weak_baseline.py"),
                "patch_intent": "attempt a reformulation without changing the weak causal attention-score implementation",
                "patch_expected_effect": "preserve the baseline implementation so the perf gate still fails and the episode remains a useful negative transform example",
                "patch_kind": "no_op",
                "transition_kind": "patch_applied",
                "eval_command": [
                    "python3",
                    "workloads/reference/triton_attention_score_weak_baseline.py",
                    "--benchmark-repeats",
                    "2",
                ],
                "pre_patch_build_spec": None,
                "post_patch_build_spec": None,
            }
        return {
            "target_file": "workloads/reference/triton_attention_score_weak_baseline.py",
            "patch_text": _load_text(root, "workloads/reference/triton_attention_score_reformulate_candidate.py"),
            "patch_intent": "replace the naive causal attention-score baseline with the tiled Triton implementation",
            "patch_expected_effect": "preserve score semantics while improving the implementation strategy and performance profile",
            "patch_kind": "perf_transform",
            "transition_kind": "reformulated",
            "eval_command": [
                "python3",
                "workloads/reference/triton_attention_score_weak_baseline.py",
                "--benchmark-repeats",
                "2",
            ],
            "pre_patch_build_spec": None,
            "post_patch_build_spec": "workloads/reference/triton_attention_score_kernel.py:get_build_spec",
        }
    return resolve_optimize_patch_plan(root, task_id, variant)


def _finalize_budget(state: AgentEnvironmentState) -> AgentEnvironmentState:
    if state.step_budget_remaining > 0:
        return state
    return state.model_copy(update={"done": True, "terminal_state": "budget_exhausted"})


def _resolve_task(root: Path, task_ref: str):
    return TaskRegistry(root).get(task_ref)


def _load_baseline(root: Path, task_ref: str) -> BaselineSpec | None:
    task = _resolve_task(root, task_ref)
    if not task.baseline_ref:
        return None
    path = root / task.baseline_ref
    if not path.exists():
        return None
    return BaselineSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _step_has_artifact_prefix(step: TrajectoryStep, prefixes: tuple[str, ...]) -> bool:
    refs = list(step.observation.artifact_refs) + list(step.action.artifact_refs)
    return any(str(ref).startswith(prefixes) for ref in refs)


def _episode_evidence_flags(steps: list[TrajectoryStep]) -> dict[str, bool]:
    build_keys = {"build_record", "tri_view", "build_projection"}
    profile_keys = {"profile_summary", "sanitizer_summary", "bottleneck_card"}
    has_build_evidence = False
    has_profile_evidence = False
    for step in steps:
        projection = step.observation.projection if isinstance(step.observation.projection, dict) else {}
        if step.action.action_type in {"build", "inspect_build"}:
            has_build_evidence = True
        if step.action.action_type in {"inspect_profile"}:
            has_profile_evidence = True
        if build_keys & set(projection):
            has_build_evidence = True
        if profile_keys & set(projection):
            has_profile_evidence = True
        if _step_has_artifact_prefix(step, ("build/",)):
            has_build_evidence = True
        if _step_has_artifact_prefix(step, ("profiles/", "traces/", "bottlenecks/", "sanitize/")):
            has_profile_evidence = True
    return {
        "has_build_evidence": has_build_evidence,
        "has_profile_evidence": has_profile_evidence,
    }


def _episode_governance_kind(training_example_kind: str, patch_bearing: bool) -> str:
    if training_example_kind in {"positive_sft_example", "positive_rl_trace"}:
        return "usable_positive_sft"
    if training_example_kind == "negative_debug_example":
        return "usable_negative_debug"
    if training_example_kind == "negative_reformulate_example":
        return "usable_negative_transition"
    if training_example_kind == "benchmark_only":
        return "benchmark_only"
    if patch_bearing:
        return "candidate_transition_only"
    return "unusable"


def _derive_episode_training_readiness(
    *,
    task_verb: str,
    terminal_state: str,
    final_readiness: dict[str, object],
    steps: list[TrajectoryStep],
) -> EpisodeReadinessReport:
    flags = _episode_evidence_flags(steps)
    patch_bearing = any(step.action.action_type == "patch_candidate" for step in steps)
    has_compare = any(step.action.action_type == "compare" for step in steps)
    benchmark_reasons = list(final_readiness.get("benchmark_reporting", {}).get("reasons", [])) if isinstance(final_readiness.get("benchmark_reporting"), dict) else []
    sft_reasons = list(final_readiness.get("sft_collection", {}).get("reasons", [])) if isinstance(final_readiness.get("sft_collection"), dict) else []
    rl_reasons = list(final_readiness.get("rl_reward_trace", {}).get("reasons", [])) if isinstance(final_readiness.get("rl_reward_trace"), dict) else []
    benchmark_ready = bool(final_readiness.get("benchmark_reporting", {}).get("eligible", False)) if isinstance(final_readiness.get("benchmark_reporting"), dict) else False
    sft_ready = bool(final_readiness.get("sft_collection", {}).get("eligible", False)) if isinstance(final_readiness.get("sft_collection"), dict) else False
    rl_ready = bool(final_readiness.get("rl_reward_trace", {}).get("eligible", False)) if isinstance(final_readiness.get("rl_reward_trace"), dict) else False
    training_example_kind = str(final_readiness.get("training_example_kind", "unusable"))
    notes: list[str] = []

    if not sft_ready:
        if terminal_state == "success":
            if task_verb in {"diagnose", "debug"} and (flags["has_build_evidence"] or flags["has_profile_evidence"]):
                sft_ready = True
                training_example_kind = "positive_sft_example"
                sft_reasons = []
                notes.append("episode_promoted_for_diagnostic_evidence")
            elif task_verb in {"optimize", "reformulate"} and flags["has_build_evidence"]:
                sft_ready = True
                training_example_kind = "positive_sft_example"
                sft_reasons = []
                notes.append("episode_promoted_for_build_evidence")
            elif task_verb == "reformulate" and patch_bearing and has_compare and training_example_kind == "benchmark_only":
                sft_ready = True
                training_example_kind = "negative_reformulate_example"
                sft_reasons = ["perf_gate_failed_or_build_evidence_missing", "reformulate_episode_contains_transition_evidence"]
                notes.append("episode_promoted_as_negative_reformulate_example")
        elif task_verb == "debug" and (flags["has_build_evidence"] or flags["has_profile_evidence"]):
            sft_ready = True
            training_example_kind = "negative_debug_example"
            sft_reasons = ["terminal_not_success", "debug_episode_contains_relevant_evidence"]
            notes.append("episode_retained_as_negative_debug_example")
        elif task_verb == "reformulate" and patch_bearing and (flags["has_build_evidence"] or has_compare):
            sft_ready = True
            training_example_kind = "negative_reformulate_example"
            sft_reasons = ["terminal_not_success", "reformulate_episode_contains_transition_evidence"]
            notes.append("episode_retained_as_negative_reformulate_example")

    governance_kind = _episode_governance_kind(training_example_kind, patch_bearing)
    reasons = sorted(
        {
            *[str(reason) for reason in benchmark_reasons],
            *[str(reason) for reason in sft_reasons],
            *[str(reason) for reason in rl_reasons],
            *(["patch_bearing"] if patch_bearing else []),
            *(["has_build_evidence"] if flags["has_build_evidence"] else []),
            *(["has_profile_evidence"] if flags["has_profile_evidence"] else []),
        }
    )
    return EpisodeReadinessReport(
        episode_governance_kind=governance_kind,
        training_example_kind=training_example_kind,
        benchmark_collection=ReadinessDecision(eligible=benchmark_ready, reasons=benchmark_reasons),
        sft_collection=ReadinessDecision(eligible=sft_ready, reasons=sft_reasons),
        rl_reward_trace=ReadinessDecision(eligible=rl_ready, reasons=rl_reasons),
        has_build_evidence=flags["has_build_evidence"],
        has_profile_evidence=flags["has_profile_evidence"],
        patch_bearing=patch_bearing,
        reasons=reasons,
        notes=notes,
    )


def _episode_optimize_trace_snapshots(steps: list[TrajectoryStep]) -> OptimizeTraceSnapshots | None:
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
            compare_payload = {
                key: projection.get(key)
                for key in ["optimize_delta_summary", "candidate_delta_brief", "recommended_next_actions", "summary_lines"]
                if projection.get(key) is not None
            }
            if compare_payload:
                compare_snapshots.append(
                    OptimizeTraceSnapshot(
                        step_index=step.step_index,
                        action_type=step.action.action_type,
                        snapshot_kind="compare",
                        run_id=step.observation.run_id,
                        payload=compare_payload,
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


def _trace_usability_from_terminal_state(terminal_state: str, *, task_success: bool, compare_used: bool, patch_bearing: bool) -> str:
    if task_success:
        return "trainable_positive"
    if terminal_state in {"negative_trace_complete", "multi_candidate_negative_complete"} and compare_used and patch_bearing:
        return "trainable_negative"
    return "analysis_only"


def _derive_reward_ledger(
    *,
    task_ref: str,
    task_verb: str,
    terminal_state: str,
    steps: list[TrajectoryStep],
    final_eval_envelope: dict[str, object],
) -> RewardLedger:
    correctness_passed = str(final_eval_envelope.get("correctness_gate", "fail")) == "pass"
    determinism_passed = str(final_eval_envelope.get("determinism_gate", "fail")) == "pass"
    perf_gate = str(final_eval_envelope.get("perf_gate", "not_run"))
    task_success = terminal_state == "success"
    compare_used = any(step.action.action_type == "compare" for step in steps)
    patch_bearing = any(step.action.action_type == "patch_candidate" for step in steps)
    trace_usability = _trace_usability_from_terminal_state(
        terminal_state,
        task_success=task_success,
        compare_used=compare_used,
        patch_bearing=patch_bearing,
    )
    total_reward_components = {
        "task_success": 0.6 if task_success else 0.0,
        "correctness": 0.25 if correctness_passed else 0.0,
        "determinism": 0.1 if determinism_passed else 0.0,
        "perf_improvement": 0.05 if perf_gate == "pass" else 0.0,
    }
    total_shaping_components = {
        "tool_cost": 0.0,
        "compare_use_bonus": 0.0,
        "candidate_regression_penalty": 0.0,
        "best_known_supersede_bonus": 0.0,
        "revert_recovery_bonus": 0.0,
        "promote_closeout_bonus": 0.0,
        "near_miss_progress_bonus": 0.0,
    }
    entries: list[RewardLedgerEntry] = []
    compare_bonus_remaining = 0.02 if task_verb in {"optimize", "reformulate"} else 0.0
    regression_penalty_cap = -0.08
    saw_regression_against_best_known = False
    for step in steps:
        reward_components: dict[str, float] = {}
        shaping_components: dict[str, float] = {}
        notes: list[str] = []
        tool_cost = float(step.reward_components.get("tool_cost", 0.0))
        if tool_cost:
            shaping_components["tool_cost"] = round(tool_cost, 4)
            total_shaping_components["tool_cost"] += shaping_components["tool_cost"]
        if step.action.action_type == "compare":
            projection = step.observation.projection if isinstance(step.observation.projection, dict) else {}
            optimize_delta = projection.get("optimize_delta_summary", {})
            candidate_delta = projection.get("candidate_delta_brief", {})
            compare_has_optimize_evidence = isinstance(optimize_delta, dict) and bool(optimize_delta)
            compare_has_candidate_evidence = isinstance(candidate_delta, dict) and bool(candidate_delta)
            if compare_bonus_remaining > 0 and patch_bearing and (compare_has_optimize_evidence or compare_has_candidate_evidence):
                shaping_components["compare_use_bonus"] = compare_bonus_remaining
                total_shaping_components["compare_use_bonus"] += compare_bonus_remaining
                notes.append("compare_bonus_awarded")
                compare_bonus_remaining = 0.0
            if isinstance(optimize_delta, dict):
                correctness_change = str(optimize_delta.get("correctness_change", "unknown"))
                perf_change = str(optimize_delta.get("perf_change", "unknown"))
                penalty = 0.0
                if correctness_change == "regressed":
                    penalty -= 0.06
                if perf_change == "regressed":
                    penalty -= 0.02
                if penalty:
                    bounded_penalty = max(regression_penalty_cap - total_shaping_components["candidate_regression_penalty"], penalty)
                    bounded_penalty = round(bounded_penalty, 4)
                    shaping_components["candidate_regression_penalty"] = bounded_penalty
                    total_shaping_components["candidate_regression_penalty"] += bounded_penalty
                    notes.append("candidate_regression_penalty_applied")
                    saw_regression_against_best_known = True
                best_known_bonus = 0.0
                if correctness_change == "recovered":
                    best_known_bonus += 0.015
                if perf_change == "improved":
                    best_known_bonus += 0.01
                if best_known_bonus:
                    bounded_bonus = round(min(0.03 - total_shaping_components["best_known_supersede_bonus"], best_known_bonus), 4)
                    if bounded_bonus > 0:
                        shaping_components["best_known_supersede_bonus"] = bounded_bonus
                        total_shaping_components["best_known_supersede_bonus"] += bounded_bonus
                        notes.append("best_known_supersede_bonus_awarded")
        if step.action.action_type == "branch_candidate":
            notes.append("branch_effect_tracked")
        if step.action.action_type == "revert_candidate":
            notes.append("revert_effect_tracked")
            if saw_regression_against_best_known:
                shaping_components["revert_recovery_bonus"] = 0.01
                total_shaping_components["revert_recovery_bonus"] += 0.01
                notes.append("revert_recovery_bonus_awarded")
                saw_regression_against_best_known = False
        if step.action.action_type == "promote_candidate":
            notes.append("promote_effect_tracked")
            if task_success:
                shaping_components["promote_closeout_bonus"] = 0.02
                total_shaping_components["promote_closeout_bonus"] += 0.02
                notes.append("promote_closeout_bonus_awarded")
        if step.terminal:
            if (
                correctness_passed
                and
                not task_success
                and terminal_state in {"two_attempt_positive_complete", "three_attempt_positive_complete", "post_patch_eval_failed"}
                and compare_used
                and patch_bearing
                and perf_gate in {"fail", "blocked", "not_run"}
            ):
                shaping_components["near_miss_progress_bonus"] = 0.03
                total_shaping_components["near_miss_progress_bonus"] += 0.03
                notes.append("near_miss_progress_bonus_awarded")
            reward_components = dict(total_reward_components)
            notes.append(f"terminal_state:{terminal_state}")
        entries.append(
            RewardLedgerEntry(
                step_index=step.step_index,
                action_type=step.action.action_type,
                reward_components=reward_components,
                shaping_components=shaping_components,
                total_delta=round(sum(reward_components.values()) + sum(shaping_components.values()), 4),
                notes=notes,
            )
        )
    return RewardLedger(
        task_id=task_ref,
        task_verb=task_verb,
        task_outcome="success" if task_success else terminal_state,
        trace_usability=trace_usability,
        entries=entries,
        total_reward_components={key: round(value, 4) for key, value in total_reward_components.items()},
        total_shaping_components={key: round(value, 4) for key, value in total_shaping_components.items()},
        total_reward=round(sum(total_reward_components.values()) + sum(total_shaping_components.values()), 4),
    )


def _derive_episode_learning_reward(
    *,
    task_ref: str,
    task_verb: str,
    terminal_state: str,
    steps: list[TrajectoryStep],
    final_eval_envelope: dict[str, object],
) -> LearningRewardTrace:
    compare_used = any(step.action.action_type == "compare" for step in steps)
    patch_bearing = any(step.action.action_type == "patch_candidate" for step in steps)
    branch_count = sum(1 for step in steps if step.action.action_type == "branch_candidate")
    revert_count = sum(1 for step in steps if step.action.action_type == "revert_candidate")
    promote_count = sum(1 for step in steps if step.action.action_type == "promote_candidate")
    correctness_passed = str(final_eval_envelope.get("correctness_gate", "fail")) == "pass"
    determinism_passed = str(final_eval_envelope.get("determinism_gate", "fail")) == "pass"
    anti_hack_passed = str(final_eval_envelope.get("anti_hack_gate", "fail")) == "pass"
    perf_gate = str(final_eval_envelope.get("perf_gate", "not_run"))
    task_success = terminal_state == "success"
    reward_ledger = _derive_reward_ledger(
        task_ref=task_ref,
        task_verb=task_verb,
        terminal_state=terminal_state,
        steps=steps,
        final_eval_envelope=final_eval_envelope,
    )
    reward_components = dict(reward_ledger.total_reward_components)
    shaping_components = dict(reward_ledger.total_shaping_components)
    notes: list[str] = []
    if perf_gate in {"blocked", "not_run"}:
        notes.append(f"perf_gate:{perf_gate}")
    if task_verb in {"optimize", "reformulate"} and not compare_used:
        notes.append("compare_missing")
    return LearningRewardTrace(
        task_id=task_ref,
        task_verb=task_verb,
        terminal_state=terminal_state,
        task_success=task_success,
        correctness_passed=correctness_passed,
        determinism_passed=determinism_passed,
        anti_hack_passed=anti_hack_passed,
        perf_gate=perf_gate,
        task_outcome="success" if task_success else terminal_state,
        trace_usability=reward_ledger.trace_usability,
        compare_used=compare_used,
        patch_bearing=patch_bearing,
        branch_count=branch_count,
        revert_count=revert_count,
        promote_count=promote_count,
        reward_components=reward_components,
        shaping_components=shaping_components,
        excluded_governance_signals=[
            "evidence_score",
            "required_artifact_completeness",
            "replay_completeness",
            "build_completeness",
            "profile_completeness",
            "provenance_completeness",
            "benchmark_reporting",
            "sft_collection",
            "rl_reward_trace_readiness",
        ],
        total_reward=reward_ledger.total_reward,
        notes=notes,
        reward_ledger=reward_ledger,
    )


def _execute_eval(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
    lineage: dict[str, object] | None = None,
) -> Path:
    task = _resolve_task(root, task_ref)
    doctor_report = collect_doctor_report()
    command_executor = make_executor(executor, root)
    run_spec = build_run_spec(
        task=task,
        backend=backend or (task.allowed_backends[0] if task.allowed_backends else "triton"),
        vendor=vendor or "nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    lineage = lineage or {}
    if lineage.get("parent_run_id") is not None:
        run_spec.parent_run_id = str(lineage["parent_run_id"])
    run_spec.run_id = build_run_id(prefix="eval")
    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)

    from gpu_cockpit.engine.command_runner import run_command

    command_summary = run_command(writer, command, executor=command_executor)
    perf_report = None
    if task.baseline_ref:
        perf_report = run_task_benchmark(
            writer,
            root=root,
            task=task,
            command=command,
            scope="tool.run_benchmark.eval",
            executor=command_executor,
        )

    correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
        writer=writer,
        root=root,
        task=task,
        command=command,
        command_summary=command_summary,
        perf_report=perf_report,
        scan_paths=[],
        determinism_runs=determinism_runs,
        executor=command_executor,
    )

    status = "ok" if envelope.final_score > 0 else "failed"
    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "correctness/correctness.json",
        "correctness/determinism.json",
        "eval/anti_hack_report.json",
        "eval/eval_envelope.json",
        "eval/gate_summary.json",
        "summary.json",
        "summary.md",
    ]
    if doctor_report.hardware_fingerprints:
        key_artifacts.insert(4, "meta/hardware_fingerprint.json")
    if perf_report is not None:
        key_artifacts.extend(["perf/benchmark.json", "perf/raw_timings.json"])
    if command_summary is not None:
        key_artifacts.extend(["command/stdout.txt", "command/stderr.txt", "command/summary.json"])
    write_run_summary(
        writer,
        RunSummary(
            run_id=run_spec.run_id,
            task_id=task.task_id,
            status=status,
            trace_enabled=False,
            backend=run_spec.target_backend,
            vendor=run_spec.target_vendor,
            parent_run_id=run_spec.parent_run_id,
            candidate_id=str(lineage["candidate_id"]) if lineage.get("candidate_id") is not None else None,
            parent_candidate_id=str(lineage["parent_candidate_id"]) if lineage.get("parent_candidate_id") is not None else None,
            patch_present=bool(lineage.get("patch_present", False)),
            patch_kind=str(lineage["patch_kind"]) if lineage.get("patch_kind") is not None else None,
            transition_kind=str(lineage["transition_kind"]) if lineage.get("transition_kind") is not None else None,
            candidate_role=str(lineage["candidate_role"]) if lineage.get("candidate_role") is not None else None,
            candidate_status=str(lineage["candidate_status"]) if lineage.get("candidate_status") is not None else None,
            candidate_origin_kind=str(lineage["candidate_origin_kind"]) if lineage.get("candidate_origin_kind") is not None else None,
            candidate_operation_kind=str(lineage["candidate_operation_kind"]) if lineage.get("candidate_operation_kind") is not None else None,
            candidate_diff_ref=str(lineage["diff_ref"]) if lineage.get("diff_ref") is not None else None,
            exit_code=command_summary.exit_code if command_summary is not None else None,
            duration_ms=command_summary.duration_ms if command_summary is not None else None,
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": run_spec.target_backend,
            "target_vendor": run_spec.target_vendor,
            "determinism_runs": determinism_runs,
        },
        lineage={
            "candidate_id": lineage.get("candidate_id"),
            "parent_candidate_id": lineage.get("parent_candidate_id"),
            "source_run_ref": lineage.get("source_run_ref"),
            "patch_ref": lineage.get("patch_ref"),
            "diff_ref": lineage.get("diff_ref"),
            "transition_ref": lineage.get("transition_ref"),
            "operation_ref": lineage.get("operation_ref"),
            "candidate_role": lineage.get("candidate_role"),
            "candidate_status": lineage.get("candidate_status"),
            "candidate_origin_kind": lineage.get("candidate_origin_kind"),
            "candidate_operation_kind": lineage.get("candidate_operation_kind"),
            "transition_kind": lineage.get("transition_kind"),
        },
    )
    writer.append_event(scope="run", kind="completed", payload={"status": status})
    return run_dir


def _execute_bench(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    warmups: int = 1,
    repeats: int = 5,
    lineage: dict[str, object] | None = None,
) -> Path:
    task = _resolve_task(root, task_ref).model_copy(deep=True)
    task.perf_protocol.warmups = warmups
    task.perf_protocol.repeats = repeats
    doctor_report = collect_doctor_report()
    command_executor = make_executor(executor, root)
    run_spec = build_run_spec(
        task=task,
        backend=task.allowed_backends[0] if task.allowed_backends else "triton",
        vendor="nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    lineage = lineage or {}
    if lineage.get("parent_run_id") is not None:
        run_spec.parent_run_id = str(lineage["parent_run_id"])
    run_spec.run_id = build_run_id(prefix="bench")
    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)
    perf = run_task_benchmark(writer, root=root, task=task, command=command, executor=command_executor)
    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "perf/benchmark.json",
        "perf/raw_timings.json",
        "summary.json",
        "summary.md",
    ]
    if doctor_report.hardware_fingerprints:
        key_artifacts.insert(4, "meta/hardware_fingerprint.json")
    write_run_summary(
        writer,
        RunSummary(
            run_id=run_spec.run_id,
            task_id=task.task_id,
            status="ok",
            trace_enabled=False,
            backend=run_spec.target_backend,
            vendor=run_spec.target_vendor,
            parent_run_id=run_spec.parent_run_id,
            candidate_id=str(lineage["candidate_id"]) if lineage.get("candidate_id") is not None else None,
            parent_candidate_id=str(lineage["parent_candidate_id"]) if lineage.get("parent_candidate_id") is not None else None,
            patch_present=bool(lineage.get("patch_present", False)),
            patch_kind=str(lineage["patch_kind"]) if lineage.get("patch_kind") is not None else None,
            transition_kind=str(lineage["transition_kind"]) if lineage.get("transition_kind") is not None else None,
            candidate_role=str(lineage["candidate_role"]) if lineage.get("candidate_role") is not None else None,
            candidate_status=str(lineage["candidate_status"]) if lineage.get("candidate_status") is not None else None,
            candidate_origin_kind=str(lineage["candidate_origin_kind"]) if lineage.get("candidate_origin_kind") is not None else None,
            candidate_operation_kind=str(lineage["candidate_operation_kind"]) if lineage.get("candidate_operation_kind") is not None else None,
            candidate_diff_ref=str(lineage["diff_ref"]) if lineage.get("diff_ref") is not None else None,
            exit_code=0,
            duration_ms=int(perf.steady_state_ms_p50),
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": run_spec.target_backend,
            "target_vendor": run_spec.target_vendor,
            "warmups": warmups,
            "repeats": repeats,
        },
        lineage={
            "candidate_id": lineage.get("candidate_id"),
            "parent_candidate_id": lineage.get("parent_candidate_id"),
            "source_run_ref": lineage.get("source_run_ref"),
            "patch_ref": lineage.get("patch_ref"),
            "diff_ref": lineage.get("diff_ref"),
            "transition_ref": lineage.get("transition_ref"),
            "operation_ref": lineage.get("operation_ref"),
            "candidate_role": lineage.get("candidate_role"),
            "candidate_status": lineage.get("candidate_status"),
            "candidate_origin_kind": lineage.get("candidate_origin_kind"),
            "candidate_operation_kind": lineage.get("candidate_operation_kind"),
            "transition_kind": lineage.get("transition_kind"),
        },
    )
    writer.append_event(scope="run", kind="completed", payload={"status": "ok"})
    return run_dir


def step_environment(
    root: Path,
    state: AgentEnvironmentState,
    *,
    action_name: str,
    task_ref: str | None = None,
    command: list[str] | None = None,
    run_ref: str | None = None,
    lhs_run_ref: str | None = None,
    rhs_run_ref: str | None = None,
    section: str = "summary",
    query: str | None = None,
    adapter_name: str | None = None,
    case_id: str | None = None,
    patch_target_file: str | None = None,
    patch_text: str | None = None,
    patch_intent: str | None = None,
    patch_expected_effect: str | None = None,
    patch_kind: str = "bug_fix",
    patch_transition_kind: str = "patch_applied",
    candidate_attempt_index: int | None = None,
    candidate_attempt_reason: str | None = None,
    branch_label: str | None = None,
    revert_target_candidate_id: str | None = None,
    promote_label: str | None = None,
    triton_build_spec: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
    warmups: int = 1,
    repeats: int = 5,
) -> tuple[AgentEnvironmentState, TrajectoryStep]:
    if state.done:
        raise RuntimeError("Environment is already terminal.")
    if state.step_budget_remaining <= 0:
        raise RuntimeError("Environment step budget is exhausted.")

    action_metadata: dict[str, object] = {"section": section}
    artifact_refs: list[str] = []
    observation: TrajectoryObservation
    reward_components: dict[str, float] = {}
    reward_total = 0.0
    next_state = state

    if action_name == "inspect_build":
        action_name = "inspect"
        section = "build"
        action_metadata["requested_action"] = "inspect_build"
    elif action_name == "inspect_quality":
        action_name = "inspect"
        section = "quality"
        action_metadata["requested_action"] = "inspect_quality"
    elif action_name == "inspect_profile":
        action_name = "inspect"
        section = "profile"
        action_metadata["requested_action"] = "inspect_profile"

    if action_name == "knowledge_query":
        task = _resolve_task(root, task_ref or state.task_id)
        rows = query_knowledge(
            root,
            query=query or f"{task.operator_family} {task.verb} {' '.join(task.feature_requirements)}",
            operator_family=task.operator_family,
            verb=task.verb,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            limit=5,
            prefer_mixed=True,
        )
        observation = TrajectoryObservation(
            observation_type="knowledge_results",
            task_id=task.task_id,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            vendor=vendor,
            projection={"results": rows},
        )
        reward_components = {"tool_cost": -_tool_cost("knowledge_query")}
        reward_total = reward_components["tool_cost"]
        next_state = _update_state_after_info(state)
    elif action_name == "adapter_show":
        if not adapter_name or not case_id:
            raise RuntimeError("adapter_show requires adapter_name and case_id.")
        adapter = get_adapter(adapter_name)
        case = adapter.load_case(root, case_id)
        task = adapter.load_task(root, case_id)
        observation = TrajectoryObservation(
            observation_type="adapter_case",
            task_id=task.task_id,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            projection={
                "case": case.model_dump(mode="json"),
                "task": task.model_dump(mode="json"),
            },
        )
        reward_components = {"tool_cost": -_tool_cost("adapter_show")}
        reward_total = reward_components["tool_cost"]
        next_state = _update_state_after_info(state)
    elif action_name == "run":
        if not task_ref or not command:
            raise RuntimeError("run requires task_ref and command.")
        lineage = _lineage_from_candidate_state(root, state)
        run_dir = run_task(
            root=root,
            task_ref=task_ref,
            command=command,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            lineage=lineage,
        )
        payload = inspect_run(root, str(run_dir), section=section)
        observation = TrajectoryObservation(
            observation_type="run_bundle_projection",
            run_id=str(payload.get("run_id")),
            task_id=str(payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(payload.get("backend")),
            vendor=str(payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(payload.get("key_artifacts", [])),
            salient_artifact_refs=_salient_artifacts("run_bundle_projection", payload, list(payload.get("key_artifacts", []))),
            projection=payload,
        )
        reward_components = {
            "completion": 1.0 if payload.get("status") == "ok" else 0.0,
            "tool_cost": -_tool_cost("run"),
        }
        action_metadata["run_ref"] = str(run_dir)
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(payload.get("run_id")))
    elif action_name == "build":
        if not task_ref:
            raise RuntimeError("build requires task_ref.")
        lineage = _lineage_from_candidate_state(root, state)
        run_dir = run_task(
            root=root,
            task_ref=task_ref,
            command=command or [],
            emit_disassembly=True,
            triton_build_spec=triton_build_spec,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            lineage=lineage,
        )
        payload = inspect_run(root, str(run_dir), section="build")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        observation = TrajectoryObservation(
            observation_type="build_projection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            salient_artifact_refs=_salient_artifacts("build_projection", payload, list(summary_payload.get("key_artifacts", []))),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("build")}
        action_metadata["run_ref"] = str(run_dir)
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id")))
        if state.current_candidate_id is not None:
            next_state = next_state.model_copy(update={"current_candidate_status": "build_passed"})
    elif action_name == "patch_candidate":
        effective_task_ref = task_ref or state.task_id
        if not patch_target_file or patch_text is None or not patch_intent:
            raise RuntimeError("patch_candidate requires patch_target_file, patch_text, and patch_intent.")
        run_dir, applied_patch, candidate_state, transition = apply_patch_candidate(
            root,
            task_ref=effective_task_ref,
            target_file=patch_target_file,
            replacement_text=patch_text,
            intent=patch_intent,
            expected_effect=patch_expected_effect,
            patch_kind=patch_kind,
            transition_kind=patch_transition_kind,  # type: ignore[arg-type]
            parent_run_ref=state.last_run_ref,
            parent_run_id=state.last_run_id,
            parent_candidate_id=state.current_candidate_id,
            policy_pack=policy_pack,
            backend=backend,
            vendor=vendor,
            executor=executor,
        )
        payload = inspect_run(root, str(run_dir), section="transition")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        artifact_refs = list(summary_payload.get("key_artifacts", []))
        action_metadata.update(
            {
                "patch_target_file": patch_target_file,
                "patch_kind": patch_kind,
                "patch_intent": patch_intent,
                "patch_expected_effect": patch_expected_effect,
                "input_candidate_id": state.current_candidate_id,
                "output_candidate_id": candidate_state.candidate_id,
                "transition_kind": transition.transition_kind,
                "patch_hash": applied_patch.patch_hash,
            }
        )
        observation = TrajectoryObservation(
            observation_type="candidate_patch",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="candidate/transition.json",
            artifact_refs=artifact_refs,
            salient_artifact_refs=_salient_artifacts("candidate_patch", payload, artifact_refs),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("patch_candidate")}
        reward_total = reward_components["tool_cost"]
        action_metadata["run_ref"] = str(run_dir)
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "current_candidate_id": candidate_state.candidate_id,
                "current_candidate_parent_id": candidate_state.parent_candidate_id,
                "current_candidate_run_ref": str(run_dir),
                "current_candidate_status": candidate_state.status,
                "current_candidate_attempt_index": candidate_attempt_index,
                "candidate_history": [*state.candidate_history, candidate_state.candidate_id],
                "candidate_run_history": [*state.candidate_run_history, str(run_dir)],
                "candidate_lineage_events": [
                    *state.candidate_lineage_events,
                    _candidate_lineage_event(
                        action_name="patch_candidate",
                        candidate_id=candidate_state.candidate_id,
                        parent_candidate_id=candidate_state.parent_candidate_id,
                        source_candidate_id=candidate_state.source_candidate_id,
                        run_ref=str(run_dir),
                        status=candidate_state.status,
                        transition_kind=transition.transition_kind,
                        summary=patch_intent,
                        candidate_role=candidate_state.candidate_role,
                        candidate_attempt_index=candidate_attempt_index,
                        metadata={
                            "patch_kind": applied_patch.patch_kind,
                            "patch_hash": applied_patch.patch_hash,
                            "candidate_attempt_reason": candidate_attempt_reason,
                        },
                    ),
                ],
                "metadata": {
                    **state.metadata,
                    "last_patch_hash": applied_patch.patch_hash,
                    "last_patch_kind": applied_patch.patch_kind,
                    "last_candidate_attempt_reason": candidate_attempt_reason,
                    "stale_perf_invalidated": True,
                    "stale_perf_invalidation_reason": "patch_candidate",
                },
            }
        )
    elif action_name == "branch_candidate":
        effective_task_ref = task_ref or state.task_id
        branch_intent = patch_intent or "branch the current candidate to explore an alternate optimization path"
        run_dir, candidate_state, transition = branch_candidate(
            root,
            task_ref=effective_task_ref,
            intent=branch_intent,
            branch_label=branch_label,
            expected_effect=patch_expected_effect,
            parent_run_ref=state.current_candidate_run_ref or state.last_run_ref,
            parent_run_id=state.current_candidate_run_ref and Path(state.current_candidate_run_ref).name or state.last_run_id,
            parent_candidate_id=state.current_candidate_id,
            policy_pack=policy_pack,
            backend=backend,
            vendor=vendor,
            executor=executor,
        )
        payload = inspect_run(root, str(run_dir), section="transition")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        artifact_refs = list(summary_payload.get("key_artifacts", []))
        action_metadata.update(
            {
                "branch_label": branch_label,
                "patch_intent": branch_intent,
                "patch_expected_effect": patch_expected_effect,
                "input_candidate_id": state.current_candidate_id,
                "output_candidate_id": candidate_state.candidate_id,
                "transition_kind": transition.transition_kind,
            }
        )
        observation = TrajectoryObservation(
            observation_type="candidate_branch",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="candidate/transition.json",
            artifact_refs=artifact_refs,
            salient_artifact_refs=_salient_artifacts("candidate_patch", payload, artifact_refs),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("branch_candidate")}
        reward_total = reward_components["tool_cost"]
        action_metadata["run_ref"] = str(run_dir)
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "current_candidate_id": candidate_state.candidate_id,
                "current_candidate_parent_id": candidate_state.parent_candidate_id,
                "current_candidate_run_ref": str(run_dir),
                "current_candidate_status": candidate_state.status,
                "current_candidate_attempt_index": state.current_candidate_attempt_index,
                "candidate_history": [*state.candidate_history, candidate_state.candidate_id],
                "candidate_run_history": [*state.candidate_run_history, str(run_dir)],
                "candidate_lineage_events": [
                    *state.candidate_lineage_events,
                    _candidate_lineage_event(
                        action_name="branch_candidate",
                        candidate_id=candidate_state.candidate_id,
                        parent_candidate_id=candidate_state.parent_candidate_id,
                        source_candidate_id=candidate_state.source_candidate_id,
                        run_ref=str(run_dir),
                        status=candidate_state.status,
                        transition_kind=transition.transition_kind,
                        summary=branch_intent,
                        candidate_role=candidate_state.candidate_role,
                        candidate_attempt_index=state.current_candidate_attempt_index,
                        metadata={"branch_label": branch_label},
                    ),
                ],
                "metadata": {
                    **state.metadata,
                    "last_candidate_operation": "branch",
                    "last_branch_label": branch_label,
                    "stale_perf_invalidated": True,
                    "stale_perf_invalidation_reason": "branch_candidate",
                },
            }
        )
    elif action_name == "revert_candidate":
        effective_task_ref = task_ref or state.task_id
        revert_intent = patch_intent or "revert the candidate back to a prior workspace state"
        effective_target_file = patch_target_file
        effective_patch_text = patch_text
        if (effective_target_file is None or effective_patch_text is None) and state.current_candidate_run_ref:
            current_payload = inspect_run(root, state.current_candidate_run_ref, section="transition")
            candidate_projection = current_payload.get("candidate_projection", {}) if isinstance(current_payload, dict) else {}
            applied_patch = candidate_projection.get("applied_patch", {}) if isinstance(candidate_projection, dict) else {}
            candidate_state_payload = candidate_projection.get("candidate_state", {}) if isinstance(candidate_projection, dict) else {}
            if effective_target_file is None:
                effective_target_file = applied_patch.get("target_file") if isinstance(applied_patch, dict) else None
                if effective_target_file is None and isinstance(candidate_state_payload, dict):
                    diff_summary = candidate_state_payload.get("diff_summary", {})
                    if isinstance(diff_summary, dict):
                        effective_target_file = diff_summary.get("primary_target_file")
            if effective_patch_text is None and state.current_candidate_run_ref:
                run_dir = resolve_run_dir(root, state.current_candidate_run_ref)
                before_path = run_dir / "patches" / "before.txt"
                if before_path.exists():
                    effective_patch_text = before_path.read_text(encoding="utf-8")
                else:
                    before_candidate_path = run_dir / "candidate" / "before.txt"
                    if before_candidate_path.exists():
                        effective_patch_text = before_candidate_path.read_text(encoding="utf-8")
        if not effective_target_file or effective_patch_text is None:
            raise RuntimeError("revert_candidate requires a target file and replacement text, or a revertable current candidate state.")
        run_dir, candidate_state, transition = revert_candidate(
            root,
            task_ref=effective_task_ref,
            target_file=effective_target_file,
            replacement_text=effective_patch_text,
            intent=revert_intent,
            expected_effect=patch_expected_effect,
            revert_target_candidate_id=revert_target_candidate_id,
            parent_run_ref=state.current_candidate_run_ref or state.last_run_ref,
            parent_run_id=state.current_candidate_run_ref and Path(state.current_candidate_run_ref).name or state.last_run_id,
            parent_candidate_id=state.current_candidate_id,
            policy_pack=policy_pack,
            backend=backend,
            vendor=vendor,
            executor=executor,
        )
        payload = inspect_run(root, str(run_dir), section="transition")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        artifact_refs = list(summary_payload.get("key_artifacts", []))
        action_metadata.update(
            {
                "patch_target_file": effective_target_file,
                "patch_intent": revert_intent,
                "patch_expected_effect": patch_expected_effect,
                "input_candidate_id": state.current_candidate_id,
                "output_candidate_id": candidate_state.candidate_id,
                "transition_kind": transition.transition_kind,
                "revert_target_candidate_id": revert_target_candidate_id,
            }
        )
        observation = TrajectoryObservation(
            observation_type="candidate_revert",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="candidate/transition.json",
            artifact_refs=artifact_refs,
            salient_artifact_refs=_salient_artifacts("candidate_patch", payload, artifact_refs),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("revert_candidate")}
        reward_total = reward_components["tool_cost"]
        action_metadata["run_ref"] = str(run_dir)
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "current_candidate_id": candidate_state.candidate_id,
                "current_candidate_parent_id": candidate_state.parent_candidate_id,
                "current_candidate_run_ref": str(run_dir),
                "current_candidate_status": candidate_state.status,
                "current_candidate_attempt_index": state.current_candidate_attempt_index,
                "candidate_history": [*state.candidate_history, candidate_state.candidate_id],
                "candidate_run_history": [*state.candidate_run_history, str(run_dir)],
                "candidate_lineage_events": [
                    *state.candidate_lineage_events,
                    _candidate_lineage_event(
                        action_name="revert_candidate",
                        candidate_id=candidate_state.candidate_id,
                        parent_candidate_id=candidate_state.parent_candidate_id,
                        source_candidate_id=candidate_state.source_candidate_id,
                        run_ref=str(run_dir),
                        status=candidate_state.status,
                        transition_kind=transition.transition_kind,
                        summary=revert_intent,
                        candidate_role=candidate_state.candidate_role,
                        candidate_attempt_index=state.current_candidate_attempt_index,
                        metadata={"revert_target_candidate_id": revert_target_candidate_id, "target_file": effective_target_file},
                    ),
                ],
                "metadata": {
                    **state.metadata,
                    "last_candidate_operation": "revert",
                    "last_revert_target_candidate_id": revert_target_candidate_id,
                    "stale_perf_invalidated": True,
                    "stale_perf_invalidation_reason": "revert_candidate",
                },
            }
        )
    elif action_name == "promote_candidate":
        effective_task_ref = task_ref or state.task_id
        promote_intent = patch_intent or "promote the current candidate as the preferred ready state"
        run_dir, candidate_state, transition = promote_candidate(
            root,
            task_ref=effective_task_ref,
            intent=promote_intent,
            promotion_label=promote_label,
            expected_effect=patch_expected_effect,
            parent_run_ref=state.current_candidate_run_ref or state.last_run_ref,
            parent_run_id=state.current_candidate_run_ref and Path(state.current_candidate_run_ref).name or state.last_run_id,
            parent_candidate_id=state.current_candidate_id,
            policy_pack=policy_pack,
            backend=backend,
            vendor=vendor,
            executor=executor,
        )
        payload = inspect_run(root, str(run_dir), section="transition")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        artifact_refs = list(summary_payload.get("key_artifacts", []))
        action_metadata.update(
            {
                "patch_intent": promote_intent,
                "patch_expected_effect": patch_expected_effect,
                "input_candidate_id": state.current_candidate_id,
                "output_candidate_id": candidate_state.candidate_id,
                "transition_kind": transition.transition_kind,
                "promote_label": promote_label,
            }
        )
        observation = TrajectoryObservation(
            observation_type="candidate_promote",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="candidate/transition.json",
            artifact_refs=artifact_refs,
            salient_artifact_refs=_salient_artifacts("candidate_patch", payload, artifact_refs),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("promote_candidate")}
        reward_total = reward_components["tool_cost"]
        action_metadata["run_ref"] = str(run_dir)
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "current_candidate_id": candidate_state.candidate_id,
                "current_candidate_parent_id": candidate_state.parent_candidate_id,
                "current_candidate_run_ref": str(run_dir),
                "current_candidate_status": candidate_state.status,
                "current_candidate_attempt_index": state.current_candidate_attempt_index,
                "candidate_history": [*state.candidate_history, candidate_state.candidate_id],
                "candidate_run_history": [*state.candidate_run_history, str(run_dir)],
                "candidate_lineage_events": [
                    *state.candidate_lineage_events,
                    _candidate_lineage_event(
                        action_name="promote_candidate",
                        candidate_id=candidate_state.candidate_id,
                        parent_candidate_id=candidate_state.parent_candidate_id,
                        source_candidate_id=candidate_state.source_candidate_id,
                        run_ref=str(run_dir),
                        status=candidate_state.status,
                        transition_kind=transition.transition_kind,
                        summary=promote_intent,
                        candidate_role=candidate_state.candidate_role,
                        candidate_attempt_index=state.current_candidate_attempt_index,
                        metadata={"promote_label": promote_label},
                    ),
                ],
                "metadata": {
                    **state.metadata,
                    "last_candidate_operation": "promote",
                    "last_promote_label": promote_label,
                    "stale_perf_invalidated": True,
                    "stale_perf_invalidation_reason": "promote_candidate",
                },
            }
        )
    elif action_name == "eval":
        if not task_ref or not command:
            raise RuntimeError("eval requires task_ref and command.")
        lineage = _lineage_from_candidate_state(root, state)
        run_dir = _execute_eval(
            root,
            task_ref,
            command,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            determinism_runs=determinism_runs,
            lineage=lineage,
        )
        payload = inspect_run(root, str(run_dir), section=section if section != "summary" else "eval")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        reward_components, reward_total = _derive_eval_rewards(run_dir)
        reward_components["tool_cost"] = -_tool_cost("eval")
        reward_total += reward_components["tool_cost"]
        action_metadata["run_ref"] = str(run_dir)
        observation = TrajectoryObservation(
            observation_type="eval_projection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            salient_artifact_refs=_salient_artifacts("eval_projection", payload, list(summary_payload.get("key_artifacts", []))),
            projection=payload,
        )
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "metadata": {**state.metadata, "last_eval_solved": summary_payload.get("status") == "ok"},
                "comparison_anchor_run_ref": state.comparison_anchor_run_ref or str(run_dir),
                "comparison_anchor_label": state.comparison_anchor_label or "primary_eval",
            }
        )
        if state.current_candidate_id is not None:
            next_state = next_state.model_copy(
                update={
                    "current_candidate_status": "eval_passed" if summary_payload.get("status") == "ok" else "eval_failed",
                    "best_known_candidate_id": state.current_candidate_id if summary_payload.get("status") == "ok" else state.best_known_candidate_id,
                    "best_known_candidate_parent_id": state.current_candidate_parent_id
                    if summary_payload.get("status") == "ok"
                    else state.best_known_candidate_parent_id,
                    "best_known_candidate_run_ref": state.current_candidate_run_ref
                    if summary_payload.get("status") == "ok"
                    else state.best_known_candidate_run_ref,
                    "best_known_candidate_reason": "eval_success" if summary_payload.get("status") == "ok" else state.best_known_candidate_reason,
                }
            )
    elif action_name == "bench":
        if not task_ref or not command:
            raise RuntimeError("bench requires task_ref and command.")
        lineage = _lineage_from_candidate_state(root, state)
        run_dir = _execute_bench(
            root,
            task_ref,
            command,
            executor=executor,
            policy_pack=policy_pack,
            warmups=warmups,
            repeats=repeats,
            lineage=lineage,
        )
        payload = inspect_run(root, str(run_dir), section="summary")
        observation = TrajectoryObservation(
            observation_type="bench_projection",
            run_id=str(payload.get("run_id")),
            task_id=str(payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(payload.get("backend")),
            vendor=str(payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(payload.get("key_artifacts", [])),
            salient_artifact_refs=_salient_artifacts("bench_projection", inspect_run(root, str(run_dir), section="full"), list(payload.get("key_artifacts", []))),
            projection=inspect_run(root, str(run_dir), section="full"),
        )
        reward_components = {"tool_cost": -_tool_cost("bench")}
        action_metadata["run_ref"] = str(run_dir)
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(payload.get("run_id")))
        if state.current_candidate_id is None and state.comparison_anchor_run_ref is None:
            next_state = next_state.model_copy(
                update={
                    "comparison_anchor_run_ref": str(run_dir),
                    "comparison_anchor_label": "baseline_bench",
                }
            )
        if state.current_candidate_id is not None:
            next_state = next_state.model_copy(update={"current_candidate_status": "benchmarked"})
    elif action_name == "inspect":
        target_run_ref = run_ref or state.last_run_ref
        if not target_run_ref:
            raise RuntimeError("inspect requires run_ref or a prior run-producing action.")
        payload = inspect_run(root, target_run_ref, section=section)
        summary_payload = inspect_run(root, target_run_ref, section="summary")
        observation = TrajectoryObservation(
            observation_type="inspection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            salient_artifact_refs=_salient_artifacts("inspection", payload, list(summary_payload.get("key_artifacts", []))),
            projection=payload,
        )
        inspect_action_name = str(action_metadata.get("requested_action", "inspect"))
        reward_components = {"tool_cost": -_tool_cost(inspect_action_name)}
        reward_total = reward_components["tool_cost"]
        next_state = _update_state_after_info(state)
        best_known_updates = _update_best_known_candidate_from_compare(state, payload)
        if best_known_updates:
            next_state = next_state.model_copy(update=best_known_updates)
    elif action_name == "compare":
        lhs = lhs_run_ref or state.last_run_ref
        rhs = rhs_run_ref
        if not lhs or not rhs:
            raise RuntimeError("compare requires lhs_run_ref and rhs_run_ref, or an existing last_run_ref plus rhs_run_ref.")
        payload = compare_runs(root, lhs, rhs).model_dump(mode="json")
        best_known_updates = _update_best_known_candidate_from_compare(state, payload)
        current_worse_than_best_known = False
        if state.best_known_candidate_run_ref and lhs == state.best_known_candidate_run_ref:
            optimize_delta = payload.get("optimize_delta_summary", {})
            if isinstance(optimize_delta, dict):
                current_worse_than_best_known = (
                    str(optimize_delta.get("correctness_change", "unknown")) == "regressed"
                    or str(optimize_delta.get("perf_change", "unknown")) == "regressed"
                )
        payload["regression_against_best_known"] = current_worse_than_best_known
        payload["best_known_candidate_reason"] = state.best_known_candidate_reason
        compare_tree_updates = _compare_tree_updates(state, payload, best_known_updates)
        action_metadata.update({"lhs_run_ref": lhs, "rhs_run_ref": rhs})
        observation = TrajectoryObservation(
            observation_type="comparison",
            run_id=str(rhs),
            task_id=state.task_id,
            salient_artifact_refs=_salient_artifacts("comparison", payload, []),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("compare")}
        reward_total = reward_components["tool_cost"]
        next_state = _update_state_after_info(state).model_copy(
            update={
                "metadata": {
                    **state.metadata,
                    "last_compare_regression_against_best_known": current_worse_than_best_known,
                    "last_compare_anchor_run_ref": lhs,
                }
            }
        )
        if best_known_updates:
            next_state = next_state.model_copy(update=best_known_updates)
        if compare_tree_updates:
            next_state = next_state.model_copy(update=compare_tree_updates)
    elif action_name == "replay":
        target_run_ref = run_ref or state.last_run_ref
        if not target_run_ref:
            raise RuntimeError("replay requires run_ref or a prior run-producing action.")
        payload = validate_run_bundle(root, target_run_ref)
        resolved_run_dir = resolve_run_dir(root, target_run_ref)
        summary_payload = inspect_run(root, str(resolved_run_dir), section="summary")
        artifact_refs = list(payload.get("required_artifacts", []))
        observation = TrajectoryObservation(
            observation_type="replay_validation",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="replay/replay_pack.json",
            artifact_refs=artifact_refs,
            salient_artifact_refs=_salient_artifacts("replay_validation", payload, artifact_refs),
            projection=payload,
        )
        reward_components = {"tool_cost": -_tool_cost("replay")}
        reward_total = reward_components["tool_cost"]
        next_state = _update_state_after_info(state)
    else:
        raise ValueError(f"Unsupported environment action: {action_name}")

    action = TrajectoryAction(
        action_type=str(action_metadata.get("requested_action", action_name)),
        step_kind="tool_call",
        command=list(command or []),
        target_run_id=observation.run_id,
        target_task_id=observation.task_id or state.task_id,
        artifact_refs=artifact_refs,
        metadata=action_metadata,
    )
    step = TrajectoryStep(
        step_index=state.steps_taken,
        step_label=_step_label_for_action(str(action_metadata.get("requested_action", action_name))),
        action=action,
        observation=observation,
        reward_components=reward_components,
        reward_total=reward_total,
        transition_kind=str(action_metadata.get("transition_kind")) if action_metadata.get("transition_kind") is not None else None,
        input_candidate_id=state.current_candidate_id,
        output_candidate_id=next_state.current_candidate_id,
        patch_hash=str(action_metadata.get("patch_hash")) if action_metadata.get("patch_hash") is not None else None,
        recommended_next_actions=_recommended_actions_from_projection(observation.observation_type, observation.projection),
    )
    next_state = _refresh_candidate_tree_state(next_state)
    next_state = _finalize_budget(next_state)
    return next_state, step


def run_scripted_reference_episode(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    policy_id: str = "scripted_reference_v1",
    step_budget: int = 5,
    section: str = "summary",
    include_knowledge: bool = True,
    include_build: bool = False,
    triton_build_spec: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
    workflow: str = "auto",
) -> TrajectoryEpisode:
    task = _resolve_task(root, task_ref)
    steps: list[TrajectoryStep] = []
    patch_variant = "negative" if workflow in {"debug_negative", "reformulate_negative", "optimize_negative"} else "positive"
    scripted_patch_plan = _scripted_patch_plan(root, task.task_id, variant=patch_variant)
    selected_workflow = workflow
    if selected_workflow == "auto":
        if task.verb == "reformulate":
            selected_workflow = "reformulate"
        elif task.verb == "optimize" and scripted_patch_plan:
            selected_workflow = "optimize_patch"
        elif task.verb == "debug":
            selected_workflow = "debug"
        elif task.verb == "diagnose":
            selected_workflow = "diagnose_first"
        else:
            selected_workflow = "standard"
    minimum_budget = 5
    if selected_workflow == "debug":
        # Debug flows now include pre/post patch verification and replay/compare tail.
        minimum_budget = 11
    elif selected_workflow == "reformulate":
        # Reformulate flows need enough room for baseline, patch, verify, and compare.
        minimum_budget = 10
    elif selected_workflow == "optimize_patch":
        minimum_budget = 10
    elif selected_workflow == "debug_negative":
        minimum_budget = 11
    elif selected_workflow == "reformulate_negative":
        minimum_budget = 10
    elif selected_workflow == "optimize_negative":
        minimum_budget = 10
    state = initialize_environment_state(root, task_ref, policy_id=policy_id, step_budget=max(step_budget, minimum_budget))

    if include_knowledge and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="knowledge_query",
            task_ref=task.task_id,
        )
        steps.append(step)

    if include_build and triton_build_spec and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="build",
            task_ref=task.task_id,
            command=[],
            section="build",
            triton_build_spec=triton_build_spec,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
        )
        steps.append(step)

    baseline = _load_baseline(root, task.task_id)
    if selected_workflow in {"debug", "debug_negative"} and scripted_patch_plan and state.step_budget_remaining > 0:
        pre_patch_build_spec = scripted_patch_plan.get("pre_patch_build_spec")
        if pre_patch_build_spec is not None:
            state, step = step_environment(
                root,
                state,
                action_name="build",
                task_ref=task.task_id,
                command=[],
                section="build",
                triton_build_spec=str(pre_patch_build_spec),
                backend=backend,
                vendor=vendor,
                executor=executor,
                policy_pack=policy_pack,
            )
            steps.append(step)

    if selected_workflow in {"reformulate", "reformulate_negative", "optimize_patch", "optimize_negative"} and baseline and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="bench",
            task_ref=task.task_id,
            command=list(baseline.command),
            section="summary",
            executor=executor,
            policy_pack=policy_pack,
        )
        state = state.model_copy(
            update={
                "comparison_anchor_run_ref": state.last_run_ref,
                "comparison_anchor_label": "baseline_bench",
            }
        )
        steps.append(step)

    if selected_workflow in {"debug", "debug_negative"} and scripted_patch_plan and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="eval",
            task_ref=task.task_id,
            command=list(scripted_patch_plan["eval_command"]),
            section="eval",
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            determinism_runs=determinism_runs,
        )
        steps.append(step)

    if selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"} and state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="inspect_quality",
            run_ref=state.last_run_ref,
            section="quality",
        )
        steps.append(step)

    if selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"} and scripted_patch_plan and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="patch_candidate",
            task_ref=task.task_id,
            patch_target_file=str(scripted_patch_plan["target_file"]),
            patch_text=str(scripted_patch_plan["patch_text"]),
            patch_intent=str(scripted_patch_plan["patch_intent"]),
            patch_expected_effect=str(scripted_patch_plan["patch_expected_effect"]),
            patch_kind=str(scripted_patch_plan["patch_kind"]),
            patch_transition_kind=str(scripted_patch_plan["transition_kind"]),
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
        )
        steps.append(step)

    if selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"} and scripted_patch_plan and state.step_budget_remaining > 0:
        post_patch_build_spec = scripted_patch_plan.get("post_patch_build_spec")
        if post_patch_build_spec is not None:
            state, step = step_environment(
                root,
                state,
                action_name="build",
                task_ref=task.task_id,
                command=[],
                section="build",
                triton_build_spec=str(post_patch_build_spec),
                backend=backend,
                vendor=vendor,
                executor=executor,
                policy_pack=policy_pack,
            )
            steps.append(step)

    if selected_workflow in {"reformulate", "reformulate_negative", "optimize_patch", "optimize_negative"} and scripted_patch_plan and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="bench",
            task_ref=task.task_id,
            command=list(scripted_patch_plan["eval_command"]),
            section="summary",
            executor=executor,
            policy_pack=policy_pack,
        )
        steps.append(step)

    if selected_workflow in {"debug_negative", "reformulate_negative", "optimize_negative"} and state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="inspect_quality",
            run_ref=state.last_run_ref,
            section="quality",
        )
        steps.append(step)

    if state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="eval",
            task_ref=task.task_id,
            command=list(scripted_patch_plan["eval_command"])
            if scripted_patch_plan and selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"}
            else command,
            section="eval",
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            determinism_runs=determinism_runs,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="inspect_quality" if selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"} else "inspect",
            run_ref=state.last_run_ref,
            section="quality" if selected_workflow in {"debug", "reformulate", "optimize_patch", "debug_negative", "reformulate_negative", "optimize_negative"} else section,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="replay",
            run_ref=state.last_run_ref,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        previous_runs = [row["run_id"] for row in list_runs(root, task_id=task.task_id, limit=8) if row["run_id"] != state.last_run_id]
        lhs_ref = state.comparison_anchor_run_ref or (previous_runs[0] if previous_runs else None)
        if lhs_ref:
            state, step = step_environment(
                root,
                state,
                action_name="compare",
                lhs_run_ref=lhs_ref,
                rhs_run_ref=state.last_run_ref,
            )
            steps.append(step)

    solved = bool(state.metadata.get("last_eval_solved"))
    saw_eval = any(step.action.action_type == "eval" for step in steps)
    terminal_state = "success" if solved else "failure"
    if state.terminal_state == "budget_exhausted" and not solved and not saw_eval:
        terminal_state = "budget_exhausted"
    if steps:
        last_step = steps[-1]
        steps[-1] = last_step.model_copy(update={"terminal": True, "terminal_state": terminal_state})
    artifact_refs = list(
        dict.fromkeys(
            artifact_ref
            for step in steps
            for artifact_ref in step.observation.artifact_refs
            if artifact_ref
        )
    )
    run_ref_map = {
        str(step.observation.run_id): str(step.action.metadata["run_ref"])
        for step in steps
        if step.observation.run_id and isinstance(step.action.metadata.get("run_ref"), str)
    }
    final_reward = sum(step.reward_total for step in steps)
    patch_steps = [step for step in steps if step.action.action_type == "patch_candidate"]
    compare_steps = [step for step in steps if step.action.action_type == "compare"]
    verification_steps = [step for step in steps if step.step_label == "verification_action"]
    source_run_dir = resolve_run_dir(root, state.last_run_ref) if state.last_run_ref else root
    final_readiness: dict[str, object] = {}
    governance_score: EvidenceQualityReport | None = None
    final_eval_envelope: dict[str, object] = {}
    if state.last_run_ref:
        quality_payload = inspect_run(root, state.last_run_ref, section="quality")
        if isinstance(quality_payload, dict):
            final_readiness = dict(quality_payload.get("evidence_quality", {}))
            governance_payload = quality_payload.get("governance_score")
            if isinstance(governance_payload, dict):
                governance_score = EvidenceQualityReport.model_validate(governance_payload)
        eval_payload = inspect_run(root, state.last_run_ref, section="eval")
        if isinstance(eval_payload, dict):
            eval_envelope = eval_payload.get("eval_envelope")
            if isinstance(eval_envelope, dict):
                final_eval_envelope = dict(eval_envelope)
    episode_readiness = _derive_episode_training_readiness(
        task_verb=task.verb,
        terminal_state=terminal_state,
        final_readiness=final_readiness,
        steps=steps,
    )
    learning_reward_trace = _derive_episode_learning_reward(
        task_ref=task.task_id,
        task_verb=task.verb,
        terminal_state=terminal_state,
        steps=steps,
        final_eval_envelope=final_eval_envelope,
    )
    optimize_trace_snapshots = _episode_optimize_trace_snapshots(steps)
    return TrajectoryEpisode(
        episode_id=state.episode_id,
        created_at=datetime.now(tz=UTC),
        policy_id=policy_id,
        task_id=task.task_id,
        task_verb=task.verb,
        operator_family=task.operator_family,
        source_run_id=state.last_run_id or "none",
        source_run_ref=str(source_run_dir),
        episode_kind="scripted_reference",
        steps=steps,
        final_reward=final_reward,
        terminal_state=terminal_state,
        artifact_refs=artifact_refs,
        governance=episode_readiness,
        governance_score=governance_score,
        learning_reward_trace=learning_reward_trace,
        optimize_trace_snapshots=optimize_trace_snapshots,
        metadata={
            **state.metadata,
            "workflow": selected_workflow,
            "action_names": [step.action.action_type for step in steps],
            "step_budget_total": state.step_budget_total,
            "step_budget_remaining": state.step_budget_remaining,
            "current_candidate_id": state.current_candidate_id,
            "current_candidate_run_ref": state.current_candidate_run_ref,
            "candidate_lineage_depth": 1 if state.current_candidate_id else 0,
            "patch_attempt_count": len(patch_steps),
            "compare_step_count": len(compare_steps),
            "verification_step_count": len(verification_steps),
            "action_cost_total": round(sum(max(0.0, -step.reward_components.get("tool_cost", 0.0)) for step in steps), 4),
            "negative_transition_trace": selected_workflow in {"debug_negative", "reformulate_negative"},
            "run_ref_map": run_ref_map,
            "training_example_kind": episode_readiness.training_example_kind,
            "episode_governance_kind": episode_readiness.episode_governance_kind,
            "episode_governance_reasons": list(episode_readiness.reasons),
            "evidence_score": final_readiness.get("overall_score", 0.0),
            "governance_score_kind": governance_score.score_kind if governance_score is not None else "governance",
            "benchmark_ready": episode_readiness.benchmark_collection.eligible,
            "sft_ready": episode_readiness.sft_collection.eligible,
            "rl_trace_ready": episode_readiness.rl_reward_trace.eligible,
            "episode_build_evidence": episode_readiness.has_build_evidence,
            "episode_profile_evidence": episode_readiness.has_profile_evidence,
            "episode_patch_bearing": episode_readiness.patch_bearing,
            "learning_reward_schema_id": learning_reward_trace.schema_id,
            "learning_reward_total": learning_reward_trace.total_reward,
        },
    )
