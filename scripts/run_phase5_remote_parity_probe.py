from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts.remote_session import RemoteSyncPolicy
from gpu_cockpit.engine.environment import initialize_environment_state, step_environment
from gpu_cockpit.engine.optimize_patch_registry import resolve_optimize_patch_harness
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.executors.local_host_remote_session import LocalHostRemoteSession


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _projection_subset(projection: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "compare_type",
        "candidate_delta_brief",
        "optimize_delta_summary",
        "recommended_next_actions",
        "perf_localization",
        "benchmark_provenance",
        "failure_localization",
    ]
    return {key: projection.get(key) for key in keys if key in projection}


def _normalized_state(summary: dict[str, Any]) -> dict[str, Any]:
    state = dict(summary["state"])
    for key in (
        "current_candidate_id",
        "best_known_candidate_id",
        "comparison_anchor_run_ref",
    ):
        state.pop(key, None)
    return state


def _normalized_compare_projection(summary: dict[str, Any]) -> dict[str, Any]:
    projection = json.loads(json.dumps(summary["compare_projection"]))
    candidate_delta = projection.get("candidate_delta_brief")
    if isinstance(candidate_delta, dict):
        for key in (
            "lhs_candidate_id",
            "rhs_candidate_id",
            "lhs_parent_candidate_id",
            "rhs_parent_candidate_id",
            "parent_candidate_ref",
        ):
            candidate_delta.pop(key, None)
    optimize_delta = projection.get("optimize_delta_summary")
    if isinstance(optimize_delta, dict):
        optimize_delta.pop("perf_p50_delta_ms", None)
    perf = projection.get("perf_localization")
    if isinstance(perf, dict):
        perf.pop("delta_ms", None)
        for side in ("lhs", "rhs"):
            packet = perf.get(side)
            if not isinstance(packet, dict):
                continue
            for key in (
                "baseline_runtime",
                "candidate_runtime",
                "p50",
                "p95",
                "delta",
                "candidate_command_sha256",
                "baseline_command_sha256",
            ):
                packet.pop(key, None)
            split = packet.get("compile_vs_runtime_split")
            if isinstance(split, dict):
                for key in ("cold_compile_ms", "baseline_cold_compile_ms", "steady_state_ms_p50"):
                    split.pop(key, None)
    return projection


def _state_subset(state: Any) -> dict[str, Any]:
    lineage = getattr(state, "candidate_lineage_events", None) or []
    return {
        "current_candidate_id": getattr(state, "current_candidate_id", None),
        "current_candidate_status": getattr(state, "current_candidate_status", None),
        "current_candidate_attempt_index": getattr(state, "current_candidate_attempt_index", None),
        "best_known_candidate_id": getattr(state, "best_known_candidate_id", None),
        "best_known_candidate_reason": getattr(state, "best_known_candidate_reason", None),
        "candidate_history_length": len(list(getattr(state, "candidate_history", []) or [])),
        "candidate_run_history_length": len(list(getattr(state, "candidate_run_history", []) or [])),
        "candidate_lineage_event_count": len(lineage) if isinstance(lineage, list) else 0,
        "comparison_anchor_label": getattr(state, "comparison_anchor_label", None),
        "comparison_anchor_run_ref": getattr(state, "comparison_anchor_run_ref", None),
    }


def _parity_summary(root: Path, task_ref: str) -> dict[str, Any]:
    registry = TaskRegistry(root)
    task = registry.get(task_ref)
    baseline_payload = _read_json(root / str(task.baseline_ref)) if task.baseline_ref else {}
    baseline_command = list(baseline_payload.get("command", []))
    patch = resolve_optimize_patch_harness(root, task_ref, "positive")
    if not isinstance(patch, dict):
        raise RuntimeError(f"No optimize patch harness for parity probe task {task_ref}")

    state = initialize_environment_state(root, task_ref, policy_id="phase5_remote_parity_probe_v1", step_budget=8)

    state, baseline_step = step_environment(
        root,
        state,
        action_name="bench",
        task_ref=task_ref,
        command=baseline_command,
        repeats=2,
        warmups=1,
    )
    baseline_run_ref = state.last_run_ref
    state, patch_step = step_environment(
        root,
        state,
        action_name="patch_candidate",
        task_ref=task_ref,
        patch_target_file=str(patch["patch_target_file"]),
        patch_text=str(patch["patch_text"]),
        patch_intent=str(patch["patch_intent"]),
        patch_expected_effect=str(patch["patch_expected_effect"]),
        patch_kind=str(patch["patch_kind"]),
        patch_transition_kind=str(patch["patch_transition_kind"]),
        candidate_attempt_index=1,
        candidate_attempt_reason="phase5_remote_parity_probe",
    )
    state, candidate_bench_step = step_environment(
        root,
        state,
        action_name="bench",
        task_ref=task_ref,
        command=list(patch["eval_command"]),
        repeats=2,
        warmups=1,
    )
    candidate_run_ref = state.last_run_ref
    state, compare_step = step_environment(
        root,
        state,
        action_name="compare",
        task_ref=task_ref,
        lhs_run_ref=str(baseline_run_ref),
        rhs_run_ref=str(candidate_run_ref),
    )
    state, eval_step = step_environment(
        root,
        state,
        action_name="eval",
        task_ref=task_ref,
        command=list(patch["eval_command"]),
        determinism_runs=2,
    )

    compare_projection = compare_step.observation.projection
    eval_projection = eval_step.observation.projection
    return {
        "task_ref": task_ref,
        "baseline_run_ref": baseline_run_ref,
        "candidate_run_ref": candidate_run_ref,
        "eval_run_ref": state.last_run_ref,
        "state": _state_subset(state),
        "compare_projection": _projection_subset(compare_projection),
        "eval_projection": _projection_subset(eval_projection),
        "reward_components": dict(eval_step.reward_components),
        "reward_total": eval_step.reward_total,
        "step_labels": [
            baseline_step.step_label,
            patch_step.step_label,
            candidate_bench_step.step_label,
            compare_step.step_label,
            eval_step.step_label,
        ],
        "transition_kinds": [
            baseline_step.transition_kind,
            patch_step.transition_kind,
            candidate_bench_step.transition_kind,
            compare_step.transition_kind,
            eval_step.transition_kind,
        ],
    }


def _compare_semantics(local_summary: dict[str, Any], remote_summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "state_equal": _normalized_state(local_summary) == _normalized_state(remote_summary),
        "compare_projection_equal": _normalized_compare_projection(local_summary)
        == _normalized_compare_projection(remote_summary),
        "eval_projection_equal": local_summary["eval_projection"] == remote_summary["eval_projection"],
        "reward_components_equal": local_summary["reward_components"] == remote_summary["reward_components"],
        "step_labels_equal": local_summary["step_labels"] == remote_summary["step_labels"],
        "transition_kinds_equal": local_summary["transition_kinds"] == remote_summary["transition_kinds"],
    }
    return {
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-summary", action="store_true")
    parser.add_argument(
        "--task-ref",
        default="task/attention_score/eval/v1",
        help="Candidate-bearing task to use for the semantic parity probe.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5_remote_parity_v1",
    )
    args = parser.parse_args()

    if args.emit_summary:
        print(json.dumps(_parity_summary(ROOT, args.task_ref), indent=2, sort_keys=True))
        return 0

    local_summary = _parity_summary(ROOT, args.task_ref)

    with tempfile.TemporaryDirectory(prefix="phase5_remote_parity_") as tmp_dir:
        workspace_root = Path(tmp_dir) / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        session = LocalHostRemoteSession(session_id="phase5_local_remote_bridge", workspace_root=workspace_root)
        sync_policy = RemoteSyncPolicy()
        copied = session.sync_tree(
            ROOT,
            Path("."),
            allowlist_roots=sync_policy.allowlist_roots,
            exclude_globs=sync_policy.exclude_globs,
        )
        python_bin = Path(sys.executable)
        remote_env = {
            "VIRTUAL_ENV": str(python_bin.parent.parent),
            "PATH": f"{python_bin.parent}:{os.environ.get('PATH', '')}",
        }
        result = session.run(
            [str(python_bin), "scripts/run_phase5_remote_parity_probe.py", "--emit-summary", "--task-ref", args.task_ref],
            cwd=workspace_root,
            env=remote_env,
            timeout=900,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Remote parity probe failed: {result.stderr.strip()}")
        remote_summary = json.loads(result.stdout)
        session.terminate()

    comparison = _compare_semantics(local_summary, remote_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_id": "phase5_remote_parity_v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "task_ref": args.task_ref,
        "sync_roots": copied,
        "status": comparison["status"],
        "checks": comparison["checks"],
        "local_summary": local_summary,
        "remote_summary": remote_summary,
    }
    (args.out_dir / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
