from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import DoctorReport, ReplayPack, RunSpec, TaskSpec
from gpu_cockpit.engine.evidence import assess_run_evidence
from gpu_cockpit.engine.inspector import resolve_run_dir
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def build_replay_pack(
    run_spec: RunSpec,
    task: TaskSpec,
    doctor_report: DoctorReport,
    command: list[str],
    required_artifacts: list[str],
    command_ref: str | None = None,
    environment_ref: str | None = None,
    lineage: dict[str, Any] | None = None,
) -> ReplayPack:
    hardware_ref = "meta/hardware_fingerprint.json" if doctor_report.hardware_fingerprints else None
    lineage = lineage or {}
    return ReplayPack(
        run_id=run_spec.run_id,
        replay_version="1.0.0",
        image_digest=run_spec.image_digest,
        seed_pack={
            "global_seed": run_spec.seed_pack.global_seed,
            "input_seed": run_spec.seed_pack.input_seed,
        },
        hardware_fingerprint_ref=hardware_ref,
        input_snapshot_ref=environment_ref,
        task_ref=task.task_id,
        commands_ref=command_ref,
        candidate_id=str(lineage["candidate_id"]) if lineage.get("candidate_id") is not None else None,
        parent_candidate_id=str(lineage["parent_candidate_id"]) if lineage.get("parent_candidate_id") is not None else None,
        source_candidate_id=str(lineage["source_candidate_id"]) if lineage.get("source_candidate_id") is not None else None,
        source_run_ref=str(lineage["source_run_ref"]) if lineage.get("source_run_ref") is not None else None,
        patch_ref=str(lineage["patch_ref"]) if lineage.get("patch_ref") is not None else None,
        diff_ref=str(lineage["diff_ref"]) if lineage.get("diff_ref") is not None else None,
        transition_ref=str(lineage["transition_ref"]) if lineage.get("transition_ref") is not None else None,
        operation_ref=str(lineage["operation_ref"]) if lineage.get("operation_ref") is not None else None,
        candidate_role=str(lineage["candidate_role"]) if lineage.get("candidate_role") is not None else None,
        candidate_role_group=str(lineage["candidate_role_group"]) if lineage.get("candidate_role_group") is not None else None,
        candidate_status=str(lineage["candidate_status"]) if lineage.get("candidate_status") is not None else None,
        candidate_tree_depth=int(lineage["candidate_tree_depth"]) if isinstance(lineage.get("candidate_tree_depth"), int) else None,
        candidate_origin_kind=str(lineage["candidate_origin_kind"]) if lineage.get("candidate_origin_kind") is not None else None,
        candidate_operation_kind=str(lineage["candidate_operation_kind"]) if lineage.get("candidate_operation_kind") is not None else None,
        transition_kind=str(lineage["transition_kind"]) if lineage.get("transition_kind") is not None else None,
        best_known_candidate_id=str(lineage["best_known_candidate_id"]) if lineage.get("best_known_candidate_id") is not None else None,
        best_known_candidate_reason=str(lineage["best_known_candidate_reason"]) if lineage.get("best_known_candidate_reason") is not None else None,
        supersede_reason=str(lineage["supersede_reason"]) if lineage.get("supersede_reason") is not None else None,
        branch_state=str(lineage["branch_state"]) if lineage.get("branch_state") is not None else None,
        endgame_recommendation=str(lineage["endgame_recommendation"]) if lineage.get("endgame_recommendation") is not None else None,
        legal_next_actions=[str(item) for item in lineage.get("legal_next_actions", []) if item is not None],
        dominated_candidate_ids=[str(item) for item in lineage.get("dominated_candidate_ids", []) if item is not None],
        active_candidate_ids=[str(item) for item in lineage.get("active_candidate_ids", []) if item is not None],
        archived_candidate_ids=[str(item) for item in lineage.get("archived_candidate_ids", []) if item is not None],
        sibling_candidate_refs=[str(item) for item in lineage.get("sibling_candidate_refs", []) if item is not None],
        required_artifacts=sorted(dict.fromkeys(required_artifacts)),
    )


def write_replay_pack(
    writer: RunBundleWriter,
    task: TaskSpec,
    doctor_report: DoctorReport,
    command: list[str],
    required_artifacts: list[str],
    environment: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> ReplayPack:
    if writer.run_spec is None:
        raise RuntimeError("Run bundle has not been initialized.")

    command_ref = None
    if command:
        artifact = writer.write_artifact(
            relative_path="replay/command.json",
            kind="command_snapshot",
            content=json.dumps({"command": command}, indent=2) + "\n",
            mime="application/json",
            semantic_tags=["replay", "command"],
        )
        command_ref = artifact.path

    environment_ref = None
    if environment:
        artifact = writer.write_artifact(
            relative_path="replay/environment.json",
            kind="environment_snapshot",
            content=json.dumps(environment, indent=2) + "\n",
            mime="application/json",
            semantic_tags=["replay", "environment"],
        )
        environment_ref = artifact.path

    replay_pack = build_replay_pack(
        run_spec=writer.run_spec,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=required_artifacts,
        command_ref=command_ref,
        environment_ref=environment_ref,
        lineage=lineage,
    )
    writer.write_artifact(
        relative_path="replay/replay_pack.json",
        kind="replay_pack",
        content=json.dumps(replay_pack.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["replay", "summary"],
    )
    return replay_pack


def validate_run_bundle(root: Path, run_ref: str) -> dict[str, Any]:
    run_dir = resolve_run_dir(root, run_ref)
    manifest_path = run_dir / "manifest.json"
    events_path = run_dir / "events.jsonl"
    replay_path = run_dir / "replay" / "replay_pack.json"

    checks = {
        "manifest_present": manifest_path.exists(),
        "events_present": events_path.exists(),
        "replay_pack_present": replay_path.exists(),
    }
    required_artifacts: list[str] = []
    missing_artifacts: list[str] = []
    replay_payload: dict[str, Any] | None = None
    if replay_path.exists():
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
        required_artifacts = list(replay_payload.get("required_artifacts", []))
        for rel_path in required_artifacts:
            if not (run_dir / rel_path).exists():
                missing_artifacts.append(rel_path)
        lineage_refs = {
            "patch_ref": replay_payload.get("patch_ref"),
            "diff_ref": replay_payload.get("diff_ref"),
            "transition_ref": replay_payload.get("transition_ref"),
            "operation_ref": replay_payload.get("operation_ref"),
        }
        for check_name, rel_path in lineage_refs.items():
            if rel_path is None:
                checks[check_name] = True
                continue
            checks[check_name] = (run_dir / str(rel_path)).exists()
        candidate_id = replay_payload.get("candidate_id")
        parent_candidate_id = replay_payload.get("parent_candidate_id")
        source_run_ref = replay_payload.get("source_run_ref")
        checks["candidate_lineage_present"] = candidate_id is not None or all(value is None for value in lineage_refs.values())
        checks["parent_candidate_linked"] = parent_candidate_id is None or source_run_ref is not None

    status = "ok" if all(checks.values()) and not missing_artifacts else "failed"
    return {
        "run_dir": str(run_dir),
        "status": status,
        "checks": checks,
        "required_artifacts": required_artifacts,
        "missing_artifacts": missing_artifacts,
        "replay_pack": replay_payload,
        "evidence_quality": assess_run_evidence(run_dir).model_dump(mode="json"),
    }


def export_proof_bundle(root: Path, run_ref: str, out_path: Path | None = None, include_raw: bool = False) -> Path:
    run_dir = resolve_run_dir(root, run_ref)
    run_id = run_dir.name
    destination = out_path or (root / "runs" / f"{run_id}_proof_bundle.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    preferred_paths = [
        "manifest.json",
        "events.jsonl",
        "summary.json",
        "summary.md",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "meta/hardware_fingerprint.json",
        "correctness/correctness.json",
        "correctness/determinism.json",
        "eval/anti_hack_report.json",
        "eval/eval_envelope.json",
        "eval/gate_summary.json",
        "perf/benchmark.json",
        "command/summary.json",
        "traces/system/summary.json",
        "profiles/kernel/summary.json",
        "profiles/kernel/summary.md",
        "sanitize/memcheck_summary.json",
        "sanitize/racecheck_summary.json",
        "sanitize/initcheck_summary.json",
        "sanitize/synccheck_summary.json",
        "bottlenecks/primary.json",
        "build/build_record.json",
        "build/tri_view.json",
        "build/source_map_summary.json",
        "candidate/state.json",
        "candidate/transition.json",
        "candidate/operation.json",
        "patches/applied_patch.json",
        "patches/unified_diff.patch",
        "replay/replay_pack.json",
        "replay/command.json",
        "replay/environment.json",
    ]
    include_paths: list[Path] = []
    if include_raw:
        include_paths = [path for path in run_dir.rglob("*") if path.is_file()]
    else:
        for rel_path in preferred_paths:
            target = run_dir / rel_path
            if target.exists():
                include_paths.append(target)
        artifact_manifest_dir = run_dir / "meta" / "artifacts"
        if artifact_manifest_dir.exists():
            include_paths.extend(sorted(path for path in artifact_manifest_dir.glob("*.json") if path.is_file()))

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(include_paths)):
            archive.write(path, arcname=path.relative_to(run_dir))
    return destination
