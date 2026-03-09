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
) -> ReplayPack:
    hardware_ref = "meta/hardware_fingerprint.json" if doctor_report.hardware_fingerprints else None
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
        required_artifacts=sorted(dict.fromkeys(required_artifacts)),
    )


def write_replay_pack(
    writer: RunBundleWriter,
    task: TaskSpec,
    doctor_report: DoctorReport,
    command: list[str],
    required_artifacts: list[str],
    environment: dict[str, Any] | None = None,
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
        "perf/benchmark.json",
        "command/summary.json",
        "traces/system/summary.json",
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
